"""Web scraper — search, fetch, extract, follow sublinks.

Robust scraping pipeline:
  1. Search via DuckDuckGo HTML endpoint (no API key)
  2. Fetch pages with httpx + proper headers
  3. Extract text with trafilatura (articles) or BeautifulSoup (fallback)
  4. Follow relevant sublinks to configurable depth
  5. Deduplication, rate limiting, content filtering

Used by WebLibrarian to feed web content into the retrieval pipeline.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# File extensions to skip
_SKIP_EXTENSIONS = frozenset(
    ".pdf .jpg .jpeg .png .gif .svg .mp4 .mp3 .zip .tar .gz "
    ".exe .dmg .css .js .woff .woff2 .ttf .ico".split()
)

# Domains to skip (ads, trackers, social)
_SKIP_DOMAINS = frozenset(
    "facebook.com twitter.com instagram.com linkedin.com "
    "youtube.com pinterest.com reddit.com tiktok.com "
    "ads. doubleclick.net googlesyndication.com "
    "analytics. tracker.".split()
)


@dataclass
class ScrapedPage:
    """A fetched and extracted web page."""
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


@dataclass
class ScrapeConfig:
    """Configuration for scraping."""
    max_search_results: int = 5
    max_sublinks: int = 3          # sublinks to follow per page
    max_depth: int = 1             # 0 = seed pages only, 1 = seed + one level
    max_pages: int = 10            # total pages to fetch
    max_chars_per_page: int = 15000
    request_delay: float = 0.5     # seconds between requests (be polite)
    timeout: int = 10


class WebScraper:
    """Searches, fetches, extracts, and follows sublinks."""

    def __init__(self, config: ScrapeConfig | None = None):
        self.config = config or ScrapeConfig()
        self._visited: set[str] = set()
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=self.config.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    # ── Search ───────────────────────────────────────────────────

    def search(self, query: str) -> list[str]:
        """Search DuckDuckGo HTML endpoint, return result URLs."""
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            resp = self._get_client().get(search_url)
            if resp.status_code != 200:
                logger.warning("  SCRAPER search returned %d", resp.status_code)
                return []
        except Exception as e:
            logger.warning("  SCRAPER search failed: %s", e)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        urls: list[str] = []
        seen: set[str] = set()

        # DDG HTML: result links are in <a class="result__a"> or uddg params
        for link in soup.find_all("a", class_="result__a"):
            href = link.get("href", "")
            url = self._extract_ddg_url(href)
            if url and self._is_valid_url(url):
                domain = urlparse(url).netloc
                if domain not in seen:
                    seen.add(domain)
                    urls.append(url)
                    if len(urls) >= self.config.max_search_results:
                        break

        # Fallback: parse uddg= from all links
        if not urls:
            for link in soup.find_all("a", href=True):
                href = link["href"]
                url = self._extract_ddg_url(href)
                if url and self._is_valid_url(url):
                    domain = urlparse(url).netloc
                    if domain not in seen:
                        seen.add(domain)
                        urls.append(url)
                        if len(urls) >= self.config.max_search_results:
                            break

        logger.info("  SCRAPER search '%s' → %d URLs", query[:60], len(urls))
        return urls

    @staticmethod
    def _extract_ddg_url(href: str) -> str | None:
        """Extract actual URL from DDG redirect link."""
        from urllib.parse import unquote, parse_qs, urlparse as up

        if "uddg=" in href:
            parsed = up(href)
            params = parse_qs(parsed.query)
            if "uddg" in params:
                return unquote(params["uddg"][0])
        elif href.startswith("http"):
            return href
        return None

    # ── Fetch + Extract ──────────────────────────────────────────

    def fetch(self, url: str) -> ScrapedPage | None:
        """Fetch a URL, extract text and links."""
        if url in self._visited:
            return None
        if not self._is_valid_url(url):
            return None

        self._visited.add(url)

        # Rate limiting
        if self.config.request_delay > 0:
            time.sleep(self.config.request_delay)

        # Fetch
        try:
            resp = self._get_client().get(url)
            if resp.status_code != 200:
                logger.debug("  SCRAPER %d from %s", resp.status_code, url[:60])
                return None
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None
        except Exception as e:
            logger.debug("  SCRAPER fetch failed %s: %s", url[:60], e)
            return None

        html = resp.text

        # Extract text — try trafilatura first (better for articles)
        text = trafilatura.extract(
            html,
            include_tables=False,
            include_comments=False,
        )

        # Fallback: BeautifulSoup
        if not text or len(text) < 100:
            text = self._bs4_extract(html)

        if not text or len(text) < 50:
            return None

        # Truncate
        if len(text) > self.config.max_chars_per_page:
            text = text[: self.config.max_chars_per_page]

        # Extract title
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Extract sublinks
        links = self._extract_links(soup, url)

        logger.info(
            "  SCRAPER fetched %d chars, %d links from %s",
            len(text), len(links), url[:60],
        )
        return ScrapedPage(url=url, title=title, text=text, links=links)

    @staticmethod
    def _bs4_extract(html: str) -> str:
        """Fallback text extraction with BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Get text from main content areas
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return ""

        text = main.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract valid outgoing links from the page."""
        links: list[str] = []
        base_domain = urlparse(base_url).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Resolve relative URLs
            url = urljoin(base_url, href)

            if not self._is_valid_url(url):
                continue

            # Prefer same-domain links (more likely to be relevant content)
            parsed = urlparse(url)
            if parsed.netloc == base_domain:
                links.insert(0, url)  # prioritize same-domain
            else:
                links.append(url)

        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for url in links:
            if url not in seen and url not in self._visited:
                seen.add(url)
                unique.append(url)

        return unique

    # ── Crawl ────────────────────────────────────────────────────

    def crawl(self, query: str) -> list[ScrapedPage]:
        """Search + fetch + follow sublinks. Returns all scraped pages."""
        self._visited.clear()
        pages: list[ScrapedPage] = []

        # Level 0: search results
        seed_urls = self.search(query)
        if not seed_urls:
            return pages

        # Fetch seed pages
        for url in seed_urls:
            if len(pages) >= self.config.max_pages:
                break
            page = self.fetch(url)
            if page:
                pages.append(page)

        # Level 1+: follow sublinks
        if self.config.max_depth >= 1:
            sublink_urls: list[str] = []
            for page in pages:
                for link in page.links[: self.config.max_sublinks]:
                    if link not in self._visited:
                        sublink_urls.append(link)

            for url in sublink_urls:
                if len(pages) >= self.config.max_pages:
                    break
                page = self.fetch(url)
                if page:
                    pages.append(page)

        logger.info(
            "  SCRAPER crawl complete: %d pages, %d chars total",
            len(pages),
            sum(len(p.text) for p in pages),
        )
        return pages

    # ── Validation ───────────────────────────────────────────────

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if URL is worth fetching."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False

        # Skip file extensions
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
            return False

        # Skip known non-content domains
        netloc = parsed.netloc.lower()
        if any(skip in netloc for skip in _SKIP_DOMAINS):
            return False

        return True
