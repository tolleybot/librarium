"""Web Librarian — searches the web, fetches pages, answers questions.

Same interface as SourceLibrarian. Uses WebScraper to:
  1. Search DuckDuckGo → get seed URLs
  2. Fetch and extract text from result pages
  3. Follow relevant sublinks (configurable depth)
  4. Index all content via SourceLibrarian (BM25 + embeddings)
  5. Answer using the same retrieval pipeline

The Planner doesn't know the difference — it asks a question,
gets facts back.
"""

from __future__ import annotations

import logging
from librarium.llm import LLMClient

from librarium.models import Fact, LibrarianAnswer
from librarium.scraper import ScrapeConfig, WebScraper
from librarium.chunking import CHUNK_OVERLAP, CHUNK_SIZE
from librarium.source import (
    CHARS_PER_TOKEN,
    CHUNK_THRESHOLD,
    EMBED_MODEL,
    EXPAND_QUERY,
    EXPANSION_STYLE,
    EXPANSION_VARIANTS,
    CITE_SOURCES,
    EXTRACT_FACTS,
    EXTRACT_FIELDS,
    FACT_MIX,
    MAX_CONCURRENT_PREFIXES,
    MAX_CONTEXT_TOKENS,
    RERANK_MODEL,
    RERANK_POOL,
    RERANK_THRESHOLD,
    RERANK_TOP,
    RESERVED_OUTPUT_TOKENS,
    RRF_K,
    TOP_K,
    SourceLibrarian,
    check_context_budget,
    check_expansion_settings,
    check_index_settings,
    check_prefix_cap,
    check_top_k,
)

logger = logging.getLogger(__name__)


class WebLibrarian:
    """Searches the web and answers questions against fetched content.

    Uses WebScraper for robust search/fetch/crawl, then indexes
    via SourceLibrarian for the same retrieval pipeline as documents.
    """

    def __init__(
        self,
        llm: "LLMClient",
        scrape_config: ScrapeConfig | None = None,
        max_concurrent_prefixes: int = MAX_CONCURRENT_PREFIXES,
        top_k: int = TOP_K,
        *,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        reserved_output_tokens: int = RESERVED_OUTPUT_TOKENS,
        strict: bool = False,
        use_reranker: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        chunk_threshold: int = CHUNK_THRESHOLD,
        rrf_k: int = RRF_K,
        rerank_top: int = RERANK_TOP,
        rerank_pool: int | None = RERANK_POOL,
        rerank_threshold: float = RERANK_THRESHOLD,
        chars_per_token: int = CHARS_PER_TOKEN,
        embed_model: str = EMBED_MODEL,
        rerank_model: str = RERANK_MODEL,
        extract_fields: bool = EXTRACT_FIELDS,
        extract_facts: bool = EXTRACT_FACTS,
        fact_mix: float = FACT_MIX,
        cite_sources: bool = CITE_SOURCES,
        expand_query: bool = EXPAND_QUERY,
        expansion_variants: int = EXPANSION_VARIANTS,
        expansion_style: str = EXPANSION_STYLE,
    ):
        """
        Args:
            llm: any LLMClient (see librarium.llm)
            scrape_config: search/fetch/crawl settings
            max_concurrent_prefixes: index-time LLM calls in flight at
                once, forwarded to every SourceLibrarian this librarian
                builds. A crawl is indexed against the same backend as
                everything else, so the cap has to be settable here too —
                otherwise a deployment tuned to 2 fans out at the default
                the moment a question falls through to the web.
            top_k: chunks retrieved per query, forwarded to every
                SourceLibrarian this librarian builds. A crawl indexes
                several pages at once, so this is the path most likely to
                overflow a tight context window. :meth:`answer` takes a
                ``top_k=`` of its own to override it for one question.

        Keyword-only, forwarded to every SourceLibrarian a crawl builds
        and documented on :class:`SourceLibrarian`:
        *max_context_tokens*, *reserved_output_tokens*, *strict*,
        *use_reranker*, *chunk_size*, *chunk_overlap*, *chunk_threshold*,
        *rrf_k*, *rerank_top*, *rerank_pool*, *rerank_threshold*,
        *chars_per_token*, *embed_model*, *rerank_model*,
        *extract_fields*, *extract_facts*, *fact_mix*, *expand_query*, *expansion_variants*. Scraped pages are not shaped like
        documents — many short ones rather than one long one — so the
        chunk geometry that suits a corpus is not automatically the
        geometry that suits a crawl, and a crawl indexes several pages at
        once against whatever *max_context_tokens* allows.

        :meth:`answer` does not expand — it retrieves for the question as
        asked, exactly as the source librarian's does. *expand_query* is
        forwarded so that the librarian a crawl builds carries the
        deployment's setting rather than the default, for a caller that
        reaches it directly.
        """
        self.llm = llm
        self.scraper = WebScraper(scrape_config)
        # Checked here, not left to the SourceLibrarian a crawl builds:
        # that one is not constructed until a question falls through to the
        # web, which can be a long way from the call that made the typo —
        # or set a cap the client's connection pool cannot honour.
        self.max_concurrent_prefixes = check_prefix_cap(
            llm, max_concurrent_prefixes
        )
        # Same reasoning: the librarian this default reaches is not built
        # until a question falls through to the web.
        self.top_k = check_top_k(top_k)
        self.use_reranker = use_reranker
        self.strict = strict
        self.max_context_tokens, self.reserved_output_tokens = (
            check_context_budget(max_context_tokens, reserved_output_tokens)
        )
        self._index_settings = check_index_settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_threshold=chunk_threshold,
            rrf_k=rrf_k,
            rerank_top=rerank_top,
            rerank_pool=rerank_pool,
            rerank_threshold=rerank_threshold,
            chars_per_token=chars_per_token,
            embed_model=embed_model,
            rerank_model=rerank_model,
            extract_fields=extract_fields,
            extract_facts=extract_facts,
            top_k=self.top_k,
            use_reranker=use_reranker,
        )
        # A second dict rather than more keys in the first: expansion is
        # a query-time setting and must not reach content_hash, which
        # decides whether a saved index is reusable.
        # Query-time like the expansion settings, so it is stored
        # here rather than folded into `check_index_settings` —
        # that dict reaches `content_hash`, and toggling a
        # prompt-shaping flag must not invalidate a saved index.
        self.cite_sources = cite_sources
        # Query-time like `cite_sources`, so it is held here
        # rather than folded into `check_index_settings`:
        # that dict reaches `content_hash`, and the fact
        # store costs one LLM call per chunk — invalidating
        # it because somebody changed a ratio would be the
        # most expensive cache miss this library can cause.
        self.fact_mix = fact_mix
        self._expansion_settings = check_expansion_settings(
            expand_query=expand_query,
            expansion_variants=expansion_variants,
            expansion_style=expansion_style,
        )
        self._cache: dict[str, SourceLibrarian] = {}

    async def answer(
        self,
        question: str,
        context_facts: list[Fact] | None = None,
        *,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Search the web, fetch pages, answer the question.

        *top_k* overrides this librarian's default for one question.
        """
        cache_key = question[:100]
        if cache_key in self._cache:
            logger.info("  WEB LIBRARIAN: using cached index for: %s", question[:80])
            lib = self._cache[cache_key]
            return await lib.answer(question, context_facts, top_k=top_k)

        # Crawl: search → fetch → follow sublinks
        pages = self.scraper.crawl(question)
        if not pages:
            logger.info("  WEB LIBRARIAN: no pages for: %s", question[:80])
            return LibrarianAnswer(items=[], coverage="uncertain")

        # Index each page as its own document, keyed by URL, so every
        # fact comes back labelled with the page it was retrieved from.
        #
        # There is deliberately no `librarian_factory=` seam here, and the
        # decision is written down because the gap has been filed three
        # times. `LibrarianRouter` takes a `source_librarian=` because a
        # corpus is indexed once and handed over; a crawl indexes
        # different pages per question, so the only thing that could be
        # injected here is a constructor. Every setting a SourceLibrarian
        # takes is already forwarded (the four-layer rule), so that
        # constructor would buy exactly one thing — substituting the
        # *class* — which no consumer has asked for. It would also not be
        # free: a factory that ignored the arguments below would silently
        # run the web path at the small-model defaults, which is the bug
        # the router needs its inert-settings warning to catch, so the
        # seam arrives carrying that machinery. Tests reach this line
        # through `stub_models`, the seam the loaders' lazy imports
        # already provide and the one the model-free guard is built on.
        # Revisit if a consumer wants a SourceLibrarian subclass here;
        # testability on its own is not the reason, it is already served.
        lib = SourceLibrarian(
            self.llm,
            use_reranker=self.use_reranker,
            max_context_tokens=self.max_context_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            strict=self.strict,
            max_concurrent_prefixes=self.max_concurrent_prefixes,
            top_k=self.top_k,
            # Already validated and normalised in __init__.
            **self._index_settings,
            cite_sources=self.cite_sources,
            fact_mix=self.fact_mix,
            **self._expansion_settings,
            # Same client, same cap: __init__ already warned, and this
            # runs once per distinct question.
            _warn_settings=False,
        )
        indexed = 0
        total_chars = 0
        for page in pages:
            if page.url in lib.documents:
                continue
            body = f"[Title: {page.title}]\n{page.text}"
            await lib.add_document(body, doc_id=page.url)
            indexed += 1
            total_chars += len(body)
        self._cache[cache_key] = lib

        logger.info(
            "  WEB LIBRARIAN: indexed %d chars from %d pages (%s) for: %s",
            total_chars, indexed, lib.retrieval_mode, question[:80],
        )

        return await lib.answer(question, context_facts, top_k=top_k)
