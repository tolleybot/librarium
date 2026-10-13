"""Librarian Router — multi-source query routing with K-line cache.

Queries sources in priority order, stopping when coverage is complete.
K-line cache sits in front of everything — verified answers return instantly.

Flow:
  Query → K-line cache
    → HIT: return immediately
    → MISS:
      Source Librarian (indexed document)
        → complete: done
        → partial/uncertain:
          Web Librarian (search + scrape — last resort)
            → merge, done

Results from all sources are merged and deduplicated.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from librarium.kline_librarian import KLineLibrarian
from librarium.klines import KLineStore
from librarium.llm import LLMClient
from librarium.models import Fact, LibrarianAnswer
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

if TYPE_CHECKING:  # WebLibrarian needs the [web] extra — keep core lean
    from librarium.web import WebLibrarian

logger = logging.getLogger(__name__)


class _QueryAllProxy:
    """Proxy that calls router._query_all, not router.answer.

    Prevents infinite recursion: KLineLibrarian → proxy._query_all
    instead of KLineLibrarian → router.answer → KLineLibrarian → ...
    """

    def __init__(self, router: LibrarianRouter):
        self._router = router

    async def answer(self, question, context_facts=None, refine=False,
                     top_k=None):
        return await self._router._query_all(
            question, context_facts, refine=refine, top_k=top_k
        )


class LibrarianRouter:
    """Routes queries to multiple librarians in priority order.

    K-line cache wraps the entire pipeline. Source → Web are queried
    in order, stopping when coverage is "complete".
    """

    def __init__(
        self,
        llm: LLMClient,
        kline_store: KLineStore | None = None,
        source_librarian: SourceLibrarian | None = None,
        web_librarian: "WebLibrarian | None" = None,
        use_web: bool = True,
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
            kline_store: verified-answer cache in front of the pipeline
            source_librarian: pre-built source librarian; one is built on
                first :meth:`index` if omitted
            web_librarian: pre-built web librarian
            use_web: query the web librarian when the source falls short
            max_concurrent_prefixes: index-time LLM calls in flight at
                once, used for the SourceLibrarian :meth:`index` builds
                lazily. The router talks to the same backend as everything
                else, so a deployment tuned to 2 must not fan out at the
                default just because nobody passed *source_librarian*.
                Ignored when *source_librarian* is given — that librarian
                keeps the cap it was built with.
            top_k: chunks retrieved per query, used for the
                SourceLibrarian :meth:`index` builds lazily. Same
                argument as the cap: the model behind the router has one
                context window, so the librarian built for you must not
                retrieve at the default when the deployment cannot take
                it. :meth:`answer` takes a ``top_k=`` of its own to
                override this for one question.

        Keyword-only, and forwarded to that same lazily built librarian
        for the same reason — a deployment's context window, chunk
        geometry and models are not something a router should quietly
        default: *max_context_tokens*, *reserved_output_tokens*, *strict*,
        *use_reranker*, *chunk_size*, *chunk_overlap*, *chunk_threshold*,
        *rrf_k*, *rerank_top*, *rerank_pool*, *rerank_threshold*,
        *chars_per_token*, *embed_model*, *rerank_model*,
        *extract_fields*, *extract_facts*, *fact_mix*, *expand_query*, *expansion_variants*. All documented on
        :class:`SourceLibrarian`, and all ignored when *source_librarian*
        is given.

        The router itself never expands: it answers through
        :meth:`SourceLibrarian.answer`, which retrieves for the question
        as asked. *expand_query* is carried so that a librarian built
        here is the one you asked for, and so that a caller reaching
        through to ``router.source.retrieve_expanded()`` gets the
        deployment's setting rather than the default.

        *max_context_tokens* matters most of the three the router used to
        drop: a question that falls through to the source librarian is
        answered against that budget, so a router aimed at a large-context
        model was capping every answer at the small-model default with no
        way to say otherwise.
        """
        self.llm = llm
        self.source = source_librarian
        self.web = web_librarian
        self.use_web = use_web
        # Checked here, not left to the SourceLibrarian this builds
        # lazily: that one is not constructed until the first index(), so a
        # typo — or a cap the client's connection pool cannot honour —
        # would otherwise surface a long way from the call that made it.
        # The validation is unconditional — a bad cap is a mistake whether
        # or not it ends up being used — but the pool warning is not: an
        # injected librarian keeps its own cap, so warning that *this* one
        # outruns the pool would tell an already-correct deployment to
        # raise max_connections to match a number nothing reads.
        self.max_concurrent_prefixes = check_prefix_cap(
            llm, max_concurrent_prefixes, warn=source_librarian is None
        )
        # Silence alone would leave the caller believing a cap they set is
        # in force, so say what actually happened. Only for a value they
        # chose: warning about a default nobody passed is noise.
        if (source_librarian is not None
                and max_concurrent_prefixes != MAX_CONCURRENT_PREFIXES):
            logger.warning(
                "  ROUTER max_concurrent_prefixes=%s is ignored — the "
                "source_librarian you passed keeps its own cap. Set it on "
                "that SourceLibrarian instead.",
                max_concurrent_prefixes,
            )
        # Same reasoning: the librarian this default reaches does not
        # exist yet, so a bad value would surface at the first query.
        self.top_k = check_top_k(top_k)
        # And the same again for the rest of what the built librarian gets:
        # the index settings additionally decide whether a saved index is
        # reusable, so a bad chunk_size would otherwise surface as a cache
        # that never hits, one index() later. The rerank advice is only
        # worth giving when the reranker is actually on, which is what
        # use_reranker carries in.
        self.use_reranker = use_reranker
        self.strict = strict
        self.max_context_tokens, self.reserved_output_tokens = (
            check_context_budget(
                max_context_tokens, reserved_output_tokens,
                warn=source_librarian is None,
            )
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
            warn=source_librarian is None,
        )
        # Query-time rather than index-shaping, hence a second dict: it
        # is validated here for the same lazily-built-librarian reason,
        # but must not be folded into the settings the cache key covers.
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
        # And the same argument as the cap's ignored-warning above: an
        # injected librarian keeps its own geometry, so silence would
        # leave the caller believing the chunk size they chose is in
        # force. Only settings that differ from the default are named —
        # warning about a default nobody passed is noise.
        if source_librarian is not None:
            defaults = check_index_settings(warn=False)
            chosen = {
                name: (value, defaults[name])
                for name, value in self._index_settings.items()
            }
            expansion_defaults = check_expansion_settings()
            chosen.update({
                name: (value, expansion_defaults[name])
                for name, value in self._expansion_settings.items()
            })
            chosen["use_reranker"] = (use_reranker, False)
            chosen["strict"] = (strict, False)
            chosen["max_context_tokens"] = (
                self.max_context_tokens, MAX_CONTEXT_TOKENS
            )
            chosen["reserved_output_tokens"] = (
                self.reserved_output_tokens, RESERVED_OUTPUT_TOKENS
            )
            ignored = [
                name for name, (value, default) in sorted(chosen.items())
                if value != default
            ]
            if ignored:
                logger.warning(
                    "  ROUTER %s %s ignored — the source_librarian you "
                    "passed keeps the settings it was built with. Set them "
                    "on that SourceLibrarian instead.",
                    ", ".join(ignored),
                    "is" if len(ignored) == 1 else "are",
                )

        # K-line cache wraps _query_all (NOT self.answer — avoids recursion)
        self.kline_store = kline_store
        self._kline_wrapper: KLineLibrarian | None = None
        if kline_store:
            self._kline_wrapper = KLineLibrarian(
                librarian=_QueryAllProxy(self),
                kline_store=kline_store,
                source="router",
            )

    async def index(self, text: str) -> None:
        """Index a document via the Source Librarian."""
        if self.source is None:
            self.source = SourceLibrarian(
                self.llm,
                use_reranker=self.use_reranker,
                max_context_tokens=self.max_context_tokens,
                reserved_output_tokens=self.reserved_output_tokens,
                strict=self.strict,
                max_concurrent_prefixes=self.max_concurrent_prefixes,
                top_k=self.top_k,
                # Already validated and normalised in __init__, which is
                # also where any advice about them was given.
                **self._index_settings,
                cite_sources=self.cite_sources,
                fact_mix=self.fact_mix,
                **self._expansion_settings,
                # Same client, same cap: __init__ already warned.
                _warn_settings=False,
            )
        await self.source.index(text)

    async def answer(
        self,
        question: str,
        context_facts: list[Fact] | None = None,
        force_refresh: bool = False,
        refine: bool = False,
        *,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Query with K-line cache → multi-source fallback.

        Args:
            force_refresh: bypass K-line cache (for re-verification)
            refine: enable iterative refinement on source retrieval
            top_k: chunks retrieved for this question, overriding the
                default of whichever librarian answers it
        """
        if self._kline_wrapper:
            return await self._kline_wrapper.answer(
                question, context_facts,
                force_refresh=force_refresh, refine=refine, top_k=top_k,
            )
        return await self._query_all(
            question, context_facts, refine=refine, top_k=top_k
        )

    async def _query_all(
        self,
        question: str,
        context_facts: list[Fact] | None = None,
        refine: bool = False,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Query sources in priority order, merge results."""
        all_facts: list[Fact] = []
        seen: set[str] = set()
        coverage = "uncertain"

        # 1. Source Librarian (indexed document or short text)
        if self.source and (self.source.use_chunking or self.source.source):
            try:
                # Pass refine and top_k only when set — keeps compatibility
                # with librarians (and test mocks) that predate either kwarg,
                # and leaves the librarian's own default in charge when the
                # caller did not ask for something else.
                extra: dict[str, object] = {"refine": True} if refine else {}
                if top_k is not None:
                    extra["top_k"] = top_k
                result = await self.source.answer(
                    question, context_facts, **extra,
                )
                all_facts, seen = self._merge(all_facts, seen, result.items)
                coverage = result.coverage
                logger.info(
                    "  ROUTER source: %d facts, coverage=%s",
                    len(result.items), result.coverage,
                )
                if coverage == "complete" and all_facts:
                    return LibrarianAnswer(items=all_facts, coverage=coverage)
            except ValueError as e:
                # Context overflow — source has too much content for the model
                logger.warning("  ROUTER source overflow: %s", str(e)[:120])
                coverage = "uncertain"

        # 2. Web Librarian (last resort — expensive)
        if self.use_web and self.web and coverage != "complete":
            logger.info("  ROUTER: source insufficient, trying web")
            # Same "only when set" idiom as the source call above: a web
            # librarian is answering from a crawl against the same model,
            # so a top_k asked for here applies there too.
            web_extra = {"top_k": top_k} if top_k is not None else {}
            result = await self.web.answer(question, context_facts, **web_extra)
            if result.items:
                all_facts, seen = self._merge(all_facts, seen, result.items)
                coverage = result.coverage
                logger.info(
                    "  ROUTER web: %d facts, coverage=%s",
                    len(result.items), result.coverage,
                )

        if not all_facts:
            coverage = "uncertain"

        return LibrarianAnswer(items=all_facts, coverage=coverage)

    # ── Verification interface ───────────────────────────────────

    def verify(self, question: str, confidence: float = 1.0) -> None:
        """Mark a pending K-line as verified."""
        if self._kline_wrapper:
            self._kline_wrapper.verify(question, confidence)

    def mark_failed(self, question: str, reason: str, urls: list[str] | None = None) -> None:
        """Record a failed query pattern."""
        if self._kline_wrapper:
            self._kline_wrapper.mark_failed(question, reason, urls)

    # ── Internal ─────────────────────────────────────────────────

    @staticmethod
    def _merge(
        existing: list[Fact], seen: set[str], new_facts: list[Fact]
    ) -> tuple[list[Fact], set[str]]:
        """Merge new facts, deduplicating by content."""
        for fact in new_facts:
            if fact.content not in seen:
                seen.add(fact.content)
                existing.append(fact)
        return existing, seen
