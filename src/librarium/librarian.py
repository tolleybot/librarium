"""Librarian — the front door.

Three ways to query, in order of preference:

``ask()`` — retrieve top-K chunks, hand them RAW to the model, one
call, get a prose answer.  This is the configuration that scored
74-88% on the librarium benchmark suites with an 8B model.  Nothing
is lost between the document and the answer.

``retrieve()`` — the chunks alone, with provenance, no LLM call.  Use
this when you want librarium's retrieval under your own prompt,
your own streaming client, or your own citation rendering.

``facts()`` — LLM extracts discrete facts from the retrieved chunks
(optionally with iterative gap-finding refinement).  Use when you
need structured facts for a downstream program rather than an answer
for a human.  Note: extraction is lossy — measured ~40-46% rubric-fact
coverage vs 73% for the raw-chunk path.  Prefer ``ask()`` unless you
specifically need structure.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from pathlib import Path

from librarium.llm import LLMClient
from librarium.models import Chunk, LibrarianAnswer
from librarium.chunking import CHUNK_OVERLAP, CHUNK_SIZE
from librarium.source import (
    CHARS_PER_TOKEN,
    CHUNK_THRESHOLD,
    DEFAULT_DOC_ID,
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
)

logger = logging.getLogger(__name__)

ASK_SYSTEM = (
    "Answer the question using the provided material. "
    "Show your reasoning. Be specific — cite numbers, names, and sources."
)

# Added to the system prompt when strict_grounding is on. A retrieval
# miss otherwise produces a confident wrong answer rather than an
# admission; on a scored benchmark declining costs a point or two, in
# production it is the difference between wrong and honest.
GROUNDING_CLAUSE = (
    " If the answer is not present in the material, say so plainly and "
    "stop — never guess, never fill in a plausible value."
)


class Librarian:
    """Index documents, then ask questions.

    Usage::

        from librarium import Librarian, OpenAICompatClient

        llm = OpenAICompatClient("http://localhost:8000/v1", "Qwen/Qwen3-8B")
        lib = Librarian(llm)
        await lib.index(open("report.md").read(), cache=".librarium/report")
        answer = await lib.ask("What is the net cash position?")
    """

    def __init__(
        self,
        llm: LLMClient,
        use_reranker: bool = False,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        think_on_ask: bool = True,
        reserved_output_tokens: int = RESERVED_OUTPUT_TOKENS,
        strict: bool = False,
        strict_grounding: bool = False,
        max_concurrent_prefixes: int = MAX_CONCURRENT_PREFIXES,
        top_k: int = TOP_K,
        *,
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
            use_reranker: enable cross-encoder reranking
            max_context_tokens: total budget for an answer call, prompt
                *and* reply
            think_on_ask: let the model think before answering in ask()
            reserved_output_tokens: budget held back for the reply
            strict: raise instead of degrading when the embedding model
                or reranker is installed but fails to load
            strict_grounding: tell the model to decline rather than
                guess when the material does not contain the answer
            max_concurrent_prefixes: index-time LLM calls in flight at
                once. Lower it for a serial backend like Ollama, raise it
                for a local vLLM; 0 restores unbounded fan-out
            top_k: chunks retrieved per query, for every method that does
                not override it. The main lever on the context/precision
                tradeoff — lower it for a tight context window

        Keyword-only, forwarded verbatim to the SourceLibrarian and
        documented there (:class:`SourceLibrarian`): *chunk_size*,
        *chunk_overlap*, *chunk_threshold*, *rrf_k*, *rerank_top*,
        *rerank_pool*, *rerank_threshold*, *chars_per_token*, *embed_model*,
        *rerank_model*, *extract_fields*, *extract_facts*, *fact_mix*, *expand_query*,
        *expansion_variants*. They shape the index and the retrieval, so a librarian tuned for a
        particular model can be built here rather than by reaching into
        the module.

        Two of them cost a call rather than shaping one, and both are off
        by default. *expand_query* is read only by :meth:`expand_query`
        and :meth:`retrieve_expanded` — :meth:`ask`, :meth:`facts` and
        :meth:`retrieve` are unaffected either way. *extract_fields* is
        read at index time, one call per document.
        """
        self.llm = llm
        self.source = SourceLibrarian(
            llm,
            use_reranker=use_reranker,
            max_context_tokens=max_context_tokens,
            reserved_output_tokens=reserved_output_tokens,
            strict=strict,
            max_concurrent_prefixes=max_concurrent_prefixes,
            top_k=top_k,
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
            cite_sources=cite_sources,
            fact_mix=fact_mix,
            expand_query=expand_query,
            expansion_variants=expansion_variants,
            expansion_style=expansion_style,
        )
        self.think_on_ask = think_on_ask
        self.strict_grounding = strict_grounding

    # ── Indexing ─────────────────────────────────────────────────

    async def index(
        self,
        text: str,
        doc_id: str = DEFAULT_DOC_ID,
        cache: str | Path | None = None,
    ) -> None:
        """Index a document, **replacing** anything indexed before.

        Text at or under ``chunk_threshold`` (3000 chars by default) is
        kept whole — no chunking, no index-time LLM calls.

        Indexing costs one LLM call for the summary plus one per chunk.
        Pass ``cache=<dir>`` to persist the result: a later run over the
        same text loads it instead of paying that again. The cache is
        keyed on a content hash, so an edited document re-indexes
        automatically.

        Use :meth:`add_document` to index several documents together.
        """
        await self.source.index(text, doc_id=doc_id, cache=cache)

    async def add_document(self, text: str, doc_id: str | None = None) -> None:
        """Add a document to the index, keeping what is already there.

        Every chunk and every fact carries its ``doc_id``, so answers
        drawn from a multi-document index stay attributable.

        Safe to call concurrently — ``asyncio.gather`` over a corpus
        indexes the documents in parallel, and ``max_concurrent_prefixes``
        bounds the librarian's index-time calls across all of them rather
        than each individual call.

        That safety is indexing against indexing: the adds overlap each
        other, and :meth:`index` is held exclusive against the ones in
        flight. It does not extend to querying. A question asked *while*
        a document is being indexed does not wait for it. It is answered
        from the part-built index, which can degrade to whole documents
        and so overflow the context budget — :meth:`ask` and
        :meth:`facts` then raise ValueError where a finished index would
        have answered. See :meth:`SourceLibrarian.add_document`.
        """
        await self.source.add_document(text, doc_id=doc_id)

    def save(self, path: str | Path) -> None:
        """Persist the index to *path* (see :meth:`SourceLibrarian.save`).

        Called while an :meth:`add_document` is in flight this writes a
        partial index and warns — ``IndexBusyError`` instead under
        ``strict``. Called while another task is inside :meth:`index` it
        raises whatever ``strict`` says, because that file records the
        document being indexed with none of its chunks and reloads as if
        it were complete. Both this and :meth:`load` are whole-index
        operations and both are synchronous, so neither can take the
        guard :meth:`index` takes. Await the indexing first.
        """
        self.source.save(path)

    def load(self, path: str | Path) -> None:
        """Load an index written by :meth:`save`.

        Raises ``IndexBusyError`` if another task is indexing this
        librarian — an :meth:`add_document` in flight, or an
        :meth:`index` replacing it. Unlike :meth:`save` there is no
        partial outcome to offer for either, so this refuses with or
        without ``strict``.
        """
        self.source.load(path)

    # ── Introspection ────────────────────────────────────────────

    @property
    def retrieval_mode(self) -> str:
        """``hybrid`` | ``bm25-only`` | ``whole-document`` | ``unindexed``.

        The project's benchmark numbers were measured on
        ``hybrid``. Check this at runtime rather than assuming the
        embedding half loaded.
        """
        return self.source.retrieval_mode

    @property
    def expansion_mode(self) -> str:
        """``off`` | ``on`` | ``inert`` — see :attr:`SourceLibrarian.
        expansion_mode`.

        Separate from :attr:`retrieval_mode` on purpose: expansion is
        orthogonal to the lexical/dense axis, and the four
        ``retrieval_mode`` strings stay exactly as they were.
        """
        return self.source.expansion_mode

    # ── Query ────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        also: Sequence[str] = (),
        exclude: Collection[int] = (),
    ) -> list[Chunk]:
        """Return the most relevant chunks, best first — no LLM call.

        Each chunk carries ``doc_id`` and ``offset``, so you can cite it.
        *top_k* defaults to the librarian's own.

        *also* fuses extra phrasings of the question into the ranking —
        from :meth:`expand_query`, or from your own client. Still
        synchronous, still no LLM call: generating phrasings is the other
        method's job (:meth:`SourceLibrarian.retrieve`).

        *exclude* drops chunks by ``index`` before the cut, which is how
        you ask for the next page when the first one missed::

            page = lib.retrieve(q)
            more = lib.retrieve(q, exclude={c.index for c in page})

        The material is usually within reach: on the benchmark corpus 70%
        of expected literals are in ranks 1-10 and only 6% are absent from
        the top 50, with seven of the eight questions that find nothing on
        the first page recovered by paging. See
        :meth:`SourceLibrarian.retrieve` for the measurement and for what
        it does not do — it filters chunks, not facts.
        """
        return self.source.retrieve(question, top_k=top_k, also=also,
                                    exclude=exclude)

    async def expand_query(self, question: str) -> list[str]:
        """Rewrite the question into alternative phrasings — one LLM call.

        Returns ``[]`` without calling the model unless the librarian was
        built with ``expand_query=True``, and ``[]`` rather than raising
        when the call or its reply is unusable (unless ``strict``). See
        :meth:`SourceLibrarian.expand_query`.
        """
        return await self.source.expand_query(question)

    async def retrieve_expanded(
        self, question: str, top_k: int | None = None
    ) -> list[Chunk]:
        """:meth:`expand_query` then :meth:`retrieve`, nothing else.

        The convenience path. Use the two halves directly when you want
        to batch, cache or precompute the phrasings.
        """
        return await self.source.retrieve_expanded(question, top_k=top_k)

    async def ask(
        self,
        question: str,
        system: str | None = None,
        *,
        top_k: int | None = None,
    ) -> str:
        """Answer a question in one LLM call over raw retrieved chunks.

        The high-accuracy path: retrieval selects, the model reads the
        original text, nothing is paraphrased in between.

        *top_k* overrides the librarian's default for this question — the
        lever to reach for when one question overflows the context budget.
        """
        material = self.source._retrieve(question, top_k=top_k)
        if not material:
            raise RuntimeError("No document indexed — call index() first")

        instructions = ASK_SYSTEM if system is None else system
        if self.strict_grounding:
            instructions += GROUNDING_CLAUSE

        system_prompt = f"{instructions}\n\nMATERIAL:\n{material}"
        user_prompt = f"QUESTION: {question}"
        self.source.check_context(system_prompt, user_prompt)

        return await self.llm.complete(
            system_prompt,
            user_prompt,
            think=self.think_on_ask,
            temperature=0.1,
        )

    async def facts(
        self,
        question: str,
        refine: bool = False,
        max_rounds: int = 2,
        *,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Extract structured facts relevant to the question.

        With ``refine=True``, after the first pass the librarian asks
        "which part of the question do these facts not answer?" and
        runs a targeted second retrieval for the gap.

        *top_k* overrides the librarian's default for this question,
        refinement rounds included.
        """
        return await self.source.answer(
            question, refine=refine, max_rounds=max_rounds, top_k=top_k
        )
