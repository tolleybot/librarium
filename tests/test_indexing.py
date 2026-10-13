"""Index-time behaviour: prefix fan-out and the HTTP pool that carries it.

Indexing is the only place librarium issues many LLM calls at once. How
many is a property of the deployment — vLLM's --max-num-seqs, Ollama's
serial default, a hosted rate limit — so it is a knob, and these tests
pin the knob's behaviour rather than a particular number.
"""

from __future__ import annotations

import asyncio
import json
import zlib
from collections import Counter

import numpy as np
import pytest

from librarium import Librarian, LibrarianRouter, SourceLibrarian
from librarium.chunking import chunk_spans, tokenize
from librarium.source import (
    CHUNK_JOINER,
    CHUNK_THRESHOLD,
    EMBED_MODEL,
    EMBED_PROBE,
    CONTEXT_PREFIX_SYSTEM,
    DOC_SUMMARY_SYSTEM,
    MAX_CONCURRENT_PREFIXES,
    RESERVED_OUTPUT_TOKENS,
    TOP_K,
    embed_prompts,
)
from tests.conftest import stub_models

# A gate that never opens is a bug in the test, not a slow machine, so the
# wait is bounded: the assertion on `peak` then reports the miscount, which
# is far easier to read than a suite that hangs.
GATE_TIMEOUT = 5.0

# Enough chunks to exceed any sane concurrency cap several times over.
MANY_CHUNKS_DOC = "\n\n".join(
    f"Section {i} discusses topic {i} in detail. " * 45 for i in range(24)
)


def prefix_for(chunk_text: str, summary: str) -> str:
    """The prefix :class:`WatchfulLLM` writes for a chunk under a summary.

    A constant prefix is the same string on every chunk of every document,
    so a prefix filed against the wrong chunk — or written from another
    document's summary — compares equal to the right one and no assertion
    can see the difference. Naming both halves of the prompt in the answer
    is the trick :class:`MarkedEmbedder` plays with vectors, one step down.
    """
    mark = zlib.crc32(chunk_text.strip().encode()) % 100_000
    return f"prefix for chunk {mark:05d} under {summary}"


def prefix_from_prompt(context: str) -> str:
    """Answer a ``_generate_prefixes`` prompt by reading it back apart.

    The prompt is all the double is given, so recovering the chunk and the
    summary from it is what lets the answer name them.
    """
    summary = context.split("DOCUMENT SUMMARY: ", 1)[1].split("\n\n", 1)[0]
    body = context.split("THIS CHUNK:\n", 1)[1]
    # Absent on the last chunk, which ends the prompt with its own text.
    body = body.split("\n\nNEXT CHUNK (beginning): ", 1)[0]
    return prefix_for(body, summary)


class WatchfulLLM:
    """Records how many calls are in flight simultaneously.

    The await is a real suspension point, so every task that is allowed
    to start does start before any finishes — an unbounded gather shows
    up as a peak equal to the chunk count.

    That is enough for the ``peak <= cap`` assertions, which cannot flake:
    a call that arrives late simply does not raise the peak. Asserting an
    *exact* peak is different — it needs every one of the overlapping
    calls to reach its await inside one `delay`, which is a race the
    machine wins on an idle laptop and can lose on a loaded CI runner.
    ``gate_at`` removes the clock from that: gated calls park on an Event
    that is released only once the Nth of them has arrived, so the overlap
    is a property of the fan-out rather than of scheduling luck.

    Only prefix calls are gated. The document summary is issued alone,
    before the fan-out, so a gate on it would wait for siblings that never
    come. For the same reason ``gate_at`` must be a count the workload
    really reaches — ``min(chunks, cap)`` — and every other test leaves it
    unset and keeps the plain sleep.
    """

    def __init__(self, delay: float = 0.005, *, gate_at: int | None = None):
        self.delay = delay
        self.gate_at = gate_at
        self.in_flight = 0
        self.peak = 0
        self.calls: list[tuple[str, str]] = []
        # Built on first use, not here: asyncio primitives bind to a loop.
        self._gate: asyncio.Event | None = None

    async def _hold(self, system: str) -> None:
        """Keep this call in flight long enough for its siblings to arrive."""
        if self.gate_at is None or system != CONTEXT_PREFIX_SYSTEM:
            await asyncio.sleep(self.delay)
            return
        if self._gate is None:
            self._gate = asyncio.Event()
        # The caller already counted itself in, so the Nth arrival is the
        # one that sees the target met and lets the whole batch go. Later
        # calls find the Event set and pass straight through, which is why
        # a batch smaller than the target at the tail of a run cannot hang.
        if self.in_flight >= self.gate_at:
            self._gate.set()
        try:
            await asyncio.wait_for(self._gate.wait(), GATE_TIMEOUT)
        except asyncio.TimeoutError:
            pass

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        self.calls.append((system, user))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self._hold(system)
            if system == CONTEXT_PREFIX_SYSTEM:
                return prefix_from_prompt(user)
            return "A context prefix."
        finally:
            self.in_flight -= 1


class FlakyLLM(WatchfulLLM):
    """A WatchfulLLM whose *prefix* calls fail for chosen documents.

    Only the prefix calls fail, not the summary: a summary failure never
    reaches the fan-out, so it cannot exercise what happens to a chunk's
    in-flight siblings when one of them raises.
    """

    def __init__(self, failing: set[str], delay: float = 0.005):
        super().__init__(delay=delay)
        self.failing = set(failing)

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        is_prefix = system == CONTEXT_PREFIX_SYSTEM
        if is_prefix and any(marker in user for marker in self.failing):
            self.calls.append((system, user))
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                await self._hold(system)
                raise RuntimeError("backend refused this chunk")
            finally:
                self.in_flight -= 1
        return await super().complete(
            system, user, think=think, temperature=temperature
        )


class SummarizingLLM(WatchfulLLM):
    """A WatchfulLLM whose *summaries* name the document they came from.

    The default answers every summary call with one constant string, which
    makes a stale `doc_summary` indistinguishable from a correct one — they
    would compare equal by accident.
    """

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        if system == DOC_SUMMARY_SYSTEM:
            self.calls.append((system, user))
            # doc_text() puts its marker at the very start of the text.
            return f"summary of {user.split()[0]}"
        return await super().complete(
            system, user, think=think, temperature=temperature
        )


class GatedLLM(WatchfulLLM):
    """A WatchfulLLM that parks one document's *summary* call until told.

    Everything the pending sweep does between its awaits is invisible to
    another task, so holding it inside `_chunk_document` is the only way
    to look at an index that is part-built. `parked` fires once the call
    is holding; setting `release` lets it through.
    """

    def __init__(self, park_on: str, delay: float = 0.005):
        super().__init__(delay=delay)
        self.park_on = park_on
        self.parked = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        if system == DOC_SUMMARY_SYSTEM and self.park_on in user:
            self.parked.set()
            await self.release.wait()
        return await super().complete(
            system, user, think=think, temperature=temperature
        )


class PooledLLM(WatchfulLLM):
    """A client that advertises a finite HTTP connection pool."""

    def __init__(self, max_connections: int = 4, delay: float = 0.005):
        super().__init__(delay=delay)
        self.max_connections = max_connections


def pool_warnings(caplog) -> list[str]:
    """The pool-vs-cap warnings recorded so far."""
    return [r.getMessage() for r in caplog.records
            if "max_connections" in r.getMessage()]


def ignored_cap_warnings(caplog) -> list[str]:
    """The "the cap you passed is not the one in force" warnings."""
    return [r.getMessage() for r in caplog.records
            if "is ignored" in r.getMessage()]


class MarkedEmbedder:
    """Vectors that carry the identity of the text they were made from.

    Plausible-looking floats would let a row filed against the wrong
    chunk pass unnoticed, which is exactly the bug being guarded.
    """

    dims = 4

    @staticmethod
    def mark(text: str) -> float:
        return float(zlib.crc32(text.encode()) % 100_000)

    def encode(self, texts, show_progress_bar: bool = False):
        return np.asarray(
            [[self.mark(t)] * self.dims for t in texts], dtype=float
        )


def embedded_text(chunk) -> str:
    """Exactly what `_refresh_embeddings` hands the encoder for *chunk*.

    One definition, used by every test that opens `chunk_embeddings` and
    checks a row against the chunk it is filed under. Those tests are about
    **correspondence** — row *i* holds chunk *i*'s vector — which is
    orthogonal to which text the encoder was given, so the input belongs in
    one place rather than repeated at each assertion. It moved once already:
    the encoder was given `c.text` and is now given `c.prefixed`, the same
    text `_build_bm25` indexes, and that turned five correspondence
    assertions red for a reason that had nothing to do with correspondence.

    Deliberately a mirror of production and not a call into it. Asking
    `_refresh_embeddings` what it would encode would make these tests pass
    against any input at all, including a reverted one — the property would
    become unfalsifiable. Keeping the mirror means a revert to `c.text`
    fails here too, which is redundant with
    :class:`TestWhatReachesTheEncoder` and harmless: that class is the one
    that fails *by name* on a revert, and it is where the requirement is
    stated.
    """
    return f"{embed_prompts(EMBED_MODEL)[0]}{chunk.prefixed}"


def doc_marker(i: int) -> str:
    """Text appearing in every chunk of :func:`doc_text` — and no other."""
    return f"DOC{i}X"


def doc_text(i: int) -> str:
    """A distinct many-chunk document whose every chunk names its origin."""
    return MANY_CHUNKS_DOC.replace("Section", f"{doc_marker(i)} Section")


def small_doc(i: int) -> str:
    """A document with far fewer chunks than any sane cap.

    The regime cross-document parallelism exists for: a scraped page or a
    wiki entry cannot saturate the cap on its own, so the only way to
    reach it is to have several documents in flight at once. Sized just
    over CHUNK_THRESHOLD so each one chunks on its own account rather
    than waiting to be swept up by the next.
    """
    return f"{doc_marker(i)} note. " + f"Line {i} of a short page. " * 130


class LockWatchingLLM(WatchfulLLM):
    """Records whether the index lock was held during each call.

    The whole point of the split is that the lock covers mutations and
    not LLM calls: a lock held across a round trip is what made a corpus
    index one document at a time.
    """

    def __init__(self, delay: float = 0.005):
        super().__init__(delay=delay)
        self.lib: SourceLibrarian | None = None   # set after construction
        self.locked_during: list[str] = []

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        lock = None if self.lib is None else self.lib._lock
        if lock is not None and lock.locked():
            self.locked_during.append(system)
        return await super().complete(
            system, user, think=think, temperature=temperature
        )


async def index_with(
    limit: int | None, *, gate_at: int | None = None
) -> tuple[SourceLibrarian, WatchfulLLM]:
    llm = WatchfulLLM(gate_at=gate_at)
    kwargs = {} if limit is None else {"max_concurrent_prefixes": limit}
    lib = SourceLibrarian(llm, **kwargs)
    lib._dense_ok = False
    await lib.index(MANY_CHUNKS_DOC)
    return lib, llm


class TestPrefixFanOut:
    async def test_document_produces_more_chunks_than_the_cap(self):
        # Guards the premise of every other test in this class.
        lib, _ = await index_with(4)
        assert len(lib.chunks) > MAX_CONCURRENT_PREFIXES

    @pytest.mark.parametrize("limit", [1, 2, 4, 8])
    async def test_concurrency_never_exceeds_the_cap(self, limit):
        lib, llm = await index_with(limit)
        assert llm.peak <= limit
        assert len(lib.chunks) > limit  # the cap actually bound something

    async def test_a_numpy_cap_bounds_a_real_index(self):
        # The type check is only half the story: a numpy scalar that got
        # stored unchanged would go on to size an asyncio.Semaphore, so
        # index once end to end rather than stopping at the constructor.
        lib, llm = await index_with(np.int64(4))
        assert len(lib.chunks) > 4  # the cap actually bound something
        assert llm.peak <= 4

    async def test_the_cap_is_reached_not_merely_respected(self):
        # A serial implementation would also pass "never exceeds", so
        # check the calls really do overlap up to the limit. The gate
        # holds the first four calls open until the fourth arrives — the
        # document has far more chunks than that, so the cap is reachable
        # and the assertion does not depend on how fast the machine is.
        _, llm = await index_with(4, gate_at=4)
        assert llm.peak == 4

    async def test_default_is_bounded(self):
        lib, llm = await index_with(None)
        assert lib.max_concurrent_prefixes == MAX_CONCURRENT_PREFIXES
        assert llm.peak <= MAX_CONCURRENT_PREFIXES

    async def test_zero_restores_unbounded_fan_out(self):
        # Every chunk in flight at once — the old behaviour, opt-in. The
        # gate is the whole chunk count, which is knowable in advance
        # because the librarian chunks with exactly this function.
        expected = len(chunk_spans(MANY_CHUNKS_DOC))
        lib, llm = await index_with(0, gate_at=expected)
        assert len(lib.chunks) == expected  # the gate targeted the real count
        assert llm.peak == len(lib.chunks)

    async def test_serial_backend_setting(self):
        _, llm = await index_with(1)
        assert llm.peak == 1

    async def test_every_chunk_still_gets_its_prefix(self):
        lib, _ = await index_with(3)
        assert lib.chunks
        # Its own prefix, not merely *a* prefix. The cap turns one gather
        # into a queue and a failure cancels the siblings mid-flight, so a
        # prefix written for one chunk landing on the next is exactly the
        # damage the fan-out is in a position to do — and every chunk still
        # has a non-empty prefix afterwards.
        for chunk in lib.chunks:
            assert chunk.prefix == prefix_for(chunk.text, lib.doc_summary)
            assert chunk.prefixed.startswith(chunk.prefix)

    async def test_prefixes_are_written_from_their_own_doc_summary(self):
        # The summary is the other half of every prefix prompt, and a
        # librarian holds one per document — `doc_summary` alone is
        # whichever was chunked last. Handing the fan-out the wrong one
        # still writes a prefix for every chunk, so only a prefix that
        # names the summary it was given can catch the mix-up.
        llm = SummarizingLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=3)
        lib._dense_ok = False
        await lib.add_document(doc_text(0), doc_id="first.md")
        await lib.add_document(doc_text(1), doc_id="second.md")

        assert lib.doc_summaries["first.md"] != lib.doc_summaries["second.md"]
        for chunk in lib.chunks:
            assert chunk.prefix == prefix_for(
                chunk.text, lib.doc_summaries[chunk.doc_id]
            )

    async def test_capping_does_not_change_the_index(self):
        bounded, _ = await index_with(2)
        unbounded, _ = await index_with(0)
        assert bounded.prefixed_chunks == unbounded.prefixed_chunks
        assert [c.offset for c in bounded.chunks] == [
            c.offset for c in unbounded.chunks
        ]
        # Identical is not the same as right: a scramble the cap does not
        # cause — one applied to both runs alike — leaves the two indexes
        # equal to each other and wrong in the same way.
        assert all(c.prefix == prefix_for(c.text, bounded.doc_summary)
                   for c in bounded.chunks)

    async def test_facade_passes_the_setting_through(self):
        llm = WatchfulLLM()
        lib = Librarian(llm, max_concurrent_prefixes=2)
        lib.source._dense_ok = False
        await lib.index(MANY_CHUNKS_DOC)
        assert llm.peak <= 2

    async def test_sequential_multi_document_indexing_stays_bounded(self):
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=3)
        lib._dense_ok = False
        await lib.index(MANY_CHUNKS_DOC, doc_id="one.md")
        await lib.add_document(
            MANY_CHUNKS_DOC.replace("Section", "Chapter"), doc_id="two.md"
        )
        assert llm.peak <= 3
        assert {c.doc_id for c in lib.chunks} == {"one.md", "two.md"}


class TestConcurrentAddDocument:
    """`asyncio.gather` over a corpus is how a library gets handed many
    documents. A per-call cap would multiply by the document count —
    four documents at cap 3 peaking at 12 — so the cap has to bound the
    librarian, not the invocation.
    """

    @staticmethod
    async def gather_documents(count: int, limit: int):
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=limit)
        lib._dense_ok = False
        await asyncio.gather(*(
            lib.add_document(
                MANY_CHUNKS_DOC.replace("Section", f"S{i}"), doc_id=f"d{i}"
            )
            for i in range(count)
        ))
        return lib, llm

    @pytest.mark.parametrize("count,limit", [(2, 2), (4, 3), (5, 1)])
    async def test_cap_bounds_the_librarian_not_the_call(self, count, limit):
        lib, llm = await self.gather_documents(count, limit)
        assert llm.peak <= limit
        assert sorted(lib.documents) == [f"d{i}" for i in range(count)]

    async def test_every_document_is_fully_indexed(self):
        lib, _ = await self.gather_documents(4, 3)
        assert {c.doc_id for c in lib.chunks} == {"d0", "d1", "d2", "d3"}
        assert all(c.prefix for c in lib.chunks)
        assert lib.bm25 is not None

    async def test_chunk_index_stays_unique_and_ordered(self):
        # `_chunk_document` reads len(self.chunks) before its awaits, so
        # interleaved documents used to hand out the same index twice.
        lib, _ = await self.gather_documents(4, 3)
        indices = [c.index for c in lib.chunks]
        assert indices == list(range(len(lib.chunks)))

    async def test_auto_generated_doc_ids_do_not_collide(self):
        # doc_id defaults to f"doc{len(documents) + 1}", read before the
        # awaits. Concurrently, every unnamed add computed "doc1" and all
        # but one document was silently dropped.
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=3)
        lib._dense_ok = False
        await asyncio.gather(*(
            lib.add_document(MANY_CHUNKS_DOC.replace("Section", f"S{i}"))
            for i in range(4)
        ))
        assert sorted(lib.documents) == ["doc1", "doc2", "doc3", "doc4"]
        assert {c.doc_id for c in lib.chunks} == set(lib.documents)

    async def test_duplicate_doc_id_still_rejected_under_concurrency(self):
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        results = await asyncio.gather(
            lib.add_document(MANY_CHUNKS_DOC, doc_id="same.md"),
            lib.add_document(MANY_CHUNKS_DOC, doc_id="same.md"),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, ValueError)]
        assert len(errors) == 1
        assert list(lib.documents) == ["same.md"]

    def test_lock_survives_a_fresh_event_loop(self):
        # An asyncio.Lock built in __init__ would bind to whichever loop
        # ran first and raise on the second asyncio.run().
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        asyncio.run(lib.add_document(MANY_CHUNKS_DOC, doc_id="a.md"))
        asyncio.run(lib.add_document(
            MANY_CHUNKS_DOC.replace("Section", "Ch"), doc_id="b.md"))
        assert sorted(lib.documents) == ["a.md", "b.md"]
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))


class TestCrossDocumentParallelism:
    """Documents overlap; the mutations they make do not.

    Indexing one document is already parallel, so overlapping *documents*
    only pays when a document has fewer chunks than the cap — a scraped
    site, a wiki, a notes directory. Getting there meant taking the
    single-writer lock off the LLM calls, which is what made four
    invariants trivially correct at once. Each is pinned below:
    `Chunk.index`, the auto-generated doc_id counter, `source`, and the
    embedding rows appended by `_refresh_embeddings`.
    """

    @staticmethod
    async def gather_small(count: int, limit: int, *, embedder=None):
        llm = LockWatchingLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=limit)
        llm.lib = lib
        if embedder is None:
            lib._dense_ok = False
        else:
            lib.embed_model = embedder
        await asyncio.gather(*(
            lib.add_document(small_doc(i), doc_id=f"p{i}.md")
            for i in range(count)
        ))
        return lib, llm

    async def test_documents_really_do_overlap(self):
        # The premise of the whole change. A two-chunk document cannot put
        # more than two calls in flight by itself, so a peak above that is
        # proof that the documents interleaved rather than queueing.
        lib, llm = await self.gather_small(12, 8)
        assert len(chunk_spans(small_doc(0))) == 2
        assert llm.peak > 2

    async def test_no_llm_call_is_made_while_the_index_lock_is_held(self):
        # What the overlap costs if it is ever undone: a lock held across
        # a round trip serializes the corpus again, silently and with
        # every test still passing.
        _, llm = await self.gather_small(6, 4)
        assert llm.locked_during == []
        assert llm.calls  # the watcher had something to watch

    @pytest.mark.parametrize("count,limit", [(12, 8), (10, 2), (6, 1)])
    async def test_the_cap_still_bounds_the_librarian(self, count, limit):
        # Including the per-document summary, which used to sit outside
        # the gate entirely: one document at a time that was one spare
        # call, but with documents overlapping it is one per document on
        # top of the cap.
        lib, llm = await self.gather_small(count, limit)
        assert llm.peak <= limit
        assert len(llm.calls) == count * 3   # a summary and two prefixes each

    async def test_the_lock_actually_excludes_a_concurrent_mutation(self):
        # Worth pinning separately, because the invariant tests below
        # cannot: every critical section is await-free, so asyncio has no
        # opportunity to interleave them and they would pass with the
        # lock deleted. What the lock buys is that this stays true for
        # free — the moment one of those steps grows an await (embedding
        # moved onto a thread, a cache write, a metrics hop) the
        # invariants go back to depending on scheduling luck. So check
        # the exclusion itself, not only its consequences.
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=4)
        lib._dense_ok = False
        async with lib._index_lock():
            queued = asyncio.create_task(
                lib.add_document(small_doc(0), doc_id="p0.md")
            )
            await asyncio.sleep(0.05)
            assert lib.documents == {}   # blocked before it could register
        await queued
        assert list(lib.documents) == ["p0.md"]

    async def test_chunk_index_is_unique_and_ordered(self):
        # `_chunk_document` used to read `start = len(self.chunks)` before
        # its awaits — a promise about where the chunks would land, made
        # before any other document had a chance to land theirs. Two
        # overlapping documents both read the same start and both numbered
        # from it.
        lib, _ = await self.gather_small(12, 8)
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))
        assert len({(c.doc_id, c.offset) for c in lib.chunks}) == len(lib.chunks)

    async def test_each_documents_chunks_stay_together_and_in_order(self):
        # Interleaving the *chunks* of two documents would keep the
        # indices unique while scrambling what `retrieve` hands back as a
        # neighbouring span.
        lib, _ = await self.gather_small(8, 4)
        for doc_id in lib.documents:
            positions = [i for i, c in enumerate(lib.chunks)
                         if c.doc_id == doc_id]
            assert positions == list(range(positions[0], positions[-1] + 1))
            offsets = [lib.chunks[i].offset for i in positions]
            assert offsets == sorted(offsets)

    async def test_auto_generated_doc_ids_do_not_collide(self):
        # doc_id defaults to f"doc{len(documents) + 1}". Claiming it has
        # to happen in the same breath as writing `documents[doc_id]`, or
        # every concurrent add computes the same number.
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=4)
        lib._dense_ok = False
        await asyncio.gather(*(
            lib.add_document(small_doc(i)) for i in range(10)
        ))
        assert sorted(lib.documents) == sorted(f"doc{i}" for i in range(1, 11))
        assert {c.doc_id for c in lib.chunks} == set(lib.documents)

    async def test_source_holds_every_document_exactly_once(self):
        # `source` is what content_hash() and the whole-document path read,
        # and it is rebuilt from `documents` on every add. A rebuild racing
        # a registration drops or duplicates a document.
        lib, _ = await self.gather_small(12, 8)
        assert lib.source == "\n\n".join(lib.documents.values())
        for i in range(12):
            assert lib.source.count(doc_marker(i)) == 1

    async def test_source_survives_a_concurrent_failure(self):
        # The rollback is the other writer of `source`, and it runs while
        # its neighbours are still registering theirs.
        llm = FlakyLLM({doc_marker(i) for i in (1, 4)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=4)
        lib._dense_ok = False
        results = await asyncio.gather(*(
            lib.add_document(small_doc(i), doc_id=f"p{i}.md")
            for i in range(8)
        ), return_exceptions=True)

        assert len([r for r in results if isinstance(r, RuntimeError)]) == 2
        assert sorted(lib.documents) == [f"p{i}.md" for i in (0, 2, 3, 5, 6, 7)]
        assert lib.source == "\n\n".join(lib.documents.values())
        for i in (1, 4):
            assert doc_marker(i) not in lib.source

    async def test_every_chunk_keeps_its_own_embedding_row(self):
        # `_refresh_embeddings` pairs vectors with chunks by position —
        # it only encodes the chunks past the end of `chunk_embeddings` —
        # so an append that races a commit files every later row against
        # the wrong chunk, with the counts still matching.
        lib, _ = await self.gather_small(10, 4, embedder=MarkedEmbedder())
        assert len(lib.chunk_embeddings) == len(lib.chunks)
        for chunk, vector in zip(lib.chunks, lib.chunk_embeddings):
            assert vector[0] == MarkedEmbedder.mark(embedded_text(chunk))

    async def test_a_rollback_renumbers_what_it_leaves_behind(self):
        # Serially, a failing document's chunks are always the last ones
        # in the list — its doc_id went into `documents` last, so it is
        # last of the pending sweep — and the rollback could drop them by
        # truncating. With documents overlapping, a document that
        # committed earlier can be the one that fails: its chunks sit in
        # the middle, and removing them opens a hole in the numbering and
        # slides every later chunk onto the wrong embedding row.
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=4)
        lib.embed_model = MarkedEmbedder()
        for i in range(3):
            await lib.add_document(small_doc(i), doc_id=f"p{i}.md")
        buried = lib.chunks[2:4]
        assert {c.doc_id for c in buried} == {"p1.md"}   # genuinely interior

        lib._roll_back("p1.md")

        assert "p1.md" not in lib.documents
        assert "p1.md" not in lib.doc_summaries
        assert "p1.md" not in {c.doc_id for c in lib.chunks}
        assert lib.source == "\n\n".join(lib.documents.values())
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))
        assert lib.bm25.corpus_size == len(lib.chunks)
        assert len(lib.chunk_embeddings) == len(lib.chunks)
        for chunk, vector in zip(lib.chunks, lib.chunk_embeddings):
            assert vector[0] == MarkedEmbedder.mark(embedded_text(chunk))

    async def test_a_concurrent_failure_leaves_a_consistent_index(self):
        # The same rollback reached the way it actually happens, with the
        # neighbours still mid-flight around it.
        llm = FlakyLLM({doc_marker(2), doc_marker(5)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=4)
        lib.embed_model = MarkedEmbedder()
        results = await asyncio.gather(*(
            lib.add_document(small_doc(i), doc_id=f"p{i}.md")
            for i in range(8)
        ), return_exceptions=True)

        assert len([r for r in results if isinstance(r, RuntimeError)]) == 2
        assert sorted(lib.documents) == [f"p{i}.md" for i in (0, 1, 3, 4, 6, 7)]
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))
        assert lib.bm25.corpus_size == len(lib.chunks)
        assert len(lib.chunk_embeddings) == len(lib.chunks)
        for chunk, vector in zip(lib.chunks, lib.chunk_embeddings):
            assert vector[0] == MarkedEmbedder.mark(embedded_text(chunk))

    async def test_a_document_is_never_chunked_twice(self):
        # The pending sweep reads `documents`, which a concurrent add has
        # already written to. Without a claim, every call in the gather
        # sweeps up every un-summarized document — the same chunks
        # appended once per caller, at one LLM call each.
        lib, llm = await self.gather_small(10, 4)
        assert len(lib.chunks) == 20
        assert len(llm.calls) == 30
        assert sorted(lib.doc_summaries) == sorted(lib.documents)

    async def test_a_swept_up_document_is_not_chunked_twice_either(self):
        # The first document of a corpus can be under CHUNK_THRESHOLD on
        # its own, so it is held whole and only chunked when a later one
        # pushes the corpus over. Concurrently, that later one is several
        # documents at once, all of which see it unsummarized.
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=4)
        lib._dense_ok = False
        tiny = f"{doc_marker(99)} held whole. " * 60
        assert len(tiny) < CHUNK_THRESHOLD
        await lib.add_document(tiny, doc_id="tiny.md")
        assert lib.chunks == []

        await asyncio.gather(*(
            lib.add_document(small_doc(i), doc_id=f"p{i}.md")
            for i in range(6)
        ))
        assert [c.doc_id for c in lib.chunks].count("tiny.md") == 1
        assert sorted(lib.doc_summaries) == sorted(lib.documents)
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))

    def test_the_gate_survives_a_fresh_event_loop(self):
        # An asyncio.Semaphore built in __init__ would bind to whichever
        # loop ran first and raise on the second asyncio.run() — the same
        # trap the index lock is built lazily to avoid.
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False

        async def add_four(start: int):
            await asyncio.gather(*(
                lib.add_document(small_doc(i), doc_id=f"p{i}.md")
                for i in range(start, start + 4)
            ))

        asyncio.run(add_four(0))
        asyncio.run(add_four(4))
        assert llm.peak <= 2
        assert len(lib.documents) == 8
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))


class TestIndexReplacingConcurrently:
    """`index()` replaces the index; an add in flight must not survive it.

    The index lock is narrow now — held across each mutation, released
    across the LLM calls — so an `index()` that only took it could
    `_reset()` in the window between a document's claim and its commit.
    That is not a scheduling curiosity: the commit then lands chunks and
    a summary for a doc_id that is in neither `documents` nor `source`,
    and `retrieve()` hands them back while `content_hash()` and `save()`
    disagree with them.
    """

    @staticmethod
    def librarian() -> tuple[SourceLibrarian, WatchfulLLM]:
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=4)
        lib._dense_ok = False
        return lib, llm

    async def test_a_concurrent_add_does_not_land_in_the_replaced_index(self):
        lib, _ = self.librarian()
        adding = asyncio.create_task(
            lib.add_document(small_doc(1), doc_id="a.md")
        )
        await asyncio.sleep(0.001)   # let it claim a.md and start calling
        await lib.index(small_doc(2), doc_id="z")
        await adding

        # index() is documented to REPLACE the index. Whatever order the
        # two ran in, what is left has to be one self-consistent index.
        assert list(lib.documents) == ["z"]
        assert set(lib.doc_summaries) == {"z"}
        assert {c.doc_id for c in lib.chunks} == {"z"}
        assert doc_marker(1) not in lib.source
        assert all(doc_marker(1) not in c.text for c in lib.chunks)
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))
        assert {c.doc_id for c in lib.retrieve("short page")} == {"z"}

    async def test_a_swept_up_document_does_not_raise_a_bare_key_error(self):
        # With a pending document swept up, the caller's `pending` has two
        # entries — so `_chunk_document` reads `self.documents[doc_id]` for
        # the second one *after* the first one's LLM calls, which is long
        # enough for a concurrent `_reset()` to have wiped it. The public
        # API raised KeyError.
        lib, _ = self.librarian()
        tiny = f"{doc_marker(9)} held whole. " * 60
        assert len(tiny) < CHUNK_THRESHOLD
        await lib.add_document(tiny, doc_id="tiny.md")
        assert lib.chunks == []          # unsummarized, waiting to be swept

        adding = asyncio.create_task(
            lib.add_document(small_doc(1), doc_id="a.md")
        )
        await asyncio.sleep(0.001)
        indexed = asyncio.create_task(lib.index(small_doc(2), doc_id="z"))
        results = await asyncio.gather(adding, indexed, return_exceptions=True)

        assert [r for r in results if isinstance(r, BaseException)] == []
        assert list(lib.documents) == ["z"]
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks} == {"z"}

    async def test_an_add_arriving_mid_replace_lands_in_the_fresh_index(self):
        # The other order. An add that starts while index() is running is
        # held off, not lost, and adds to the index that replaced the old
        # one — which is what the whole-document lock did before it was
        # narrowed.
        lib, _ = self.librarian()
        await lib.add_document(small_doc(0), doc_id="old.md")

        indexed = asyncio.create_task(lib.index(small_doc(2), doc_id="z"))
        await asyncio.sleep(0.001)
        adding = asyncio.create_task(
            lib.add_document(small_doc(1), doc_id="a.md")
        )
        await asyncio.gather(indexed, adding)

        assert sorted(lib.documents) == ["a.md", "z"]
        assert "old.md" not in lib.documents
        assert doc_marker(0) not in lib.source
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))

    async def test_two_replaces_do_not_interleave(self):
        lib, _ = self.librarian()
        await asyncio.gather(
            lib.index(small_doc(1), doc_id="one"),
            lib.index(small_doc(2), doc_id="two"),
        )
        assert len(lib.documents) == 1
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}
        assert set(lib.documents) == {c.doc_id for c in lib.chunks}
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))

    async def test_replacing_does_not_serialize_the_adds(self):
        # The guard is between index() and the adds, not among the adds:
        # the cross-document overlap has to survive it.
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=8)
        lib._dense_ok = False
        await lib.index(small_doc(0), doc_id="first.md")
        peak_after_index = llm.peak

        await asyncio.gather(*(
            lib.add_document(small_doc(i), doc_id=f"p{i}.md")
            for i in range(1, 9)
        ))
        assert peak_after_index <= 2   # one two-chunk document on its own
        assert llm.peak > 2            # the adds still overlapped
        assert llm.peak <= 8


class TestWhatReachesTheEncoder:
    """The encoder is given `c.prefixed`, stated as a requirement.

    Five tests in this suite and `test_stale_vectors.py` assert it
    *incidentally* — they open `chunk_embeddings` and check each row against
    the chunk it is filed under, so they happen to notice the input as well.
    That is not the same as pinning it: their subject is correspondence, they
    would be satisfied by any input applied consistently, and the reason they
    catch a revert is `embedded_text`, which exists to keep the input in one
    place rather than to defend it. This class is where the requirement lives.

    **Why it is a requirement and not a preference.** `add_document` spends
    one LLM call per chunk writing a prefix that situates the chunk in its
    parent document; giving it to BM25 and withholding it from the encoder
    meant the dense half worked from strictly less text than the lexical
    half, so a chunk reading "the rate increased 3% year over year" was
    embedded with no idea which rate. Measured three times on one index with
    the prefixes held constant, each against a flat `bm25-only` control:
    recall at 10 rose in all three and more of the expected literals were
    found in all three (evidence: internal benchmark runs).
    The magnitude is deliberately not quoted here — each run resamples
    prefixes, so `@1`, `@3` and `@5` changed sign between runs and only the
    `@10` direction and the literal count reproduced.

    **Asserted on the encoder's input rather than on retrieval.** A
    behavioural test would have to show a query matching only prefix text
    ranking its chunk, and RRF hides a dead dense half — this suite already
    records that BM25 alone ranks these documents correctly, which is why a
    misaligned dense half is silent. So the input is the observable, and it
    is deterministic.

    Catches: the encoder input reverting to `c.text`, by name and with the
    missing prefix quoted; and the two halves being fed different text,
    which is the property the change is for.
    """

    class Recording(MarkedEmbedder):
        """A `MarkedEmbedder` that also keeps every batch it was handed.

        Subclassed rather than replaced so the rows stay content-derived:
        a test can assert what was encoded *and* that each row still
        carries the identity of its own text.
        """

        def __init__(self):
            self.batches: list[list[str]] = []

        def encode(self, texts, show_progress_bar: bool = False):
            self.batches.append(list(texts))
            return super().encode(texts, show_progress_bar=show_progress_bar)

    # Per-model now, not a librarium constant: nomic wants a task
    # prefix, bge-m3 wants none. What this class actually pins is that
    # the encoder is handed `c.prefixed` rather than `c.text`, which is
    # independent of the prefix and is the part that must not regress.
    INSTRUCTION = embed_prompts(EMBED_MODEL)[0]

    @staticmethod
    async def indexed():
        """A two-document index, and the chunk texts its encoder was given.

        **The fingerprint probe is separated out rather than skipped**, and
        it is why this helper exists. `embed_fingerprint` encodes
        `EMBED_PROBE` once per model object, so a double that records its
        batches sees one call that is not a chunk, and reading the batches as if they were all chunks
        misaligns every later row by one. Asserting the probe is exactly
        what is expected pins that documented extra call instead of
        quietly dropping whatever does not fit.
        """
        embedder = TestWhatReachesTheEncoder.Recording()
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=2)
        lib.embed_model = embedder
        await lib.add_document(doc_text(0), doc_id="first.md")
        await lib.add_document(doc_text(1), doc_id="second.md")

        instruction = TestWhatReachesTheEncoder.INSTRUCTION
        chunks, other = [], []
        for batch in embedder.batches:
            for text in batch:
                (chunks if text != EMBED_PROBE else other).append(text)
        assert other == [EMBED_PROBE], (
            f"the encoder saw calls this test cannot account for: {other!r}"
        )
        assert all(text.startswith(instruction) for text in chunks)
        return lib, chunks

    async def test_the_encoder_is_given_the_prefix_and_not_the_bare_chunk(self):
        lib, encoded = await self.indexed()
        assert len(encoded) == len(lib.chunks) > 4

        # Content-derived, and this is the half that makes a revert legible:
        # `WatchfulLLM` writes a prefix naming the chunk it was written for,
        # so each chunk's own prefix is a string that appears in what the
        # encoder saw only if the prefix was given to it.
        for chunk, text in zip(lib.chunks, encoded):
            assert chunk.prefix, "the double wrote no prefix, so this proves nothing"
            assert chunk.prefix in text, (
                f"chunk {chunk.index} of {chunk.doc_id} was embedded without "
                f"its prefix — the encoder saw {text[:80]!r}, which is "
                f"missing {chunk.prefix[:60]!r}"
            )
            assert text == f"{embed_prompts(EMBED_MODEL)[0]}{chunk.prefixed}"

        # And the bare chunk is *not* what was sent, stated separately so a
        # future input that merely contains the prefix cannot pass quietly.
        assert all(
            text != f"{embed_prompts(EMBED_MODEL)[0]}{chunk.text}"
            for chunk, text in zip(lib.chunks, encoded)
        )

    async def test_both_retrieval_halves_index_the_same_text(self):
        # The property the change exists for, and the reason the prefix is
        # worth encoding at all: the fusion's two halves read the same
        # material. Asserted against BM25's own per-chunk term frequencies
        # rather than against a second copy of the tokenizer call, so this
        # fails if either half is changed without the other.
        lib, encoded = await self.indexed()

        for i, (chunk, text) in enumerate(zip(lib.chunks, encoded)):
            dense_text = text[len(self.INSTRUCTION):]
            assert Counter(tokenize(dense_text)) == lib.bm25.doc_freqs[i], (
                f"chunk {i} reached the encoder and BM25 as different text"
            )


class TestFailedDocuments:
    """The cap has to hold when a call *fails*, not only when it succeeds.

    `asyncio.gather` re-raises the first exception but leaves the other
    children running. Those siblings keep holding slots on the document's
    semaphore and keep calling the backend for a document nobody is
    waiting for, while the next document builds a fresh semaphore with a
    full allowance. The allowances add and the cap is exceeded.
    """

    @staticmethod
    async def gather_documents(count: int, limit: int, failing: set[int]):
        llm = FlakyLLM({doc_marker(i) for i in failing})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=limit)
        lib._dense_ok = False
        results = await asyncio.gather(*(
            lib.add_document(doc_text(i), doc_id=f"d{i}") for i in range(count)
        ), return_exceptions=True)
        return lib, llm, results

    @pytest.mark.parametrize("count,limit,failing", [
        (6, 2, {1, 3, 4}),
        (6, 4, {0, 2, 5}),
        (4, 1, {1}),
    ])
    async def test_a_failure_does_not_leak_its_siblings(
        self, count, limit, failing
    ):
        lib, llm, results = await self.gather_documents(count, limit, failing)
        errors = [r for r in results if isinstance(r, RuntimeError)]
        assert len(errors) == len(failing)  # the failures really happened
        assert llm.peak <= limit

    async def test_a_failed_document_leaves_no_trace(self):
        lib, _, _ = await self.gather_documents(6, 2, {1, 3, 4})
        assert sorted(lib.documents) == ["d0", "d2", "d5"]
        for i in (1, 3, 4):
            assert f"d{i}" not in lib.documents
            assert f"d{i}" not in lib.doc_summaries
            # Still in `source` would mean still in content_hash(), and so
            # persisted into any index saved afterwards.
            assert doc_marker(i) not in lib.source

    async def test_no_document_is_left_summarized_but_unchunked(self):
        # The ghost the pending sweep cannot see: `_add_document` finds work
        # by looking for documents *without* a summary, so a document that
        # has one and no chunks is never re-chunked.
        lib, _, _ = await self.gather_documents(6, 2, {1, 3, 4})
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}

    async def test_the_documents_that_survived_are_fully_indexed(self):
        lib, _, _ = await self.gather_documents(6, 2, {1, 3, 4})
        assert {c.doc_id for c in lib.chunks} == {"d0", "d2", "d5"}
        assert all(c.prefix for c in lib.chunks)
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))

    async def test_an_earlier_pending_document_is_not_ghosted(self):
        # The ghost the caller's rollback cannot reach. A document under
        # CHUNK_THRESHOLD is held unchunked; a later, larger one pushes the
        # corpus over and the sweep chunks BOTH. When the earlier one's
        # prefixes fail, only the document being added gets rolled back —
        # so the earlier one's summary must never have been written.
        llm = FlakyLLM({doc_marker(7)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False

        small = f"{doc_marker(7)} short note. " * 150   # under the threshold
        await lib.add_document(small, doc_id="small.md")
        assert lib.chunks == [] and lib.doc_summaries == {}

        with pytest.raises(RuntimeError):
            await lib.add_document(doc_text(8), doc_id="big.md")

        # A summary with no chunks makes the document invisible to the
        # pending sweep, which finds work by looking for the absence of one.
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}
        assert "small.md" not in lib.doc_summaries

        llm.failing = set()  # backend recovers — the sweep must pick it up
        await lib.add_document(doc_text(9), doc_id="other.md")
        assert {c.doc_id for c in lib.chunks} == {"small.md", "other.md"}
        assert all(c.prefix for c in lib.chunks)

    async def test_a_failed_document_can_be_indexed_again(self):
        llm = FlakyLLM({doc_marker(0)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False

        with pytest.raises(RuntimeError):
            await lib.add_document(doc_text(0), doc_id="doc.md")
        assert "doc.md" not in lib.documents

        llm.failing = set()  # backend recovers
        await lib.add_document(doc_text(0), doc_id="doc.md")
        assert lib.doc_summaries["doc.md"]
        assert all(c.prefix for c in lib.chunks)
        assert {c.doc_id for c in lib.chunks} == {"doc.md"}

    async def test_a_failure_after_chunking_does_not_orphan_chunks(self):
        # Rolling the document back has to take its chunks with it, or a
        # failure landing after it chunked — a strict embedding error, say
        # — leaves chunks pointing at a doc_id no longer in the index.
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=2)
        lib._dense_ok = False
        await lib.add_document(doc_text(0), doc_id="good.md")
        before = list(lib.chunks)

        def boom():
            raise RuntimeError("embedding model exploded")

        lib._refresh_embeddings = boom
        with pytest.raises(RuntimeError, match="exploded"):
            await lib.add_document(doc_text(1), doc_id="late.md")

        assert "late.md" not in lib.documents
        assert lib.chunks == before
        assert {c.doc_id for c in lib.chunks} == {"good.md"}
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}

    async def test_a_rollback_does_not_leave_doc_summary_on_a_removed_doc(self):
        # `_chunk_document` writes `doc_summaries[doc_id]` and `doc_summary`
        # together. A rollback that pops only the former leaves
        # `doc_summary` holding the summary of a document that is no longer
        # in `documents` — the same staleness the under-threshold branch
        # already clears.
        llm = SummarizingLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        # Two survivors, not one: with a single document left behind, the
        # newest surviving summary and the oldest are the same string, so
        # nothing says which end of `doc_summaries` the fallback reads. The
        # newest is the one `_chunk_document` would have written.
        await lib.add_document(doc_text(0), doc_id="first.md")
        await lib.add_document(doc_text(1), doc_id="second.md")
        survivor = lib.doc_summary
        assert survivor == lib.doc_summaries["second.md"]
        assert survivor != lib.doc_summaries["first.md"]

        def boom():
            raise RuntimeError("reranker exploded")

        lib._load_reranker = boom
        with pytest.raises(RuntimeError, match="exploded"):
            await lib.add_document(doc_text(2), doc_id="late.md")

        assert doc_marker(2) not in lib.doc_summary
        assert lib.doc_summary in lib.doc_summaries.values()
        assert lib.doc_summary == lib.doc_summaries["second.md"] == survivor

    async def test_a_stale_doc_summary_is_not_persisted(self, tmp_path):
        # `save` writes `doc_summary` verbatim and `load` restores it, so a
        # stale value outlives the process that produced it.
        llm = SummarizingLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        await lib.add_document(doc_text(0), doc_id="good.md")

        def boom():
            raise RuntimeError("reranker exploded")

        lib._load_reranker = boom
        with pytest.raises(RuntimeError, match="exploded"):
            await lib.add_document(doc_text(1), doc_id="late.md")

        lib.save(tmp_path)
        payload = json.loads((tmp_path / "index.json").read_text())
        assert doc_marker(1) not in payload["doc_summary"]

        restored = SourceLibrarian(SummarizingLLM(), max_concurrent_prefixes=2)
        restored.load(tmp_path)
        assert restored.doc_summary in restored.doc_summaries.values()

    async def test_a_rollback_that_shrinks_the_corpus_leaves_no_ghost(self):
        # The rollback is the only thing in the library that makes `source`
        # smaller, and shrinking can drop it back under CHUNK_THRESHOLD —
        # reaching the hand-it-over-whole branch with summaries already
        # recorded, which nothing could do before. That branch drops the
        # chunks, so it has to drop the summaries with them: a summary
        # without chunks is invisible to the pending sweep forever.
        llm = FlakyLLM({doc_marker(1)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False

        small = f"{doc_marker(0)} short note. " * 150
        assert len(small) < CHUNK_THRESHOLD      # held whole on its own
        await lib.add_document(small, doc_id="small.md")

        with pytest.raises(RuntimeError):        # pushes over, then fails
            await lib.add_document(doc_text(1), doc_id="big.md")
        # The sweep chunked small.md on the way, and the rollback shrank
        # the corpus back under the threshold.
        assert lib.chunks and len(lib.source) <= CHUNK_THRESHOLD

        llm.failing = set()
        await lib.add_document(f"{doc_marker(2)} tiny. " * 5, doc_id="tiny.md")
        assert lib.chunks == []
        assert set(lib.doc_summaries) == {c.doc_id for c in lib.chunks}
        # `doc_summary` is written beside `doc_summaries`, so emptying one
        # without the other leaves a summary naming a document this branch
        # has just stopped chunking — and `save` persists it.
        assert lib.doc_summary == ""

        # Which is what keeps small.md reachable: the next document to push
        # the corpus over must re-chunk it.
        await lib.add_document(doc_text(3), doc_id="last.md")
        assert {c.doc_id for c in lib.chunks} == {
            "small.md", "tiny.md", "last.md"
        }
        assert all(c.prefix for c in lib.chunks)

    async def test_a_rollback_takes_the_embedding_rows_with_the_chunks(self):
        # `_refresh_embeddings` only encodes the chunks past the end of
        # `chunk_embeddings`, so rows a rollback leaves behind get handed
        # to the *next* document's chunks. The counts line up again at
        # that point, so the length guard in `_embedding_ranking` never
        # fires and save() persists the misfiled vectors.
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=2)
        lib.embed_model = MarkedEmbedder()
        await lib.add_document(doc_text(0), doc_id="good.md")
        assert len(lib.chunk_embeddings) == len(lib.chunks)

        def boom():
            raise RuntimeError("reranker exploded")

        lib._load_reranker = boom   # the step right after _refresh_embeddings
        with pytest.raises(RuntimeError, match="exploded"):
            await lib.add_document(doc_text(1), doc_id="late.md")
        assert len(lib.chunk_embeddings) == len(lib.chunks)

        lib._load_reranker = lambda: None
        await lib.add_document(doc_text(2), doc_id="third.md")
        assert len(lib.chunk_embeddings) == len(lib.chunks)
        assert {c.doc_id for c in lib.chunks} == {"good.md", "third.md"}
        # Matching counts are not enough — every row must be its chunk's.
        for chunk, vector in zip(lib.chunks, lib.chunk_embeddings):
            assert vector[0] == MarkedEmbedder.mark(embedded_text(chunk))

    @staticmethod
    async def emptied_by_a_rollback() -> tuple[SourceLibrarian, WatchfulLLM]:
        """A librarian whose only document was rolled back after chunking.

        The failure has to land *after* the pending sweep — a strict
        embedding error, a reranker that will not load — because that is
        where `use_chunking` has already been set to True.
        """
        llm = WatchfulLLM()
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False

        def boom():
            raise RuntimeError("reranker exploded")

        lib._load_reranker = boom
        with pytest.raises(RuntimeError, match="exploded"):
            await lib.add_document(doc_text(0), doc_id="only.md")
        return lib, llm

    async def test_a_rollback_that_empties_the_index_says_it_is_unindexed(self):
        # `use_chunking` is set before the index is built, so a failure in
        # any of the steps after it left the flag standing over an empty
        # index: `retrieval_mode` said "unindexed" while the flag said the
        # librarian had a chunked index to search.
        lib, _ = await self.emptied_by_a_rollback()
        assert lib.documents == {} and lib.source == "" and lib.chunks == []
        assert lib.use_chunking is False
        assert lib.retrieval_mode == "unindexed"
        assert lib.retrieve("anything") == []

    async def test_the_router_does_not_query_an_emptied_librarian(self):
        # What the stale flag cost: the router gates on
        # `use_chunking or source`, so it called answer() with empty source
        # material and took whatever the model invented back as facts —
        # with coverage "complete", which also short-circuits the web.
        lib, llm = await self.emptied_by_a_rollback()
        router = LibrarianRouter(llm, source_librarian=lib, use_web=False)

        result = await router.answer("what colour is the sky?")

        assert [s for s, _ in llm.calls if "SOURCE MATERIAL" in s] == []
        assert result.items == []
        assert result.coverage == "uncertain"

    async def test_a_rollback_leaves_bm25_covering_every_surviving_chunk(self):
        # The failing document contributed no chunks of its own, but the
        # pending sweep appended an EARLIER document's before it got there.
        # Rebuilding bm25 only when the failing document had chunks left
        # the lexical index covering fewer entries than `chunks`, and
        # `retrieve` indexes bm25_scores by chunk position.
        llm = FlakyLLM({doc_marker(1)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False

        small = f"{doc_marker(0)} short note. " * 150
        assert len(small) < CHUNK_THRESHOLD   # held whole: no chunks, no bm25
        await lib.add_document(small, doc_id="small.md")
        assert lib.bm25 is None

        with pytest.raises(RuntimeError):     # pushes over, then fails
            await lib.add_document(doc_text(1), doc_id="big.md")

        # The sweep chunked small.md on the way through.
        assert lib.chunks and {c.doc_id for c in lib.chunks} == {"small.md"}
        assert lib.bm25 is not None
        assert lib.bm25.corpus_size == len(lib.chunks)

    async def test_the_original_exception_type_reaches_the_caller(self):
        # Not an ExceptionGroup: cancelling the siblings must not change
        # what callers and `except` clauses see.
        llm = FlakyLLM({doc_marker(0)})
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        with pytest.raises(RuntimeError, match="backend refused"):
            await lib.index(doc_text(0))


class TestQueryingDuringAnIndex:
    """What a question asked mid-index gets.

    Indexing serializes against indexing; querying was never part of
    that bargain — `retrieve` is synchronous and takes no lock, so it
    reads whatever the index holds at the time. That is documented on
    `add_document` and `index`. These pin the two things it must be: a
    partial answer, and never an exception.
    """

    @staticmethod
    async def save_with_a_pending_document(tmp_path) -> None:
        """Persist an index whose second document was never chunked.

        `save` takes no lock either, so a checkpoint written while
        `add_document` is waiting on a summary call records that
        document as indexed — it joins `documents` on the first line of
        the add — with no summary and no chunks of its own. `load` reads
        it back and rebuilds bm25 from the chunks it did find, which is
        how a librarian starts life a document behind its own corpus
        with `use_chunking` still true.

        That `save` now *warns* about what it is writing (and would
        refuse under `strict`) rather than doing it silently. It still
        writes it, which is why this helper still works — and why it is
        cited in `test_load_concurrency` as the reason `save` warns
        instead of refusing: the partial file is a state the library
        tolerates, and this is the test that says so.
        """
        llm = GatedLLM(doc_marker(1))
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        await lib.index(doc_text(0), doc_id="d0")

        indexing = asyncio.ensure_future(
            lib.add_document(doc_text(1), doc_id="d1")
        )
        await llm.parked.wait()
        lib.save(tmp_path)
        llm.release.set()
        await indexing

    async def test_a_bm25_index_behind_the_chunks_does_not_raise(
        self, tmp_path, caplog
    ):
        # The window: a sweep with two pending documents extends `chunks`
        # for the first and only rebuilds bm25 after the last, so a query
        # landing on the await in between ranks positions bm25 never
        # scored. `retrieve` indexed straight into the score array.
        await self.save_with_a_pending_document(tmp_path)

        llm = GatedLLM(doc_marker(2))
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        lib.load(tmp_path)
        assert [d for d in lib.documents if d not in lib.doc_summaries] == ["d1"]

        indexing = asyncio.ensure_future(
            lib.add_document(doc_text(2), doc_id="d2")
        )
        await llm.parked.wait()
        assert len(lib.chunks) > lib.bm25.corpus_size   # d1's chunks, unscored

        chunks = lib.retrieve("topic 17", top_k=1)      # was an IndexError

        assert any("BM25 index covers" in r.getMessage() for r in caplog.records)
        # What the degradation IS, pinned so the docstrings cannot drift
        # away from it: neither ranking covers these chunks, so this is
        # the whole-document path — one entire document each, `top_k`
        # not applied, and d2, still being indexed, among them.
        assert [c.doc_id for c in chunks] == ["d0", "d1", "d2"] == list(lib.documents)
        assert [c.text for c in chunks] == list(lib.documents.values())
        # Which is far more material than the answering path will take:
        # a question a finished index answers raises here instead.
        with pytest.raises(ValueError, match="exceeds max_context_tokens"):
            await lib.answer("topic 17")

        # And the degradation lasts exactly as long as the window does.
        llm.release.set()
        await indexing
        assert lib.bm25.corpus_size == len(lib.chunks)
        assert doc_marker(2) in lib.retrieve(doc_marker(2), top_k=1)[0].text

    async def test_a_matching_bm25_index_ranks_every_chunk(self, caplog):
        # The guard fires on a length mismatch and nothing else: when the
        # lengths agree it must rank the whole corpus, not a prefix of it.
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=2)
        lib._dense_ok = False
        await lib.index(doc_text(0), doc_id="d0")
        await lib.add_document(doc_text(1), doc_id="d1")

        # Not the doc marker: it appears in half the corpus, so its IDF
        # term is zero and every score ties. A section number is rare
        # enough for the ranking to mean something.
        ranked = lib.retrieve("topic 17", top_k=len(lib.chunks))

        assert {c.index for c in ranked} == set(range(len(lib.chunks)))
        assert "topic 17" in ranked[0].text        # BM25's order...
        assert "topic 17" not in ranked[-1].text   # ...not chunk order
        assert not [r for r in caplog.records
                    if "BM25 index covers" in r.getMessage()]

    async def test_a_query_during_index_sees_the_replacement_in_progress(self):
        # `index()` replaces under the lock, and the query does not take
        # it: the documents being replaced are gone before the new one is
        # chunked. A wrong answer rather than an error, which is why the
        # docstrings say indexing-vs-indexing and this pins the rest.
        llm = GatedLLM(doc_marker(1))
        lib = SourceLibrarian(llm, max_concurrent_prefixes=2)
        lib._dense_ok = False
        await lib.index(doc_text(0), doc_id="d0")
        assert doc_marker(0) in lib.retrieve(doc_marker(0), top_k=1)[0].text

        indexing = asyncio.ensure_future(lib.index(doc_text(1), doc_id="d1"))
        await llm.parked.wait()
        during = lib.retrieve(doc_marker(0), top_k=3)

        assert lib.retrieval_mode == "whole-document"
        assert [c.doc_id for c in during] == ["d1"]
        assert doc_marker(0) not in during[0].text   # d0 is already gone

        llm.release.set()
        await indexing
        assert doc_marker(1) in lib.retrieve(doc_marker(1), top_k=1)[0].text


class TestSettingValidation:
    def test_negative_cap_is_rejected(self):
        # Only 0 is documented to mean unbounded; -3 reads like a mistake.
        with pytest.raises(ValueError, match="must be >= 0"):
            SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=-3)

    def test_zero_is_accepted_as_unbounded(self):
        assert SourceLibrarian(
            WatchfulLLM(), max_concurrent_prefixes=0
        ).max_concurrent_prefixes == 0

    def test_warns_when_the_pool_is_narrower_than_the_cap(self, caplog):
        class PooledLLM(WatchfulLLM):
            max_connections = 4

        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(PooledLLM(), max_concurrent_prefixes=16)
        assert "max_connections" in caplog.text

    def test_quiet_when_the_pool_is_wide_enough(self, caplog):
        class PooledLLM(WatchfulLLM):
            max_connections = 32

        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(PooledLLM(), max_concurrent_prefixes=8)
        assert "max_connections" not in caplog.text

    def test_quiet_for_clients_without_a_pool(self, caplog):
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=16)
        assert "max_connections" not in caplog.text

    def test_quiet_when_the_pool_is_unbounded(self, caplog):
        # An unbounded pool cannot be the narrower of the two.
        class PooledLLM(WatchfulLLM):
            max_connections = 0

        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(PooledLLM(), max_concurrent_prefixes=16)
        assert "max_connections" not in caplog.text

    def test_warns_when_an_unbounded_cap_meets_a_finite_pool(self, caplog):
        # The loudest narrowing there is, and the one that was silent: a
        # 2MB document fires roughly a thousand requests into 32 slots.
        class PooledLLM(WatchfulLLM):
            max_connections = 32

        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(PooledLLM(), max_concurrent_prefixes=0)
        assert "unbounded fan-out" in caplog.text
        assert "max_connections=32" in caplog.text

    def test_the_pool_warning_cannot_be_silenced_positionally(self):
        # `_warn_settings` trails six public arguments. As a
        # positional-or-keyword parameter a caller who miscounted would
        # land a stray seventh value on it and turn the warning off
        # without ever naming it — a private flag flipped by accident,
        # and silently, since the only symptom is a warning that stops
        # appearing. Keyword-only turns the miscount into a TypeError.
        with pytest.raises(TypeError, match="positional argument"):
            SourceLibrarian(
                PooledLLM(4), False, 6000, RESERVED_OUTPUT_TOKENS, False,
                16, False,
            )

    def test_the_pool_warning_is_still_suppressible_by_keyword(self, caplog):
        # The router and the web librarian check the cap themselves and
        # pass the flag by name to keep the warning from being emitted
        # twice; keyword-only must not break that.
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(PooledLLM(4), max_concurrent_prefixes=16,
                            _warn_settings=False)
        assert not pool_warnings(caplog)

    @pytest.mark.parametrize("bad", ["8", 8.0, None, True, False])
    def test_a_non_int_cap_gets_a_readable_error(self, bad):
        # Bare `< 0` turns a mistyped cap into "'<' not supported between
        # instances of 'str' and 'int'", which names neither the setting
        # nor the fix. The pool's own check already reads better.
        #
        # bool is in the list on purpose: it is an int as far as Python is
        # concerned, but True as a fan-out cap is a typo, not a request for
        # one slot at a time.
        with pytest.raises(TypeError, match="max_concurrent_prefixes"):
            SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=bad)

    @pytest.mark.parametrize("cap", [np.int64(4), np.int32(4), np.uint8(4)])
    def test_a_numpy_cap_is_accepted(self, cap):
        # numpy is a hard dependency, so a cap that fell out of a numpy
        # computation — a sizing heuristic, a value read out of an array —
        # is an ordinary thing to pass. Rejecting it on the type name alone
        # would be a regression against callers who already do.
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=cap)
        assert lib.max_concurrent_prefixes == 4

    def test_a_numpy_cap_is_stored_as_a_builtin_int(self):
        # Normalised at the boundary: the stored value goes on to size an
        # asyncio.Semaphore and to be formatted into warnings, and nothing
        # downstream should have to know where it came from.
        lib = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=np.int64(4))
        assert type(lib.max_concurrent_prefixes) is int

    def test_a_negative_numpy_cap_is_still_out_of_range(self):
        # Widening the accepted types must not widen the accepted values.
        with pytest.raises(ValueError, match="must be >= 0"):
            SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=np.int64(-3))

    def test_a_numpy_pool_still_triggers_the_narrowing_warning(self, caplog):
        # The warning reads `max_connections` off whatever client it is
        # given, so a client whose pool size came out of a computation must
        # not slip past it silently.
        class PooledLLM(WatchfulLLM):
            max_connections = np.int64(4)

        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(PooledLLM(), max_concurrent_prefixes=16)
        assert "max_connections=4" in caplog.text


class TestConnectionPool:
    """The pool is the other half of the cap: narrowing fan-out achieves
    nothing if the requests just queue in httpx instead.
    """

    def test_pool_is_sized_and_fully_keepalive(self):
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        client = OpenAICompatClient("http://localhost:8000/v1", "m",
                                    max_connections=24)
        assert client.max_connections == 24
        assert client.limits.max_connections == 24
        # Reconnecting per call against one endpoint is pure overhead.
        assert client.limits.max_keepalive_connections == 24

    def test_default_pool_admits_the_default_prefix_cap(self):
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        client = OpenAICompatClient("http://localhost:8000/v1", "m")
        assert client.max_connections >= MAX_CONCURRENT_PREFIXES

    def test_zero_is_an_unbounded_pool_not_a_closed_one(self):
        # 0 is the librarian's "unbounded", and the README points users at
        # this knob to match it. httpx spells unbounded None and reads a
        # literal 0 as a pool that admits nothing — with no default
        # pool_timeout that is not an error, it is a permanent hang.
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        client = OpenAICompatClient("http://localhost:8000/v1", "m",
                                    max_connections=0)
        assert client.max_connections == 0
        assert client.limits.max_connections is None
        assert client.limits.max_keepalive_connections is None

    def test_none_is_still_an_unbounded_pool(self):
        # httpx's own spelling of unbounded, and what this argument
        # accepted before 0 meant it. The validation must not turn a
        # working call into a TypeError from `None < 0`.
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        client = OpenAICompatClient("http://localhost:8000/v1", "m",
                                    max_connections=None)
        assert client.limits.max_connections is None
        assert client.limits.max_keepalive_connections is None

    def test_negative_pool_is_rejected(self):
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        with pytest.raises(ValueError, match="must be >= 0"):
            OpenAICompatClient("http://localhost:8000/v1", "m",
                               max_connections=-1)

    @pytest.mark.parametrize("bad", ["8", 8.0, True, False])
    def test_a_non_int_pool_gets_a_readable_error(self, bad):
        # None is absent from this list on purpose — it is the accepted
        # spelling of unbounded. bool is present: True is a typo, not a
        # one-connection pool.
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        with pytest.raises(TypeError, match="max_connections"):
            OpenAICompatClient("http://localhost:8000/v1", "m",
                               max_connections=bad)

    @pytest.mark.parametrize("pool", [np.int64(8), np.int32(8), np.uint8(8)])
    def test_a_numpy_pool_is_accepted(self, pool):
        # Same reasoning as the cap: numpy is a hard dependency, and the
        # two settings are documented as being sized against each other, so
        # whatever sizes one may well have sized the other.
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        client = OpenAICompatClient("http://localhost:8000/v1", "m",
                                    max_connections=pool)
        assert client.max_connections == 8
        assert client.limits.max_connections == 8

    def test_a_numpy_pool_is_stored_as_a_builtin_int(self):
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        client = OpenAICompatClient("http://localhost:8000/v1", "m",
                                    max_connections=np.int64(8))
        assert type(client.max_connections) is int

    def test_a_negative_numpy_pool_is_still_out_of_range(self):
        pytest.importorskip("httpx")
        from librarium.llm import OpenAICompatClient

        with pytest.raises(ValueError, match="must be >= 0"):
            OpenAICompatClient("http://localhost:8000/v1", "m",
                               max_connections=np.int64(-1))


class TestWebLibrarianCap:
    """A crawl is indexed against the same backend as everything else, so
    the cap has to reach the librarian WebLibrarian builds per question.
    """

    @staticmethod
    def web_module(monkeypatch):
        """`librarium.web` with the [web] extra's imports stubbed if absent.

        The scraper's third-party imports are module-level but only used
        inside fetch paths this test never reaches; stubbing them keeps
        the cap covered in the default dev environment, where the extra
        is not installed. The stubs go in through monkeypatch so they come
        back out again — a real trafilatura installed for a later test in
        the same session must not find ours sitting in `sys.modules`.

        The embedding model is stubbed on the same argument, one layer
        down. A crawl index is a full `SourceLibrarian`, built per
        question inside `answer()` with no handle to configure — so every
        test here that lets a crawl index used to download and build a
        real `nomic-embed-text-v1.5`, seven of the suite's sixteen. The
        stub leaves the dense half switched *on*, which is the state
        these tests were running in; only the weights are supplied
        locally.
        """
        import sys
        import types

        # Before any stubbing, so a skip here leaves sys.modules untouched.
        pytest.importorskip("httpx")

        stub_models(monkeypatch)

        for name, attr in (("trafilatura", "extract"), ("bs4", "BeautifulSoup")):
            try:
                __import__(name)
            except ImportError:
                stub = types.ModuleType(name)
                setattr(stub, attr, lambda *a, **k: None)
                monkeypatch.setitem(sys.modules, name, stub)
        import librarium.web as web

        return web

    @staticmethod
    def fake_crawl(lib):
        """Point the librarian's scraper at canned pages.

        Deliberately small pages: enough to chunk several times over the
        cap, small enough that the answer call at the end of `answer()`
        stays inside the default context budget.
        """
        class _Page:
            def __init__(self, url, title, text):
                self.url, self.title, self.text = url, title, text

        body = "\n\n".join(
            f"Paragraph {i} covers subject {i} at length. " * 12
            for i in range(6)
        )
        lib.scraper.crawl = lambda q: [
            _Page(f"https://example.test/{i}", f"Page {i}",
                  body.replace("Paragraph", f"P{i}"))
            for i in range(2)
        ]

    def test_the_cap_is_forwarded_to_the_crawl_index(self, monkeypatch):
        web = self.web_module(monkeypatch)
        llm = WatchfulLLM()
        lib = web.WebLibrarian(llm, max_concurrent_prefixes=2)
        self.fake_crawl(lib)

        asyncio.run(lib.answer("anything"))

        indexed = next(iter(lib._cache.values()))
        assert indexed.max_concurrent_prefixes == 2
        assert len(indexed.chunks) > 2  # the cap actually bound something
        assert llm.peak <= 2

    def test_the_default_matches_the_source_librarian(self, monkeypatch):
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM())
        assert lib.max_concurrent_prefixes == MAX_CONCURRENT_PREFIXES

    def test_negative_cap_is_rejected_at_construction(self, monkeypatch):
        # The librarian carrying the cap is not built until a question
        # falls through to the web, so without a check here the mistake
        # surfaces deep inside answer(), far from the call that made it.
        web = self.web_module(monkeypatch)
        with pytest.raises(ValueError, match="must be >= 0"):
            web.WebLibrarian(WatchfulLLM(), max_concurrent_prefixes=-3)

    def test_a_non_int_cap_is_rejected_at_construction(self, monkeypatch):
        web = self.web_module(monkeypatch)
        with pytest.raises(TypeError, match="max_concurrent_prefixes"):
            web.WebLibrarian(WatchfulLLM(), max_concurrent_prefixes="8")

    def test_a_narrow_pool_warns_at_construction(self, monkeypatch, caplog):
        # Same argument as the validation above: the librarian carrying the
        # cap is not built until a question falls through to the web.
        web = self.web_module(monkeypatch)
        with caplog.at_level("WARNING", logger="librarium.source"):
            web.WebLibrarian(PooledLLM(4), max_concurrent_prefixes=16)
        assert len(pool_warnings(caplog)) == 1

    def test_the_crawl_index_does_not_repeat_the_warning(
        self, monkeypatch, caplog
    ):
        web = self.web_module(monkeypatch)
        with caplog.at_level("WARNING", logger="librarium.source"):
            lib = web.WebLibrarian(PooledLLM(4), max_concurrent_prefixes=16)
            self.fake_crawl(lib)
            asyncio.run(lib.answer("anything"))
        assert len(pool_warnings(caplog)) == 1


class TestWebLibrarianTopK:
    """Same argument as the cap, and the crawl sharpens it: a question
    that falls through to the web indexes several pages at once, so this
    is the path most likely to overflow a tight context window.

    Borrows the module stub and the canned crawl from the cap tests
    rather than inheriting them, which would re-run that whole class.
    """

    web_module = staticmethod(TestWebLibrarianCap.web_module)
    fake_crawl = staticmethod(TestWebLibrarianCap.fake_crawl)

    def test_the_default_is_forwarded_to_the_crawl_index(self, monkeypatch):
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM(), top_k=3)
        self.fake_crawl(lib)

        asyncio.run(lib.answer("anything"))

        indexed = next(iter(lib._cache.values()))
        assert indexed.top_k == 3
        assert len(indexed.chunks) > 3  # the default actually bound something

    def test_a_per_call_top_k_reaches_the_crawl_index(self, monkeypatch):
        web = self.web_module(monkeypatch)
        llm = WatchfulLLM()
        lib = web.WebLibrarian(llm, top_k=8)
        self.fake_crawl(lib)

        asyncio.run(lib.answer("anything", top_k=2))

        indexed = next(iter(lib._cache.values()))
        # The librarian keeps its default; only this question was narrowed.
        assert indexed.top_k == 8
        answer_system, _ = llm.calls[-1]
        assert answer_system.count(CHUNK_JOINER) + 1 == 2

    def test_the_default_matches_the_source_librarian(self, monkeypatch):
        web = self.web_module(monkeypatch)
        assert web.WebLibrarian(WatchfulLLM()).top_k == TOP_K

    def test_a_bad_top_k_is_rejected_at_construction(self, monkeypatch):
        # The librarian carrying it is not built until a question falls
        # through to the web, so without a check here the mistake surfaces
        # deep inside answer().
        web = self.web_module(monkeypatch)
        with pytest.raises(ValueError, match="must be >= 1"):
            web.WebLibrarian(WatchfulLLM(), top_k=0)

    def test_a_non_int_top_k_is_rejected_at_construction(self, monkeypatch):
        web = self.web_module(monkeypatch)
        with pytest.raises(TypeError, match="top_k"):
            web.WebLibrarian(WatchfulLLM(), top_k="5")


class TestWebLibrarianIndexSettings:
    """The chunk geometry has to reach a crawl too. Scraped pages are not
    shaped like documents — many short ones rather than one long one — so
    a deployment that tuned chunk_size for its corpus and got the module
    default the moment a question fell through to the web was indexing
    the two halves differently and had no way to say otherwise.

    Borrows the module stub and the canned crawl from the cap tests, as
    the top_k tests above do.
    """

    web_module = staticmethod(TestWebLibrarianCap.web_module)
    fake_crawl = staticmethod(TestWebLibrarianCap.fake_crawl)

    def test_the_geometry_is_forwarded_to_the_crawl_index(self, monkeypatch):
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM(), chunk_size=600, chunk_overlap=50)
        self.fake_crawl(lib)

        asyncio.run(lib.answer("anything"))

        indexed = next(iter(lib._cache.values()))
        assert indexed.chunk_size == 600
        assert indexed.chunk_overlap == 50
        # Cut to that geometry, not merely carrying the number.
        assert len(indexed.chunks) > 2
        assert all(len(c.text) <= 600 for c in indexed.chunks)

    def test_the_context_budget_is_forwarded_to_the_crawl_index(
        self, monkeypatch
    ):
        # A crawl indexes several pages at once, so this is the path most
        # likely to need a budget other than the small-model default — and
        # it is built too late to set one on afterwards.
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM(), max_context_tokens=32_000,
                               reserved_output_tokens=2048, strict=True)
        self.fake_crawl(lib)

        asyncio.run(lib.answer("anything"))

        indexed = next(iter(lib._cache.values()))
        assert indexed.max_context_tokens == 32_000
        assert indexed.reserved_output_tokens == 2048
        assert indexed.strict is True

    def test_the_defaults_match_the_source_librarian(self, monkeypatch):
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM())
        direct = SourceLibrarian(WatchfulLLM())
        assert lib._index_settings["chunk_size"] == direct.chunk_size
        assert lib._index_settings["embed_model"] == direct.embed_model_name
        assert lib.max_context_tokens == direct.max_context_tokens
        assert lib.reserved_output_tokens == direct.reserved_output_tokens

    def test_a_bad_setting_is_rejected_at_construction(self, monkeypatch):
        # The librarian carrying it is not built until a question falls
        # through to the web, so without a check here the mistake surfaces
        # deep inside answer().
        web = self.web_module(monkeypatch)
        with pytest.raises(ValueError, match="chunk_size"):
            web.WebLibrarian(WatchfulLLM(), chunk_size=0)


class TestWebLibrarianExpansionSettings:
    """The query-expansion knobs, on the same argument as the geometry
    above: the librarian that carries them is built per question, so a
    caller reaching `WebLibrarian`'s crawl index would otherwise get the
    module default rather than the deployment's setting.

    `answer()` itself does not expand, so these assert on the librarian
    the crawl builds, not on what the model was sent.
    """

    web_module = staticmethod(TestWebLibrarianCap.web_module)
    fake_crawl = staticmethod(TestWebLibrarianCap.fake_crawl)

    def test_they_are_forwarded_to_the_crawl_index(self, monkeypatch):
        # Catches the layer that drops a knob: WebLibrarian holds these
        # in a dict of its own and splats it, so an addition made only to
        # SourceLibrarian goes inert here with no error anywhere.
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM(), expand_query=True,
                               expansion_variants=4)
        self.fake_crawl(lib)

        asyncio.run(lib.answer("anything"))

        indexed = next(iter(lib._cache.values()))
        assert indexed.expansion_enabled is True
        assert indexed.expansion_variants == 4

    def test_the_defaults_match_the_source_librarian(self, monkeypatch):
        web = self.web_module(monkeypatch)
        lib = web.WebLibrarian(WatchfulLLM())
        direct = SourceLibrarian(WatchfulLLM())
        assert lib._expansion_settings["expand_query"] \
            == direct.expansion_enabled
        assert lib._expansion_settings["expansion_variants"] \
            == direct.expansion_variants

    def test_a_bad_setting_is_rejected_at_construction(self, monkeypatch):
        # The librarian carrying it is not built until a question falls
        # through to the web, so without a check here the mistake surfaces
        # deep inside answer().
        web = self.web_module(monkeypatch)
        with pytest.raises(TypeError, match="expand_query"):
            web.WebLibrarian(WatchfulLLM(), expand_query=1)
        with pytest.raises(ValueError, match="expansion_variants"):
            web.WebLibrarian(WatchfulLLM(), expansion_variants=0)


class TestRouterCap:
    """`index()` builds a librarian when none was injected, and that one
    talks to the same backend as every other half of the router.
    """

    def test_the_cap_reaches_the_lazily_built_librarian(self, monkeypatch):
        # Built inside index(), so there is nothing to switch the dense
        # half off on — which is why this indexed against a real embedder
        # until the double was supplied through the loader's own import.
        stub_models(monkeypatch)
        llm = WatchfulLLM()
        router = LibrarianRouter(llm, use_web=False, max_concurrent_prefixes=2)
        assert router.source is None      # nothing injected: built on index()

        asyncio.run(router.index(MANY_CHUNKS_DOC))

        assert router.source.max_concurrent_prefixes == 2
        assert len(router.source.chunks) > 2  # the cap bound something real
        assert llm.peak <= 2

    def test_an_injected_librarian_keeps_its_own_cap(self):
        injected = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=3)
        router = LibrarianRouter(WatchfulLLM(), source_librarian=injected,
                                 max_concurrent_prefixes=2)
        assert router.source is injected
        assert router.source.max_concurrent_prefixes == 3

    def test_the_default_matches_the_source_librarian(self):
        router = LibrarianRouter(WatchfulLLM())
        assert router.max_concurrent_prefixes == MAX_CONCURRENT_PREFIXES

    def test_an_injected_librarian_silences_the_pool_warning(self, caplog):
        # A deployment tuned to 2 all the way through: pool 2, injected cap
        # 2. Nothing is misconfigured, but the router's own default of 8 is
        # what the check would see — and the advice it gives ("raise
        # max_connections") would undo the tuning.
        injected = SourceLibrarian(PooledLLM(2), max_concurrent_prefixes=2)
        with caplog.at_level("WARNING", logger="librarium.source"):
            LibrarianRouter(PooledLLM(2), source_librarian=injected,
                            use_web=False)
        assert pool_warnings(caplog) == []

    def test_a_cap_that_will_be_ignored_is_still_validated(self):
        # Ignored is not the same as unread: a typo is a mistake whether or
        # not the value ends up governing anything.
        injected = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=3)
        with pytest.raises(ValueError, match="must be >= 0"):
            LibrarianRouter(WatchfulLLM(), source_librarian=injected,
                            max_concurrent_prefixes=-3)
        with pytest.raises(TypeError, match="max_concurrent_prefixes"):
            LibrarianRouter(WatchfulLLM(), source_librarian=injected,
                            max_concurrent_prefixes="8")

    def test_an_ignored_cap_says_that_it_is_ignored(self, caplog):
        # Silence would leave the caller believing the 2 is in force.
        injected = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=3)
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(WatchfulLLM(), source_librarian=injected,
                            use_web=False, max_concurrent_prefixes=2)
        assert len(ignored_cap_warnings(caplog)) == 1

    def test_a_cap_nobody_passed_is_not_worth_a_warning(self, caplog):
        # Injecting a librarian is the documented way to set the cap; being
        # told the untouched default is ignored is noise, not news.
        injected = SourceLibrarian(WatchfulLLM(), max_concurrent_prefixes=3)
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(WatchfulLLM(), source_librarian=injected,
                            use_web=False)
        assert ignored_cap_warnings(caplog) == []

    def test_negative_cap_is_rejected_at_construction(self):
        # The librarian that validates the cap is built on the first
        # index(), so without a check here a typo survives construction.
        with pytest.raises(ValueError, match="must be >= 0"):
            LibrarianRouter(WatchfulLLM(), max_concurrent_prefixes=-3)

    def test_a_non_int_cap_is_rejected_at_construction(self):
        with pytest.raises(TypeError, match="max_concurrent_prefixes"):
            LibrarianRouter(WatchfulLLM(), max_concurrent_prefixes="8")

    def test_a_narrow_pool_warns_at_construction(self, caplog):
        # Same argument as the validation above: a router built with a cap
        # its client's pool cannot honour stayed silent until index().
        with caplog.at_level("WARNING", logger="librarium.source"):
            LibrarianRouter(PooledLLM(4), use_web=False,
                            max_concurrent_prefixes=16)
        assert len(pool_warnings(caplog)) == 1

    def test_the_lazily_built_librarian_does_not_repeat_the_warning(
        self, caplog, monkeypatch
    ):
        stub_models(monkeypatch)
        with caplog.at_level("WARNING", logger="librarium.source"):
            router = LibrarianRouter(PooledLLM(4), use_web=False,
                                     max_concurrent_prefixes=16)
            asyncio.run(router.index(MANY_CHUNKS_DOC))
        assert len(pool_warnings(caplog)) == 1
