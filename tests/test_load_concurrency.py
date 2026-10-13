"""`save()` and `load()` against indexing that is still running.

`index()` and `load()` do the same thing — throw away the index and put
another one in its place — and only `index()` was guarded. These pin what
the unguarded pair actually did (it is not theoretical: every shape
reproduces here), what the refusal replaced it with, and the two places the
refusal must *not* fire: `index(..., cache=...)`, which loads and saves
from inside the exclusive hold, and any librarian whose count belongs to a
loop that has since finished.

Two halves, because indexing arrives two ways. An `add_document` in
flight is *counted* (`_adds_in_flight_here`); an `index()` replacing the
librarian is not counted at all and is *owned* (`_replacing_elsewhere`),
because `asyncio.Lock` records no owner and "is the lock held" is the
wrong question — `index()` loads and saves from inside its own hold.

Determinism: every race here is sequenced by `GatedLLM`, which parks one
named document's *summary* call on an Event and releases it on command —
no sleeps, no clock. `WatchfulLLM.gate_at` is deliberately not used: it is
single-shot (CLAUDE.md's testing notes), so it cannot sequence the
two-document cases below, and none of these tests are about peak
concurrency anyway. Where the ordering of a *task* matters rather than of
an LLM call — `index()` having reached its wait before the add is let go —
:func:`wait_until_replacing` spins on the recorded owner rather than
sleeping, so the ordering is a fact rather than a hope.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import pytest

import librarium
from librarium import IndexBusyError, Librarian, SourceLibrarian
from tests.test_indexing import (
    GATE_TIMEOUT,
    GatedLLM,
    WatchfulLLM,
    doc_marker,
    doc_text,
    small_doc,
)

SOURCE_LOGGER = "librarium.source"

#: A word that appears in exactly one document of any index built here.
#:
#: `doc_marker` cannot do this job for a retrieval assertion. Two
#: `doc_text` documents put it in exactly half the corpus, where
#: BM25Okapi's IDF term is precisely zero, every score ties and the
#: ranking degenerates to index order — the trap `test_indexing` already
#: documents beside its own queries. "The orphaned chunks are handed back
#: by `retrieve`" has to be a real ranking to mean anything.
ODD_MARK = "ZORBLAX"


def odd_doc() -> str:
    """A many-chunk document whose vocabulary appears in no other."""
    return "\n\n".join(
        f"{ODD_MARK} marmalade cartography paragraph {i}. " * 45
        for i in range(24)
    )


def librarian(llm=None, **kwargs) -> SourceLibrarian:
    """A BM25-only librarian, model-free like the rest of the suite.

    `_dense_ok = False` rather than `stub_models`: nothing here is about
    vectors, and the flag keeps the embedding loader out of reach
    entirely. `MODEL_GUARD` is what proves that claim rather than this
    comment — a real construction fails the session.
    """
    lib = SourceLibrarian(llm or WatchfulLLM(), max_concurrent_prefixes=4,
                          **kwargs)
    lib._dense_ok = False
    return lib


async def park_an_add(lib, llm: GatedLLM, text: str, doc_id: str):
    """Start an `add_document` and return once it is parked mid-flight.

    Parked on the *summary* call, which is the window that matters:
    `_chunk_document` has already read `documents[doc_id]` and has not
    yet committed its chunks, so a reset landing here is what strands
    them. The returned task must be released (`llm.release.set()`) and
    awaited by the caller.

    The registration is read through `_adds_registered_on` and asked of
    **this** loop rather than of the librarian, which is both the accessor
    the source offers and the stronger claim: the count is keyed by the
    loop the add was taken on, so naming the loop here is what would catch
    a registration landing under somebody else's key.
    """
    adding = asyncio.ensure_future(lib.add_document(text, doc_id=doc_id))
    await asyncio.wait_for(llm.parked.wait(), GATE_TIMEOUT)
    here = asyncio.get_running_loop()
    assert lib._adds_registered_on(here) == 1, (
        "the add should be registered by now"
    )
    return adding


async def park_an_index(lib, llm: GatedLLM, text: str, doc_id: str):
    """Start an `index()` and return once it is parked mid-replacement.

    The `index()` sibling of :func:`park_an_add`, and a different window:
    the `_reset()` has already happened and the replace hold is already
    taken, so the librarian holds *only* the incoming document, with none
    of its chunks. That is the state a `save()` or `load()` landing here
    sees. Parked on the incoming document's summary call, which is inside
    `_replacing()` and inside `_add_document`.

    The returned task must be released (`llm.release.set()`) and awaited.
    """
    indexing = asyncio.ensure_future(lib.index(text, doc_id=doc_id))
    await asyncio.wait_for(llm.parked.wait(), GATE_TIMEOUT)
    here = asyncio.get_running_loop()
    assert lib._replace_owner_on(here) is indexing, (
        "index() should hold the replace"
    )
    return indexing


async def wait_until_replacing(lib) -> None:
    """Spin until `index()` is inside `_replacing()`.

    Spins on the recorded owner rather than on `_replace_lock.locked()`,
    which is what it used to watch: the owner is what the synchronous
    guard actually reads, and `_replacing` records it in the same step
    that acquires the lock — no await in between — so the two flip
    together and this watches the one that matters. Asserting on it
    rather than sleeping is the difference between a test that sequences
    the two tasks and one that usually does.

    Through `_replace_owner_on` and **not** by testing the field for
    `None`, which is the shape this had to change for: the owner is a
    `dict` keyed by loop now, and `{} is None` is False, so the raw test
    exited on its first evaluation and every caller was sequencing
    nothing while still looking as though it did.
    """
    async def spin():
        here = asyncio.get_running_loop()
        while lib._replace_owner_on(here) is None:
            await asyncio.sleep(0)

    await asyncio.wait_for(spin(), GATE_TIMEOUT)


def warnings_from(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING]


class TestLoadRefusesMidAdd:
    """`load()` raises rather than resetting under a running add.

    Catches: the guard being removed or weakened. Without it none of
    these raise — they corrupt instead, which is what
    :class:`TestTheCorruptionTheRefusalPrevents` shows.
    """

    async def test_load_raises_index_busy_error_while_an_add_is_in_flight(
        self, tmp_path
    ):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)

        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        with pytest.raises(IndexBusyError) as excinfo:
            lib.load(tmp_path)

        message = str(excinfo.value)
        assert "cannot load safely" in message
        assert "1 add_document call(s)" in message
        # The damage, not just the rule: the caller is holding a
        # librarian that still looks fine and has to be told why.
        assert "strand" in message

        # And it refused without touching anything — the librarian is the
        # one the caller had, not a half-replaced one.
        assert set(lib.documents) == {"d0", "d1"}
        assert {c.doc_id for c in lib.chunks} == {"d0"}

        llm.release.set()
        await adding
        # The add lands normally afterwards: the refusal cost it nothing.
        assert set(lib.doc_summaries) == {"d0", "d1"}
        assert {c.doc_id for c in lib.chunks} == {"d0", "d1"}

    async def test_it_refuses_before_it_even_looks_for_the_file(self, tmp_path):
        # Ordering, pinned: the refusal is the first thing `load` does, so
        # there is no argument about how far a bad call got. A guard
        # placed after the read would raise FileNotFoundError here.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        with pytest.raises(IndexBusyError):
            lib.load(tmp_path / "no-index-was-ever-written-here")

        llm.release.set()
        await adding

    async def test_the_refusal_lasts_exactly_as_long_as_the_add(self, tmp_path):
        # The other half of the contract, and the one an over-broad guard
        # breaks: a librarian that has finished its adds loads normally.
        # A guard reading a count that never came back down passes the
        # test above and fails this one.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)

        adding = await park_an_add(lib, llm, doc_text(1), "d1")
        llm.release.set()
        await adding
        assert lib._adds_registered_on(asyncio.get_running_loop()) == 0

        lib.load(tmp_path)
        assert list(lib.documents) == ["d0"]

    async def test_the_facade_raises_it_too(self, tmp_path):
        # `Librarian` is the class most callers touch, and it forwards
        # rather than reimplementing — but the exception has to arrive
        # with the name the docstring promises.
        llm = GatedLLM(doc_marker(1))
        lib = Librarian(llm, max_concurrent_prefixes=4)
        lib.source._dense_ok = False
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)

        adding = asyncio.ensure_future(
            lib.add_document(doc_text(1), doc_id="d1")
        )
        await asyncio.wait_for(llm.parked.wait(), GATE_TIMEOUT)

        with pytest.raises(IndexBusyError):
            lib.load(tmp_path)

        llm.release.set()
        await adding


class TestSaveMidAdd:
    """`save()` warns and writes; `strict` turns that into a refusal.

    Catches: a `save()` that goes back to silence (the state before this
    change), and a `save()` that refuses unconditionally — the design
    `test_indexing.TestQueryingDuringAnIndex` already rules out, since it
    manufactures a partial index with exactly this call.
    """

    async def test_it_warns_and_still_writes_the_partial_file(
        self, tmp_path, caplog
    ):
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        lib.save(tmp_path)

        assert any("cannot save safely" in m for m in warnings_from(caplog))
        # WARNING and not INFO: the library installs no handler, so an
        # INFO would report a partial file indistinguishably from not
        # reporting it.
        assert any(r.levelno == logging.WARNING and "cannot save" in r.getMessage()
                   for r in caplog.records)

        # Written, and partial in exactly the documented way: the file
        # claims d1 — it joined `documents`, and so `content_hash`, one
        # locked phase early — and holds no chunk and no summary for it.
        meta = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
        assert set(meta["documents"]) == {"d0", "d1"}
        assert {c["doc_id"] for c in meta["chunks"]} == {"d0"}
        assert "d1" not in meta["doc_summaries"]

        llm.release.set()
        await adding

    async def test_the_partial_file_recovers_on_the_next_add(
        self, tmp_path, caplog
    ):
        # The reason `save` warns instead of refusing: what it wrote is
        # recoverable. The un-chunked document keeps no summary, so it is
        # still pending when the file is read back and the next add
        # chunks it. If that were false the warn/raise split would be
        # wrong, not merely lenient.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        adding = await park_an_add(lib, llm, doc_text(1), "d1")
        lib.save(tmp_path)
        llm.release.set()
        await adding

        restored = librarian(WatchfulLLM())
        restored.load(tmp_path)
        assert [d for d in restored.documents
                if d not in restored.doc_summaries] == ["d1"]

        await restored.add_document(doc_text(2), doc_id="d2")
        assert set(restored.doc_summaries) == {"d0", "d1", "d2"}
        assert {c.doc_id for c in restored.chunks} == {"d0", "d1", "d2"}

    async def test_strict_refuses_and_writes_nothing(self, tmp_path):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm, strict=True)
        await lib.index(doc_text(0), doc_id="d0")
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        target = tmp_path / "never-created"
        with pytest.raises(IndexBusyError) as excinfo:
            lib.save(target)
        assert "cannot save safely" in str(excinfo.value)
        # It raises before `mkdir`, so `strict` really does mean no file
        # rather than a file plus an exception.
        assert not target.exists()

        llm.release.set()
        await adding

    async def test_a_drained_librarian_saves_quietly(self, tmp_path, caplog):
        # The over-broad-guard half again, for `save`: an ordinary
        # checkpoint after the adds must not warn about anything.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        lib = librarian(WatchfulLLM())
        await lib.index(doc_text(0), doc_id="d0")
        await lib.add_document(doc_text(1), doc_id="d1")

        lib.save(tmp_path)

        assert not [m for m in warnings_from(caplog) if "cannot save" in m]


class TestTheCachePathIsUnaffected:
    """`index(..., cache=...)` loads and saves from inside `_replacing()`.

    This is the regression that matters most: the guard sits on the hot
    path of every cached index, and it must read zero there because the
    adds are already drained. Catches a guard that consults the count
    without loop- or drain-awareness, and catches `_replacing()` being
    weakened so that the drain no longer happens.
    """

    @staticmethod
    async def a_cache_of(text: str, tmp_path) -> Path:
        cache = tmp_path / "cache"
        builder = librarian(WatchfulLLM())
        await builder.index(text, doc_id="cached", cache=cache)
        assert (cache / "index.json").exists()
        return cache

    async def test_a_cached_index_drains_the_add_and_still_reuses_the_cache(
        self, tmp_path, caplog
    ):
        cached_text = doc_text(2)
        cache = await self.a_cache_of(cached_text, tmp_path)

        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        caplog.set_level(logging.INFO, logger=SOURCE_LOGGER)
        indexing = asyncio.ensure_future(
            lib.index(cached_text, doc_id="z", cache=cache)
        )
        # index() is now parked on the add rather than racing it — which
        # is precisely why the load inside it reads zero in flight. The
        # claim is asserted rather than left to the helper: both halves
        # true at once is what "parked on the add" means, and while
        # `wait_until_replacing` was silently exiting early this test
        # passed by finishing the add first instead.
        await wait_until_replacing(lib)
        assert lib._replacing_elsewhere() is True
        assert lib._adds_in_flight_here() == 1
        assert "d1" in lib.documents      # nothing has been reset yet

        llm.release.set()
        await asyncio.gather(adding, indexing)   # no IndexBusyError

        messages = [r.getMessage() for r in caplog.records]
        assert any("reused cached index" in m for m in messages), (
            "the cache was not actually loaded, so this proves nothing"
        )
        assert not [m for m in messages if "cannot load safely" in m]
        assert not [m for m in messages if "cannot save safely" in m]

        # And the replacement is a self-consistent index: the add drained
        # into it and was then replaced, leaving nothing stranded.
        assert list(lib.documents) == ["z"]
        assert {c.doc_id for c in lib.chunks} == {"z"}
        assert doc_marker(1) not in lib.source

    async def test_a_cache_miss_saves_from_inside_the_hold_too(
        self, tmp_path, caplog
    ):
        # The other branch of `index(cache=)`: no cache to reuse, so it
        # indexes and then `save`s — also inside `_replacing()`, so also
        # silent, even with an add having been in flight when it started.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        cache = tmp_path / "fresh"
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        indexing = asyncio.ensure_future(
            lib.index(doc_text(2), doc_id="z", cache=cache)
        )
        await wait_until_replacing(lib)
        # Same ordering claim as the test above, asserted for the same
        # reason: the `save` this is about happens inside a hold taken
        # while the add was still counted.
        assert lib._replacing_elsewhere() is True
        assert lib._adds_in_flight_here() == 1
        llm.release.set()
        await asyncio.gather(adding, indexing)

        assert (cache / "index.json").exists()
        assert not [m for m in warnings_from(caplog) if "cannot save" in m]

    async def test_the_guard_is_live_on_that_path_if_the_count_is_not_zero(
        self, tmp_path, monkeypatch
    ):
        # The mutation check for the two above. They assert an absence,
        # which is only evidence if the presence was reachable: force the
        # observation non-zero and `index(cache=)` does raise, out of
        # `_load_if_current`'s `load()`. So the tests above are pinning
        # the drain, not a guard that happens to be inert here.
        cached_text = doc_text(2)
        cache = await self.a_cache_of(cached_text, tmp_path)

        lib = librarian(WatchfulLLM())
        monkeypatch.setattr(lib, "_adds_in_flight_here", lambda: 1)

        with pytest.raises(IndexBusyError):
            await lib.index(cached_text, doc_id="z", cache=cache)


class TestTheCorruptionTheRefusalPrevents:
    """The defect itself, pinned — not only the guard over it.

    `_adds_in_flight_here` is forced to 0, which is exactly what `save`
    and `load` observed before this change. What follows is what the
    library did then. Both tests are mutation checks by construction:
    they assert the damage the guard removes, so they say what the guard
    is worth rather than that it exists.
    """

    @staticmethod
    def unguard(monkeypatch, lib) -> None:
        monkeypatch.setattr(lib, "_adds_in_flight_here", lambda: 0)

    async def test_shape_1_load_strands_the_in_flight_documents_chunks(
        self, tmp_path, monkeypatch
    ):
        clean = tmp_path / "clean"
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(clean)

        adding = await park_an_add(lib, llm, odd_doc(), "d1")
        self.unguard(monkeypatch, lib)
        lib.load(clean)          # the reset the guard now refuses
        llm.release.set()
        await adding             # ...and the commit that lands on top of it

        # No exception anywhere. What is left is an index holding chunks
        # for a document it does not know it has.
        orphans = [c for c in lib.chunks if c.doc_id not in lib.documents]
        assert {c.doc_id for c in orphans} == {"d1"}
        assert "d1" in lib.doc_summaries and "d1" not in lib.documents
        assert ODD_MARK not in lib.source

        # They are not inert: retrieval ranks them and hands them back —
        # content from a document the index does not know it holds.
        hits = lib.retrieve(ODD_MARK, top_k=3)
        assert {c.doc_id for c in hits} == {"d1"}
        assert all(ODD_MARK in c.text for c in hits)

        # And `content_hash` cannot see them, so they survive to disk and
        # past the cache check — the corruption outlives the process.
        torn = tmp_path / "torn"
        lib.save(torn)
        written = json.loads((torn / "index.json").read_text(encoding="utf-8"))
        assert "d1" in {c["doc_id"] for c in written["chunks"]}
        assert "d1" not in written["documents"]

        fresh = librarian(WatchfulLLM())
        assert fresh._load_if_current(torn, doc_text(0)) is True
        assert any(c.doc_id == "d1" for c in fresh.chunks)

    async def test_the_guard_makes_shape_1_unreachable(self, tmp_path):
        # The same script with the guard intact. Nothing is stranded,
        # because the reset never happens.
        clean = tmp_path / "clean"
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(clean)

        adding = await park_an_add(lib, llm, odd_doc(), "d1")
        with pytest.raises(IndexBusyError):
            lib.load(clean)
        llm.release.set()
        await adding

        assert [c for c in lib.chunks if c.doc_id not in lib.documents] == []
        assert set(lib.documents) == set(lib.doc_summaries) == {"d0", "d1"}
        assert ODD_MARK in lib.source
        assert {c.doc_id for c in lib.retrieve(ODD_MARK, top_k=3)} == {"d1"}

    async def test_shape_2_a_swept_up_document_raises_a_bare_key_error(
        self, tmp_path, monkeypatch
    ):
        # Two documents in one pending sweep: the second
        # `_chunk_document` reads `self.documents[doc_id]` after the
        # reset removed it. The failure `_replace_guard`'s docstring
        # predicts for an unguarded reset, reached through `load`.
        clean = tmp_path / "clean"
        llm = GatedLLM(doc_marker(9))
        lib = librarian(llm)
        tiny = f"{doc_marker(9)} held whole. " * 60
        await lib.add_document(tiny, doc_id="tiny.md")
        assert lib.chunks == []          # unsummarized, waiting to be swept
        lib.save(clean)

        # The sweep chunks tiny.md first, so parking on *its* summary is
        # what puts the reset between the two documents.
        adding = await park_an_add(lib, llm, doc_text(1), "d1")
        self.unguard(monkeypatch, lib)
        lib.load(clean)
        llm.release.set()

        with pytest.raises(KeyError) as excinfo:
            await adding
        assert "d1" in str(excinfo.value)

    async def test_the_guard_makes_shape_2_unreachable(self, tmp_path):
        clean = tmp_path / "clean"
        llm = GatedLLM(doc_marker(9))
        lib = librarian(llm)
        tiny = f"{doc_marker(9)} held whole. " * 60
        await lib.add_document(tiny, doc_id="tiny.md")
        lib.save(clean)

        adding = await park_an_add(lib, llm, doc_text(1), "d1")
        with pytest.raises(IndexBusyError):
            lib.load(clean)
        llm.release.set()
        await adding                     # no KeyError

        assert set(lib.documents) == {"tiny.md", "d1"}
        assert set(lib.doc_summaries) == {"tiny.md", "d1"}


class TestTheErrorType:
    """`IndexBusyError`'s base class and its export are both load-bearing."""

    def test_it_is_exported_from_the_package(self):
        # A caller cannot opt out of needing this name — `load` raises it
        # with no flag set — so it has to be importable from the top.
        assert librarium.IndexBusyError is IndexBusyError
        assert "IndexBusyError" in librarium.__all__

    def test_it_is_a_runtime_error_and_not_a_value_error(self):
        assert issubclass(IndexBusyError, RuntimeError)
        assert not issubclass(IndexBusyError, ValueError)

    async def test_a_value_error_there_would_be_swallowed_into_a_re_index(
        self, tmp_path, caplog
    ):
        # Why the base class is not the module's usual ValueError, shown
        # rather than asserted: `_load_if_current` catches ValueError
        # around `load()`. As one, a refusal would become a silent,
        # expensive re-index — the opposite of converting corruption into
        # a loud error.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        cache = tmp_path / "cache"
        builder = librarian(WatchfulLLM())
        await builder.index(doc_text(0), doc_id="d0", cache=cache)

        def refuse(error):
            def load(path):
                raise error
            return load

        as_value_error = librarian(WatchfulLLM())
        as_value_error.load = refuse(ValueError("adds are in flight"))
        assert as_value_error._load_if_current(cache, doc_text(0)) is False
        assert any("failed to load" in m for m in warnings_from(caplog))

        as_runtime_error = librarian(WatchfulLLM())
        as_runtime_error.load = refuse(IndexBusyError("adds are in flight"))
        with pytest.raises(IndexBusyError):
            as_runtime_error._load_if_current(cache, doc_text(0))


class TestTheCountIsLoopKeyed:
    """A count that belongs to a finished loop must not wedge `load`.

    A registration survives the loop it was counted on. Read raw it would
    make every later `load()` refuse forever over an add that ended with
    that loop — a much worse failure than the one being fixed, and one no
    caller could clear. Both branches of the loop check are here.

    The state is planted **under the loop's own key**, which is the shape
    the registrations are kept in (`dict[loop, int]`, read through
    `_adds_registered_on`). There is no accessor that writes, and a
    hand-built entry is the point of this class — `TestALoopThatEndedUnderPendingWork`
    is the same state reached through the public API.
    """

    def test_no_running_loop_means_nothing_to_race(self, tmp_path):
        lib = librarian(WatchfulLLM())
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)
        # What an add cancelled at loop teardown leaves behind.
        lib._adds_in_flight = {lib._replace_loop: 2}

        lib.load(tmp_path)      # the ordinary way an index is loaded

        assert list(lib.documents) == ["d0"]

    def test_a_count_from_another_loop_is_not_this_loops_business(
        self, tmp_path
    ):
        lib = librarian(WatchfulLLM())
        first = asyncio.new_event_loop()
        try:
            first.run_until_complete(lib.index(doc_text(0), doc_id="d0"))
        finally:
            first.close()
        lib.save(tmp_path)
        lib._adds_in_flight = {first: 2}
        assert lib._replace_loop is first

        # A loop *is* running here — just not the one that counted those
        # adds, so they cannot be racing this call either.
        async def load_from_a_different_loop():
            lib.load(tmp_path)

        second = asyncio.new_event_loop()
        try:
            second.run_until_complete(load_from_a_different_loop())
        finally:
            second.close()

        assert list(lib.documents) == ["d0"]

    def test_save_reads_the_same_count(self, tmp_path, caplog):
        # Same helper, both callers: a stranded count must not make every
        # later `save` announce a partial file that is not partial.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        lib = librarian(WatchfulLLM())
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib._adds_in_flight = {lib._replace_loop: 2}

        lib.save(tmp_path)

        assert not [m for m in warnings_from(caplog) if "cannot save" in m]


class TestTheObserversReadEveryLiveLoop:
    """The synchronous pair sums *all* live loops, not the guard's alone.

    "How much indexing can still commit into this index" is a fact about
    the librarian; which loop it started on is not part of the question.
    Asking only `_replace_loop` is what a single `int` forced, and it is
    what made the reading destroyable — a handover moved the guard to a new
    loop and everything registered on the old one became invisible, so a
    `load()` was admitted and a live document's chunks were left orphaned
    (D2, measured in `test_loop_handover.py` with the refusal off).

    Hand-built entries, because reaching two live loops holding
    registrations at once through the API means getting past the handover
    refusal, which is that other file's subject. What is isolated here is
    the arithmetic: every live entry counts, a closed one does not, and the
    guard's own loop is not privileged among them.

    Catches: either observer being narrowed back to `_adds_registered_on()`
    / `_replace_owner_on()` — the guard's loop only — and the per-entry
    liveness being dropped so a closed loop's registration is summed in.
    """

    @staticmethod
    def three_loops():
        """Two open loops and one closed, none of them running."""
        first, second = asyncio.new_event_loop(), asyncio.new_event_loop()
        dead = asyncio.new_event_loop()
        dead.close()
        return first, second, dead

    def test_the_count_is_the_sum_over_the_live_loops(self):
        lib = librarian(WatchfulLLM())
        first, second, dead = self.three_loops()
        try:
            lib._replace_loop = second
            lib._adds_in_flight = {first: 1, second: 2, dead: 4}

            # 1 + 2, and emphatically not 2 (the guard's loop alone), 3+4
            # (liveness ignored) or 4.
            assert lib._adds_in_flight_here() == 3
        finally:
            first.close()
            second.close()

    def test_an_owner_on_a_loop_that_is_not_the_guards_still_refuses(self):
        lib = librarian(WatchfulLLM())
        first, second, dead = self.three_loops()
        try:
            # The guard has moved on to `second`, which holds nothing; the
            # `index()` recorded on `first` is still live and still owns the
            # librarian. A reader that only asked `_replace_owner_on()`
            # would answer False here and admit a `save()`/`load()` into
            # the window `TestLoadRefusesMidIndex` measures corrupting.
            lib._replace_loop = second
            lib._replace_owner = {first: object()}
            assert lib._replacing_elsewhere() is True

            # And an owner belonging to a loop that has closed is ignored,
            # which is the wedge staying closed.
            lib._replace_owner = {dead: object()}
            assert lib._replacing_elsewhere() is False
        finally:
            first.close()
            second.close()


class TestTheObserversSurviveAnEntryVanishingMidWalk:
    """Neither observer raises when the dict changes under its walk.

    Both walk `list(self._adds_in_flight.items())` /
    `list(self._replace_owner.items())` rather than the live view, and the
    reason is **not** `_forget_closed_loops`'s: those two do not mutate.
    What they are avoiding is a resize by *another thread's* loop
    mid-walk — an `_adding` on that loop popping its entry at zero between
    two steps of this iteration. `save` and `load` are ordinary `def`s that
    a plain worker thread calls (`TestLoadAndSaveFromAnotherThread`), so
    without the copy a caller who did nothing wrong gets
    `RuntimeError: dictionary changed size during iteration` out of a
    checkpoint instead of an answer.

    **Injected rather than raced.** Winning that window by timing would be
    a flaky test that proved nothing on the runs it lost, so the mutation
    is applied at a boundary instead: `_observe_replace_guard` is the one
    call each walk makes per entry, and a double that pops an entry on its
    first call puts the resize exactly where a thread would. That is the
    same technique `identity_observer` uses in `test_guard_liveness.py`,
    and the double delegates to the real observer so the walk's own logic
    is unchanged.

    Both tests assert through the public API as well as on the observer,
    because the damage is not a wrong number — it is the wrong *exception*
    reaching a caller, which is what
    `TestTheOwnerCheckFromAThreadWithNoLoop
    ::test_a_thread_gets_the_refusal_and_not_asyncios_complaint` already
    pins for the other way this pair can raise something irrelevant.

    Catches: either `list(...)` being dropped from the two observers.
    Verified both directions — with the copies these pass, and with either
    removed the matching test fails on the `RuntimeError`.
    """

    @staticmethod
    def three_loops():
        """Two open loops and one closed, none of them running."""
        first, second = asyncio.new_event_loop(), asyncio.new_event_loop()
        dead = asyncio.new_event_loop()
        dead.close()
        return first, second, dead

    @staticmethod
    def observer_that_drops(lib, field: str, victim):
        """The real observer, plus one pop from *field* on the first call.

        Stands in for the other thread: `_adding` there reaching zero and
        popping its own entry while this walk is between two steps. The
        real bound method is captured first, so the answers are the
        librarian's own and only the timing is fabricated.
        """
        real = lib._observe_replace_guard
        dropped = False

        def observe(loop=None):
            nonlocal dropped
            if not dropped:
                dropped = True
                getattr(lib, field).pop(victim, None)
            return real(loop)

        return observe

    def test_the_count_observer_answers_while_an_entry_is_dropped(
        self, tmp_path, monkeypatch
    ):
        lib = librarian(WatchfulLLM())
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)
        first, second, dead = self.three_loops()
        try:
            lib._replace_loop = second
            # `dead` is the entry that vanishes, and it is last so the walk
            # still has a step left to take when it goes. Closed, so it
            # contributes nothing either way and the total is unambiguous.
            lib._adds_in_flight = {first: 1, second: 2, dead: 4}
            monkeypatch.setattr(
                lib, "_observe_replace_guard",
                self.observer_that_drops(lib, "_adds_in_flight", dead),
            )

            assert lib._adds_in_flight_here() == 3
            assert dead not in lib._adds_in_flight, "the double did not fire"

            # And the answer reaches the caller as a refusal about adds,
            # not as an iteration error about a dict they cannot see.
            lib._adds_in_flight = {first: 1, second: 2, dead: 4}
            monkeypatch.setattr(
                lib, "_observe_replace_guard",
                self.observer_that_drops(lib, "_adds_in_flight", dead),
            )
            with pytest.raises(IndexBusyError) as excinfo:
                lib.load(tmp_path)
            assert "3 add_document call(s)" in str(excinfo.value)
        finally:
            first.close()
            second.close()

    def test_the_owner_observer_answers_while_an_entry_is_dropped(
        self, tmp_path, monkeypatch
    ):
        lib = librarian(WatchfulLLM())
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        first, second, dead = self.three_loops()
        spare = asyncio.new_event_loop()
        spare.close()
        try:
            lib._replace_loop = second
            # `dead` first so the walk `continue`s past it and keeps going;
            # `spare` is what vanishes; `first` is the live owner the walk
            # must still reach and refuse over. Without the snapshot the
            # step after the pop raises instead.
            owners = {dead: object(), spare: object(), first: object()}
            lib._replace_owner = dict(owners)
            monkeypatch.setattr(
                lib, "_observe_replace_guard",
                self.observer_that_drops(lib, "_replace_owner", spare),
            )

            assert lib._replacing_elsewhere() is True
            assert spare not in lib._replace_owner, "the double did not fire"

            lib._replace_owner = dict(owners)
            monkeypatch.setattr(
                lib, "_observe_replace_guard",
                self.observer_that_drops(lib, "_replace_owner", spare),
            )
            target = tmp_path / "never-created"
            with pytest.raises(IndexBusyError) as excinfo:
                lib.save(target)
            assert "index() is replacing this librarian" in str(excinfo.value)
            assert not target.exists()
        finally:
            first.close()
            second.close()


class TestTheEntriesDoNotAccumulate:
    """The bookkeeping is O(loops still open), not O(loops ever used).

    `asyncio.run()` per request over one long-lived librarian is a
    realistic shape, and an entry per loop that ever indexed would be a
    leak for the life of the process. Two mechanisms bound it and they
    catch different things, so both are measured: a balanced loop drops its
    own entry (`_adding` pops at zero, `_replacing` pops in its `finally`),
    and `_forget_closed_loops` drops what nothing gives back.

    Catches: the pop-at-zero being replaced by leaving a `0` behind, which
    the first test fails; and `_forget_closed_loops` being dropped, which
    `TestALoopThatEndedUnderPendingWork` fails, since the tidy case never
    needs it.
    """

    def test_many_short_lived_loops_leave_nothing_behind(self):
        lib = librarian(WatchfulLLM())
        asyncio.run(lib.index(small_doc(0), doc_id="d0"))

        # `small_doc`, not `doc_text`: what is under test is the *loop*
        # bookkeeping, and a document's chunk count is not part of it. The
        # many-chunk document cost 0.44s an add against 0.043s and made
        # this one test 27% of the whole suite, buying nothing — the
        # entries this asserts about are per `asyncio.run`, not per chunk.
        # Both still chunk, which is the property that matters: `small_doc`
        # is sized just over CHUNK_THRESHOLD for exactly that reason.
        #
        # Enough runs that a leak is unambiguous rather than arguable, and
        # each `asyncio.run` closes its loop — the ordinary legal pattern,
        # not a stranding. Checked after *every* run rather than only at
        # the end, which is what makes this "O(live loops)" and not "a
        # number that happens to come out small": between two
        # `asyncio.run` calls there is no live loop at all, so the right
        # answer is always nothing. Asserting only the endpoint would also
        # pass on a version that grew and was pruned back.
        for n in range(1, 26):
            asyncio.run(lib.add_document(small_doc(n % 4), doc_id=f"d{n}"))
            assert lib._adds_in_flight == {}, (
                f"run {n} left {lib._adds_in_flight} behind"
            )
            assert lib._replace_owner == {}, (
                f"run {n} left {lib._replace_owner} behind"
            )
        # And the librarian still works, which is the other half of "these
        # were dropped because they were finished".
        assert len(lib.doc_summaries) == 26
        assert lib._adds_in_flight_here() == 0

    def test_an_index_gives_its_own_entry_back_too(self):
        # `_replacing` records an owner under its loop's key, so the same
        # question applies to it: 25 cached indexes must not leave 25
        # owners behind.
        # `small_doc` for its sibling's reason: what accumulates is one
        # entry per `asyncio.run`, and the document's chunk count is not
        # part of that.
        lib = librarian(WatchfulLLM())
        for _ in range(25):
            asyncio.run(lib.index(small_doc(0), doc_id="d0"))

        assert lib._replace_owner == {}
        assert lib._replacing_elsewhere() is False


class TestTheHoldIsObservedByOwnerNotByLock:
    """`_replacing_elsewhere` asks *who* holds, not *whether* it is held.

    The distinction is the whole design. `index()` calls
    `_load_if_current` and `save(cache)` from inside its own
    `_replacing()`, so a guard written as `self._replace_lock.locked()`
    passes every "a stranger is refused" test in this file and breaks the
    one path that must never be refused.
    """

    async def test_the_owner_is_not_a_stranger_to_its_own_hold(self, tmp_path):
        # The mutation check, stated directly: the lock *is* held and the
        # holder is *not* elsewhere, at the same instant. `locked()` as
        # the predicate flips the second assertion and both calls below.
        lib = librarian(WatchfulLLM())
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)

        async with lib._replacing():
            assert lib._replace_lock.locked()
            assert lib._replacing_elsewhere() is False
            # And so both whole-index operations are admitted from here,
            # which is exactly what `_load_if_current` and the trailing
            # `save(cache)` do inside `index()`.
            lib.save(tmp_path)
            lib.load(tmp_path)

        assert list(lib.documents) == ["d0"]

    async def test_another_task_inside_the_hold_is_a_stranger(self):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")

        indexing = await park_an_index(lib, llm, doc_text(1), "fresh")

        # Counted indexing is zero here — this is the half a counter
        # cannot see, which is why the owner exists at all.
        assert lib._adds_in_flight_here() == 0
        assert lib._replacing_elsewhere() is True

        llm.release.set()
        await indexing

    async def test_the_hold_is_given_back_when_index_returns(self, tmp_path):
        # The over-broad-guard half: a recorded owner that is never
        # cleared makes every later `load()` refuse forever, which is a
        # worse failure than the one being fixed and one no caller can
        # clear.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)

        indexing = await park_an_index(lib, llm, doc_text(1), "fresh")
        llm.release.set()
        await indexing

        assert lib._replace_owner_on(asyncio.get_running_loop()) is None
        # And given back by being *removed*, not by being set to None: an
        # entry per loop that ever ran an `index()` would never be
        # collected, which is what `_forget_closed_loops` documents.
        assert lib._replace_owner == {}
        assert lib._replacing_elsewhere() is False
        lib.load(tmp_path)
        assert list(lib.documents) == ["d0"]

    async def test_a_cancelled_index_gives_the_hold_back_too(self, tmp_path):
        # Why the clear sits in an await-free `finally`: a cancelled
        # replace still has to give the librarian back. An `await` on
        # that path could be cancelled in turn and leave a finished
        # `index()` recorded as the owner.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)

        indexing = await park_an_index(lib, llm, doc_text(1), "fresh")
        indexing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await indexing

        assert lib._replace_owner == {}
        assert not lib._replace_lock.locked()
        lib.load(tmp_path)              # no IndexBusyError
        assert list(lib.documents) == ["d0"]

    async def test_a_loop_callback_holds_no_task_and_is_a_stranger(
        self, tmp_path
    ):
        # `asyncio.current_task()` is None outside a task. That is not
        # the owner either, so it is refused like any other stranger —
        # the safe direction, since a callback provably is not the
        # `index()` that took the hold. Pinned because the other reading
        # ("no task, so nothing to compare, so admit") would let a
        # `call_soon` write the lying file measured in
        # `TestSaveRefusesMidIndex`.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        indexing = await park_an_index(lib, llm, doc_text(1), "fresh")

        seen: dict[str, object] = {}
        done = asyncio.Event()

        def from_a_callback():
            seen["task"] = asyncio.current_task()
            seen["elsewhere"] = lib._replacing_elsewhere()
            try:
                lib.save(tmp_path / "from-a-callback")
            except IndexBusyError as error:
                seen["error"] = error
            done.set()

        asyncio.get_running_loop().call_soon(from_a_callback)
        await asyncio.wait_for(done.wait(), GATE_TIMEOUT)

        assert seen["task"] is None, "a call_soon callback runs in no task"
        assert seen["elsewhere"] is True
        assert "cannot save safely" in str(seen.get("error"))
        assert not (tmp_path / "from-a-callback").exists()

        llm.release.set()
        await indexing

    async def test_a_plain_cached_index_is_never_refused(self, tmp_path, caplog):
        # The hot path in its purest form: no add anywhere, nothing
        # concurrent, the ordinary two-run cached index. It goes through
        # `save()` (miss) and `load()` (hit) from inside `_replacing()`,
        # and a `locked()` guard refuses both. This is the test that
        # fails first if anyone simplifies the owner check away.
        caplog.set_level(logging.INFO, logger=SOURCE_LOGGER)
        cache = tmp_path / "cache"
        text = doc_text(3)

        first = librarian(WatchfulLLM())
        await first.index(text, doc_id="cached", cache=cache)     # save inside
        assert (cache / "index.json").exists()

        second = librarian(WatchfulLLM())
        await second.index(text, doc_id="cached", cache=cache)    # load inside

        messages = [r.getMessage() for r in caplog.records]
        assert any("reused cached index" in m for m in messages)
        assert not [m for m in messages if "cannot load safely" in m]
        assert not [m for m in messages if "cannot save safely" in m]
        assert list(second.documents) == ["cached"]


class TestLoadRefusesMidIndex:
    """`load()` raises against a concurrent `index()`, not only an add.

    The counter half of the guard cannot see this: `_adds_in_flight` is
    raised by `add_document` alone, and `index()` — the very thing the
    replace guard exists for — never touches it. Catches the guard
    reverting to the counter alone, whose cost
    :meth:`test_the_corruption_the_refusal_prevents` measures.
    """

    async def test_load_refuses_while_another_task_is_replacing(self, tmp_path):
        clean = tmp_path / "clean"
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(clean)

        indexing = await park_an_index(lib, llm, odd_doc(), "fresh")
        assert lib._adds_in_flight_here() == 0, "no add is in flight at all"

        with pytest.raises(IndexBusyError) as excinfo:
            lib.load(clean)

        message = str(excinfo.value)
        assert "cannot load safely" in message
        assert "index() is replacing this librarian from another task" in message
        # The damage, not just the rule.
        assert "stranding the document it is indexing" in message

        llm.release.set()
        await indexing

        # The index() finished into a self-consistent librarian: nothing
        # was stranded, because the reset never happened underneath it.
        assert list(lib.documents) == ["fresh"]
        assert {c.doc_id for c in lib.chunks} == {"fresh"}
        assert [c for c in lib.chunks if c.doc_id not in lib.documents] == []

    async def test_the_corruption_the_refusal_prevents(
        self, tmp_path, monkeypatch
    ):
        # The defect itself, with the replace half of the guard forced
        # off — which is the guard exactly as it shipped before this
        # round, since the counter reads zero throughout. What follows is
        # what `load()` mid-`index()` did then.
        clean = tmp_path / "clean"
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(clean)

        indexing = await park_an_index(lib, llm, odd_doc(), "fresh")
        monkeypatch.setattr(lib, "_replacing_elsewhere", lambda: False)
        assert lib._adds_in_flight_here() == 0

        lib.load(clean)          # the reset the guard now refuses
        llm.release.set()
        await indexing           # ...and the commit that lands on top of it

        # No exception anywhere. What is left is an index holding chunks
        # for a document it does not know it has.
        orphans = [c for c in lib.chunks if c.doc_id not in lib.documents]
        assert {c.doc_id for c in orphans} == {"fresh"}
        assert len(orphans) >= 20, "the whole document was stranded"
        assert "fresh" in lib.doc_summaries and "fresh" not in lib.documents
        assert ODD_MARK not in lib.source

        # And they are not inert: retrieval ranks them first for a term
        # only they contain, so answers cite a document the index does
        # not know it holds.
        hits = lib.retrieve(ODD_MARK, top_k=3)
        assert {c.doc_id for c in hits} == {"fresh"}


class TestSaveRefusesMidIndex:
    """`save()` refuses a replace outright — `strict` does not enter into it.

    Deliberately not the add case's warn/raise split, and not an
    extension of it. Mid-`index()` the reset has already run, so the file
    records the incoming document and *nothing else*: nothing marks it as
    partial, `_load_if_current` calls it current, and no later
    `add_document` is coming to chunk it. There is no degraded outcome
    for `strict` to select between.
    """

    @staticmethod
    async def parked_index(lib, llm):
        await lib.index(doc_text(0), doc_id="d0")
        return await park_an_index(lib, llm, odd_doc(), "fresh")

    async def test_it_refuses_and_writes_nothing_without_strict(
        self, tmp_path, caplog
    ):
        # `strict` defaults to False, and the add case would only warn
        # here. This one raises: catches the replace case being folded
        # into the add case's split.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        indexing = await self.parked_index(lib, llm)
        assert lib.strict is False

        target = tmp_path / "never-created"
        with pytest.raises(IndexBusyError) as excinfo:
            lib.save(target)

        message = str(excinfo.value)
        assert "cannot save safely" in message
        assert "index() is replacing this librarian from another task" in message
        assert "no later add repairs" in message
        # It raises before `mkdir`: no file, not a file plus an exception.
        assert not target.exists()
        assert not [m for m in warnings_from(caplog) if "cannot save" in m], (
            "it refused rather than warning"
        )

        llm.release.set()
        await indexing

    async def test_strict_selects_nothing_here(self, tmp_path):
        # The same call on a strict librarian gets the same answer. The
        # add case's split survives (TestSaveMidAdd pins it); this one
        # never consults the flag, and a guard that did would let the
        # default-strict librarian above write the lying file.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm, strict=True)
        indexing = await self.parked_index(lib, llm)

        target = tmp_path / "never-created-either"
        with pytest.raises(IndexBusyError) as excinfo:
            lib.save(target)
        assert "index() is replacing this librarian" in str(excinfo.value)
        assert not target.exists()

        llm.release.set()
        await indexing

    async def test_the_file_it_refuses_to_write_reloads_as_if_complete(
        self, tmp_path, monkeypatch, caplog
    ):
        # What the refusal is worth, measured with the replace half of
        # the guard forced off — the guard as it shipped before this
        # round. On a *strict* librarian, which is the sharp end: the
        # flag that exists to turn degradation into an error stayed
        # silent, because the counter it consulted read zero.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm, strict=True)
        indexing = await self.parked_index(lib, llm)
        monkeypatch.setattr(lib, "_replacing_elsewhere", lambda: False)

        torn = tmp_path / "torn"
        lib.save(torn)                                  # silent, under strict
        assert not [m for m in warnings_from(caplog) if "cannot save" in m]

        written = json.loads((torn / "index.json").read_text(encoding="utf-8"))
        assert list(written["documents"]) == ["fresh"]
        assert written["chunks"] == []
        assert written["doc_summaries"] == {}

        # And that file is not visibly partial to anything that reads it:
        # its content hash matches, so `_load_if_current` calls it current
        # and serves a many-chunk document whole.
        reader = librarian(WatchfulLLM())
        assert reader._load_if_current(torn, odd_doc()) is True
        assert reader.use_chunking is False
        assert reader.chunks == []
        assert list(reader.documents) == ["fresh"]

        llm.release.set()
        await indexing

    async def test_an_add_draining_under_a_waiting_index_is_the_replace_case(
        self, tmp_path, caplog
    ):
        # Both observations true at once, and the precedence has to be
        # the replace: the add's repair depends on the document staying
        # pending in this librarian until the next add reaches it, and
        # the `index()` holding the guard is about to discard it. Checked
        # in the other order, `save` would warn and write the file.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")

        adding = await park_an_add(lib, llm, doc_text(1), "d1")
        indexing = asyncio.ensure_future(lib.index(doc_text(2), doc_id="fresh"))
        await wait_until_replacing(lib)

        # The overlap, asserted rather than assumed: an add is counted
        # *and* another task owns the replace, waiting for it to drain.
        assert lib._adds_in_flight_here() == 1
        assert lib._replacing_elsewhere() is True

        target = tmp_path / "never-created"
        with pytest.raises(IndexBusyError) as excinfo:
            lib.save(target)
        assert "index() is replacing this librarian" in str(excinfo.value)
        assert "add_document call(s)" not in str(excinfo.value)
        assert not target.exists()
        assert not [m for m in warnings_from(caplog) if "cannot save" in m]

        # `load` refuses on either half, so it needs no precedence.
        with pytest.raises(IndexBusyError):
            lib.load(target)

        llm.release.set()
        await asyncio.gather(adding, indexing)
        assert list(lib.documents) == ["fresh"]


class TestALoopThatEndedUnderPendingWork:
    """Indexing stranded by a loop closing under it, produced not simulated.

    `TestTheCountIsLoopKeyed` sets `_adds_in_flight` by hand. This is the
    same state reached through the public API alone — park indexing, then
    close the loop without awaiting the task — which is the scenario the
    loop keying exists for, and it strands the *owner* as well as the
    count. Keeping a reference to the task is what makes it stick: an
    abandoned coroutine that is collected runs its `finally` and cleans
    up after itself; one that is still referenced never does.

    Catches: the loop check dropped from either observer, and
    `_forget_closed_loops` dropped from `_replace_guard`'s new-loop branch
    — the pruning that replaced the `_adds_in_flight = 0` /
    `_replace_owner = None` resets, and the only thing that gives back an
    entry nothing else will. Without any of those, a librarian whose loop
    ended under parked indexing refuses every later `save()` and `load()`
    forever, with nothing a caller could do about it.
    """

    @staticmethod
    def strand(start):
        """Run *start* to its parked point, then close the loop under it.

        Returns the abandoned task and the loop, both of which the caller
        must keep referenced: a collected coroutine would run the very
        `finally` this is arranging to skip, and the assertions would
        then be measuring the tidy case.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(start()), loop
        finally:
            # A pending task reports itself to its loop's exception
            # handler when it is finally collected, and the loop is dead
            # by then. Silencing that handler keeps the noise out of
            # whichever *other* test happens to be running at the time —
            # it is not part of what is being measured here.
            loop.set_exception_handler(lambda dead_loop, context: None)
            loop.close()

    @staticmethod
    def dispose(task) -> None:
        """Close the abandoned coroutine, now that it has been measured.

        Left to the garbage collector this runs during an unrelated test
        and raises out of an `async with` whose loop is gone, which
        pytest reports as an unraisable-exception warning against that
        test. Doing it here keeps the mess local. It also runs the
        `finally` the test spent its setup avoiding, so it must come
        last.
        """
        with contextlib.suppress(BaseException):
            task.get_coro().close()

    def test_a_stranded_add_count_does_not_wedge_later_calls(self, tmp_path):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        stranded, dead = self.strand(
            lambda: park_an_add(lib, llm, doc_text(1), "d1")
        )
        # No private assignment: the count is high because the task that
        # would have decremented it was destroyed pending.
        assert lib._adds_registered_on(dead) == 1
        assert lib._replace_loop is dead and dead.is_closed()

        lib.load(tmp_path)                       # no loop running at all
        assert list(lib.documents) == ["d0"]

        async def on_a_new_loop():
            await lib.add_document(doc_text(2), doc_id="d2")
            here = asyncio.get_running_loop()
            assert lib._adds_registered_on(here) == 0, (
                "this loop's count, not the dead one"
            )
            # And the dead loop's entry is *gone*, not merely discounted.
            # Nothing else ever gives it back — the `finally` that would
            # have died with the loop — so without `_forget_closed_loops`
            # the librarian grows one entry per loop that ever ended under
            # a parked add, for the life of the process.
            assert lib._adds_in_flight == {}, "the dead loop's entry was kept"
            lib.load(tmp_path)

        asyncio.run(on_a_new_loop())
        assert list(lib.documents) == ["d0"]
        self.dispose(stranded)                   # referenced until here

    def test_a_stranded_replace_owner_does_not_wedge_later_calls(self, tmp_path):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        stranded, dead = self.strand(
            lambda: park_an_index(lib, llm, doc_text(1), "fresh")
        )
        # The owner outlives the loop it was recorded on, and the task it
        # names can never run again.
        assert lib._replace_owner_on(dead) is stranded
        assert lib._replace_loop is dead and dead.is_closed()

        # With no loop running — the ordinary way an index is loaded.
        lib.load(tmp_path)
        assert list(lib.documents) == ["d0"]

        # And from a live loop that is simply not the one that recorded it.
        async def load_from_a_different_loop():
            assert lib._replacing_elsewhere() is False
            lib.load(tmp_path)

        second = asyncio.new_event_loop()
        try:
            second.run_until_complete(load_from_a_different_loop())
        finally:
            second.close()

        # Once a new loop builds the guard the stale owner is forgotten
        # rather than carried — it belongs to a loop that has closed, so
        # nothing it recorded can commit, and leaving it would have every
        # later `save`/`load` refuse over an `index()` that ended with its
        # loop. That is `_forget_closed_loops`, and asserting the entry is
        # gone rather than that the *reading* is clear is what makes this
        # fail if the pruning is dropped.
        async def on_a_new_loop():
            await lib.add_document(doc_text(2), doc_id="d2")
            assert lib._replace_owner == {}, "the dead loop's owner was kept"
            assert lib._replacing_elsewhere() is False
            lib.load(tmp_path)
            lib.save(tmp_path)

        asyncio.run(on_a_new_loop())
        assert list(lib.documents) == ["d0"]
        self.dispose(stranded)                   # referenced until here

    def test_the_reset_clears_a_claim_the_dead_loop_left_behind(self, tmp_path):
        # `_reset()` clears `_chunking`, and the comment above it now
        # calls that load-bearing rather than belt and braces. This is
        # the case it means, and the one thing the refusals do *not*
        # cover: a claim stranded by the loop ending under a pending
        # task, which no guard can see because the registration that
        # would have flagged it belongs to a loop that has closed and is
        # forgiven by every reader of it.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))

        stranded, dead = self.strand(
            lambda: park_an_add(lib, llm, doc_text(1), "d1")
        )
        # Saved after that loop died, so the file is the partial one
        # `save` warns about: d1 registered, no chunks, no summary —
        # still pending, and waiting to be swept up.
        lib.save(tmp_path)
        assert lib._chunking == {"d1"}, "the claim outlived its loop"

        # A fresh double: the old one's gate belongs to the dead loop.
        lib.llm = WatchfulLLM()
        lib.load(tmp_path)
        assert lib._chunking == set()
        assert [d for d in lib.documents if d not in lib.doc_summaries] == ["d1"]

        # The next add sweeps d1 up with it. Left in place the claim
        # would exclude d1 from that sweep and nothing would ever chunk
        # it — a document silently missing from the index for good.
        asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
        assert set(lib.doc_summaries) == {"d0", "d1", "d2"}
        assert {c.doc_id for c in lib.chunks} == {"d0", "d1", "d2"}
        self.dispose(stranded)                   # referenced until here
