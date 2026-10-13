"""The replace guard's observers tell a *dead* loop from a *distant* one.

`test_load_concurrency.py` pins the guard against the loop it was built
on. This file pins the axis that guard was blind to: **which thread**.

Both synchronous observers used to open with one identity test —
`asyncio.get_running_loop() is self._replace_loop`, else "nothing to
race" — and that single comparison was doing two jobs. Ignoring state
stranded by a *finished* loop is right, and refusing over it would wedge
every later `save()`/`load()` forever. Ignoring an add running on
another thread's *live* loop is wrong: that add is going to commit into
whatever index is here when it wakes. `_observe_replace_guard()` splits
the two by asking `loop.is_closed()` instead, and everything below is
one side or the other of that split:

* :class:`TestLoadAndSaveFromAnotherThread` — symptom A, refused.
* :class:`TestWhatTheLivenessCheckIsWorth` — the same script with the
  identity test put back, which is what the library did before.
* :class:`TestTheWedgeStaysClosed` — both directions of the regression
  the identity test existed to prevent.
* :class:`TestALoopThatIsOpenButNotRunning` — the in-between reading.
* :class:`TestTheOwnerCheckFromAThreadWithNoLoop` — `current_task()`.
* :class:`TestAClaimStrandedByAClosedLoop` — part B, through the route
  that produces the state rather than by assigning to `_chunking`.
* :class:`TestAClaimHeldByAnAbandonedOpenLoop` — part B's other side,
  where the claim must *survive*, since the loop holding it can resume.
* :class:`TestWhatTheCachePathPromiseRestsOn` — `_load_if_current`, both
  the case it must stay quiet for and the case it must refuse.

**Since `LoopHandoverError`, three tests below run with the refusal off.**
Taking a librarian onto a new loop while a live one still holds work is
refused now, and that transition is how half this file produced the state
it measures. `test_loop_handover.py` owns the refusal; what is left here
is the machinery behind it, reached with :func:`without_the_handover_refusal`
and marked as such at every site. Anything that installs it is a test
about a state the library will not enter on its own — kept because the
code that handles that state is still there, and because the refusal's
coverage is a decision that can change.

**Determinism.** Real threads plus asyncio is where flaky suites come
from, so nothing here sleeps and nothing races on timing. Every ordering
is a fact before it is used: `park_an_add` / `park_an_index` assert the
registration they were waiting for before returning, `GatedLLM` holds a
named document's summary call until it is released by name, and the one
test that runs a loop on a second thread hands the parked state back over
a `threading.Event` that is set *after* those assertions and released
only from the test's own `finally`. Every `join`/`wait` carries
`GATE_TIMEOUT` and asserts it did not expire, so a hang fails rather than
hanging.

**The measured premise.** On CPython 3.11.6 `asyncio.current_task()`
*raises* `RuntimeError("no running event loop")` off a loop — it resolves
the running loop before the task lookup — rather than returning None.
That is why `_replacing_elsewhere` may only ask it when the guard's loop
is this thread's running one, and
:meth:`TestTheOwnerCheckFromAThreadWithNoLoop.test_the_owner_check_answers_from_a_thread_with_no_loop`
asserts it on the thread where it would bite.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading

import pytest

from librarium import IndexBusyError, LoopHandoverError
from librarium.source import SourceLibrarian
from tests.test_indexing import (
    GATE_TIMEOUT,
    GatedLLM,
    WatchfulLLM,
    doc_marker,
    doc_text,
)
from tests.test_load_concurrency import (
    ODD_MARK,
    SOURCE_LOGGER,
    librarian,
    odd_doc,
    park_an_add,
    park_an_index,
    warnings_from,
)


def abandon(start, *, close: bool):
    """Run *start* to its parked point on a private loop and walk away.

    The route that produces stranded state, rather than a simulation of
    it: the parked task's `finally` never runs, so whatever it holds —
    the count, the owner, the `_chunking` claim — is left exactly as a
    real loop teardown leaves it. `asyncio.run()` is deliberately not
    used, because it cancels and gathers on exit and that `finally` then
    runs; this is the case it tidies away.

    *close* picks which of the two states this leaves behind: a closed
    loop (dead — stranded state must be ignored) or an open, idle one
    (the in-between reading). Returns ``(task, loop)``, both of which the
    caller must keep referenced until :func:`dispose`: a collected
    coroutine runs the very `finally` this is arranging to skip, and the
    assertions would then be measuring the tidy case.
    """
    loop = asyncio.new_event_loop()
    # A task destroyed pending reports itself to its loop's exception
    # handler when it is finally collected, which is long after this test
    # and inside whichever other one is running then. Silencing it keeps
    # the noise local; it is not part of what is being measured.
    loop.set_exception_handler(lambda dead_loop, context: None)
    task = loop.run_until_complete(start())
    if close:
        loop.close()
    return task, loop


def dispose(task, loop) -> None:
    """Close the abandoned coroutine and its loop, after measuring them.

    Left to the garbage collector this runs during an unrelated test and
    raises out of an `async with` whose loop is gone, which pytest
    reports against *that* test. It also runs the `finally` the setup
    spent its effort avoiding, so it must come last.
    """
    if not loop.is_closed():
        loop.close()
    with contextlib.suppress(BaseException):
        task.get_coro().close()


def on_a_plain_thread(call):
    """Run *call* on a bare thread and hand back its result or its error.

    `asyncio.to_thread` needs a running loop, and half of what is being
    measured here is the caller that has none — the synchronous `ask()`
    wrapper, the plain worker thread. `join(GATE_TIMEOUT)` plus the
    liveness assertion means a call that never returns fails the test
    instead of hanging the suite.
    """
    box: dict[str, object] = {}

    def run():
        try:
            box["value"] = call()
        except BaseException as error:      # noqa: BLE001 - re-raised below
            box["error"] = error

    thread = threading.Thread(target=run, name="librarium-test-observer")
    thread.start()
    thread.join(GATE_TIMEOUT)
    assert not thread.is_alive(), "the call never returned"
    if "error" in box:
        raise box["error"]                  # type: ignore[misc]
    return box["value"]


def identity_observer(lib):
    """`_observe_replace_guard` as the bare identity test it replaced.

    Faithful rather than approximate: off the guard's loop the old code
    returned `0` from `_adds_in_flight_here` and `False` from
    `_replacing_elsewhere`, which is exactly `(live=False, ours=False)`
    here; on it, the count and `owner is not current_task()`, which is
    `(True, True)`. So a test that installs this is measuring the library
    as it shipped before this change, and is a mutation check by
    construction.

    It takes the *loop being asked about* now, because the bookkeeping is
    keyed by loop and the observers ask this question once per entry they
    find. The identity test is applied to whichever loop is named — the
    entry's own key, or `_replace_loop` when a caller asks about the guard
    at large, which is the default the source itself uses. That keeps the
    simulation faithful where it matters: the old code compared the running
    loop against the one loop it had, and every entry these tests plant
    belongs to that same loop, so a walk over the entries reaches exactly
    the answer the single comparison used to give.
    """
    def observe(loop=None) -> tuple[bool, bool]:
        if loop is None:
            loop = lib._replace_loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            return False, False
        if running is not loop:
            return False, False
        return True, True

    return observe


def without_the_handover_refusal(monkeypatch):
    """`_replace_guard`'s new-loop branch as it was before the refusal.

    :func:`identity_observer`'s trick for the other half of the guard, and
    faithful in the same way: the only thing `LoopHandoverError` added to
    that branch is a call to `_handover_would_strand` and a `raise` of
    whatever sentence comes back, so returning None unconditionally *is*
    the branch as it shipped. A test that installs this measures the old
    behaviour rather than approximating it.

    Two kinds of test need it, and both are about the same fact: taking a
    librarian onto a new loop while a live one still holds work is now
    refused, so every state that transition used to produce is out of
    reach through the public API.

    * `_replace_guard`'s conditional `_chunking.clear()` is a backstop. It
      is unreachable while the refusal holds and load-bearing again the
      moment the refusal's coverage changes, so a test that pins it has to
      run with the refusal off or it pins nothing at all.
    * **Proving a fix is a fix and not a fence.** The count and the owner
      the branch used to reset are keyed by loop now, so D1, D2 and D3 die
      at the root rather than behind this refusal — and the only way to
      say that rather than hope it is to take the refusal away and watch
      them *not* reproduce. `test_loop_handover.py`'s
      `TestTheDeferredDefectsDoNotReproduce` is that measurement.

    What it is **no longer** for is reaching those defects. An earlier
    version of this docstring said the resets below the refusal were still
    wrong and still deferred; they are gone, so a test that installs this
    expecting -1, a destroyed count or a dropped owner is asserting a
    fiction.

    Using it is therefore a claim that the test is about the machinery
    *behind* the refusal. A test about the refusal itself must not
    install it.
    """
    monkeypatch.setattr(
        SourceLibrarian,
        "_handover_would_strand",
        lambda self, previous_live, action: None,
    )


class TestLoadAndSaveFromAnotherThread:
    """Symptom A: the whole-index pair, called off the librarian's loop.

    The `personal-librarian` pattern — a synchronous `ask()`/`load()`
    wrapped in `asyncio.to_thread` — puts the caller on a thread with no
    running loop while the librarian's own loop is mid-add. That used to
    read as "nothing to race" and be admitted.

    Catches: the liveness check reverting to the identity test, in the
    direction that matters. Every one of these passes vacuously against
    the old code except that no exception arrives and no warning is
    logged, which is what :class:`TestWhatTheLivenessCheckIsWorth`
    measures the cost of.
    """

    async def test_load_from_another_thread_refuses_mid_add(self, tmp_path):
        clean = tmp_path / "clean"
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(clean)

        adding = await park_an_add(lib, llm, odd_doc(), "d1")

        with pytest.raises(IndexBusyError) as excinfo:
            await asyncio.to_thread(lib.load, clean)

        message = str(excinfo.value)
        assert "cannot load safely" in message
        assert "1 add_document call(s)" in message
        # The damage, not just the rule — and not asyncio's own
        # complaint, which is what an unguarded `current_task()` would
        # have produced from a thread instead of a refusal.
        assert "strand" in message
        assert "no running event loop" not in message

        # It refused without touching anything: the caller still holds
        # the librarian it had, not a half-replaced one.
        assert set(lib.documents) == {"d0", "d1"}
        assert {c.doc_id for c in lib.chunks} == {"d0"}

        llm.release.set()
        await adding
        # And the add landed normally: the refusal cost it nothing.
        assert [c for c in lib.chunks if c.doc_id not in lib.documents] == []
        assert set(lib.doc_summaries) == {"d0", "d1"}
        assert ODD_MARK in lib.source

    async def test_save_from_another_thread_sees_the_add_too(
        self, tmp_path, caplog
    ):
        # `save`'s answer to an add in flight is the warn-and-write one
        # (`TestSaveMidAdd` pins the split); what is new is that it
        # reaches that answer from a thread at all.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        await asyncio.to_thread(lib.save, tmp_path)

        assert any("cannot save safely" in m for m in warnings_from(caplog))
        # The file really is the partial one it announced, so the warning
        # is about this write rather than a stray message.
        meta = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
        assert set(meta["documents"]) == {"d0", "d1"}
        assert {c["doc_id"] for c in meta["chunks"]} == {"d0"}

        llm.release.set()
        await adding

    async def test_strict_save_from_another_thread_refuses_and_writes_nothing(
        self, tmp_path
    ):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm, strict=True)
        await lib.index(doc_text(0), doc_id="d0")
        adding = await park_an_add(lib, llm, doc_text(1), "d1")

        target = tmp_path / "never-created"
        with pytest.raises(IndexBusyError) as excinfo:
            await asyncio.to_thread(lib.save, target)
        assert "cannot save safely" in str(excinfo.value)
        # `strict` still means no file rather than a file plus an
        # exception, from a thread as from the loop.
        assert not target.exists()

        llm.release.set()
        await adding


class TestWhatTheLivenessCheckIsWorth:
    """The defect itself, with the identity test put back.

    `_observe_replace_guard` is replaced by :func:`identity_observer`,
    which is the library exactly as it shipped before this change. What
    follows is what a `load()` from `asyncio.to_thread` did then, and it
    is the full shape-1 corruption: chunks for a document `documents`,
    `source` and `content_hash` do not hold, no exception anywhere,
    `retrieve` ranking them first, and a file `_load_if_current` calls
    current.

    Catches: the check being removed or narrowed back to identity. This
    is the test that says what the refusals above are *worth* rather than
    that they exist.
    """

    async def test_the_identity_test_admits_the_cross_thread_load(
        self, tmp_path, monkeypatch
    ):
        clean = tmp_path / "clean"
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(clean)

        adding = await park_an_add(lib, llm, odd_doc(), "d1")
        monkeypatch.setattr(
            lib, "_observe_replace_guard", identity_observer(lib),
        )

        # Admitted. No exception, no warning, nothing for a caller to
        # notice — the reset the guard now refuses.
        await asyncio.to_thread(lib.load, clean)
        llm.release.set()
        await adding                  # the commit that lands on top of it

        orphans = [c for c in lib.chunks if c.doc_id not in lib.documents]
        assert {c.doc_id for c in orphans} == {"d1"}
        assert len(orphans) >= 20, "the whole document was stranded"
        assert "d1" in lib.doc_summaries and "d1" not in lib.documents
        assert ODD_MARK not in lib.source

        # Not inert: retrieval ranks them first for a term only they
        # contain, so answers cite a document the index does not know it
        # holds.
        hits = lib.retrieve(ODD_MARK, top_k=3)
        assert {c.doc_id for c in hits} == {"d1"}

        # And `content_hash` cannot see them, so they outlive the process
        # and survive the cache check on the way back in.
        torn = tmp_path / "torn"
        lib.save(torn)
        written = json.loads((torn / "index.json").read_text(encoding="utf-8"))
        assert "d1" in {c["doc_id"] for c in written["chunks"]}
        assert "d1" not in written["documents"]

        fresh = librarian(WatchfulLLM())
        assert fresh._load_if_current(torn, doc_text(0)) is True
        assert any(c.doc_id == "d1" for c in fresh.chunks)


class TestTheWedgeStaysClosed:
    """Both directions of the regression the identity test guarded.

    The identity test was not arbitrary: state stranded by a loop that
    has since closed *must* keep being ignored, or every later `save()`
    and `load()` refuses forever over work that can never run, with
    nothing a caller could do about it. The new check has to keep that
    while gaining the other half. Both are here, because a version that
    only refuses passes the cross-thread tests above and wedges the
    library.

    The two closed-loop tests are the only ones in this file that also
    pass against the code before this change, deliberately: they pin an
    invariant the change had to *preserve*, so a revert is not what they
    catch. Mutation-checked against what they do catch — dropping the
    `is_closed()` test from `_observe_replace_guard`, so that `live` is
    always true — which fails both of them.

    Not *only* them, and an earlier version of this docstring said so
    wrongly. Every test that turns on stranded state being ignored fails
    with that mutation: here,
    :meth:`TestALoopThatIsOpenButNotRunning.test_closing_the_abandoned_loop_lifts_the_refusal`
    and :class:`TestAClaimStrandedByAClosedLoop`, and in
    `test_load_concurrency.py` the whole of `TestTheCountIsLoopKeyed` and
    `TestALoopThatEndedUnderPendingWork`. No count is carried in place of
    that list, because a count moves whenever such a test is added — this
    docstring already quoted one that had.
    """

    def test_a_count_stranded_by_a_closed_loop_is_still_ignored(self, tmp_path):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        stranded, dead = abandon(
            lambda: park_an_add(lib, llm, doc_text(1), "d1"), close=True,
        )
        # No private assignment: the count is high because the task that
        # would have decremented it was destroyed pending.
        assert lib._adds_registered_on(dead) == 1
        assert lib._replace_loop is dead and dead.is_closed()

        lib.load(tmp_path)                     # no loop on this thread
        assert list(lib.documents) == ["d0"]

        # And from a plain worker thread, which is the caller this change
        # taught the guard to see. A version that refuses whenever the
        # guard's loop is not the caller's would wedge exactly here.
        on_a_plain_thread(lambda: lib.load(tmp_path))
        assert list(lib.documents) == ["d0"]
        assert on_a_plain_thread(lib._adds_in_flight_here) == 0

        dispose(stranded, dead)

    def test_an_owner_stranded_by_a_closed_loop_is_still_ignored(self, tmp_path):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        stranded, dead = abandon(
            lambda: park_an_index(lib, llm, doc_text(1), "fresh"), close=True,
        )
        assert lib._replace_owner_on(dead) is stranded
        assert lib._replace_loop is dead and dead.is_closed()

        # The owner half wedges the same way, and `save()` refuses it
        # outright rather than warning, so an over-broad version here
        # takes checkpointing away entirely.
        assert on_a_plain_thread(lib._replacing_elsewhere) is False
        on_a_plain_thread(lambda: lib.load(tmp_path))
        on_a_plain_thread(lambda: lib.save(tmp_path))
        assert list(lib.documents) == ["d0"]

        dispose(stranded, dead)

    def test_an_add_on_a_live_loop_on_another_thread_refuses(self, tmp_path):
        # The other direction, in its sharpest form: the librarian's loop
        # is genuinely running, on a thread that is not this one, with an
        # add parked on it. Sequenced by two `threading.Event`s rather
        # than by timing — `parked` is set only after `park_an_add` has
        # asserted the registration, and `finish` is set only from this
        # test's own `finally`, so the add cannot progress while the
        # assertions run.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        parked = threading.Event()
        finish = threading.Event()

        async def park_and_hold():
            adding = await park_an_add(lib, llm, doc_text(1), "d1")
            parked.set()
            # Keeps this loop genuinely *running* while the main thread
            # observes it, without spinning: the loop is servicing an
            # await, which is what a real busy librarian looks like.
            await asyncio.to_thread(finish.wait, GATE_TIMEOUT)
            llm.release.set()
            await adding

        worker = threading.Thread(
            target=lambda: asyncio.run(park_and_hold()),
            name="librarium-test-loop",
        )
        worker.start()
        try:
            assert parked.wait(GATE_TIMEOUT), "the add never parked"
            assert lib._adds_in_flight_here() == 1
            with pytest.raises(IndexBusyError) as excinfo:
                lib.load(tmp_path)             # no loop on this thread
            assert "1 add_document call(s)" in str(excinfo.value)
        finally:
            finish.set()
            worker.join(GATE_TIMEOUT)

        assert not worker.is_alive(), "the worker loop never finished"
        # The add committed on its own loop, which is why refusing was
        # right: an admitted `load()` would have reset the index out from
        # under this.
        assert set(lib.doc_summaries) == {"d0", "d1"}
        assert {c.doc_id for c in lib.chunks} == {"d0", "d1"}


class TestALoopThatIsOpenButNotRunning:
    """The in-between reading, pinned: open and idle counts as LIVE.

    A loop between `run_until_complete` calls is not executing, so
    `is_running()` reports False for it — and for the `add_document`
    parked on it, which is suspended rather than dead. Reading that as
    "nothing to race" would admit the reset that strands it, which is the
    silent failure the guard exists to convert into a loud one. So
    `is_closed()` is the test and this errs toward refusing.

    Catches: `is_running()` (or any liveness test other than
    `is_closed()`) being substituted for it.
    """

    def test_an_add_parked_on_an_idle_open_loop_still_refuses(self, tmp_path):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        parked, idle = abandon(
            lambda: park_an_add(lib, llm, doc_text(1), "d1"), close=False,
        )
        try:
            # The state under test, asserted rather than assumed.
            assert not idle.is_closed()
            assert idle.is_running() is False
            assert lib._adds_registered_on(idle) == 1

            with pytest.raises(IndexBusyError) as excinfo:
                lib.load(tmp_path)
            assert "1 add_document call(s)" in str(excinfo.value)

            # Why that is the right reading, shown rather than argued:
            # the next turn of that loop resumes the parked add and
            # commits it. Had the `load()` above been admitted, these
            # chunks would have landed in an index that no longer held
            # d1 at all.
            llm.release.set()
            idle.run_until_complete(parked)
            assert set(lib.doc_summaries) == {"d0", "d1"}
            assert {c.doc_id for c in lib.chunks} == {"d0", "d1"}

            # And the refusal lasts exactly as long as the add: once it
            # has drained, the same call on the same idle loop's
            # librarian is admitted.
            assert lib._adds_registered_on(idle) == 0
            lib.load(tmp_path)
            assert list(lib.documents) == ["d0"]
        finally:
            dispose(parked, idle)

    def test_save_is_refused_by_the_idle_loop_too(self, tmp_path, caplog):
        # Same observation, the other caller. `strict` so the answer is a
        # refusal rather than a warning, which is unambiguous to assert.
        caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm, strict=True)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))

        parked, idle = abandon(
            lambda: park_an_add(lib, llm, doc_text(1), "d1"), close=False,
        )
        try:
            target = tmp_path / "never-created"
            with pytest.raises(IndexBusyError) as excinfo:
                lib.save(target)
            assert "cannot save safely" in str(excinfo.value)
            assert not target.exists()
        finally:
            dispose(parked, idle)

    def test_closing_the_abandoned_loop_lifts_the_refusal(self, tmp_path):
        # The documented cost of erring toward refusing, and its escape
        # hatch: an open loop abandoned under a parked add refuses every
        # later save/load *until it is closed*. `asyncio.run` always
        # closes one; a hand-rolled loop wants a `close()`. Without this
        # the refusal would have no way out, which is the wedge again.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        parked, idle = abandon(
            lambda: park_an_add(lib, llm, doc_text(1), "d1"), close=False,
        )
        with pytest.raises(IndexBusyError):
            lib.load(tmp_path)

        idle.close()

        lib.load(tmp_path)                     # no exception
        assert list(lib.documents) == ["d0"]
        dispose(parked, idle)

    def test_a_replace_parked_on_an_idle_loop_refuses_from_a_different_loop(
        self, tmp_path
    ):
        # The owner half of the same reading, and the case where
        # `current_task()` would answer about the wrong loop's task: the
        # caller is inside a running loop, just not the one that recorded
        # the owner. Under the identity test this read False and the
        # `save` below wrote the file `_load_if_current` calls current.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        parked, idle = abandon(
            lambda: park_an_index(lib, llm, odd_doc(), "fresh"), close=False,
        )
        try:
            assert lib._replace_owner_on(idle) is parked
            assert not idle.is_closed()

            async def from_a_different_loop():
                assert asyncio.get_running_loop() is not idle
                assert lib._replacing_elsewhere() is True
                target = tmp_path / "never-created"
                with pytest.raises(IndexBusyError) as excinfo:
                    lib.save(target)
                assert "index() is replacing this librarian" in str(
                    excinfo.value
                )
                assert not target.exists()
                with pytest.raises(IndexBusyError):
                    lib.load(tmp_path)

            other = asyncio.new_event_loop()
            try:
                other.run_until_complete(from_a_different_loop())
            finally:
                other.close()
        finally:
            dispose(parked, idle)


class TestTheOwnerCheckFromAThreadWithNoLoop:
    """`current_task()` is asked only when the guard's loop is ours.

    Off a loop it raises rather than returning None, so an unconditional
    comparison would turn a synchronous `save`/`load` from a plain thread
    into `RuntimeError("no running event loop")` — a crash out of a `def`
    that has nothing to do with what the caller did wrong. From a
    *different* loop it answers about that loop's task, which can never
    be an owner recorded elsewhere, so there is nothing to ask.

    Catches: the observer reverting to the identity test (the answers
    below flip to False), and the `ours` gate being dropped (they become
    a `RuntimeError` with asyncio's message instead of a refusal).
    """

    async def test_the_owner_check_answers_from_a_thread_with_no_loop(self):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        indexing = await park_an_index(lib, llm, doc_text(1), "fresh")

        def from_a_plain_thread():
            # The premise, asserted on the thread where it would bite:
            # CPython resolves the running loop before the task lookup.
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.current_task()
            return lib._replacing_elsewhere()

        assert await asyncio.to_thread(from_a_plain_thread) is True

        llm.release.set()
        await indexing

    async def test_a_thread_gets_the_refusal_and_not_asyncios_complaint(
        self, tmp_path
    ):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        await lib.index(doc_text(0), doc_id="d0")
        lib.save(tmp_path)
        indexing = await park_an_index(lib, llm, doc_text(1), "fresh")

        target = tmp_path / "never-created"
        with pytest.raises(IndexBusyError) as excinfo:
            await asyncio.to_thread(lib.save, target)
        message = str(excinfo.value)
        assert "index() is replacing this librarian" in message
        assert "no running event loop" not in message
        assert not target.exists()

        with pytest.raises(IndexBusyError) as excinfo:
            await asyncio.to_thread(lib.load, tmp_path)
        assert "no running event loop" not in str(excinfo.value)

        llm.release.set()
        await indexing
        assert list(lib.documents) == ["fresh"]


class TestAClaimStrandedByAClosedLoop:
    """Part B: `_replace_guard` clears `_chunking` for a new loop.

    A `_chunking` claim is given back in `_add_document`'s `finally`,
    which a task destroyed at loop teardown never runs. Left in place it
    does not wedge a *call* the way a stranded count or owner does — it
    wedges a *document*: the pending sweep skips anything already
    claimed, so that document sits in `documents` and `source` with no
    summary and no chunks until some `_reset()` happens to run.

    Reached through the route that produces it, and observed through
    `add_document` alone — no `load()`, no `index()`, no assignment to
    `_chunking` — because "a caller who only ever adds never triggers a
    `_reset()`" is the whole reason this was silent.

    Catches: the `_chunking.clear()` being dropped from
    `_replace_guard`'s new-loop branch.

    `_reset`'s own `_chunking.clear()` is not re-pinned here.
    `test_load_concurrency.TestALoopThatEndedUnderPendingWork
    .test_the_reset_clears_a_claim_the_dead_loop_left_behind` already
    fails without it — checked, not assumed — and it is the only path
    that reaches it: an `index()` resetting on its own loop has drained
    the adds, and a claim is released in `_add_document`'s `finally`
    before `_adding` decrements, so `_chunking` is empty there by
    construction.
    """

    def test_the_next_loops_add_sweeps_up_the_document_the_claim_would_hide(
        self, tmp_path
    ):
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))

        stranded, dead = abandon(
            lambda: park_an_add(lib, llm, doc_text(1), "d1"), close=True,
        )
        # The stranded state, produced rather than assigned: d1 is
        # registered and claimed, and the task that would have released
        # the claim was destroyed pending.
        assert lib._chunking == {"d1"}
        assert dead.is_closed()
        assert "d1" in lib.documents and "d1" not in lib.doc_summaries
        assert [c for c in lib.chunks if c.doc_id == "d1"] == []

        # A fresh double: the old one's gate belongs to the dead loop.
        lib.llm = WatchfulLLM()
        asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

        assert lib._chunking == set()
        assert set(lib.doc_summaries) == {"d0", "d1", "d2"}
        assert {c.doc_id for c in lib.chunks} == {"d0", "d1", "d2"}
        # And d1 is retrievable, which is what "swept up" has to mean:
        # with the claim in place it stays in `documents` and
        # `content_hash` while contributing nothing to any answer.
        assert {c.doc_id for c in lib.retrieve(doc_marker(1), top_k=3)} == {"d1"}

        dispose(stranded, dead)


class TestAClaimHeldByAnAbandonedOpenLoop:
    """Part B's other route: the claim survives a loop that is still open.

    :class:`TestAClaimStrandedByAClosedLoop` covers the case the clear
    exists for — the loop *closed* under the add, so its `finally` can
    never run and the claim is rubbish. This is the case the clear must
    not touch. An add parked on a loop abandoned **open** is suspended,
    not dead: the next turn of that loop resumes it and commits its own
    chunks. Dropping its claim lets the next loop's pending sweep chunk
    the same document a second time, and the parked add then commits on
    top — one document, two full sets of chunks, no exception, nothing a
    caller could see except a corpus that has silently doubled.

    Single-threaded and `add_document`-only: no `load()`, no `index()`,
    no assignment to `_chunking`. The two loops are the whole mechanism.

    Catches: the `_chunking.clear()` in `_replace_guard`'s new-loop
    branch being made unconditional again — which is what it was, and
    which the entire suite passed with.

    The two resets that used to sit beside it are gone: the count and the
    owner are keyed by loop, so a handover has no other loop's state to
    zero and there is nothing left there to gate. What remains beside this
    clear is `_forget_closed_loops`, which asks the same liveness question
    and so cannot drift from it — a live previous loop keeps its
    registrations, its owner *and*, here, its claim. The condition on this
    line is what says the third of those out loud.

    **Both tests install :func:`without_the_handover_refusal`, and that
    is the point of them now.** The second loop's `add_document` is the
    handover `LoopHandoverError` declines, so with the refusal in force
    the conditional clear is unreachable and these tests would only be
    measuring the refusal — which
    `test_loop_handover.py` already does, once, on its own terms. What is
    left to pin here is that the clear is a working backstop and not
    decoration: if a later change narrows what the refusal covers and
    lets this handover through again, the condition is what stops the
    96-chunk doubling coming back, and these are the only tests that
    would notice it had been removed.
    """

    @staticmethod
    def park_on_an_open_loop(lib, llm):
        """d1 parked mid-add on a loop left open, asserted into that state."""
        parked, idle = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        # Produced rather than assigned: d1 is claimed by an add that is
        # suspended on a loop which is open and idle — live by
        # `_observe_replace_guard`'s reading.
        assert lib._chunking == {"d1"}
        assert not idle.is_closed()
        assert idle.is_running() is False
        assert "d1" in lib.documents and "d1" not in lib.doc_summaries
        return parked, idle

    def test_the_next_loops_sweep_leaves_the_parked_document_alone(
        self, monkeypatch
    ):
        without_the_handover_refusal(monkeypatch)
        # The observable half of "the claim survives", stated before the
        # claim itself so a failure names the damage: d1 is mid-add, so
        # the sweep on the new loop must not chunk or summarize it.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        parked, idle = self.park_on_an_open_loop(lib, llm)
        try:
            # An unrelated add on a *fresh* loop, which is what sends
            # `_replace_guard` down its new-loop branch. A fresh double
            # too: the parked one's gate belongs to the abandoned loop,
            # and d1's summary call would park a second time on it.
            lib.llm = WatchfulLLM()
            asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

            assert {c.doc_id for c in lib.chunks} == {"d0", "d2"}
            assert set(lib.doc_summaries) == {"d0", "d2"}
            assert lib._chunking == {"d1"}, "the claim was dropped"
        finally:
            dispose(parked, idle)

    def test_the_parked_add_then_commits_its_document_once_not_twice(
        self, monkeypatch
    ):
        without_the_handover_refusal(monkeypatch)
        # The consequence, all the way through: resume the loop that was
        # abandoned and d1 ends up in the index exactly once. Under the
        # unconditional clear this is the same text twice over, with no
        # exception raised anywhere along the way.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        parked, idle = self.park_on_an_open_loop(lib, llm)
        try:
            lib.llm = WatchfulLLM()
            asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            llm.release.set()
            idle.run_until_complete(parked)
        finally:
            dispose(parked, idle)

        d1_chunks = [c for c in lib.chunks if c.doc_id == "d1"]
        offsets = [c.offset for c in d1_chunks]
        assert offsets, "d1 never committed its chunks at all"
        assert len(offsets) == len(set(offsets)), (
            f"d1 was chunked twice: {len(offsets)} chunks covering "
            f"{len(set(offsets))} distinct offsets"
        )

        # Against a control that did the same work on one loop, so the
        # expected count is a measurement rather than a number copied out
        # of a run — and so a *missing* second half fails too.
        control = librarian(WatchfulLLM())

        async def build_control():
            await control.index(doc_text(0), doc_id="d0")
            await control.add_document(odd_doc(), doc_id="d1")
            await control.add_document(doc_text(2), doc_id="d2")

        asyncio.run(build_control())
        assert len(d1_chunks) == len(
            [c for c in control.chunks if c.doc_id == "d1"]
        )
        assert set(lib.doc_summaries) == set(control.doc_summaries)
        assert lib._chunking == set()

        # Visible to a caller, which is what makes the duplication a bug
        # rather than internal untidiness: doubled chunks are doubled
        # hits, so every answer about d1 pays for the same text twice.
        hits = lib.retrieve(ODD_MARK, top_k=6)
        assert {c.doc_id for c in hits} == {"d1"}
        assert len({c.offset for c in hits}) == len(hits)


class TestWhatTheCachePathPromiseRestsOn:
    """`_load_if_current` must not acquire, block or raise — and *why* not.

    It runs inside `index()`'s own `_replacing()`, in the same task on the
    same loop, and it is on the hot path of every cached index. The two
    halves of the guard reach it differently, and only one of them is
    unconditional:

    * **The replace half cannot fire here at all.**
      `_replacing_elsewhere` asks whether somebody *else* holds the
      replace, and on this path the caller is the owner.
    * **The add half can.** `_replacing`'s drain empties *this loop's*
      entry (`while self._adds_registered_on(loop)`) while
      `_adds_in_flight_here` sums **every live loop's**, so a registration
      on another live loop is visible from here and refuses. What makes
      the hot path quiet is therefore not the drain — it is that no other
      live loop can be holding one, because `LoopHandoverError` declined
      the handover that would have left one there.

    So the promise rests on the refusal, and the second test below is what
    says so: take the refusal away and the same cached index raises
    `IndexBusyError` out of `_load_if_current`.

    **This class used to claim it caught "the liveness read being taken
    against a remembered loop rather than the guard's current one".** It
    did not. That mutation — `_observe_replace_guard` always reading
    `self._replace_loop` instead of the entry's own loop — fails five
    tests in `test_load_concurrency.py` and `test_loop_handover.py` and
    none of them was this one; measured, twice, by a reviewer and again
    here. The clause was left behind when the first test was restructured
    onto a live-but-idle previous loop, and a "Catches" line that names a
    mutation the test survives is worse than none.

    Catches, now, one clause per test:

    * the refusal firing on a previous loop that is live but holds nothing,
      which would break every script that leaves a loop open (test 1);
    * the add half being exempted on the cache path — or
      `_adds_in_flight_here` narrowed so a registration on another live
      loop goes unseen from here (test 2). Verified both directions:
      dropping `_busy_with_adds`'s complaint, or narrowing the observer to
      the guard's own loop, each fails test 2.

    That first clause used to add "the tempting 'fix' for a message that
    tells the reader this path is unaffected", and the message stopped
    saying that: `_busy_with_adds` now scopes "unaffected" to
    `index(..., cache=...)`'s *own* loop's adds and says an add on another
    live loop refuses there too. So the motivation is gone while the
    mutation is unchanged — the exemption is still worth catching, just no
    longer invited by a sentence in the source.
    """

    def test_a_cached_index_is_not_refused_by_an_abandoned_open_loop(
        self, tmp_path, caplog
    ):
        # The negative half: a previous loop that is live but *idle* is an
        # ordinary handover — a script that called `run_until_complete` and
        # moved on — and it must stay quiet all the way through the cached
        # path. An open loop reads as live, so a guard that refused on
        # liveness alone rather than on liveness *plus* a holding would
        # break exactly this.
        cached_text = doc_text(3)
        cache = tmp_path / "cache"
        builder = librarian(WatchfulLLM())
        asyncio.run(builder.index(cached_text, doc_id="cached", cache=cache))
        assert (cache / "index.json").exists()

        lib = librarian(WatchfulLLM())
        idle = asyncio.new_event_loop()
        try:
            # An add that *finished* on a loop nobody closed — the state a
            # script leaves behind by calling `run_until_complete` and
            # moving on. `run_until_complete` rather than `asyncio.run`
            # because the point is a loop that is still open.
            idle.run_until_complete(lib.add_document(doc_text(1), doc_id="d1"))

            # The precondition that makes this test mean something: the
            # guard's loop reads live from here, and holds nothing.
            assert lib._replace_loop is idle
            assert lib._observe_replace_guard() == (True, False)
            assert lib._adds_in_flight_here() == 0
            assert lib._chunking == set()
            # Empty rather than "holds None": the add that finished on this
            # loop gave its registration back by removing the entry, and so
            # did the `index()` inside it, which is what keeps a librarian
            # used across many loops from growing one entry per loop.
            assert lib._adds_in_flight == {}
            assert lib._replace_owner == {}

            caplog.set_level(logging.INFO, logger=SOURCE_LOGGER)
            asyncio.run(lib.index(cached_text, doc_id="cached", cache=cache))
        finally:
            if not idle.is_closed():
                idle.close()

        messages = [r.getMessage() for r in caplog.records]
        assert any("reused cached index" in m for m in messages), (
            "the cache was not actually loaded, so this proves nothing"
        )
        assert not [m for m in messages if "cannot load safely" in m]
        assert not [m for m in messages if "cannot save safely" in m]
        assert list(lib.documents) == ["cached"]

    def test_a_cached_index_is_refused_when_another_live_loop_holds_an_add(
        self, tmp_path, monkeypatch
    ):
        # The positive half, and the one nothing else in the suite pins:
        # `_load_if_current` *does* raise when a live loop somewhere else
        # holds a registered add. This is what the hot path's quiet rests
        # on — not the drain, which only empties this loop's entry.
        #
        # The refusal is off because with it on this state cannot be
        # reached: taking the librarian onto a new loop while the old one
        # holds a registered add is precisely what `LoopHandoverError`
        # declines, and it fires from `_replacing` before `index()` gets
        # anywhere near the cache. That is the point rather than a
        # workaround — installing the hatch is how "the promise rests on
        # the refusal" becomes an assertion instead of a sentence.
        without_the_handover_refusal(monkeypatch)
        cached_text = doc_text(3)
        cache = tmp_path / "cache"
        builder = librarian(WatchfulLLM())
        asyncio.run(builder.index(cached_text, doc_id="cached", cache=cache))
        assert (cache / "index.json").exists()

        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            # The premise, produced rather than assumed: a registration on a
            # loop that is open and can still resume, and it is not the loop
            # the cached index will run on.
            assert lib._adds_registered_on(held) == 1
            assert not held.is_closed()
            assert lib._adds_in_flight_here() == 1

            lib.llm = WatchfulLLM()
            with pytest.raises(IndexBusyError) as excinfo:
                asyncio.run(
                    lib.index(cached_text, doc_id="cached", cache=cache)
                )

            message = str(excinfo.value)
            # Out of the *add* half, from inside `_load_if_current` — not
            # the replace half, which cannot fire for the owner of the hold.
            assert "cannot load safely" in message
            assert "1 add_document call(s) are in flight" in message
            assert "index() is replacing this librarian" not in message
            # Not the handover refusal either: that is the thing switched
            # off, and mistaking one for the other would make this test
            # pass for the wrong reason.
            assert not isinstance(excinfo.value, LoopHandoverError)

            # The remedy line is no longer wrong here. It used to tell this
            # caller flatly that `index(..., cache=...)` is unaffected, which
            # at this one call site it is not, and that was recorded in the
            # source as a known rough edge; it now scopes the clause to the
            # adds this path drains — its own loop's — and says an add on
            # another live loop refuses here too. Still deliberately not
            # asserted, and for the reason that outlives the fix rather than
            # the one that motivated it: pinning wording makes the next
            # rewording fail a test that is not about it.

            # And the damage is real, not cosmetic: `index()` resets before
            # it consults the cache, so the refusal lands on an already
            # emptied librarian. That is what makes this worth refusing
            # loudly rather than warning about.
            assert list(lib.documents) == []

            llm.release.set()
            held.run_until_complete(parked)
        finally:
            dispose(parked, held)
