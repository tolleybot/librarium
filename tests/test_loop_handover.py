"""Taking a librarian onto a new loop while a live one holds work is refused.

`_replace_guard`'s new-loop branch exists so a librarian can outlive the
loop it was used on: meeting a loop it has not seen, it rebuilds the
asyncio primitives — which bind to the loop they are first awaited on —
and forgets the bookkeeping of loops that have closed. What it cannot do
is *move* work: an `index()` here waits out the adds registered on this
loop and no others, because an `asyncio.Event` can only signal the loop it
is bound to.

A loop abandoned **open** with an `add_document` or an `index()` parked on
it is therefore holding work this loop can neither wait for nor see
finish. The parked task is suspended, not dead: the next turn of that loop
commits it into whatever index is here by then. So the transition is
refused.

* :class:`TestTheHandoverIsRefused` — the transition, through the public
  API, for each shape of held work.
* :class:`TestWhatProceedingWouldCost` — the refusal's own justification:
  with it off, the parked add resuming after the new loop's `index()`
  orphans its chunks. Why this raises instead of warning.
* :class:`TestWhereTheRefusalSurfaces` — it lands on the caller doing the
  illegal thing, and nowhere else. `save`/`load` keep their own refusals.
* :class:`TestLegalMultiLoopUseStaysLegal` — the tests that matter most.
  Two `asyncio.run()` calls, a live-but-idle previous loop, one loop
  throughout, and a previous loop that died holding work. A guard that
  cried wolf on ordinary script usage would be worse than the bug.
* :class:`TestTheDeferredDefectsDoNotReproduce` — D1, D2 and D3, measured
  with the refusal **off**, which is the only way to tell a defect fixed
  from a defect fenced.
* :class:`TestTheComplaint` — what the sentence has to say.
* :class:`TestWhatEachHoldingIsWorth` — the mutation check. The count is
  reachable alone through the API and its check is load-bearing; the claim
  and the owner are not, and are isolated by hand.
* :class:`TestOnlyTheRefusalKeepsTheRebuiltEventClosed` — a recorded
  limit, made a guarded invariant: `_replace_lock` and
  `_adds_idle` are rebuilt per handover, and only this refusal keeps a loop
  from being re-adopted while an add parked on it still holds the `Event` its
  drain would no longer be watching.

**The refusal is preventive.** Nothing is corrupted at the transition; the
harm lands later, when the parked task resumes. So every positive test
here also asserts that the librarian came out of the refusal unchanged — a
refusal that half-adopted the new loop would be worse than none.

**Determinism.** `test_guard_liveness.py`'s discipline, borrowed whole:
nothing sleeps, `abandon` produces stranded state by walking away from a
real loop rather than by assigning to private attributes, `park_an_add` /
`park_an_index` assert the registration they waited for before returning,
and the tests that could hang run in a subprocess under a timeout.
`asyncio.wait_for` was measured not to bound that hang — the drain never
yields, so nothing else on the loop ever runs, the timeout callback
included.

**What changed under this file, and why the class above is the headline.**
This used to say the refusal closed the *route* to D1 and D2 while leaving
the defects in place — the resets untouched, which loop owned the count
undecided. That is no longer true: the count and the owner are keyed by
loop, so a decrement finds its own increment, a handover has no other
loop's state to zero, and a parked `index()` keeps its recorded hold.
:class:`TestTheDeferredDefectsDoNotReproduce` takes the refusal away and
shows all three failing to reproduce on the scripts that used to produce
them, and its predecessor asserted the opposite. The refusal stays, and
stays permanent: it is what makes "wait only for your own loop" a rule
rather than a hole.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import subprocess
import sys
import textwrap

import pytest

from librarium import IndexBusyError, LoopHandoverError
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
)
from tests.test_guard_liveness import (
    abandon,
    dispose,
    on_a_plain_thread,
    without_the_handover_refusal,
)

#: Long enough for a refused handover to be reported and a suite to run,
#: short enough that the hang it is bounding fails a test rather than a CI
#: job. The spinning drain never yields, so only a process boundary can
#: interrupt it.
SUBPROCESS_TIMEOUT = 30.0


def bookkeeping(lib) -> tuple:
    """The three holdings plus the adopted loop, as one comparable value.

    Compared before and against after rather than asserted piecemeal: the
    claim is that a refused handover changes *nothing*, and naming the
    fields one at a time invites the next reader to add a fourth field to
    the guard and not to this tuple.

    **Every mutable piece is copied.** The registrations and the owners are
    `dict`s keyed by loop, so holding the live objects would compare a
    dict against itself: `before == after` would pass however much the
    handover had rebound inside them, and this whole helper would be
    measuring nothing. `_chunking` was already frozen for that reason.
    """
    return (
        lib._replace_loop,
        dict(lib._adds_in_flight),
        frozenset(lib._chunking),
        dict(lib._replace_owner),
    )


#: The bound for a child that is *expected* not to finish, where the timeout
#: is the measurement rather than the failure. Shorter than
#: `SUBPROCESS_TIMEOUT` because a longer wait adds no information to "it
#: never finishes" while costing every run of the suite that much: the spin
#: starts on the first turn of the drain, and the same script with the
#: refusal in force exits in about a second (evidence:
#: `TestOnlyTheRefusalKeepsTheRebuiltEventClosed
#: ::test_the_refusal_is_what_keeps_the_hazard_out_of_reach`, which is that
#: script run in-process and passes in well under a second). Ten times the
#: legitimate runtime is enough headroom that a slow machine cannot turn a
#: finishing child into a false "it hung".
SPIN_TIMEOUT = 10.0


def run_in_a_subprocess(
    body: str, *, timeout: float = SUBPROCESS_TIMEOUT,
    expect_timeout: bool = False,
) -> subprocess.CompletedProcess | str:
    """Run *body* against this checkout, bounded by a wall-clock timeout.

    *expect_timeout* inverts the verdict for the one case where not
    finishing is the thing being measured: the child must time out, and what
    comes back is **the output it managed to print first**, as a `str`. A
    child that *does* finish then fails the test, because a hazard that has
    stopped reproducing is a finding rather than a pass. Default is the
    original behaviour — a timeout is the failure — so no existing caller
    changes.

    Handing back the partial output is what keeps that mode honest, and it
    is not a convenience. "It timed out" is also what a merely *slow* child
    looks like, so on a loaded machine — this one has run the same suite
    between 110s and 694s — a timeout could be reported as the hazard
    reproducing when the child had not yet reached it. The caller asserts on
    markers the child prints as it goes, which turns the verdict from a
    timing guess into a position in the script. `TimeoutExpired.stdout` is
    **bytes even under `text=True`** on CPython 3.11, hence the decode.

    `test_degradation.py`'s shape, for its reason: some failures cannot be
    caught in-process. Here it is a busy-wait — `_replacing`'s drain reads
    a negative count as busy and waits on an Event that is already set, so
    it never yields and no timeout scheduled on that loop can fire.

    Runs `sys.executable` and relies on the editable install resolving,
    like the subprocess tests in `test_degradation.py`. Do not add
    PYTHONPATH=src — it masks a broken install — but the repository root does have to
    be importable for `tests.*`, so it is passed as an argument rather
    than assumed.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(root)!r})
        """
    ) + textwrap.dedent(body)
    try:
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as expired:
        if expect_timeout:
            partial = expired.stdout or b""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            return partial
        # The failure this exists to catch, reported as the thing it is
        # rather than as a raw timeout out of the plumbing. Measured: with
        # the refusal removed, this is the branch that fires.
        pytest.fail(
            f"the child did not finish within {timeout}s — it is "
            f"spinning, which is the hang the refusal exists to keep out of "
            f"reach. stdout so far: "
            f"{(expired.stdout or b'').decode(errors='replace')!r}"
        )
    if expect_timeout:
        pytest.fail(
            f"the child finished within {timeout}s, and it was expected not "
            f"to. That is a better outcome than the alternative, but it means "
            f"the hazard this measures has stopped reproducing and the "
            f"paragraph describing it is now wrong. stdout: {done.stdout!r} "
            f"stderr: {done.stderr[-800:]!r}"
        )
    return done


class TestTheHandoverIsRefused:
    """The transition itself, once per shape of work the old loop holds.

    Each case is produced by `abandon(..., close=False)` — a real loop run
    to a parked point and walked away from — so the state under test is
    the state a real teardown leaves, and the assertions before the
    refusal are what makes it a reproduction rather than a mock.

    The two shapes are the ones `abandon` can leave behind, and they are
    not the three holdings taken one at a time. An `add_document` parked in
    its **LLM phase** holds the count *and* a `_chunking` claim; a parked
    `index()` holds the replace owner *and* a claim, with the count at
    zero.

    Scoped to that phase deliberately: it describes where these doubles
    park, not an invariant. `_adding` registers before `_add_document`
    claims anything, and the under-`chunk_threshold` path never claims at
    all — so the count *does* occur alone, which
    :meth:`TestWhatEachHoldingIsWorth.test_a_registration_with_no_claim_is_reachable_after_all`
    reaches through the API. An earlier version of this paragraph said
    "neither the count nor the claim occurs alone — measured, not assumed",
    which read a fact about `GatedLLM`'s parking point as a property of the
    code.

    Catches: the refusal being dropped from `_replace_guard`'s new-loop
    branch, and a refusal that fires but leaves the guard partly rebound.
    """

    def test_an_add_on_a_new_loop_refuses_while_the_old_loop_holds_an_add(self):
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))

        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            # Produced rather than assigned: d1 is registered and claimed
            # by an add suspended on a loop that is open and can resume.
            assert lib._adds_registered_on(held) == 1
            assert lib._chunking == {"d1"}
            assert not held.is_closed()
            before = bookkeeping(lib)

            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError) as excinfo:
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

            # The whole point of refusing *here*: nothing was rebound, so
            # the parked add still has the count and the claim it is
            # holding, and the guard still belongs to the loop it is on.
            assert bookkeeping(lib) == before
            assert "cannot add_document on this event loop" in str(excinfo.value)
            # And d2 did not land: a refused add is a refused add.
            assert "d2" not in lib.documents
        finally:
            dispose(parked, held)

    def test_an_index_on_a_new_loop_refuses_too(self):
        # `index()` reaches the guard through `_replacing` rather than
        # `_adding`, which is a different call site for the same branch.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            before = bookkeeping(lib)
            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError) as excinfo:
                asyncio.run(lib.index(doc_text(2), doc_id="d2"))

            assert bookkeeping(lib) == before
            # Named for the call the reader made, not for the guard they
            # have never heard of.
            assert "cannot index on this event loop" in str(excinfo.value)
            # `index()` resets before it does anything else, so a refusal
            # that arrived one line late would have emptied the index.
            assert "d1" in lib.documents
        finally:
            dispose(parked, held)

    def test_a_parked_index_is_itself_a_holding(self):
        # The third piece of bookkeeping: `_replace_owner`. A parked
        # `index()` holds it with the count at zero, so a guard that only
        # counted adds would wave this handover through.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_index(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            assert lib._replace_owner_on(held) is parked
            assert lib._adds_registered_on(held) == 0
            before = bookkeeping(lib)

            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError) as excinfo:
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

            assert bookkeeping(lib) == before
            assert "an index() holding it exclusively there" in str(excinfo.value)
        finally:
            dispose(parked, held)

    def test_the_refusal_repeats_and_then_lifts_when_the_loop_closes(self):
        # A refusal that fired once and then let the second attempt
        # through would be the corruption with an extra step, and the
        # message promises the opposite: "nothing here has changed, so the
        # same call succeeds once it is".
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            for attempt in range(3):
                with pytest.raises(LoopHandoverError):
                    asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
        finally:
            # The remedy the message names, taken: close the loop the
            # work is parked on. `dispose` also runs the parked
            # coroutine's `finally`, which is what a real teardown does.
            dispose(parked, held)

        # And now the same call goes through, on a librarian nothing else
        # has touched.
        asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
        assert "d2" in lib.doc_summaries
        assert {c.doc_id for c in lib.retrieve(doc_marker(2), top_k=3)} == {"d2"}

    def test_it_is_an_index_busy_error(self):
        # A caller who already wraps a shared librarian in `except
        # IndexBusyError` keeps catching this, and `_load_if_current`'s
        # `except ValueError` still cannot swallow it into a silent,
        # expensive re-index.
        assert issubclass(LoopHandoverError, IndexBusyError)
        assert issubclass(LoopHandoverError, RuntimeError)
        assert not issubclass(LoopHandoverError, ValueError)

        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            with pytest.raises(IndexBusyError):
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
        finally:
            dispose(parked, held)


class TestWhatProceedingWouldCost:
    """The refusal's central claim: proceeding *is* the corruption.

    `LoopHandoverError`'s docstring rests its whole shape on this — the
    reason it raises rather than warns, and the reason it ignores `strict`,
    is that "a flag can only choose between raising and *degrading*, and
    there is no degraded outcome on offer to select". If proceeding merely
    risked something, a warning would be defensible.

    That claim was carried by two things and not by a test of the case it
    describes. `TestTheCorruptionTheRefusalPrevents` measures the
    **one-loop** mechanism — a `load()` resetting under a parked add — and
    the cross-loop instance was inferred from it plus the observation that
    `_replacing`'s reset is the same reset. `source.py` marked it
    `(unverified: no test resumes the parked add after the new loop's
    reset)`, which was accurate:
    `TestTheDeferredDefectsDoNotReproduce` takes the refusal away and
    reaches the state, but its cases assert the count, the owner and the
    drain returning, and **none of them resumes the parked add**. The
    `index()` in `test_the_drain_waits_for_its_own_loops_adds_and_no_others`
    performs exactly the reset at issue and the test ends there.

    So this resumes it, and measures what lands: 24 chunks for a document
    `documents` no longer names, `retrieve` handing them back first, and no
    exception anywhere. The refusal is off, because with it on the `index()`
    never happens — which is the point, and the same fix-versus-fence
    discipline as `TestTheDeferredDefectsDoNotReproduce`. Unlike those, this
    one asserts the damage is **still there**: it is the justification, not
    a defect.

    Catches: the justification quietly becoming false — if a later change
    made a parked add's commit land harmlessly (rolled back, re-swept,
    refused at commit time), `LoopHandoverError` would need to be re-argued
    as a warning and nothing would have said so. It also catches the
    `_replacing` reset ceasing to be the reset `load()` performs, which is
    the inference the prose was standing on.
    """

    def test_a_parked_add_resuming_after_the_new_loops_index_orphans_its_chunks(
        self, monkeypatch
    ):
        without_the_handover_refusal(monkeypatch)
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            # The reset at issue, on a second loop, admitted only because
            # the refusal is off. `index()` resets before it does anything
            # else, so d1 is gone from `documents` from here on.
            asyncio.run(lib.index(doc_text(3), doc_id="fresh"))
            assert list(lib.documents) == ["fresh"]
            assert lib._adds_registered_on(held) == 1, (
                "the parked add must still be registered, or this measures "
                "nothing"
            )

            # The step nothing else in the suite takes.
            llm.release.set()
            held.run_until_complete(parked)
        finally:
            dispose(parked, held)

        # Shape 1, cross-loop: chunks for a document the index does not
        # know it holds. No exception was raised anywhere above.
        orphans = [c for c in lib.chunks if c.doc_id not in lib.documents]
        assert {c.doc_id for c in orphans} == {"d1"}
        assert len(orphans) >= 20, "the whole document was stranded"
        assert list(lib.documents) == ["fresh"]
        assert "d1" in lib.doc_summaries and "d1" not in lib.documents
        assert ODD_MARK not in lib.source

        # Not inert, which is what makes it corruption rather than
        # untidiness: retrieval ranks them first for a term only they
        # contain, so an answer cites a document the index does not hold.
        hits = lib.retrieve(ODD_MARK, top_k=3)
        assert {c.doc_id for c in hits} == {"d1"}
        assert all(ODD_MARK in c.text for c in hits)


class TestWhereTheRefusalSurfaces:
    """It lands on the caller doing the handover, and on nobody else.

    `_replace_guard` is reached from `_adding` and `_replacing`, so from
    `add_document` and `index`. The synchronous `save`/`load` pair does not
    call it — they only *read* through `_observe_replace_guard` — and they
    have refusals of their own, with a per-case split (`load` raises, `save`
    warns mid-add and raises mid-`index`) that this change must not
    disturb.

    Catches: the refusal being moved into `_observe_replace_guard` or into
    the synchronous observers, where it would turn every later
    `save`/`load` into a `LoopHandoverError` and re-open the question the
    per-case split already answered.
    """

    def test_load_still_raises_the_base_error_and_not_the_handover_one(
        self, tmp_path
    ):
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)

        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            with pytest.raises(IndexBusyError) as excinfo:
                lib.load(tmp_path)
            # `LoopHandoverError` is an `IndexBusyError`, so the exact type
            # is what distinguishes the two refusals — `pytest.raises`
            # alone would pass either way.
            assert type(excinfo.value) is IndexBusyError
            assert "1 add_document call(s) are in flight" in str(excinfo.value)
        finally:
            dispose(parked, held)

    def test_save_still_warns_mid_add_rather_than_raising_anything(
        self, tmp_path, caplog
    ):
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            caplog.set_level(logging.WARNING, logger=SOURCE_LOGGER)
            lib.save(tmp_path)             # writes, warns, does not raise
            messages = [r.getMessage() for r in caplog.records]
            assert any("cannot save safely" in m for m in messages)
            assert (tmp_path / "index.json").exists()
        finally:
            dispose(parked, held)

    def test_a_synchronous_caller_off_any_loop_is_unaffected(self, tmp_path):
        # The `personal-librarian` pattern: a plain worker thread with no
        # loop of its own. `_handover_would_strand` is never reached from
        # there, and the observers still answer, so what comes back is the
        # base refusal.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        lib.save(tmp_path)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            with pytest.raises(IndexBusyError) as excinfo:
                on_a_plain_thread(lambda: lib.load(tmp_path))
            assert not isinstance(excinfo.value, LoopHandoverError)
        finally:
            dispose(parked, held)


class TestLegalMultiLoopUseStaysLegal:
    """The negative half, and the half that matters more.

    A librarian used across two `asyncio.run()` calls is legal, common in
    scripts, and is the reason `_replace_guard`'s new-loop branch is
    written the way it is. A guard that refused there would break every
    such script to prevent a bug none of them have.

    Every case here is a handover the refusal must wave through:

    * two `asyncio.run()` calls — the loop closes, so it reads dead;
    * a previous loop that is open but **idle** — live, holding nothing;
    * one loop throughout — the branch is never taken at all;
    * a previous loop that closed while still holding work — the case the
      resets exist for, and the wedge that must stay closed.

    Catches: the refusal being written against loop *identity* rather than
    against what the previous loop holds (every test here fails), and the
    liveness reading being replaced by `is_running()` (the idle case
    fails, and so would the third one).
    """

    def test_two_asyncio_run_calls_over_one_librarian(self):
        lib = librarian(WatchfulLLM())
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        # A second, entirely separate loop — the pattern a script written
        # with two `asyncio.run()` calls produces, and the one the
        # new-loop branch was added for.
        asyncio.run(lib.add_document(doc_text(1), doc_id="d1"))
        asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

        assert set(lib.doc_summaries) == {"d0", "d1", "d2"}
        assert {c.doc_id for c in lib.retrieve(doc_marker(1), top_k=3)} == {"d1"}

    def test_a_previous_loop_that_is_open_but_idle_hands_over(self):
        lib = librarian(WatchfulLLM())
        idle = asyncio.new_event_loop()
        try:
            # `run_until_complete` and no `close()`: the loop is open, and
            # reads live, but the add it ran has finished and given back
            # everything it held.
            idle.run_until_complete(lib.add_document(doc_text(0), doc_id="d0"))
            assert lib._replace_loop is idle
            assert lib._observe_replace_guard() == (True, False)
            assert (
                lib._adds_registered_on(idle),
                lib._chunking,
                lib._replace_owner_on(idle),
            ) == (0, set(), None)

            asyncio.run(lib.add_document(doc_text(1), doc_id="d1"))
        finally:
            idle.close()

        assert set(lib.doc_summaries) == {"d0", "d1"}

    def test_one_loop_throughout_is_never_a_handover(self):
        lib = librarian(WatchfulLLM())

        async def build():
            await lib.index(doc_text(0), doc_id="d0")
            # Concurrent adds on the one loop: the case the replace guard
            # was built for, which must not be mistaken for a handover.
            await asyncio.gather(
                lib.add_document(doc_text(1), doc_id="d1"),
                lib.add_document(doc_text(2), doc_id="d2"),
            )
            await lib.index(doc_text(3), doc_id="d3")

        asyncio.run(build())
        assert set(lib.doc_summaries) == {"d3"}

    def test_a_loop_that_closed_holding_work_still_hands_over(self):
        # The wedge the liveness reading exists to keep closed, restated
        # against the refusal: a count and a claim stranded by a loop that
        # has *closed* cannot commit anything, so refusing over them would
        # wedge the librarian for good — there is no loop left to await or
        # to close.
        llm = GatedLLM(doc_marker(1))
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        stranded, dead = abandon(
            lambda: park_an_add(lib, llm, doc_text(1), "d1"), close=True,
        )
        # The state that would trip the refusal if it read the holdings
        # without asking whether the loop they belong to is alive.
        assert lib._adds_registered_on(dead) == 1
        assert lib._chunking == {"d1"}
        assert dead.is_closed()

        lib.llm = WatchfulLLM()
        asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))    # no refusal

        assert set(lib.doc_summaries) == {"d0", "d1", "d2"}
        dispose(stranded, dead)


class TestTheDeferredDefectsDoNotReproduce:
    """D1, D2 and D3 are gone at the root — measured with the refusal off.

    All three reproductions start the same way: work parked on a loop
    abandoned open, then an `add_document` on a second loop. With the
    refusal in force that second call is a `LoopHandoverError` and none of
    them gets past its first step, which is what the first three tests
    below pin. But a refusal in front of a defect proves nothing about the
    defect, so the rest of this class takes it away
    (:func:`without_the_handover_refusal`) and runs the same scripts.

    * **D1** — the branch used to zero a count the parked add was still
      holding, and the add's own `finally` then decremented it to -1, which
      `_replacing`'s drain read as busy while waiting on an Event already
      set: it never yielded, so `index()` spun. Keyed by loop, a decrement
      lands in the entry its increment came from, so **-1 is unreachable by
      construction** rather than unreachable because something refuses
      first.
    * **D2** — the same zeroing destroyed the reading a synchronous
      `load()` refuses on, admitting a `load()` that stranded the parked
      document's chunks in the index that replaced it. The entry survives
      the handover and the observers read every live loop's, so the
      `load()` stays refused.
    * **D3** — the branch also dropped a parked `index()`'s
      `_replace_owner`, letting a later `save()`/`load()` past a replace
      that had not finished. The owner is per loop too, so the hold
      survives.

    This class's predecessor asserted the opposite of the three tests at
    the end — that D2 came straight back with the refusal removed — and it
    was right to, at the fork point. That inversion is the evidence this
    cycle did something the last one did not.

    Catches: the refusal ceasing to cover the handover (the first three);
    and the registrations or the owner reverting to a single shared value,
    which is what the refusal-off tests exist for and what nothing else in
    the suite would notice, because the refusal would hide it again.
    """

    def test_the_count_is_not_spent_by_the_refusal(self):
        # D1's first two steps, and then the resumption that used to
        # produce -1. The count goes 1 -> (refused) -> 0.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError):
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            # Intact because nothing zeroes another loop's entry any more —
            # not because the refusal handed it back. The refusal fires
            # before a field is rebound, so it never had it to spend.
            assert lib._adds_registered_on(held) == 1

            # Resume the parked add: with its registration intact, its
            # `finally` gives back what it took.
            llm.release.set()
            held.run_until_complete(parked)
            assert lib._adds_registered_on(held) == 0
            assert lib._chunking == set()
        finally:
            dispose(parked, held)

    def test_index_does_not_spin_where_it_used_to_hang(self):
        # In a subprocess because the failure mode is a busy-wait: the
        # drain never yields, so an in-process timeout can never fire.
        # `asyncio.wait_for` was measured not to bound it.
        result = run_in_a_subprocess(
            """
            import asyncio
            from tests.test_indexing import GatedLLM, WatchfulLLM, doc_text
            from tests.test_load_concurrency import (
                ODD_MARK, librarian, odd_doc, park_an_add,
            )
            from librarium import LoopHandoverError

            def a_loop():
                loop = asyncio.new_event_loop()
                loop.set_exception_handler(lambda l, c: None)
                return loop

            llm = GatedLLM(ODD_MARK)
            lib = librarian(llm)
            asyncio.run(lib.index(doc_text(0), doc_id="d0"))

            first = a_loop()
            parked = first.run_until_complete(
                park_an_add(lib, llm, odd_doc(), "d1")
            )
            lib.llm = WatchfulLLM()

            # The second loop stays open: the hang needs the damaged count
            # to be read on the loop the guard has adopted, and a third
            # loop would reset it away before the drain saw it.
            second = a_loop()
            try:
                second.run_until_complete(lib.add_document(doc_text(2), "d2"))
            except LoopHandoverError:
                print("REFUSED")
            else:
                llm.release.set()
                first.run_until_complete(parked)
                print("ADMITTED, count now", lib._adds_registered_on())
                # This is the call that spins when the count is negative.
                second.run_until_complete(lib.index(doc_text(3), doc_id="d3"))
                print("index() returned")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "REFUSED" in result.stdout, result.stdout
        assert "ADMITTED" not in result.stdout

    def test_load_stays_refused_because_the_count_survives(self, tmp_path):
        # D2, all the way to the corruption it produced: with the handover
        # refused the count is never zeroed, so `load()` keeps refusing and
        # the parked document cannot be stranded in a replacement index.
        cache = tmp_path / "cache"
        builder = librarian(WatchfulLLM())
        asyncio.run(builder.index(doc_text(3), doc_id="cached", cache=cache))

        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            with pytest.raises(IndexBusyError):
                lib.load(cache)            # refused, as it should be

            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError):
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

            # The step that used to be admitted here, still refused.
            with pytest.raises(IndexBusyError):
                lib.load(cache)

            llm.release.set()
            held.run_until_complete(parked)
        finally:
            dispose(parked, held)

        # And d1 committed into its own index rather than into a
        # replacement: no chunk belongs to a document `documents` does not
        # name, which is the corruption D2 produced.
        assert [c for c in lib.chunks if c.doc_id not in lib.documents] == []
        assert {c.doc_id for c in lib.retrieve(ODD_MARK, top_k=3)} == {"d1"}

    def test_d2_does_not_reproduce_with_the_refusal_removed(
        self, tmp_path, monkeypatch
    ):
        # **The headline.** This is the same script that used to produce
        # D2's corruption with the refusal taken away, and the previous
        # version of this test asserted that it did. It must not any more:
        # the handover is admitted, and the `load()` is *still* refused,
        # because the parked add's registration belongs to its own loop and
        # the observers read every live loop's rather than the guard's.
        #
        # That is the whole fix-versus-fence distinction. With a single
        # shared `int` restored, the handover zeroes the count, the `load()`
        # below is admitted, and d1's chunks end up orphaned in the index
        # that replaced them — which is what the three assertions at the
        # end would then report.
        without_the_handover_refusal(monkeypatch)
        cache = tmp_path / "cache"
        builder = librarian(WatchfulLLM())
        asyncio.run(builder.index(doc_text(3), doc_id="cached", cache=cache))

        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            # Admitted: the refusal is off, so this is the handover that
            # used to destroy the reading.
            asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            assert lib._replace_loop is not held, "no handover happened"

            # The reading survived it, and survived *where it was taken*.
            assert lib._adds_registered_on(held) == 1
            # And it is visible to the synchronous pair even though the
            # guard has moved on: this is the sum over every live loop, not
            # the guard's own entry, which reads zero here.
            assert lib._adds_registered_on() == 0
            assert lib._adds_in_flight_here() == 1

            with pytest.raises(IndexBusyError) as excinfo:
                lib.load(cache)
            assert "1 add_document call(s) are in flight" in str(excinfo.value)
            # Refused means refused: the cached index did not land.
            assert "cached" not in lib.documents

            llm.release.set()
            held.run_until_complete(parked)
            # Its own `finally` gave back its own registration. Never -1.
            assert lib._adds_registered_on(held) == 0
            assert lib._adds_in_flight_here() == 0
        finally:
            dispose(parked, held)

        assert [c for c in lib.chunks if c.doc_id not in lib.documents] == [], (
            "D2 reproduced: the fix is a fence"
        )
        assert {c.doc_id for c in lib.retrieve(ODD_MARK, top_k=3)} == {"d1"}
        assert set(lib.doc_summaries) == {"d1", "d2"}

    def test_d1_does_not_reproduce_with_the_refusal_removed(self, monkeypatch):
        # D1's other half, on the same terms: the resuming parked add gives
        # back the registration it took rather than one that was taken away
        # from it, so no entry ever goes negative. Under a shared `int` the
        # count here is -1 and `_adds_in_flight_here` reports it.
        without_the_handover_refusal(monkeypatch)
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        asyncio.run(lib.index(doc_text(0), doc_id="d0"))
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            llm.release.set()
            held.run_until_complete(parked)

            assert lib._adds_registered_on(held) == 0
            assert lib._adds_in_flight_here() == 0
            # Every entry, not just a reading that happens to come out at
            # zero: a -1 hiding behind a +1 elsewhere is the same defect.
            assert all(n >= 0 for n in lib._adds_in_flight.values())
            # Balanced entries are removed, so after this there is nothing
            # left at all.
            assert lib._adds_in_flight == {}
        finally:
            dispose(parked, held)

    def test_index_does_not_spin_with_the_refusal_removed_either(self):
        # The half of D1 no in-process assertion can reach: with a negative
        # count, `_replacing`'s drain reads busy and waits on an Event that
        # is already set, so it never yields and no timeout scheduled on
        # that loop can fire. The refusal is removed *inside the child*, so
        # what bounds the hang is the process boundary.
        result = run_in_a_subprocess(
            """
            import asyncio
            from librarium.source import SourceLibrarian
            from tests.test_indexing import GatedLLM, WatchfulLLM, doc_text
            from tests.test_load_concurrency import (
                ODD_MARK, librarian, odd_doc, park_an_add,
            )

            # `without_the_handover_refusal`, by hand: no monkeypatch in a
            # bare interpreter, and the patch is one lambda either way.
            SourceLibrarian._handover_would_strand = (
                lambda self, previous_live, action: None
            )

            def a_loop():
                loop = asyncio.new_event_loop()
                loop.set_exception_handler(lambda l, c: None)
                return loop

            llm = GatedLLM(ODD_MARK)
            lib = librarian(llm)
            asyncio.run(lib.index(doc_text(0), doc_id="d0"))

            first = a_loop()
            parked = first.run_until_complete(
                park_an_add(lib, llm, odd_doc(), "d1")
            )
            lib.llm = WatchfulLLM()

            second = a_loop()
            second.run_until_complete(lib.add_document(doc_text(2), "d2"))
            print("ADMITTED")
            llm.release.set()
            first.run_until_complete(parked)
            print("count now", lib._adds_registered_on(second))
            # The call that spins when a resumed add drove the count below
            # zero. It has its own registration back, so there is nothing
            # for this drain to wait on.
            second.run_until_complete(lib.index(doc_text(3), doc_id="d3"))
            print("index() returned")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        # The premise: the handover really was admitted, so this is
        # measuring the bookkeeping and not the refusal.
        assert "ADMITTED" in result.stdout, result.stdout
        assert "count now 0" in result.stdout, result.stdout
        assert "index() returned" in result.stdout, result.stdout

    def test_the_drain_waits_for_its_own_loops_adds_and_no_others(self):
        # The rule the refusal makes sound, pinned where it can fail:
        # `_replacing` waits out the adds registered on *its* loop, because
        # an `asyncio.Event` binds to the loop that first awaits it and so
        # an add finishing anywhere else cannot wake this task.
        #
        # Here `first` is open and still holds a parked add, and this
        # `index()` runs on `second`. It must return. A drain that read the
        # librarian-wide total instead — which is what the synchronous
        # observers correctly read, so the swap looks like tidying — waits
        # on `second`'s Event for an add that will only ever set `first`'s,
        # and since that Event is already set the wait never yields: a
        # busy-spin no in-process timeout can interrupt, which is why this
        # is a subprocess. Measured: this is the test that fails.
        #
        # The refusal is off, because with it on this handover never
        # happens — which is exactly why the refusal is permanent rather
        # than scaffolding, and this is the test that says so.
        result = run_in_a_subprocess(
            """
            import asyncio
            from librarium.source import SourceLibrarian
            from tests.test_indexing import GatedLLM, WatchfulLLM, doc_text
            from tests.test_load_concurrency import (
                ODD_MARK, librarian, odd_doc, park_an_add,
            )

            SourceLibrarian._handover_would_strand = (
                lambda self, previous_live, action: None
            )

            def a_loop():
                loop = asyncio.new_event_loop()
                loop.set_exception_handler(lambda l, c: None)
                return loop

            llm = GatedLLM(ODD_MARK)
            lib = librarian(llm)

            first = a_loop()
            parked = first.run_until_complete(
                park_an_add(lib, llm, odd_doc(), "d1")
            )
            lib.llm = WatchfulLLM()

            second = a_loop()
            second.run_until_complete(lib.add_document(doc_text(2), "d2"))
            # The premise: `first` is live and still registered, and the
            # guard has moved to `second`, which is not.
            print("held elsewhere", lib._adds_in_flight_here(),
                  "here", lib._adds_registered_on(second))

            second.run_until_complete(lib.index(doc_text(3), doc_id="d3"))
            print("the drain returned")
            """
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "held elsewhere 1 here 0" in result.stdout, result.stdout
        assert "the drain returned" in result.stdout, result.stdout

    def test_d3_a_parked_index_keeps_its_hold_across_a_handover(
        self, tmp_path, monkeypatch
    ):
        # D3, with the refusal off for the same reason as the two above:
        # the branch used to drop `_replace_owner`, so a `save()`/`load()`
        # after the handover was let past a replace that had not finished —
        # the window `TestLoadRefusesMidIndex
        # ::test_the_corruption_the_refusal_prevents` measures corrupting.
        # Under a single shared owner, `_replacing_elsewhere()` below reads
        # False and the `save` writes the file.
        without_the_handover_refusal(monkeypatch)
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_index(lib, llm, odd_doc(), "fresh"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            assert lib._replace_loop is not held, "no handover happened"

            # The hold is still recorded, still against the loop that took
            # it, and still visible to the synchronous pair.
            assert lib._replace_owner_on(held) is parked
            assert lib._replace_owner_on() is None
            assert lib._replacing_elsewhere() is True

            target = tmp_path / "never-created"
            with pytest.raises(IndexBusyError) as excinfo:
                lib.save(target)
            assert "index() is replacing this librarian" in str(excinfo.value)
            assert not target.exists()

            llm.release.set()
            held.run_until_complete(parked)
            # And given back when it finishes, so the refusal is not a new
            # wedge: the entry is popped rather than left behind.
            assert lib._replace_owner_on(held) is None
            assert lib._replacing_elsewhere() is False
        finally:
            dispose(parked, held)


class TestTheComplaint:
    """What the sentence says, because the sentence is the whole remedy.

    A caller who hits this is holding a librarian that still looks fine
    and a call that failed for a reason nothing in their code mentions.
    The message has to carry three things: what is holding the librarian,
    what taking it anyway would have cost, and which of the two exits to
    take.

    Catches: the message losing the remedy or the holding.
    """

    def test_it_names_the_holdings_the_damage_and_both_exits(self):
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError) as excinfo:
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            message = str(excinfo.value)
        finally:
            dispose(parked, held)

        # The holdings, both of them: a parked add holds the count and a
        # claim, and naming only the count would send the reader looking
        # for one thing while two are stranded.
        assert "1 add_document call(s) registered there" in message
        assert "claimed for chunking there: d1" in message
        # The damage, so it reads as a refusal with a reason rather than
        # as a rule.
        assert "commit into whatever index is here by then" in message
        # Both exits. Neither is optional: awaiting is right when the work
        # is still wanted, closing when the loop is simply finished.
        assert "Await the work parked on that loop" in message
        assert "close it if it is finished" in message

    def test_the_damage_it_names_is_the_one_that_is_still_real(self):
        # The sentence used to promise two things a handover would cost:
        # stranded chunks, and "an in-flight count driven below zero that
        # makes a later index() spin". The second cannot happen now — a
        # decrement finds its own increment — so the message must not still
        # be describing it, and it has to say why indexing here cannot
        # simply wait for indexing there instead.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError) as excinfo:
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            message = str(excinfo.value)
        finally:
            dispose(parked, held)

        assert "below zero" not in message
        assert "Indexing here cannot wait for indexing there" in message
        assert "asyncio.Event can signal" in message


class TestWhatEachHoldingIsWorth:
    """The mutation check: which holdings the API can isolate, and which not.

    The spec asks that weakening the condition to ignore any one of the
    three holdings fails something. Where a holding is reachable alone that
    is a test through the API; where it is not, it is a hand-built state
    here, and the two are kept apart because only one of them says the
    state can happen.

    **The count is reachable alone**, which an earlier version of this
    docstring denied: it claimed "the holdings do not occur alone" and
    "every state the API can reach carries two of the three", on the
    strength of where `GatedLLM` parks. `_adding` registers before
    `_add_document` claims anything, and the under-`chunk_threshold` path
    claims nothing for the whole of its life, so the first test below holds
    a registration with an empty `_chunking` and refuses on it. Reading a
    property of the doubles as a property of the code is the mistake that
    paragraph made.

    **The claim and the owner are not**, for the reason the old paragraph
    was reaching for: an add parked in its LLM phase holds the count *and* a
    claim — the claim is taken inside `_add_document` and given back in its
    `finally`, one step before `_adding` decrements — and a parked `index()`
    holds the owner *and* a claim. So dropping either of those two from the
    condition leaves the other to refuse on, and the weakening is invisible
    from outside.

    So **the count check is load-bearing, not redundant-today**, and that is
    the sharper statement the source now makes at
    `_handover_would_strand`: drop it and a state the API reaches stops
    being refused. Only the claim and the owner are redundant *today*, and
    they are kept anyway, because the three are the three ways a loop can
    still be holding indexing — a property of the guard rather than of what
    `_add_document` happens to hold when it awaits. A condition covering
    only what is currently reachable would be correct and would silently
    stop being correct the first time the phases move; the first test below
    is what that looks like once it has already happened.

    For the two the API cannot isolate, `_handover_would_strand` is a pure
    reader of the guard's state, so a hand-built state is a legitimate way
    to ask what each check contributes — as long as nobody mistakes those
    for tests that the state is reachable.

    Catches: any one of the three holdings being dropped from the
    condition.
    """

    async def test_a_registration_with_no_claim_is_reachable_after_all(self):
        # `_adding` registers before `_add_document` claims anything, so an
        # add suspended between the two holds the count *alone* — and the
        # under-`chunk_threshold` path never claims at all, because it
        # returns before `_chunking.update(pending)` (source.py, "Small
        # enough to hand to the model whole").
        #
        # So the count check is not merely redundant-today: it is the only
        # one of the three that answers here, which is what makes this
        # worth an assertion rather than a caveat. Measured: dropping the
        # count check from `_handover_would_strand` fails this test on its
        # own message, from a state the API reached.
        #
        # Deterministic without sleeping: the add is held by a real
        # `_index_lock` this test owns, and the spin waits on the
        # registration itself — `wait_until_replacing`'s discipline. No
        # `GatedLLM`, because this path makes no LLM call to park on; the
        # lock *is* the only suspension point it has.
        lib = librarian(WatchfulLLM())
        here = asyncio.get_running_loop()
        short = "Held whole, well under the threshold. "

        async with lib._index_lock():
            adding = asyncio.ensure_future(
                lib.add_document(short, doc_id="small")
            )

            async def until_registered():
                while lib._adds_registered_on(here) == 0:
                    await asyncio.sleep(0)

            await asyncio.wait_for(until_registered(), GATE_TIMEOUT)

            # One holding, not two — the state this class used to say the
            # API could not produce.
            assert lib._adds_registered_on(here) == 1
            assert lib._chunking == set(), "a claim was taken after all"
            assert lib._replace_owner_on(here) is None

            message = lib._handover_would_strand(True, "add_document")
            assert message is not None, "the count alone did not refuse"
            assert "1 add_document call(s) registered there" in message
            assert "claimed for chunking" not in message

        await adding
        # The premise, confirmed after the fact: this really was the
        # whole-document path, which is the one that never claims.
        assert lib.use_chunking is False
        assert lib._chunking == set()

    @staticmethod
    def a_librarian_whose_previous_loop_is_live():
        """A librarian reporting a live previous loop and nothing else.

        The loop is real and open — `_observe_replace_guard` asks it
        whether it is closed — so `previous_live` below is the librarian's
        own answer rather than the argument's. It is also the **key** each
        holding has to be planted under: `_handover_would_strand` reads the
        previous loop's own entry, so a holding filed anywhere else is not
        the state being described.
        """
        lib = librarian(WatchfulLLM())
        loop = asyncio.new_event_loop()
        lib._replace_loop = loop
        assert lib._observe_replace_guard() == (True, False)
        return lib, loop

    def test_nothing_held_is_no_complaint(self):
        lib, loop = self.a_librarian_whose_previous_loop_is_live()
        try:
            assert lib._handover_would_strand(True, "index") is None
        finally:
            loop.close()

    @pytest.mark.parametrize(
        "holding,expected",
        [
            ("count", "2 add_document call(s) registered there"),
            ("claim", "claimed for chunking there: d1, d2"),
            ("owner", "an index() holding it exclusively there"),
        ],
    )
    def test_each_holding_alone_is_enough_to_refuse(self, holding, expected):
        lib, loop = self.a_librarian_whose_previous_loop_is_live()
        try:
            if holding == "count":
                lib._adds_in_flight = {loop: 2}
            elif holding == "claim":
                lib._chunking = {"d2", "d1"}       # reported sorted
            else:
                lib._replace_owner = {loop: object()}

            message = lib._handover_would_strand(True, "add_document")
            assert message is not None, f"{holding} alone did not refuse"
            assert expected in message
        finally:
            loop.close()

    def test_a_dead_previous_loop_holds_nothing_whatever_it_says(self):
        # The other half of the condition, and the one that keeps the
        # wedge closed: `previous_live` false short-circuits every
        # holding, because state a closed loop left cannot commit.
        lib = librarian(WatchfulLLM())
        lib._adds_in_flight = {lib._replace_loop: 3}
        lib._chunking = {"d1"}
        lib._replace_owner = {lib._replace_loop: object()}
        assert lib._handover_would_strand(False, "index") is None

    def test_the_action_it_is_given_is_the_action_it_names(self):
        lib, loop = self.a_librarian_whose_previous_loop_is_live()
        try:
            lib._adds_in_flight = {loop: 1}
            for action in ("add_document", "index"):
                message = lib._handover_would_strand(True, action)
                assert message.startswith(f"cannot {action} on this event loop")
        finally:
            loop.close()

    def test_the_two_call_sites_pass_the_action_they_are(self):
        # The pairing the parametrised test above cannot see: `_adding`
        # says "add_document" and `_replacing` says "index". Swapping them
        # would leave every message grammatical and every one of them
        # pointing at the wrong call.
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, held = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            lib.llm = WatchfulLLM()
            with pytest.raises(LoopHandoverError) as from_add:
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
            with pytest.raises(LoopHandoverError) as from_index:
                asyncio.run(lib.index(doc_text(2), doc_id="d2"))
        finally:
            dispose(parked, held)

        assert "cannot add_document" in str(from_add.value)
        assert "cannot index" in str(from_index.value)


class TestOnlyTheRefusalKeepsTheRebuiltEventClosed:
    """A recorded limit, turned into a guarded invariant.

    The limit is that `_replace_lock` and `_adds_idle` are **rebuilt** per
    handover rather than kept per loop, and that re-adopting a loop which
    still holds registrations of its own would hand its drain a fresh
    `Event` while the add parked over there sets the old one — with **only
    the refusal** keeping it out of reach. It was a note nobody checked, and
    the thing it depends on is a refusal a later cycle could narrow for
    reasons that look unrelated.

    **The route, traced against the code rather than assumed.**
    `_handover_would_strand` reads `_adds_registered_on()` and
    `_replace_owner_on()` with no argument, so both default to
    `self._replace_loop` — and `_replace_guard` assigns `_replace_loop =
    loop` *after* the strand check, so the question is always about the
    **previous** loop and never about the one being adopted. For L1 to hold
    registrations while the guard belongs to L2, the L1→L2 handover must
    therefore have happened while L1 held work, which is exactly what
    `LoopHandoverError` declines. Re-adopting L1 afterwards is *not* itself
    refused **once L2 has closed** — which is what `asyncio.run` does, and how
    the script below gets there. The count and the owner are asked of the
    previous loop, so L1's own registration is invisible from L2's vantage;
    `_chunking` is not loop-keyed and does still refuse while L2 is open
    (measured). So the refusal's whole contribution is blocking the step
    before.

    **What the hazard actually does, measured.** Not a blocked wait, which
    is what "wait for something nobody can signal" suggests: the rebuilt
    `Event` is `.set()` at build time, so `await idle.wait()` returns
    without yielding, the `while` re-reads a count only the *old* Event's
    setter could lower, and the loop spins without ever giving the parked
    add a turn. That is why it has to be a subprocess —
    `asyncio.wait_for` cannot bound a drain that never yields, a fact this
    file's `run_in_a_subprocess` already records for the same shape.

    Catches: `LoopHandoverError` being narrowed so that a handover away
    from a loop holding registrations is admitted. **Nothing here is
    reachable today** — the first test installs
    `without_the_handover_refusal` to reach it at all, and the second is
    why. Read this as defending a hazard that a future narrowing would
    make live, not as protecting against something that happens now.

    **How much narrowing it takes, measured, because that is what a future
    narrower needs.** Dropping the count check alone from
    `_handover_would_strand` does *not* open this: a parked `add_document`
    holds a `_chunking` claim as well as a registration, and the claim still
    refuses — so the second test below keeps passing. What opens it is
    dropping the count *and* the claim, or neutralising the refusal
    outright; both were forced and both fail the second test on `DID NOT
    RAISE LoopHandoverError`. So the hazard sits behind two of the three
    holdings, which is why `TestWhatEachHoldingIsWorth` keeping all three
    even where one looks redundant is load-bearing here and not tidiness.
    """

    #: The one script, parameterised only by whether the refusal is in
    #: force, so the pair differs in nothing else. Step 2 is the handover the
    #: refusal declines; step 3 is the re-adoption that spins.
    SCRIPT = """
        import asyncio
        from librarium.source import SourceLibrarian
        from tests.test_indexing import GatedLLM, WatchfulLLM, doc_text
        from tests.test_load_concurrency import (
            ODD_MARK, librarian, odd_doc, park_an_add,
        )

        SourceLibrarian._handover_would_strand = (
            lambda self, previous_live, action: None
        )

        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)

        first = asyncio.new_event_loop()
        first.set_exception_handler(lambda l, c: None)
        parked = first.run_until_complete(
            park_an_add(lib, llm, odd_doc(), "d1")
        )
        assert lib._adds_registered_on(first) == 1
        lib.llm = WatchfulLLM()

        # Step 2: the handover the refusal declines. Admitted here.
        asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))
        assert lib._replace_loop is not first
        print("HANDOVER ADMITTED", flush=True)

        # Step 3: re-adopt `first`, which rebuilds its lock and Event while
        # the add parked on it still holds a registration and still holds the
        # Event it captured. The drain then spins.
        print("RE-ADOPTING", flush=True)
        first.run_until_complete(lib.index(doc_text(3), doc_id="fresh"))
        print("INDEX RETURNED", flush=True)
    """

    def test_the_hazard_reproduces_with_the_refusal_removed(self):
        # The measurement the limit paragraph rests on. If this ever fails
        # by the child *finishing*, the hazard has gone and the paragraph
        # needs rewriting rather than this test relaxing —
        # `run_in_a_subprocess` says so at the failure site.
        printed = run_in_a_subprocess(
            self.SCRIPT, timeout=SPIN_TIMEOUT, expect_timeout=True,
        )
        # Where it got to before it stopped finishing, asserted rather than
        # assumed: a timeout alone is also what a slow child looks like, and
        # this is what distinguishes "spinning in the drain" from "had not
        # reached the drain yet".
        assert "HANDOVER ADMITTED" in printed, printed
        assert "RE-ADOPTING" in printed, printed
        assert "INDEX RETURNED" not in printed, printed

    def test_the_refusal_is_what_keeps_the_hazard_out_of_reach(self):
        """The same script with the refusal in force, in-process.

        Safe in-process precisely because the refusal fires at step 2: the
        re-adoption that spins is never reached, so there is nothing here a
        timeout would have to bound. That asymmetry *is* the claim.

        Catches: the refusal ceasing to cover a handover away from a loop
        that holds registrations — which is the single event that makes the
        test above's hazard reachable through the public API.
        """
        llm = GatedLLM(ODD_MARK)
        lib = librarian(llm)
        parked, first = abandon(
            lambda: park_an_add(lib, llm, odd_doc(), "d1"), close=False,
        )
        try:
            assert lib._adds_registered_on(first) == 1
            captured = lib._adds_idle
            lib.llm = WatchfulLLM()

            with pytest.raises(LoopHandoverError):
                asyncio.run(lib.add_document(doc_text(2), doc_id="d2"))

            # The consequence that matters, and the reason this is not a
            # duplicate of `TestTheHandoverIsRefused`: the guard never left
            # `first`, so the primitives were never rebuilt under a live
            # registration, so the re-adoption in the test above has no way
            # to arise. Same object, not merely an equal reading.
            assert lib._replace_loop is first
            assert lib._adds_idle is captured
            assert lib._adds_registered_on(first) == 1

            llm.release.set()
            first.run_until_complete(parked)
            # And the add drains normally on the loop that owns its Event,
            # which is what the rebuild would have taken away.
            assert lib._adds_registered_on(first) == 0
            assert "d1" in lib.doc_summaries
        finally:
            dispose(parked, first)
