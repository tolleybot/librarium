"""A protection that is switched off has to say so.

``embed_fingerprint`` returns ``None`` on four conditions and three of
them used to say nothing at all, which left the whole stale-vector check
inert for that librarian with no signal anywhere that it was. The one
signal that did exist — ``load()``'s "records no embedding fingerprint …
re-index to record one" — was worse than nothing here, because
re-indexing does not record one when the model cannot produce one.

Part A below pins the report: level, count, which reason fired, and the
two load-time cases being told apart. Part C pins ``_load_reranker``'s
two failure branches, which no test had ever run.

Offline and model-free. The doubles are content-derived:
:class:`MarkerCountEmbedder` is degenerate on the probe *because of what
it is* — it scores marker words, and ``EMBED_PROBE`` contains none — not
because it returns a constant, and the same is true of the non-finite and
unpooled variants below.

**On the warning this change adds to the rest of the suite.** Measured
over a full run at ``cbc0729``: 21 records, one per test in 21 tests of
704, every one of them a stub embedder being probed once. Nothing is
masked — the existing "silent" assertions
(`test_dense_honesty.py::test_it_is_silent_when_the_vectors_load`,
`test_expansion.py`'s two ``caplog.records == []``) scope their capture to
a ``load()`` or to a pure function, and the probe happens during indexing,
outside those windows. The doubles were deliberately left degenerate: a
double that fingerprints is a different double (it activates the drift
comparison in tests that are not about drift), and ``WeightedEmbedder`` in
``test_stale_vectors.py`` already carries that property for the tests that
need it. What is *not* safe is a future assertion that no warning is
emitted across an ``add_document`` with a stub embedder, and
:class:`TestItIsSaidOncePerModel` is what keeps the count at one.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import logging
import sys
import types

import numpy as np
import pytest

from librarium.source import (
    EMBED_PROBE,
    RERANK_MODEL,
    SourceLibrarian,
    embed_fingerprint,
)
from tests.conftest import MockLLM

# ── The doubles ──────────────────────────────────────────────────

#: None of these appears in ``EMBED_PROBE``, which is what makes the
#: marker-counting doubles below degenerate on the probe by construction
#: rather than by a special case.
MARKERS = ("motor", "vehicle", "deductible", "collision", "claim",
           "dentist", "cleaning", "hygienist", "molar", "bakery")


class MarkerCountEmbedder:
    """Scores marker words — the shape of every older double in the suite.

    Content-derived: a chunk's vector is a function of the words in it,
    so a vector filed against the wrong chunk ranks differently. It
    encodes ``EMBED_PROBE`` to all zeros because the probe contains no
    marker, and a zero vector has no direction, so cosine cannot tell it
    from any other model's zeros.
    """

    DIMS = len(MARKERS)

    def __init__(self):
        #: Every batch handed to ``encode``, so a test can say that a
        #: code path took no probe rather than infer it from a log.
        self.batches: list[list[str]] = []

    def encode(self, texts, show_progress_bar: bool = False):
        self.batches.append(list(texts))
        return np.array(
            [[float(text.lower().count(m)) for m in MARKERS] for text in texts],
            dtype=np.float32,
        )


class NormalisingMarkerEmbedder(MarkerCountEmbedder):
    """Divides the marker counts by their total, so a text with no
    markers divides by zero and comes back as NaN.

    The non-finite case as it would actually arise: a normalising step
    with nothing to normalise, not a hard-coded ``nan``.
    """

    def encode(self, texts, show_progress_bar: bool = False):
        rows = super().encode(texts, show_progress_bar=show_progress_bar)
        with np.errstate(invalid="ignore", divide="ignore"):
            return rows / rows.sum(axis=1, keepdims=True)


class UnpooledEmbedder(MarkerCountEmbedder):
    """Returns one row per word instead of one vector per text.

    The wrong-shape case as it would actually arise: a model wired up
    without its pooling layer still encodes, and what comes back is a
    token matrix rather than a sentence vector.
    """

    def encode(self, texts, show_progress_bar: bool = False):
        self.batches.append(list(texts))
        return [
            np.array(
                [[float(word.lower().count(m)) for m in MARKERS]
                 for word in text.split()],
                dtype=np.float32,
            )
            for text in texts
        ]


class WordAxisEmbedder:
    """A healthy encoder: every word lands on an axis of its own.

    Content-derived like the others, and non-zero for *any* non-empty
    text — including ``EMBED_PROBE`` — which is the one property the
    marker counters lack and the whole reason a fingerprint can be taken
    at all.
    """

    DIMS = 12

    def __init__(self, weights: str = "healthy-2026"):
        self.weights = weights
        self.batches: list[list[str]] = []

    def encode(self, texts, show_progress_bar: bool = False):
        self.batches.append(list(texts))
        rows = []
        for text in texts:
            row = np.zeros(self.DIMS)
            for word in text.lower().split():
                seed = hashlib.sha256(f"{self.weights}|{word}".encode()).digest()
                row[seed[0] % self.DIMS] += 1.0 + seed[1] / 256.0
            rows.append(row)
        return np.asarray(rows, dtype="float32")


class ExplodingEmbedder:
    """Cannot encode anything — the one ``None`` that stays silent."""

    def encode(self, texts, show_progress_bar: bool = False):
        raise RuntimeError("no backend here")


# ── The corpus ───────────────────────────────────────────────────

DOCS = {
    "policy.md": "Northwind Mutual auto cover 41-B. The motor vehicle "
                 "deductible is 500 per collision claim.",
    "dental.md": "Brightsmile dentist appointments. A cleaning is booked "
                 "every six months and the hygienist checks each molar.",
}


async def build(embedder=None, **kwargs):
    """A chunked, two-document index, hybrid or BM25-only."""
    lib = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                          use_reranker=False, **kwargs)
    if embedder is None:
        lib._dense_ok = False
    else:
        lib.embed_model = embedder
    for doc_id, text in DOCS.items():
        await lib.add_document(text, doc_id=doc_id)
    return lib


def source_records(caplog, level=logging.DEBUG):
    return [r for r in caplog.records
            if r.name == "librarium.source" and r.levelno >= level]


UNFINGERPRINTABLE = "cannot fingerprint the embedding weights"
NO_FINGERPRINT_TEXT = "records no embedding fingerprint"


def probe_capturing(model, caplog):
    """Fingerprint *model*, returning (result, librarium.source records).

    Captures from DEBUG so that a report demoted below WARNING is still
    seen — a level assertion that only captured WARNING would read a
    demotion as "removed" and pass a test written to catch removal.
    """
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="librarium.source"):
        result = embed_fingerprint(model)
    return result, source_records(caplog)


# ── Part A: the report at the probe ──────────────────────────────


class TestADegenerateProbeIsReported:
    """The three ``None``s where the encode *worked* and nothing else
    will ever complain."""

    def test_a_zero_vector_is_reported_exactly_once_at_warning(self, caplog):
        # The whole of Part A in one assertion set, and every clause of it
        # is load-bearing. Catches the report being deleted (no records),
        # catches it being logged at INFO or DEBUG — where, with no
        # handler installed, Python's last-resort handler drops it and the
        # user sees exactly the silence this exists to end — and catches it
        # firing twice for one probe.
        model = MarkerCountEmbedder()
        result, found = probe_capturing(model, caplog)

        assert result is None
        assert len(found) == 1, [r.getMessage() for r in found]
        assert found[0].levelno == logging.WARNING
        assert UNFINGERPRINTABLE in found[0].getMessage()
        # Names the model, so a process holding two of them says which.
        assert "MarkerCountEmbedder" in found[0].getMessage()

    @pytest.mark.parametrize("model,expect", [
        (NormalisingMarkerEmbedder(), "non-finite"),
        (UnpooledEmbedder(), "shape"),
        (MarkerCountEmbedder(), "zeros"),
    ])
    def test_every_degenerate_reason_is_reported(self, model, expect, caplog):
        # Each of the three branches, one record each at WARNING. Catches
        # any one of them losing its call — before this change all three
        # were silent, and the non-finite and wrong-shape pair shared a
        # single `if`, so a merge back into one condition costs the reason
        # even while the count stays right.
        result, found = probe_capturing(model, caplog)

        assert result is None
        assert len(found) == 1, [r.getMessage() for r in found]
        assert found[0].levelno == logging.WARNING
        assert expect in found[0].getMessage(), found[0].getMessage()

    def test_the_three_reasons_are_told_apart(self, caplog):
        # The reasons have different remedies — an unpooled model is
        # wired up wrong, an all-zero probe is a scorer rather than an
        # encoder — so one shared message would leave the reader to guess.
        # Catches the three calls being collapsed onto one reason string.
        messages = {
            probe_capturing(model, caplog)[1][0].getMessage()
            for model in (MarkerCountEmbedder(), NormalisingMarkerEmbedder(),
                          UnpooledEmbedder())
        }
        assert len(messages) == 3, messages

    def test_it_says_the_check_is_off_and_not_that_retrieval_broke(
        self, caplog
    ):
        # The line fires on a path a user may have chosen deliberately, so
        # it has to be readable as "a check is unavailable" and not as
        # "retrieval is broken" — the second reading sends someone
        # debugging a working system. Catches the message being rewritten
        # into a retrieval failure, and catches the remedy being stated as
        # "re-index", which is the misdirection this cycle exists to end.
        _, found = probe_capturing(MarkerCountEmbedder(), caplog)
        message = found[0].getMessage()

        assert "Retrieval is unaffected" in message
        assert "re-indexing will not turn it on" in message

    def test_a_model_that_fingerprints_says_nothing(self, caplog):
        # The control, and the assertion that keeps every test above from
        # passing on a report that fires unconditionally. Captured from
        # DEBUG, so an INFO "fingerprinted OK" added here would fail too:
        # a healthy load is not an event.
        result, found = probe_capturing(WordAxisEmbedder(), caplog)

        assert result is not None and result["probe"] == EMBED_PROBE
        assert [r.getMessage() for r in found] == []

    def test_a_model_that_cannot_encode_stays_silent(self, caplog):
        # Deliberate, and the one `None` that keeps its silence: the
        # object is about to fail at the real work, loudly, and a
        # complaint from here arrives first and buries it. Catches the
        # report being wired into the `except` branch along with the rest.
        result, found = probe_capturing(ExplodingEmbedder(), caplog)

        assert result is None
        assert [r.getMessage() for r in found] == []


class TestItIsSaidOncePerModel:
    """Once at the probe — not once per document, not once per query."""

    async def test_indexing_two_documents_and_querying_says_it_once(
        self, caplog
    ):
        # "Once" is the acceptance criterion, and it is the caller's doing
        # rather than the log site's: `_current_fingerprint` memoises on
        # the model object. Catches that memo being dropped or keyed on
        # something that changes — a per-document or per-query repeat
        # turns a real signal into a wall of noise on a large corpus,
        # which is how the warning it sits beside gets filtered out.
        model = MarkerCountEmbedder()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="librarium.source"):
            lib = await build(model)
            lib.retrieve("motor vehicle deductible", top_k=2)
            lib.retrieve("hygienist", top_k=2)
            await lib.add_document(DOCS["policy.md"], doc_id="third.md")

        said = [r for r in source_records(caplog)
                if UNFINGERPRINTABLE in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in said]

    async def test_re_indexing_the_same_model_does_not_say_it_again(
        self, caplog
    ):
        # `index()` resets and re-embeds from scratch, which is the path
        # that re-enters the probe site with `have == 0`. Catches `_reset`
        # clearing `_model_fingerprint` or `_fingerprint_of` — which would
        # also break the load-time disambiguation below, silently, because
        # `_model_declined_probe` would start answering "no".
        model = MarkerCountEmbedder()
        lib = await build(model)
        # Cleared, not merely re-levelled: `caplog.at_level` scopes the
        # level and not the capture, so the probe from `build` above
        # would otherwise still be in `records` and this would pass while
        # measuring nothing.
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="librarium.source"):
            await lib.index(DOCS["dental.md"])

        said = [r for r in source_records(caplog)
                if UNFINGERPRINTABLE in r.getMessage()]
        assert said == [], [r.getMessage() for r in said]


# ── Part A: the two load-time cases ──────────────────────────────


def load_capturing(lib, path, caplog):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="librarium.source"):
        lib.load(path)
    return [r for r in source_records(caplog, logging.INFO)
            if NO_FINGERPRINT_TEXT in r.getMessage()]


class TestTheTwoUnrecordedCasesAreDistinguishable:
    """"This index predates fingerprints" and "this model cannot produce
    one" have opposite remedies, so they must not read alike."""

    async def declining(self, tmp_path):
        """An index with vectors, saved by a librarian whose model was
        probed and declined. Returned with that same librarian, because
        having probed is what lets it tell which case it is in."""
        lib = await build(MarkerCountEmbedder())
        assert lib.chunk_embeddings is not None
        lib.save(tmp_path / "declined")
        assert json.loads(
            (tmp_path / "declined" / "index.json").read_text()
        )["embed_fingerprint"] is None
        return lib, tmp_path / "declined"

    async def pre_existing(self, tmp_path):
        """An index written before fingerprints existed: the key absent,
        not null, because that is literally what is on disk."""
        lib = await build(WordAxisEmbedder())
        lib.save(tmp_path / "old")
        meta = json.loads((tmp_path / "old" / "index.json").read_text())
        del meta["embed_fingerprint"]
        (tmp_path / "old" / "index.json").write_text(json.dumps(meta),
                                                     encoding="utf-8")
        return tmp_path / "old"

    async def test_they_do_not_emit_the_same_message(self, tmp_path, caplog):
        # The acceptance criterion, stated as the comparison it is.
        # Catches the disambiguation being removed — one shared message
        # tells the declining user to re-index forever, which changes
        # nothing and never reveals that the check is off.
        lib, declined_path = await self.declining(tmp_path)
        declined = load_capturing(lib, declined_path, caplog)

        old_path = await self.pre_existing(tmp_path)
        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WordAxisEmbedder()
        predates = load_capturing(victim, old_path, caplog)

        assert len(declined) == 1, [r.getMessage() for r in declined]
        assert len(predates) == 1, [r.getMessage() for r in predates]
        # The directory is masked out of both before comparing. Both
        # messages name the index they are about, so two loads of two
        # different directories differ on that alone — compared raw, this
        # passes with the disambiguation deleted, which is the one thing
        # it exists to catch.
        assert (declined[0].getMessage().replace(str(declined_path), "<index>")
                != predates[0].getMessage().replace(str(old_path), "<index>"))

    async def test_only_the_upgrade_case_is_told_to_re_index(
        self, tmp_path, caplog
    ):
        # The remedies, not just the wording: re-indexing records a
        # fingerprint for the old index and cannot for the declining
        # model. Catches the two branches being swapped, which the
        # message-inequality test above would happily pass.
        lib, declined_path = await self.declining(tmp_path)
        declined = load_capturing(lib, declined_path, caplog)[0].getMessage()

        old_path = await self.pre_existing(tmp_path)
        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WordAxisEmbedder()
        predates = load_capturing(victim, old_path, caplog)[0].getMessage()

        assert "re-index to record one" in predates
        assert "re-indexing will not record one" in declined
        assert str(declined_path) in declined  # names which index

    async def test_the_declining_case_stays_at_info(self, tmp_path, caplog):
        # Not a second WARNING: the probe's own warning has already fired
        # for this model — being probed is the precondition for reaching
        # this branch — and repeating it on every load of every index is
        # how the one that matters gets filtered out. Catches the level
        # being raised here.
        lib, path = await self.declining(tmp_path)
        found = load_capturing(lib, path, caplog)

        assert found[0].levelno == logging.INFO

    async def test_load_takes_no_probe_of_its_own(self, tmp_path, caplog):
        # `load()` must stay free of encodes — an offline process stays
        # offline, and a load stays as cheap as it was. Catches
        # `_model_declined_probe` being implemented by probing, which is
        # the obvious way to write it and puts a model encode on every
        # load.
        lib, path = await self.declining(tmp_path)
        before = len(lib.embed_model.batches)
        load_capturing(lib, path, caplog)

        assert len(lib.embed_model.batches) == before

    async def test_the_probe_load_does_take_is_reported_from_there(
        self, tmp_path, caplog
    ):
        # `load()` is free of encodes only while nothing is comparable:
        # an index that *does* carry a fingerprint sends
        # `_check_embed_fingerprint` through `_current_fingerprint`, and
        # a degenerate model probed there reports from inside `load()`.
        # Pinned because it is the one place the new WARNING lands in a
        # window where existing tests assert on load-time records, and
        # because a report made conditional on "we are indexing" would
        # leave this path silent again.
        lib = await build(WordAxisEmbedder())
        lib.save(tmp_path / "recorded")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = MarkerCountEmbedder()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="librarium.source"):
            victim.load(tmp_path / "recorded")

        said = [r for r in source_records(caplog)
                if UNFINGERPRINTABLE in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in said]
        assert said[0].levelno == logging.WARNING
        # And the vectors are kept: an unfingerprintable model cannot
        # convict them of anything.
        assert victim.chunk_embeddings is not None

    async def test_a_librarian_that_never_loaded_a_model_gets_the_upgrade_note(
        self, tmp_path, caplog
    ):
        # The guard on `model is not None`. Without it, a librarian with
        # no model at all matches "probed, and it declined" — `None is
        # None` — and every core-install user loading a pre-fingerprint
        # index would be told their model cannot produce one, which is
        # both false and unactionable. Catches that guard being dropped.
        path = await self.pre_existing(tmp_path)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        assert victim.embed_model is None
        found = load_capturing(victim, path, caplog)

        assert len(found) == 1, [r.getMessage() for r in found]
        assert "re-index to record one" in found[0].getMessage()

    async def test_an_unprobed_model_gets_the_upgrade_note_too(
        self, tmp_path, caplog
    ):
        # A model assigned but never used: the librarian cannot yet know
        # which case it is in, and guessing "it declined" would be a
        # claim about a model nothing has run. Catches the identity test
        # (`_fingerprint_of is model`) being dropped in favour of "is the
        # fingerprint None", which is None for both.
        path = await self.pre_existing(tmp_path)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = MarkerCountEmbedder()  # would decline, unprobed
        found = load_capturing(victim, path, caplog)

        assert len(found) == 1, [r.getMessage() for r in found]
        assert "re-index to record one" in found[0].getMessage()
        assert victim.embed_model.batches == []


# ── Part C: the reranker's two failure branches ──────────────────


def without_the_extra(monkeypatch) -> None:
    """A machine where ``import sentence_transformers`` fails.

    ``None`` in ``sys.modules`` is how the import system spells "this
    import is halted" — it raises ImportError from the import statement
    itself, which is what a core install does, rather than from an
    attribute lookup somewhere later.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


def with_a_broken_cross_encoder(monkeypatch, error: Exception) -> list[str]:
    """The extra installed, and its ``CrossEncoder`` failing to load —
    a half-downloaded model, an incompatible torch, no disk.

    *error* is re-raised as-is on every attempt: **one object**, so its
    text is identical across retries by construction. That is what most
    tests here want, but it is also the single shape that cannot tell a
    deduplication key over the error's *text* apart from one over
    anything richer — under this helper both look equally stable. A test
    drawing that distinction wants the sibling
    `with_a_cross_encoder_failing_afresh`, which takes a factory and
    builds a fresh error carrying the attempt number.

    Returns the list of model names it was asked for, so a test can say
    that a *second* load was attempted rather than infer it from a
    second log line.
    """
    module = types.ModuleType("sentence_transformers")
    attempts: list[str] = []

    def explode(name, **kwargs):
        attempts.append(name)
        raise error

    module.SentenceTransformer = lambda name, **kwargs: object()
    module.CrossEncoder = explode
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return attempts


class MarkerCountReranker:
    """A cross-encoder stand-in that scores a chunk by its markers.

    Content-derived and spread, on the same argument as every other
    double in this file: a constant score would leave BM25's order
    untouched, so a test could not tell a loaded reranker from no
    reranker at all — which is precisely the distinction the recovery
    cases below are drawing. The base of 1.0 keeps every score clear of
    the default ``rerank_threshold``.
    """

    def __init__(self):
        #: Every pair it was asked to score, so a test can say the
        #: reranker ran rather than infer it from an ordering.
        self.pairs: list[list[str]] = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return np.array(
            [1.0 + sum(text.lower().count(m) for m in MARKERS)
             for _, text in pairs],
            dtype=np.float32,
        )


def with_a_working_cross_encoder(monkeypatch, reranker) -> None:
    """The extra present and its ``CrossEncoder`` loading.

    The recovery the retry exists for: an extra installed, or a hub that
    came back, part-way through a librarian's life.
    """
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, **kwargs: object()
    module.CrossEncoder = lambda name, **kwargs: reranker
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


#: A document for the add that happens *after* the extra appears, so the
#: recovery is a fresh add rather than a replace of one of ``DOCS``.
LATE_DOC = "The bakery sells 120 loaves before the cleaning at noon."


async def reranking_librarian(**kwargs):
    """A librarian that will try to load a reranker, and no embedder.

    ``_dense_ok = False`` before anything indexes, so the embedding
    loader never reaches the same import and the only record in the
    capture is the reranker's. Without that, both halves of the extra
    report at once and "exactly one warning" stops meaning anything.
    ``top_k`` is under ``rerank_top`` so construction has nothing of its
    own to warn about either.
    """
    lib = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                          use_reranker=True, top_k=3, **kwargs)
    lib._dense_ok = False
    return lib


async def add_capturing(lib, caplog, doc_id="policy.md"):
    """Index one document, returning librarium.source's WARNING+ records.

    One document per call because the reranker is loaded from
    ``_add_document``, so the report below is per-add rather than per
    librarian; two documents in one capture would make every count
    assertion here read "two" for no reason the test is about.
    """
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="librarium.source"):
        await lib.add_document(DOCS[doc_id], doc_id=doc_id)
    return source_records(caplog, logging.WARNING)


class TestTheRerankerExtraIsMissing:
    """``ImportError`` — the documented core-install path, which no test
    had ever run."""

    async def test_it_warns_and_leaves_reranking_off(self, caplog, monkeypatch):
        # Catches the `except ImportError` branch being narrowed or
        # removed, which turns a core install's first `add_document` into
        # an ImportError out of the public API; and catches the report
        # being logged below WARNING, where a user with no logging
        # configured never sees that the reranker they asked for is not
        # running.
        without_the_extra(monkeypatch)
        lib = await reranking_librarian()

        warned = await add_capturing(lib, caplog)

        assert len(warned) == 1, [r.getMessage() for r in warned]
        assert warned[0].levelno == logging.WARNING
        assert "reranking disabled" in warned[0].getMessage()
        assert lib.reranker is None

    async def test_retrieval_still_works_without_it(self, caplog, monkeypatch):
        # "Disabled", not "broken": `use_reranker` stays True and
        # `_rerank` has to fall through on a falsy reranker. Catches a
        # future `self.reranker.predict` losing that guard, which would
        # make every query on a core install raise AttributeError.
        without_the_extra(monkeypatch)
        lib = await reranking_librarian()
        await add_capturing(lib, caplog)

        assert lib.use_reranker is True
        assert [c.doc_id for c in lib.retrieve("motor vehicle deductible")]

    async def test_it_does_not_raise_even_under_strict(self, caplog,
                                                       monkeypatch):
        # The asymmetry with the branch below, and it matches
        # `_ensure_embed_model`: a missing extra is a deployment fact,
        # not a failure, and strict has never turned one into a stop.
        # Catches `if self.strict: raise` being copied up into this
        # branch, which would make every core install that asked for
        # reranking fail to index at all.
        without_the_extra(monkeypatch)
        lib = await reranking_librarian(strict=True)

        warned = await add_capturing(lib, caplog)

        assert len(warned) == 1, [r.getMessage() for r in warned]
        assert list(lib.documents) == ["policy.md"]

    async def test_it_is_said_once_for_the_librarian_not_once_per_document(
        self, caplog, monkeypatch
    ):
        # This test used to pin the opposite — one warning per
        # `add_document`, so a 500-document corpus on a core install got
        # 500 identical records — and named itself as the test to change
        # if that were ever made once-per-librarian. It was, so it is.
        #
        # The number on its own would not be worth much: a loader that
        # simply gave up after the first failure would also produce one
        # warning, and would be a far worse librarian — an extra
        # installed mid-life could never be picked up. So the assertion
        # is on the *reason* the second add is silent. The report is
        # deduplicated; the load is still attempted, and the proof is
        # that the third add, with the extra now present, loads through
        # the very same call and comes back with a reranker.
        without_the_extra(monkeypatch)
        lib = await reranking_librarian()

        first = await add_capturing(lib, caplog, "policy.md")
        second = await add_capturing(lib, caplog, "dental.md")

        assert len(first) == 1, [r.getMessage() for r in first]
        assert "reranking disabled" in first[0].getMessage()
        assert second == [], [r.getMessage() for r in second]

        reranker = MarkerCountReranker()
        with_a_working_cross_encoder(monkeypatch, reranker)
        await lib.add_document(LATE_DOC, doc_id="bakery.md")

        assert lib.reranker is reranker
        assert lib.retrieve("motor vehicle deductible")
        assert reranker.pairs, "the recovered reranker was never consulted"


class TestTheRerankerIsInstalledButWillNotLoad:
    """``Exception`` — installed, and failing for some other reason."""

    async def test_it_is_an_error_naming_the_model_and_the_cause(
        self, caplog, monkeypatch
    ):
        # ERROR, not WARNING: the extra is installed, so this is a real
        # failure rather than a documented degradation, and it is the
        # only signal that the precision funnel the user paid for is not
        # running. Catches the level being lowered to match the branch
        # above, and catches the cause being swallowed — without the
        # exception text there is nothing to act on.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian()

        found = await add_capturing(lib, caplog)

        assert len(found) == 1, [r.getMessage() for r in found]
        assert found[0].levelno == logging.ERROR
        message = found[0].getMessage()
        assert lib.rerank_model in message
        assert "RuntimeError" in message and "no weights" in message
        assert lib.reranker is None

    async def test_retrieval_still_works_after_it(self, caplog, monkeypatch):
        # `reranker = None` is what makes the failure survivable: BM25
        # and dense still rank, only the funnel is gone. Catches the
        # assignment being dropped, which would leave a half-constructed
        # object behind on some future CrossEncoder that fails late.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian()
        await add_capturing(lib, caplog)

        assert lib.reranker is None
        assert [c.doc_id for c in lib.retrieve("motor vehicle deductible")]

    async def test_it_is_retried_on_the_next_add(self, caplog, monkeypatch):
        # `reranker = None` also means "still not loaded", so the next
        # add tries the loader again — a transient failure (a disk that
        # filled, a hub that was down) recovers without rebuilding the
        # librarian. Asserted on the loader being *called* twice rather
        # than on two log lines, so it still holds if the reporting is
        # ever made once-per-librarian. Catches the branch setting a
        # "gave up" flag instead of None.
        attempts = with_a_broken_cross_encoder(monkeypatch,
                                               RuntimeError("no weights"))
        lib = await reranking_librarian()
        await add_capturing(lib, caplog, "policy.md")
        await add_capturing(lib, caplog, "dental.md")

        assert attempts == [lib.rerank_model, lib.rerank_model]

    async def test_strict_raises_the_original_error(self, caplog, monkeypatch):
        # The branch that makes `strict` mean anything for the reranker,
        # and it had never run. Catches `if self.strict: raise` being
        # dropped — under which a deployment that asked to be stopped on
        # a broken component would index and serve without one, which is
        # the exact silence `strict` exists to prevent. Re-raised bare,
        # so the caller sees the real cause rather than a wrapper.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian(strict=True)

        with pytest.raises(RuntimeError, match="no weights"):
            await add_capturing(lib, caplog)

    async def test_the_strict_raise_rolls_the_document_back(
        self, caplog, monkeypatch
    ):
        # Documented consequence rather than a preference: the reranker
        # loads inside `add_document`'s try, so a strict raise unwinds
        # the document that had just been committed. Catches that
        # rollback being lost — a librarian left holding a document with
        # no chunks after a call that raised is the inconsistent state
        # `_roll_back` exists to prevent.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian(strict=True)

        with pytest.raises(RuntimeError):
            await lib.add_document(DOCS["policy.md"], doc_id="policy.md")

        assert lib.documents == {}
        assert lib.chunks == []


# ── Part D: the report is made once, the load is not ─────────────


#: Part D needs more documents than ``DOCS`` holds: every one of its
#: cases turns on what the *third* and fourth add say, and re-adding a
#: doc_id would be a replace rather than an add.
MORE_DOCS = {
    "bakery.md": LATE_DOC,
    "ledger.md": "The ledger records one claim and one cleaning per month.",
    "roster.md": "The roster lists the hygienist on molar duty this week.",
}


async def add_more_capturing(lib, caplog, doc_id):
    """`add_capturing` over :data:`MORE_DOCS`."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="librarium.source"):
        await lib.add_document(MORE_DOCS[doc_id], doc_id=doc_id)
    return source_records(caplog, logging.WARNING)


class TestTheFailureIsSaidOncePerLibrarian:
    """`_load_reranker` retries on every add and used to re-report on
    every add with it. Deduplicating the *report* without touching the
    retry is a narrow line to walk, and these are the four ways to fall
    off it: silencing the retry, silencing a genuinely different
    failure, silencing `strict`, and silencing a failure that follows a
    recovery.

    "Once" is keyed on the record that would be written — level, message
    template and stringified args — held per librarian. Every test here
    is really about that key being the record rather than a "have I
    already said something" flag.
    """

    async def test_the_unloadable_model_is_reported_once_and_still_retried(
        self, caplog, monkeypatch
    ):
        # The other failure branch, on the same argument as the
        # missing-extra one above, and here the retry is directly
        # visible: the loader is entered twice and writes one record.
        # Catches a fix that deduplicated by skipping the load — under
        # which `attempts` would hold one name and a transient failure
        # would be permanent.
        attempts = with_a_broken_cross_encoder(monkeypatch,
                                               RuntimeError("no weights"))
        lib = await reranking_librarian()

        first = await add_capturing(lib, caplog, "policy.md")
        second = await add_capturing(lib, caplog, "dental.md")

        assert len(first) == 1, [r.getMessage() for r in first]
        assert first[0].levelno == logging.ERROR
        assert second == [], [r.getMessage() for r in second]
        assert attempts == [lib.rerank_model, lib.rerank_model]

    async def test_a_different_cause_is_said_again(self, caplog, monkeypatch):
        # The whole reason the key is the record and not a flag. A disk
        # that fills after a version mismatch is a different fact about
        # the deployment, and the second one is the actionable one.
        # Catches the deduplication being reduced to "have I reported
        # anything for this librarian", which would swallow every
        # failure after the first for the life of the process.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian()
        first = await add_capturing(lib, caplog, "policy.md")

        with_a_broken_cross_encoder(monkeypatch, OSError("no space left"))
        second = await add_capturing(lib, caplog, "dental.md")
        third = await add_more_capturing(lib, caplog, "ledger.md")

        assert len(first) == 1 and "RuntimeError" in first[0].getMessage()
        assert len(second) == 1, [r.getMessage() for r in second]
        assert "OSError" in second[0].getMessage()
        assert "no space left" in second[0].getMessage()
        # And the new cause takes the old one's place rather than being
        # said forever: the key is the last record, not a set of every
        # record ever written, so this stays one line per cause however
        # long the corpus is.
        assert third == [], [r.getMessage() for r in third]

    async def test_the_extra_appearing_and_then_failing_is_said_again(
        self, caplog, monkeypatch
    ):
        # A change of circumstance across the two branches rather than
        # within one: the core install gets the extra, and the model then
        # fails for its own reason. Two different sentences at two
        # different levels, and a user who only ever heard the first
        # would go on believing the extra was the problem.
        without_the_extra(monkeypatch)
        lib = await reranking_librarian()
        first = await add_capturing(lib, caplog, "policy.md")

        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        second = await add_capturing(lib, caplog, "dental.md")
        third = await add_more_capturing(lib, caplog, "ledger.md")

        assert len(first) == 1 and first[0].levelno == logging.WARNING
        assert len(second) == 1, [r.getMessage() for r in second]
        assert second[0].levelno == logging.ERROR
        assert lib.rerank_model in second[0].getMessage()
        # The new branch is then itself said once, not once per add.
        assert third == [], [r.getMessage() for r in third]

    async def test_strict_raises_on_the_add_that_is_no_longer_logged(
        self, caplog, monkeypatch
    ):
        # The one that matters most. `strict` means "stop me rather than
        # serve without the component I asked for", so it has to fire on
        # every failed add, not only on the one that happened to be
        # worth a log line. Catches the raise being moved inside the
        # deduplication — under which the second document and every
        # document after it would index silently with no reranker, in a
        # deployment that explicitly asked not to be left in that state.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian(strict=True)

        with pytest.raises(RuntimeError, match="no weights"):
            await lib.add_document(DOCS["policy.md"], doc_id="policy.md")

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="librarium.source"):
            with pytest.raises(RuntimeError, match="no weights"):
                await lib.add_document(DOCS["dental.md"], doc_id="dental.md")

        # Raised, and this time without a word about it: the two are
        # independent, which is the property being pinned.
        assert source_records(caplog, logging.WARNING) == []

    async def test_a_failure_after_a_recovery_is_news_again(self, caplog,
                                                            monkeypatch):
        # A load that works ends the previous report's life, so the same
        # sentence is said again if the same failure comes back after the
        # reranker has once been running. Catches the successful branch
        # forgetting to clear the record — under which the failure that
        # ended a working reranker would be compared against a record
        # from before it ever worked, and silently dropped.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian()
        first = await add_capturing(lib, caplog, "policy.md")
        assert len(first) == 1

        with_a_working_cross_encoder(monkeypatch, MarkerCountReranker())
        await lib.add_document(MORE_DOCS["bakery.md"], doc_id="bakery.md")
        assert lib.reranker is not None

        # The reranker is gone again — the state `_load_reranker` reads
        # to decide whether to load at all — and the same failure
        # returns.
        lib.reranker = None
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        after = await add_capturing(lib, caplog, "dental.md")
        again = await add_more_capturing(lib, caplog, "ledger.md")

        assert len(after) == 1, [r.getMessage() for r in after]
        assert after[0].getMessage() == first[0].getMessage()
        # Said again once, not returned to saying it per document.
        assert again == [], [r.getMessage() for r in again]

    async def test_the_report_survives_a_load(self, caplog, monkeypatch,
                                              tmp_path):
        # `load()` resets the index and deliberately keeps the loaded
        # models; the record of a *model* failure belongs with them, the
        # same way `_model_fingerprint` does. Catches `_reset` being
        # given a `self._reranker_report = None` — under which every
        # `load()` in a long-lived process would re-open the repetition
        # this change closed.
        #
        # The capture is around `load()` itself and not only around the
        # add after it, because `load()` calls the loader on its own way
        # through. A test that watched only the next `add_document`
        # would see the record re-armed *and* consumed inside `load()`
        # and report silence either way — it passed against a `_reset`
        # that cleared the field, which is the one mutation it exists
        # for.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("no weights"))
        lib = await reranking_librarian()
        first = await add_capturing(lib, caplog, "policy.md")
        assert len(first) == 1
        lib.save(tmp_path)

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="librarium.source"):
            lib.load(tmp_path)
        during_load = source_records(caplog, logging.WARNING)

        after = await add_capturing(lib, caplog, "dental.md")

        assert during_load == [], [r.getMessage() for r in during_load]
        assert after == [], [r.getMessage() for r in after]


# ── Part E: the key holds an exception by type, not by its text ───


#: Five documents, because what this part measures is a count across a
#: *corpus*: the defect it pins is one record per document, and two
#: documents cannot tell "one per document" from "one, and then one more
#: cause". Reuses the documents already defined above rather than
#: inventing more, and every one still names its own subject — the
#: content-derived rule, so a chunk filed under the wrong doc_id shows.
CORPUS = {
    "policy.md": DOCS["policy.md"],
    "dental.md": DOCS["dental.md"],
    "bakery.md": LATE_DOC,
    "ledger.md": MORE_DOCS["ledger.md"],
    "roster.md": MORE_DOCS["roster.md"],
}


async def add_corpus_capturing(lib, caplog):
    """Index all of :data:`CORPUS` inside **one** capture window.

    Deliberately unlike `add_capturing`, which insists on one document
    per call: there the question is what a single add says, and a window
    spanning two adds would make every count read "two". Here the
    question is what a corpus costs in total, which a per-add window
    cannot answer at all — it can only ever report one or zero, per add,
    with nothing summing them.
    """
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="librarium.source"):
        for doc_id, text in CORPUS.items():
            await lib.add_document(text, doc_id=doc_id)
    return source_records(caplog, logging.WARNING)


def with_a_cross_encoder_failing_afresh(monkeypatch, make_error):
    """A ``CrossEncoder`` raising a **newly built** error every attempt.

    `with_a_broken_cross_encoder` raises one error object over and over,
    which is the single shape that cannot show what this part is about:
    its text is identical across retries by construction, so a key over
    that text looks stable no matter what it does. *make_error* is
    called with the attempt number instead, so the exception carries the
    attempt it came from — the content-derived rule applied to a
    failure, which is what lets an assertion below say *which* attempt
    wrote the record that survived.
    """
    module = types.ModuleType("sentence_transformers")
    attempts: list[str] = []

    def explode(name, **kwargs):
        attempts.append(name)
        raise make_error(len(attempts))

    module.SentenceTransformer = lambda name, **kwargs: object()
    module.CrossEncoder = explode
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return attempts


def with_a_half_installed_extra(monkeypatch, make_error):
    """``import sentence_transformers`` working, ``CrossEncoder`` not.

    The ImportError branch reached with a *fresh* error each attempt.
    A module-level ``__getattr__`` raising ImportError is how ``from
    sentence_transformers import CrossEncoder`` fails when the package
    imports but the symbol behind it does not — a broken native
    extension underneath, rather than the package being absent, which
    `without_the_extra` already covers with a ``None`` in ``sys.modules``.
    """
    module = types.ModuleType("sentence_transformers")
    attempts: list[str] = []

    def raise_on_attribute(attr):
        # Dunders answer honestly: the import statement asks for
        # `__path__` before the name it wants, and a module-level
        # `__getattr__` that raised ImportError at *that* would fail the
        # import for a reason the test is not about — and would count an
        # attempt that never reached `CrossEncoder`.
        if attr.startswith("__"):
            raise AttributeError(attr)
        attempts.append(attr)
        raise make_error(len(attempts))

    # `find_spec` raises on a module whose `__spec__` is None, and this
    # stub says "the extra is here", so the honest spec is one that
    # resolves. `loader=None`: nothing may ever execute this module.
    module.__spec__ = importlib.machinery.ModuleSpec(
        "sentence_transformers", loader=None,
    )
    module.__getattr__ = raise_on_attribute
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return attempts


def hub_rate_limit_error(attempt: int) -> OSError:
    """What a rate-limited hub hands back through ``CrossEncoder(...)``.

    The shape is huggingface_hub's, not invented here:
    ``utils/_http.py`` appends ``(Request ID: ...)`` to every HTTP error
    it raises, and ``transformers/utils/hub.py`` copies that text
    verbatim into the ``OSError`` it re-raises. The token is fresh per
    *request*, so a retry that fails the same way still stringifies
    differently; the attempt number stands in for it so a record can be
    traced to the attempt that wrote it.
    """
    return OSError(
        "There was a specific connection error when trying to load "
        f"{RERANK_MODEL}: 429 Client Error: Too Many Requests "
        f"(Request ID: Root=1-68a0b1c2-attempt{attempt})"
    )


def broken_native_import_error(attempt: int) -> ImportError:
    """An ImportError whose text carries a path that is not fixed."""
    return ImportError(
        "dlopen(/var/folders/9x/T/pip-target-"
        f"{attempt}f3a/torch/lib/libtorch_cpu.dylib, 0x0006): "
        "symbol not found in flat namespace"
    )


class TestAnExceptionEntersTheKeyByTypeNotByItsText:
    """The dedup key holds ``type(e).__name__``, not ``str(e)``.

    The key used to be ``tuple(str(a) for a in args)`` and one of those
    args is the exception itself. That is stable only if the exception's
    text is, and the failures that actually reach here are the hub's:
    huggingface_hub stamps a fresh ``(Request ID: ...)`` into every HTTP
    error and transformers re-raises it verbatim. So against a
    rate-limited hub every retry stringified differently, every retry
    got its own key, and the per-document repetition the class above
    removed came straight back — silently, since the deduplication
    looked present the whole time.

    The accepted cost is pinned here too rather than left implied: two
    failures of the **same type** with different text now collapse into
    one record, and the second cause's text is genuinely lost. No test
    stated that in either direction before this one —
    `test_a_different_cause_is_said_again` varies the type as well, and
    the type is the half the key still keeps.
    """

    async def test_a_fresh_request_id_per_attempt_is_still_one_record(
        self, caplog, monkeypatch
    ):
        # The measurement the change exists for, through the public API:
        # five documents behind a hub that fails the same way with a
        # different request ID each time. Five records on the old key,
        # one on this one. Catches the key going back to `str(a)` — under
        # which a rate-limited hub buries the log in a message that is
        # the same sentence five hundred times over, and the
        # deduplication reports nothing about having stopped working.
        attempts = with_a_cross_encoder_failing_afresh(monkeypatch,
                                                       hub_rate_limit_error)
        lib = await reranking_librarian()

        found = await add_corpus_capturing(lib, caplog)

        assert len(found) == 1, [r.getMessage() for r in found]
        assert found[0].levelno == logging.ERROR
        assert "OSError" in found[0].getMessage()
        # The retry is untouched — the load was attempted once per
        # document and only the report was withheld, which is the line
        # this whole mechanism has to walk. Asserted on the loader being
        # entered, not on log lines, so it cannot be satisfied by a
        # loader that gave up.
        assert attempts == [lib.rerank_model] * len(CORPUS)
        assert lib.reranker is None

    async def test_the_cause_that_is_reported_is_the_first_one(
        self, caplog, monkeypatch
    ):
        # The accepted cost, stated rather than implied: the four later
        # failures are not merely un-repeated, their text is *gone*. A
        # reader who needs the request ID of the attempt that finally
        # exhausted a retry budget will not find it here. Catches the
        # collapse being widened into "report the latest instead", which
        # would rewrite one record per document under a different name
        # and reopen the flood in a form the count assertion above would
        # not see.
        with_a_cross_encoder_failing_afresh(monkeypatch,
                                            hub_rate_limit_error)
        lib = await reranking_librarian()

        found = await add_corpus_capturing(lib, caplog)

        # Over every record the corpus produced, not only the first:
        # "the later causes are absent" is the claim, and reading one
        # record can only say that record does not carry them.
        written = " ".join(r.getMessage() for r in found)
        assert "attempt1)" in written
        for later in range(2, len(CORPUS) + 1):
            assert f"attempt{later})" not in written, written

    async def test_a_half_installed_extra_is_also_one_record(
        self, caplog, monkeypatch
    ):
        # The other branch, and the one with the least holding it steady:
        # the ImportError report passes the exception as its *only* arg,
        # so before the change the key was that text and nothing else —
        # the ERROR branch at least had `type(e).__name__` and the model
        # name beside it. The claim is not that some real ImportError
        # varies per attempt; it is that this branch no longer depends on
        # none of them doing so. Catches the type substitution being
        # applied to the ERROR branch's args alone, which would leave the
        # documented core-install path keyed on text.
        attempts = with_a_half_installed_extra(monkeypatch,
                                               broken_native_import_error)
        lib = await reranking_librarian()

        found = await add_corpus_capturing(lib, caplog)

        assert len(found) == 1, [r.getMessage() for r in found]
        assert found[0].levelno == logging.WARNING
        assert "reranking disabled" in found[0].getMessage()
        assert attempts == ["CrossEncoder"] * len(CORPUS)

    async def test_the_record_still_carries_the_text_the_key_dropped(
        self, caplog, monkeypatch
    ):
        # What changed is the key, not the record. Catches the
        # stringified key being logged in place of the args —
        # `logger.log(level, message, *report[2])` is a one-character
        # slip from what is written — under which every reranker failure
        # in the library would read "(OSError: OSError)" and the cause a
        # user has to act on would be dropped from the only line that
        # ever mentions it.
        with_a_cross_encoder_failing_afresh(monkeypatch,
                                            hub_rate_limit_error)
        lib = await reranking_librarian()

        found = await add_corpus_capturing(lib, caplog)

        message = found[0].getMessage()
        assert "429 Client Error: Too Many Requests" in message
        assert "Request ID: Root=1-68a0b1c2-attempt1" in message

    async def test_a_different_type_with_the_same_text_is_said_again(
        self, caplog, monkeypatch
    ):
        # The half of the old key that survives, isolated: the text is
        # byte-identical across the two failures and only the type
        # differs, where `test_a_different_cause_is_said_again` varies
        # both and so cannot say which one carried it. Holds on the old
        # key as well — it is the retained behaviour, not the changed one
        # — and what it catches is the accepted cost being over-applied:
        # a key reduced to `(level, message)`, or to a "have I said
        # anything yet" flag, silences the second failure and passes
        # every count assertion above.
        with_a_broken_cross_encoder(monkeypatch, RuntimeError("cannot load"))
        lib = await reranking_librarian()
        first = await add_capturing(lib, caplog, "policy.md")

        with_a_broken_cross_encoder(monkeypatch, OSError("cannot load"))
        second = await add_capturing(lib, caplog, "dental.md")

        assert len(first) == 1 and "RuntimeError" in first[0].getMessage()
        assert len(second) == 1, [r.getMessage() for r in second]
        assert "OSError" in second[0].getMessage()

    async def test_strict_raises_on_every_add_the_key_collapsed(
        self, caplog, monkeypatch
    ):
        # `strict` and the deduplication stay independent, and the new
        # key makes that independence wider: one record now stands for
        # five failures, so five documents raise while four of them go
        # unlogged. Catches the raise being folded into the report — a
        # deployment that asked to be stopped rather than served without
        # its reranker would index four of five documents in silence,
        # which is the exact degradation `strict` exists to prevent.
        attempts = with_a_cross_encoder_failing_afresh(monkeypatch,
                                                       hub_rate_limit_error)
        lib = await reranking_librarian(strict=True)

        raised = 0
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="librarium.source"):
            for doc_id, text in CORPUS.items():
                with pytest.raises(OSError, match="Too Many Requests"):
                    await lib.add_document(text, doc_id=doc_id)
                raised += 1
        found = source_records(caplog, logging.WARNING)

        assert raised == len(CORPUS)
        assert attempts == [lib.rerank_model] * len(CORPUS)
        assert len(found) == 1, [r.getMessage() for r in found]
        # The strict raise rolls each document back, so nothing was
        # served without the component the deployment asked for.
        assert lib.documents == {}
