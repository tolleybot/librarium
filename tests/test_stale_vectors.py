"""Stored vectors must not be searched with weights that did not make them.

A saved index used to load clean however far the embedding weights had
moved underneath it, so every query was embedded under the new weights
and scored, by cosine, against vectors embedded under the old ones — a
meaningless number that still sorts, so retrieval got quietly worse and
nothing said anything.

What catches it is a *behavioural* fingerprint: ``save()`` records the
model's own embedding of :data:`~librarium.source.EMBED_PROBE` beside the
vectors, and the comparison happens wherever the model actually loads.
That placement is forced — ``_load_if_current`` hashes before any model
exists, so ``content_hash()`` structurally cannot carry this.

Everything here is offline and model-free, per CLAUDE.md.
:class:`WeightedEmbedder` is content-derived *and* identity-carrying: the
vector is a function of the words in the text, so a vector filed against
the wrong chunk ranks differently, and the ``weights`` name is mixed into
every component, so "the same text under different weights" — the whole
subject of this file — is expressible without a real encoder.

The one property it has that the suite's older doubles do not is that it
embeds ``EMBED_PROBE`` to something **non-zero**. A marker-word counter
scores that string zero, and a zero vector has no direction, so it
fingerprints as ``None`` — "cannot tell", silent. That is why those
doubles still pass unchanged, and it is why a double that has to reach
the drift branch cannot be one of them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import types

import numpy as np
import pytest

import librarium
from librarium.source import (
    EMBED_FINGERPRINT_TOLERANCE,
    EMBED_PROBE,
    INDEX_FORMAT_VERSION,
    SourceLibrarian,
    StaleEmbeddingsError,
    embed_fingerprint,
    fingerprint_drift,
)
from tests.conftest import MockLLM
#: Imported rather than re-spelled: the one definition of what
#: `_refresh_embeddings` hands the encoder, so a change to that input is a
#: single edit across both files that check a row against its chunk.
from tests.test_indexing import embedded_text

# ── The doubles ──────────────────────────────────────────────────


class WeightedEmbedder:
    """A pretend set of embedding weights, named.

    Two instances with the same ``weights`` are bit-identical; two with
    different ones send the same text to different axes, which is how a
    test states "upstream moved" without moving anything.

    ``jitter`` is the other half: the *same* weights running on a
    different device or dtype. It perturbs every component by a fixed
    relative ramp, so the components genuinely differ — far above the
    six decimals the fingerprint is stored to — while the direction
    barely moves. That asymmetry is the design's whole argument for a
    cosine threshold over a digest, so it is a double a test can point
    at rather than a claim in a comment.
    """

    DIMS = 12

    def __init__(self, weights: str, *, jitter: float = 0.0):
        self.weights = weights
        self.jitter = jitter
        #: Every batch handed to encode, so a test can say *which* strings
        #: were embedded rather than only how many calls happened.
        self.batches: list[list[str]] = []

    def encode(self, texts, show_progress_bar: bool = False):
        self.batches.append(list(texts))
        rows = []
        for text in texts:
            row = np.zeros(self.DIMS)
            for word in text.lower().split():
                seed = hashlib.sha256(f"{self.weights}|{word}".encode()).digest()
                row[seed[0] % self.DIMS] += 1.0 + seed[1] / 256.0
            if self.jitter:
                row = row * (
                    1.0 + self.jitter * np.arange(1, self.DIMS + 1) / self.DIMS
                )
            rows.append(row)
        return np.asarray(rows, dtype="float32")


class ZeroProbeEmbedder:
    """The shape every pre-existing double in this suite has.

    It scores ``EMBED_PROBE`` at zero — no word of it is a marker — so it
    cannot be fingerprinted, which is the property that keeps the rest of
    the suite unchanged. Pinned here so that stays a decision.
    """

    #: Twelve, matching :class:`WeightedEmbedder`, so a test can hand an
    #: index built by one to a librarian holding the other without the
    #: width mismatch getting in the way of what it is measuring. Not one
    #: of them appears in ``EMBED_PROBE``, which is the whole point.
    MARKERS = ("motor", "vehicle", "deductible", "collision", "claim",
               "cover", "dentist", "cleaning", "hygienist", "molar",
               "bakery", "loaves")
    DIMS = len(MARKERS)

    def encode(self, texts, show_progress_bar: bool = False):
        return np.array(
            [[float(text.lower().count(m)) for m in self.MARKERS]
             for text in texts],
            dtype=np.float32,
        )


class ExplodingEmbedder:
    def encode(self, texts, show_progress_bar: bool = False):
        raise RuntimeError("no backend here")


#: The query prefix of the model `test_a_query_failure_leaves_no_identity_behind`
#: pins itself to. nomic marks queries; the default bge-m3 does not.
QUERY_MARK = "search_query: "
MARKING_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class QueryBreakingEmbedder(WeightedEmbedder):
    """Embeds documents and the probe, and fails on a *query*.

    Half a working encoder, because only half a working encoder reaches
    the state this is here for: the probe and the chunks embed, so an
    identity is recorded for real vectors, and then the first query flips
    ``_dense_ok`` off under it. A double that failed on everything never
    gets a fingerprint recorded in the first place, and one that failed
    on nothing never drops the vectors — the counter-state needs exactly
    this split. ``EMBED_PROBE`` is a ``search_document:`` string, so the
    fingerprint still goes in.
    """

    def encode(self, texts, show_progress_bar: bool = False):
        texts = list(texts)
        # Split on the *configured* query prefix, not on nomic's literal.
        # The default embedder is bge-m3, which uses no prefixes at all,
        # so a hardcoded "search_query: " matched nothing and this double
        # silently stopped breaking anything — the premise assertion below
        # is what caught it. Where the prefix is empty every text matches,
        # so fall back to the shape instead: a query encode is one text,
        # a document encode is the pending chunks.
        # The caller pins `embed_model` to a model that marks queries, so
        # this can tell a query encode from a document one. It used to
        # hardcode nomic's literal, which stopped matching anything when
        # the default moved to bge-m3 — a double that silently breaks
        # nothing, caught only by the premise assertion in the test.
        # Shape is not an alternative: a one-chunk index passes a single
        # text on both paths.
        if any(t.startswith(QUERY_MARK) for t in texts):
            raise RuntimeError("query encode is broken here")
        return super().encode(texts, show_progress_bar=show_progress_bar)


class DentalPreferringReranker:
    """A cross-encoder stand-in that disagrees with BM25 on purpose.

    Content-derived and spread, like every other double here: the score
    counts dental marker words, so the two documents in :data:`DOCS` land
    on different scores and the chunk BM25 puts *first* for a motor-
    vehicle question lands *last*. That disagreement is the only thing
    that makes "reranking ran" observable from outside — a constant-score
    double would leave BM25's order in place, and a test built on one
    could not tell a loaded reranker from no reranker at all, which is
    precisely the state this is here to catch. The base of 1.0 keeps
    every score clear of the default ``rerank_threshold``.
    """

    MARKERS = ("dentist", "cleaning", "hygienist", "molar")

    def __init__(self):
        #: Every pair it was asked to score, so a test can say the
        #: reranker was consulted rather than infer it from an ordering.
        self.pairs: list[list[str]] = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return np.array(
            [1.0 + sum(text.lower().count(m) for m in self.MARKERS)
             for _, text in pairs],
            dtype=np.float32,
        )


def stub_cross_encoder(monkeypatch, reranker) -> None:
    """Let ``_load_reranker`` succeed offline, handing back *reranker*.

    The suite is model-free, so the only way to observe that
    ``_load_reranker`` ran is to put a module where it imports from.
    """
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, **kwargs: object()
    module.CrossEncoder = lambda name, **kwargs: reranker
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


# ── The corpus ───────────────────────────────────────────────────

DOCS = {
    "policy.md": "Northwind Mutual auto cover 41-B. The motor vehicle "
                 "deductible is 500 per collision claim.",
    "dental.md": "Brightsmile dentist appointments. A cleaning is booked "
                 "every six months and the hygienist checks each molar.",
}

SHORT = "The bakery sells 120 loaves."


async def build(*, embedder=None, order=("policy.md", "dental.md"), **kwargs):
    """A chunked index over *order*, hybrid or BM25-only."""
    lib = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                          use_reranker=False, **kwargs)
    if embedder is None:
        lib._dense_ok = False  # as if the embeddings extra were absent
    else:
        lib.embed_model = embedder
    for doc_id in order:
        await lib.add_document(DOCS[doc_id], doc_id=doc_id)
    return lib


def records(caplog, level=logging.WARNING):
    return [r for r in caplog.records
            if r.name == "librarium.source" and r.levelno >= level]


def load_capturing(lib, path, caplog, level=logging.INFO):
    """Load *path* into *lib*, returning librarium.source's LogRecords."""
    caplog.clear()
    with caplog.at_level(level, logger="librarium.source"):
        lib.load(path)
    return records(caplog, level)


def written_under(policy, monkeypatch, lib, path):
    """Save *lib* to *path* as a build with `EMBED_SOURCE = policy` would.

    `EMBED_SOURCE` is read inside `content_hash` and again inside `save()` at
    call time rather than passed in — deliberately, since nothing configures
    it — so moving the module attribute is exactly what a differently-built
    version of this code looks like both to the cache key and to the file it
    writes. That is the only handle a test has on "written before the change"
    without keeping a fixture index in the repository, and it is a faithful
    one: the two files differ in the two keys the policy decides and in
    nothing else, which
    :meth:`TestACacheOfADifferentVectorSpaceIsNotServed
    .test_the_policy_is_the_only_difference_between_the_two` asserts — down to
    `embeddings.npy` being byte-identical — rather than assumes.

    Shared by the two classes below because the two guarantees are separate
    and both need the same fixture: `content_hash` decides whether a *cache*
    is current, which only `_load_if_current` asks, and the `embed_source`
    key decides what a direct `load()` does with the vectors it finds.
    """
    monkeypatch.setattr(librarium.source, "EMBED_SOURCE", policy)
    try:
        lib.save(path)
    finally:
        monkeypatch.undo()


def meta_of(path) -> dict:
    return json.loads((path / "index.json").read_text(encoding="utf-8"))


DRIFT_TEXT = "were not produced by the embedding weights now loaded"
NO_FINGERPRINT_TEXT = "records no embedding fingerprint"


# ── Part A: the fingerprint primitives ───────────────────────────


class TestEmbedFingerprint:
    """One encode of one constant string, or an honest ``None``."""

    def test_it_records_the_probe_it_used_alongside_the_vector(self):
        # Catches the probe being dropped from the payload. Without it two
        # fingerprints taken of *different* strings compare as drift, and
        # every index written by a librarium whose EMBED_PROBE differed
        # would report a false alarm on load rather than declining to
        # answer.
        model = WeightedEmbedder("nomic-2024")
        fingerprint = embed_fingerprint(model)

        assert fingerprint["probe"] == EMBED_PROBE
        assert len(fingerprint["vector"]) == WeightedEmbedder.DIMS
        # The cost claim in the spec: one encode, of one constant string.
        assert model.batches == [[EMBED_PROBE]]

    def test_the_same_weights_fingerprint_identically(self):
        # The premise the no-false-positive tests rest on.
        assert (embed_fingerprint(WeightedEmbedder("nomic-2024"))
                == embed_fingerprint(WeightedEmbedder("nomic-2024")))

    def test_different_weights_fingerprint_differently(self):
        # The premise the drift tests rest on, stated rather than assumed:
        # if these two ever fingerprinted alike, every drift test below
        # would pass for the wrong reason.
        drift = fingerprint_drift(
            embed_fingerprint(WeightedEmbedder("nomic-2024")),
            embed_fingerprint(WeightedEmbedder("nomic-2025")),
        )
        assert drift > EMBED_FINGERPRINT_TOLERANCE, drift

    def test_it_declines_when_the_model_cannot_encode(self):
        # Catches the try/except being narrowed or removed: an object that
        # cannot encode is about to fail at the real work, loudly, and a
        # crash from the fingerprint would arrive first and bury it.
        assert embed_fingerprint(ExplodingEmbedder()) is None

    def test_it_declines_a_zero_vector(self):
        # A zero vector has no direction, so cosine cannot compare it to
        # anything. Catches the norm guard being dropped: the fingerprint
        # would then be recorded, `load()` would read it as "verified",
        # and the INFO that says nothing was verified would stop firing.
        assert embed_fingerprint(ZeroProbeEmbedder()) is None

    def test_it_declines_a_non_finite_vector(self):
        # Catches the isfinite guard being dropped: a NaN component makes
        # every later comparison NaN, `NaN <= tolerance` is False, and the
        # librarian would discard good vectors on every load.
        class NaNEmbedder:
            def encode(self, texts, show_progress_bar=False):
                return np.array([[1.0, np.nan, 2.0]] * len(texts))

        assert embed_fingerprint(NaNEmbedder()) is None

    def test_it_declines_a_vector_with_no_components(self):
        # Catches the size guard being dropped — an empty vector would be
        # stored as an identity that matches nothing.
        class EmptyEmbedder:
            def encode(self, texts, show_progress_bar=False):
                return np.zeros((len(texts), 0))

        assert embed_fingerprint(EmptyEmbedder()) is None


class TestFingerprintDrift:
    """``None`` means "cannot tell", and every caller treats it as silence."""

    FP = {"probe": EMBED_PROBE, "vector": [1.0, 2.0, 3.0]}

    def test_identical_fingerprints_are_zero_apart(self):
        assert fingerprint_drift(self.FP, dict(self.FP)) == 0.0

    @pytest.mark.parametrize("saved,current", [
        (None, {"probe": EMBED_PROBE, "vector": [1.0]}),
        ({"probe": EMBED_PROBE, "vector": [1.0]}, None),
        ("not a dict", {"probe": EMBED_PROBE, "vector": [1.0]}),
    ])
    def test_a_missing_fingerprint_is_incomparable(self, saved, current):
        # The pre-upgrade case, and the no-model-loaded case. Catches
        # either being turned into a verdict: a missing identity read as
        # drift would discard the vectors of every index saved before this
        # change existed.
        assert fingerprint_drift(saved, current) is None

    def test_a_fingerprint_of_a_different_probe_is_incomparable(self):
        # An index written when EMBED_PROBE said something else. Comparing
        # the vectors of two different strings would report drift on every
        # load. Catches the probe equality check being dropped.
        other = {"probe": "some earlier probe", "vector": [1.0, 2.0, 3.0]}
        assert fingerprint_drift(other, self.FP) is None

    @pytest.mark.parametrize("vector", [[], None])
    def test_an_empty_vector_is_incomparable(self, vector):
        assert fingerprint_drift({"probe": EMBED_PROBE, "vector": vector},
                                 self.FP) is None

    def test_a_zero_vector_is_incomparable(self):
        # Zero norm, so the cosine is 0/0. Catches the norms guard being
        # dropped, which would return NaN — and NaN is not <= tolerance,
        # so it would read as drift and discard.
        zero = {"probe": EMBED_PROBE, "vector": [0.0, 0.0, 0.0]}
        assert fingerprint_drift(zero, self.FP) is None

    def test_a_different_width_is_conclusive_not_incomparable(self):
        # Nothing that changes the width of the embedding space leaves the
        # old vectors usable, so this is the one malformed-looking case
        # that must *not* be silent. Catches it being folded in with the
        # `return None` cases above, which would let a model swap that
        # changed the dimensionality load in silence.
        wider = {"probe": EMBED_PROBE, "vector": [1.0, 2.0, 3.0, 4.0]}
        assert fingerprint_drift(wider, self.FP) == 1.0

    def test_device_noise_stays_far_inside_the_tolerance(self):
        # The design's central claim, measured rather than asserted in a
        # comment: components differing by 1e-3 *relative* — ten times the
        # tolerance, and a thousand times the six decimals the vector is
        # stored to — move the cosine by ~d²/2, which is orders of
        # magnitude inside it. Catches the comparison being changed to a
        # per-component or digest one, under which this reads as drift and
        # every device change becomes a false alarm.
        drift = fingerprint_drift(
            embed_fingerprint(WeightedEmbedder("nomic-2024")),
            embed_fingerprint(WeightedEmbedder("nomic-2024", jitter=1e-3)),
        )
        assert 0.0 < drift < EMBED_FINGERPRINT_TOLERANCE, drift


# ── Part B: recording it ─────────────────────────────────────────


class TestSaveRecordsTheFingerprint:
    async def test_it_survives_save_and_load_unchanged(self, tmp_path):
        # The round trip. Catches the key being written under a different
        # name, JSON-mangled, or read back from the wrong place — any of
        # which leaves a check that never has anything to compare and so
        # never fires, silently.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")

        on_disk = meta_of(tmp_path / "idx")["embed_fingerprint"]
        assert on_disk == embed_fingerprint(WeightedEmbedder("nomic-2024"))

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.load(tmp_path / "idx")
        assert victim._vector_fingerprint == on_disk

        # And back out again: a load/save cycle must not lose it. If it
        # did, every index would silently downgrade itself to
        # "unrecorded" the first time it was re-saved, and the check would
        # go quiet for good on exactly the indexes it had already
        # verified once.
        victim.save(tmp_path / "again")
        assert meta_of(tmp_path / "again")["embed_fingerprint"] == on_disk

    async def test_a_bm25_only_index_records_none(self, tmp_path):
        # Gated on the vectors the same way `embed_model` is. Catches the
        # gate being dropped: an identity with nothing to identify would
        # outlive its array and be compared against the next one.
        lib = await build()
        assert lib.chunk_embeddings is None
        lib.save(tmp_path / "idx")

        assert meta_of(tmp_path / "idx")["embed_fingerprint"] is None

    async def test_a_query_failure_leaves_no_identity_behind(self, tmp_path):
        # The one sequence that reached a fingerprint with no vectors, and
        # it needs nothing private to reach it: index (identity recorded
        # for real vectors), query (the embed raises, `_dense_ok` goes
        # False, the vectors are still there), add again (`_refresh_
        # embeddings` bails on `_dense_ok` and drops the array). The
        # identity described the array being dropped, so it has to go with
        # it.
        #
        # Catches the drop being left out of that early return, which is
        # what the code did: the librarian then held an identity for
        # vectors that no longer existed, and nothing in the module put it
        # back in agreement — `_dense_ok` never goes True again, so no
        # later add can re-embed and overwrite it.
        #
        # Two assertions because there are two defences, and either alone
        # makes the file come out right: the state assertion is on the
        # invariant, the file assertions are on `save()`'s own gate. The
        # file is where a wrong pairing would stop being process state and
        # become a claim on disk that no later `load()` can see to correct
        # — `index.json` naming an identity for an `embeddings.npy` that
        # `save()` has just deleted.
        lib = await build(embedder=QueryBreakingEmbedder("nomic-2024"),
                          order=("policy.md",), embed_model=MARKING_MODEL)
        assert lib.chunk_embeddings is not None
        assert lib._vector_fingerprint is not None, "premise: identity recorded"

        assert lib.retrieve("deductible", top_k=1), "the query still answers"
        assert lib.retrieval_mode == "bm25-only", "premise: the embed failed"
        assert lib.chunk_embeddings is not None, "premise: vectors not yet dropped"

        await lib.add_document(DOCS["dental.md"], doc_id="dental.md")
        assert lib.chunk_embeddings is None
        assert lib._vector_fingerprint is None

        lib.save(tmp_path / "idx")
        assert not (tmp_path / "idx" / "embeddings.npy").exists()
        assert meta_of(tmp_path / "idx")["embed_fingerprint"] is None

    async def test_a_whole_document_index_records_none(self, tmp_path):
        lib = SourceLibrarian(MockLLM(), use_reranker=False)
        lib.embed_model = WeightedEmbedder("nomic-2024")
        await lib.add_document(SHORT, doc_id="bakery.md")
        assert not lib.use_chunking
        lib.save(tmp_path / "idx")

        assert meta_of(tmp_path / "idx")["embed_fingerprint"] is None

    async def test_a_model_that_cannot_be_probed_records_none(self, tmp_path):
        # The double every other test file uses. Catches a fingerprint
        # being invented for it — a zero vector recorded as an identity
        # matches every other zero vector and nothing else, so it would
        # read as "same weights" against an entirely different model.
        lib = await build(embedder=ZeroProbeEmbedder())
        assert lib.chunk_embeddings is not None
        lib.save(tmp_path / "idx")

        assert meta_of(tmp_path / "idx")["embed_fingerprint"] is None

    async def test_an_unrecorded_index_is_never_re_stamped_on_re_save(
        self, tmp_path
    ):
        # The decision the spec asks to be deliberate: an index from before
        # fingerprints existed stays unrecorded even after a librarian with
        # a perfectly good model loads it and extends it. Catches
        # opportunistic recording, which would stamp the current model's
        # identity onto vectors it did not produce and launder "unverified"
        # into "verified" — after which real drift on those rows is
        # undetectable forever.
        first = await build(embedder=ZeroProbeEmbedder(),
                            order=("policy.md",))
        first.save(tmp_path / "idx")
        assert meta_of(tmp_path / "idx")["embed_fingerprint"] is None

        second = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                                 use_reranker=False)
        second.embed_model = WeightedEmbedder("nomic-2024")
        second.load(tmp_path / "idx")
        await second.add_document(DOCS["dental.md"], doc_id="dental.md")
        assert len(second.chunk_embeddings) == len(second.chunks)

        second.save(tmp_path / "idx2")
        assert meta_of(tmp_path / "idx2")["embed_fingerprint"] is None

    async def test_re_embedding_the_whole_array_does_record_one(self, tmp_path):
        # The other side of the rule above: when every row came from this
        # model the librarian can vouch for all of it. Catches the `have
        # == 0` gate being inverted or dropped, which would either never
        # record anything (the check becomes dead) or record on every
        # append (the unverified rows above become falsely verified).
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")

        assert (meta_of(tmp_path / "idx")["embed_fingerprint"]
                == embed_fingerprint(WeightedEmbedder("nomic-2024")))


# ── Part C: the check ────────────────────────────────────────────


class TestDriftIsCaught:
    async def saved_under(self, tmp_path, weights: str):
        lib = await build(embedder=WeightedEmbedder(weights))
        assert lib.retrieval_mode == "hybrid"
        lib.save(tmp_path / "idx")
        return tmp_path / "idx"

    async def test_new_weights_over_old_vectors_warn_and_degrade(
        self, tmp_path, caplog
    ):
        # The bug this whole change exists for. Catches the check being
        # removed, and catches it being downgraded to "warn and keep
        # serving": wrong-space vectors are worse than no vectors, because
        # they look like dense retrieval and rank like noise.
        path = await self.saved_under(tmp_path, "nomic-2024")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        warned = load_capturing(victim, path, caplog, logging.WARNING)

        assert len(warned) == 1, [r.getMessage() for r in warned]
        assert DRIFT_TEXT in warned[0].getMessage()
        assert "re-index" in warned[0].getMessage()   # names the remedy
        assert victim.chunk_embeddings is None
        assert victim.retrieval_mode == "bm25-only"

    async def test_the_model_is_not_disabled_by_a_drift_verdict(
        self, tmp_path, caplog
    ):
        # `_dense_ok` means "the model is unusable" and here it is not —
        # only the vectors are. Catches drift being handled the way the
        # saved-model-name mismatch is: with `_dense_ok = False` the next
        # `add_document` would refuse to re-embed, and the remedy the
        # warning prescribes would silently not work.
        path = await self.saved_under(tmp_path, "nomic-2024")

        victim = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                                 use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        load_capturing(victim, path, caplog, logging.WARNING)
        assert victim._dense_ok is True

        await victim.add_document(SHORT * 3, doc_id="bakery.md")
        assert victim.retrieval_mode == "hybrid"

    async def test_re_embedding_after_a_discard_covers_every_chunk(
        self, tmp_path, caplog
    ):
        # The discard can land in the middle of `_refresh_embeddings`, and
        # this is the sequence that puts it there: load with no model, so
        # the check defers, then assign one and index — `have` is counted
        # from the loaded array, and `_ensure_embed_model` throws that
        # array away on the very next line.
        #
        # The count must therefore be re-read after the load. Carrying the
        # earlier one across embeds only `chunks[have:]` into an array
        # that now starts empty, so it comes out `len(chunks) - have` rows
        # long with row *i* holding chunk *have + i*'s vector — every row
        # shifted, and the count short by `have`, which is what this test
        # asserts on. `_embedding_ranking`'s length guard does see that
        # disagreement and drops the dense half; the reason to fix it here
        # rather than lean on the guard is that the misalignment outlives
        # it, which the test below measures. Catches the re-read being
        # folded back up into the first one.
        path = await self.saved_under(tmp_path, "nomic-2024")

        victim = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                                 use_reranker=False)
        load_capturing(victim, path, caplog, logging.WARNING)
        before = len(victim.chunks)
        # The premise: vectors are loaded and unexamined, because no model
        # has loaded for them to be examined against.
        assert before and len(victim.chunk_embeddings) == before

        victim.embed_model = WeightedEmbedder("nomic-2025")
        await victim.add_document(SHORT * 3, doc_id="bakery.md")

        assert len(victim.chunks) > before, "premise: chunks were appended"
        assert len(victim.chunk_embeddings) == len(victim.chunks)
        assert victim.retrieval_mode == "hybrid"

        # And the array is vouched for by the weights that made all of it,
        # which the mutation above cannot produce either: with `have`
        # still non-zero nothing is recorded at all.
        assert (victim._vector_fingerprint
                == embed_fingerprint(WeightedEmbedder("nomic-2025")))

    async def test_every_row_still_belongs_to_its_own_chunk(
        self, tmp_path, caplog
    ):
        # Why the re-read above is a fix and not a duplicate of
        # `_embedding_ranking`'s length guard. That guard only ever sees a
        # count, and the count comes back: the short array is appended to
        # by the next add, and two adds after the discard it is exactly as
        # long as `chunks` again while every row is still shifted. From
        # there the guard passes and the wrong vector is scored as the
        # right one, silently and for good.
        #
        # So this asserts correspondence, not arity — it is the only test
        # here that opens the array and checks each row against the chunk
        # it is filed under. Catches the re-read being dropped in the
        # state where nothing else would: with it dropped, the second add
        # restores the count and row 0 holds a later chunk's vector.
        # `retrieve()` cannot be the assertion, because BM25 ranks these
        # documents correctly on its own and RRF hides the wrong dense
        # half — which is the point, and the reason the misalignment is
        # silent.
        path = await self.saved_under(tmp_path, "nomic-2024")

        victim = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                                 use_reranker=False)
        load_capturing(victim, path, caplog, logging.WARNING)
        embedder = WeightedEmbedder("nomic-2025")
        victim.embed_model = embedder

        await victim.add_document(SHORT * 3, doc_id="bakery.md")
        await victim.add_document("Zeppelin mooring mast rates." * 3,
                                  doc_id="airship.md")

        assert len(victim.chunk_embeddings) == len(victim.chunks) > 3
        for i, chunk in enumerate(victim.chunks):
            expected = embedder.encode([embedded_text(chunk)])[0]
            assert np.allclose(victim.chunk_embeddings[i], expected), (
                f"row {i} does not hold chunk {i}'s vector ({chunk.doc_id})"
            )

    async def test_a_query_time_discard_does_not_crash_retrieve(
        self, tmp_path, caplog
    ):
        # The model can arrive *after* the vectors: load with none loaded,
        # then assign one, and the first query is where the check runs.
        # `_embedding_ranking` re-reads `chunk_embeddings` after
        # `_ensure_embed_model` for exactly that reason. Catches the
        # re-read being dropped — the cosine then runs against None and
        # `retrieve()` raises through the public API.
        path = await self.saved_under(tmp_path, "nomic-2024")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        assert load_capturing(victim, path, caplog, logging.WARNING) == []
        assert victim.retrieval_mode == "hybrid"   # nothing checked yet

        victim.embed_model = WeightedEmbedder("nomic-2025")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="librarium.source"):
            got = victim.retrieve("deductible", top_k=2)

        assert [c.doc_id for c in got], got
        assert victim.retrieval_mode == "bm25-only"
        assert any(DRIFT_TEXT in r.getMessage() for r in records(caplog))

    async def test_it_says_so_once_not_on_every_call(self, tmp_path, caplog):
        # The discard clears what it compared, so there is nothing left to
        # re-report. Catches a version that keeps the fingerprint around:
        # every retrieval would then re-warn, which is how a real warning
        # gets filtered out.
        path = await self.saved_under(tmp_path, "nomic-2024")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        assert len(load_capturing(victim, path, caplog, logging.WARNING)) == 1

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="librarium.source"):
            victim.retrieve("deductible", top_k=2)
            victim.retrieve("hygienist", top_k=2)
        assert records(caplog) == []


class TestNoFalsePositives:
    async def test_the_same_weights_reload_in_silence(self, tmp_path, caplog):
        # The case that runs on every ordinary load. Catches the tolerance
        # being tightened to an equality, or the fingerprint being taken
        # of something that varies run to run — either turns every load of
        # every hybrid index into a warning and a discarded array.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2024")
        assert load_capturing(victim, tmp_path / "idx", caplog,
                              logging.INFO) != []  # ordinary INFO chatter
        assert [r.getMessage() for r in records(caplog)] == []
        assert victim.retrieval_mode == "hybrid"
        assert victim.chunk_embeddings is not None

    async def test_the_same_weights_on_another_device_reload_in_silence(
        self, tmp_path, caplog
    ):
        # The residual the design accepts, pinned in the direction it must
        # not fail: fp32 across cpu/mps/cuda kernels perturbs every
        # component and barely moves the direction. Catches the tolerance
        # being replaced by a digest of the rounded vector, under which
        # this is drift — a re-index demanded of every user who changed
        # device.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2024", jitter=1e-3)
        assert load_capturing(victim, tmp_path / "idx", caplog,
                              logging.WARNING) == []
        assert victim.retrieval_mode == "hybrid"

    async def test_a_bm25_only_index_is_silent(self, tmp_path, caplog):
        # Nothing stored, so nothing to be stale. Catches the
        # `chunk_embeddings is None` guard being dropped.
        #
        # Scoped to the drift message rather than to "no warnings at all",
        # because such an index legitimately draws the shipped
        # was-built-without-embeddings warning on a machine with the extra
        # installed — a different warning about a different thing, pinned
        # in test_dense_honesty.py.
        lib = await build()
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        warned = load_capturing(victim, tmp_path / "idx", caplog,
                                logging.WARNING)
        assert not [r for r in warned if DRIFT_TEXT in r.getMessage()], \
            [r.getMessage() for r in warned]

    async def test_a_whole_document_index_is_silent(self, tmp_path, caplog):
        # Below the chunk threshold there are no vectors by design, and
        # re-indexing the same short text would produce none either.
        lib = SourceLibrarian(MockLLM(), use_reranker=False)
        lib.embed_model = WeightedEmbedder("nomic-2024")
        await lib.add_document(SHORT, doc_id="bakery.md")
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        warned = load_capturing(victim, tmp_path / "idx", caplog,
                                logging.WARNING)
        assert [r.getMessage() for r in warned] == []
        assert victim.retrieval_mode == "whole-document"

    async def test_a_model_that_cannot_be_probed_is_silent(
        self, tmp_path, caplog
    ):
        # "Cannot tell" changes nothing and says nothing, the way
        # `_check_code_revision` treats a model with no sha. Catches an
        # unprobeable model being treated as drift, which would discard
        # the vectors of every librarian using a custom encoder object.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = ExplodingEmbedder()
        warned = load_capturing(victim, tmp_path / "idx", caplog,
                                logging.WARNING)
        assert [r.getMessage() for r in warned] == []
        assert victim.chunk_embeddings is not None


class TestAPreExistingIndexIsNotNagged:
    """The ordinary state on upgrade: vectors, no recorded identity."""

    async def pre_existing(self, tmp_path):
        """An index with vectors and no ``embed_fingerprint`` key at all.

        Hand-edited rather than saved, because that is literally what an
        index written before this change looks like — the key absent, not
        null. The two must behave alike, and only one of them is
        reachable through `save()`.
        """
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")
        meta = meta_of(tmp_path / "idx")
        del meta["embed_fingerprint"]
        (tmp_path / "idx" / "index.json").write_text(json.dumps(meta),
                                                     encoding="utf-8")
        return tmp_path / "idx"

    async def test_it_says_so_once_at_info_and_never_warns(
        self, tmp_path, caplog
    ):
        # Asserted on the LogRecord, not on the text, because the level is
        # the decision: a WARNING nobody can act on — the vectors may be
        # perfectly current and nothing here can tell — fires on every
        # load for every upgrading user and is how the real warning two
        # branches up gets filtered out. Catches the level being raised,
        # and catches the branch being deleted so the upgrade is wholly
        # silent.
        path = await self.pre_existing(tmp_path)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        found = load_capturing(victim, path, caplog, logging.INFO)

        said = [r for r in found if NO_FINGERPRINT_TEXT in r.getMessage()]
        assert len(said) == 1, [r.getMessage() for r in found]
        assert said[0].levelno == logging.INFO
        assert str(path) in said[0].getMessage()   # names which index
        assert "re-index" in said[0].getMessage()  # names the remedy
        assert [r.getMessage() for r in records(caplog)] == []

    async def test_the_vectors_are_kept_and_still_serve(self, tmp_path, caplog):
        # Unverifiable is not unusable. Catches "no fingerprint" being
        # folded in with "different fingerprint", which would discard the
        # vectors of every index on disk the first time a user upgraded.
        path = await self.pre_existing(tmp_path)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        load_capturing(victim, path, caplog, logging.INFO)

        assert victim.chunk_embeddings is not None
        assert victim.retrieval_mode == "hybrid"

    async def test_a_recorded_index_does_not_get_the_notice(
        self, tmp_path, caplog
    ):
        # The control: with an identity on disk there is nothing to say.
        # Catches the INFO being emitted unconditionally.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2024")
        found = load_capturing(victim, tmp_path / "idx", caplog, logging.INFO)
        assert not [r for r in found if NO_FINGERPRINT_TEXT in r.getMessage()]

    async def test_a_bm25_only_index_does_not_get_the_notice(
        self, tmp_path, caplog
    ):
        # No vectors, so "cannot be checked" is not news. Catches the
        # `chunk_embeddings is not None` term being dropped from the elif.
        lib = await build()
        lib.save(tmp_path / "idx")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        found = load_capturing(victim, tmp_path / "idx", caplog, logging.INFO)
        assert not [r for r in found if NO_FINGERPRINT_TEXT in r.getMessage()]


class TestStrictRaises:
    async def drifted(self, tmp_path, **kwargs):
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")
        victim = SourceLibrarian(MockLLM(), use_reranker=False, **kwargs)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        return victim, tmp_path / "idx"

    async def test_strict_stops_on_drift(self, tmp_path):
        # The existing convention: raise when set, degrade otherwise.
        # Catches `strict` being ignored here, which would leave the one
        # setting whose entire purpose is "do not paper over it" papering
        # over the worst case in the module.
        victim, path = await self.drifted(tmp_path, strict=True)
        with pytest.raises(StaleEmbeddingsError) as excinfo:
            victim.load(path)
        assert DRIFT_TEXT in str(excinfo.value)
        assert "strict=True" in str(excinfo.value)

    async def test_without_strict_it_degrades_instead(self, tmp_path):
        # The control that makes the test above about `strict` rather than
        # about drift.
        victim, path = await self.drifted(tmp_path)
        victim.load(path)
        assert victim.retrieval_mode == "bm25-only"

    def test_it_is_a_runtime_error_and_not_a_value_error(self):
        # The type is the design. `_load_if_current` catches ValueError
        # around `load()` to treat a corrupt cache as a miss; a ValueError
        # here would be swallowed into a silent, and expensive, re-index —
        # the precise opposite of what `strict` was asked for. Catches the
        # base class being "tidied up" to match the rest of the module.
        assert issubclass(StaleEmbeddingsError, RuntimeError)
        assert not issubclass(StaleEmbeddingsError, ValueError)

    def test_it_is_exported_from_the_package(self):
        # `strict=True` is the only way to raise it, and a caller who
        # asked for that has to be able to name it in an `except`.
        assert librarium.StaleEmbeddingsError is StaleEmbeddingsError
        assert "StaleEmbeddingsError" in librarium.__all__

    async def test_strict_is_not_swallowed_into_a_silent_re_index(
        self, tmp_path
    ):
        # The type's reason for existing, exercised through the path that
        # would swallow it. `index(cache=)` catches ValueError from
        # `load()` and quietly re-indexes — one LLM call per chunk, and
        # the stop the caller asked for never happens. Catches the base
        # class change *behaviourally*, which the issubclass test above
        # cannot: it would still pass if `load()` raised something else.
        text = DOCS["policy.md"]
        first = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                                use_reranker=False)
        first.embed_model = WeightedEmbedder("nomic-2024")
        await first.index(text, cache=tmp_path / "idx")
        assert first.chunk_embeddings is not None

        llm = MockLLM(["A prefix."])
        victim = SourceLibrarian(llm, chunk_threshold=0, use_reranker=False,
                                 strict=True)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        with pytest.raises(StaleEmbeddingsError):
            await victim.index(text, cache=tmp_path / "idx")
        assert llm.calls == [], "strict must stop, not re-index"

    async def test_without_strict_the_cache_is_still_a_hit(self, tmp_path):
        # The other half: drift is not corruption, so a non-strict caller
        # keeps the cached chunks and pays no LLM calls — it just loses
        # the dense half. Catches drift being made to invalidate the
        # cache, which is the rejected "re-embed automatically on
        # mismatch" arriving by the back door.
        text = DOCS["policy.md"]
        first = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                               use_reranker=False)
        first.embed_model = WeightedEmbedder("nomic-2024")
        await first.index(text, cache=tmp_path / "idx")

        llm = MockLLM(["A prefix."])
        victim = SourceLibrarian(llm, chunk_threshold=0, use_reranker=False)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        await victim.index(text, cache=tmp_path / "idx")

        assert llm.calls == []
        assert victim.retrieval_mode == "bm25-only"


class TestTheStrictRaiseLeavesAWorkingLibrarian:
    """What ``load()``'s docstring promises behind the raise.

    ``StaleEmbeddingsError`` is documented as catchable and as leaving a
    "fully loaded, usable index" — but ``_load_reranker()`` used to run
    at the *tail* of ``load()``, past the point the fingerprint check
    raises, and it has no other call site any loaded librarian reaches.
    A caller that caught the error therefore kept ``use_reranker=True``
    with ``self.reranker is None`` for the life of the librarian, and
    ``_rerank`` skips a falsy reranker without a word: the branch that
    exists to stop silent degradation introduced some. The order of
    those two calls is the whole fix and only a comment holds it.
    """

    async def drifted(self, tmp_path, monkeypatch, **kwargs):
        """A librarian whose reranker can load, over a drifted index."""
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")
        reranker = DentalPreferringReranker()
        stub_cross_encoder(monkeypatch, reranker)
        victim = SourceLibrarian(MockLLM(), use_reranker=True, **kwargs)
        victim.embed_model = WeightedEmbedder("nomic-2025")
        return victim, reranker, tmp_path / "idx"

    async def test_a_caught_strict_raise_still_reranks(
        self, tmp_path, monkeypatch
    ):
        # Catches `_load_reranker()` moving back below the fingerprint
        # check: the librarian the caller keeps would then rerank nothing
        # it ever retrieves, silently, forever.
        victim, reranker, path = await self.drifted(
            tmp_path, monkeypatch, strict=True
        )
        with pytest.raises(StaleEmbeddingsError):
            victim.load(path)

        question = "motor vehicle deductible collision claim"
        order = [c.doc_id for c in victim.retrieve(question)]
        assert reranker.pairs, "the reranker was never consulted"
        assert order[0] == "dental.md", order

        # The premise, asserted rather than assumed: BM25 alone ranks
        # policy.md first for this question, so dental.md at the top
        # above cannot be an accident of the corpus.
        control = SourceLibrarian(MockLLM(), use_reranker=False)
        control.embed_model = WeightedEmbedder("nomic-2025")
        control.load(path)
        assert [c.doc_id for c in control.retrieve(question)][0] \
            == "policy.md"


# ── Part D: scope discipline ─────────────────────────────────────


class TestACacheOfADifferentVectorSpaceIsNotServed:
    """A cache written when the encoder read `c.text` must not load now.

    This is the stale-vector class the rest of this file is structurally
    unable to see, and it is why the change needed more than one line.
    `_check_embed_fingerprint` compares the *model* that produced the
    vectors, so identical weights over a *different text* pass it: the
    fingerprint matches, the chunk count matches, `load()` is silent, and
    every dense score afterwards is computed between a `search_query:`
    vector and vectors of a text this version never embeds. Nothing in
    Parts A-C can notice, because nothing there is about the input.

    `EMBED_SOURCE` closes it by joining `content_hash`, which is the
    documented mechanism for settings that shaped the index — the same one
    `chunk_size` uses, for the same reason. The migration cost is that every
    cache on disk re-indexes once.

    **Demonstrated through `save()`/`load()` and never by comparing hash
    strings.** A hash assertion would pass on a digest that no caller
    consults, which is the mistake of testing the mechanism instead of the
    guarantee: what a user experiences is `_load_if_current` returning False
    and `index()` rebuilding. So the policy is moved, a cache is written
    under the old one, and the question asked is the one `index(cache=...)`
    asks.

    Catches: `EMBED_SOURCE` being dropped from `content_hash`, which makes
    the first test below serve a stale vector space silently; and
    `content_hash` becoming insensitive in general, which the second test
    would notice by refusing an ordinary cache.
    """

    #: One document, because `_load_if_current` compares
    #: `list(cached_docs.values()) != [text]` before it ever reaches the
    #: hash. A two-document fixture is refused for having two documents,
    #: which is a False that says nothing about `EMBED_SOURCE` — the first
    #: draft of this class used one and passed for that reason. Every test
    #: below also asserts the logged *reason*, so neither direction can pass
    #: on the wrong branch again.
    ONLY = "policy.md"

    @staticmethod
    def a_reader():
        return SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                               use_reranker=False)

    async def test_a_text_policy_cache_is_not_current_for_a_prefixed_build(
        self, tmp_path, monkeypatch, caplog
    ):
        path = tmp_path / "written-before"
        lib = await build(embedder=WeightedEmbedder("nomic-2024"),
                          order=(self.ONLY,))
        written_under("text", monkeypatch, lib, path)
        # The premise: a real, complete, loadable index — so a refusal below
        # is about the policy and not about a broken or empty file.
        assert meta_of(path)["chunks"], "the fixture cache is empty"

        reader = self.a_reader()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="librarium.source"):
            current = reader._load_if_current(path, DOCS[self.ONLY])

        assert current is False
        # For the *hash* reason, which is the whole claim. Without this the
        # test would pass on "different text" or "older format" too.
        messages = [r.getMessage() for r in caplog.records]
        assert any("is stale — re-indexing" in m for m in messages), messages
        assert not [m for m in messages if "different text" in m], messages
        # And nothing was half-loaded on the way to saying no.
        assert reader.chunks == []
        assert reader.chunk_embeddings is None

    async def test_a_cache_written_by_this_build_is_still_served(
        self, tmp_path, caplog
    ):
        # The other direction, and the one that makes the refusal above mean
        # something: ordinary caching is untouched. Without this, folding
        # anything at all into `content_hash` — or breaking it outright —
        # would satisfy the test above.
        path = tmp_path / "written-now"
        lib = await build(embedder=WeightedEmbedder("nomic-2024"),
                          order=(self.ONLY,))
        lib.save(path)

        reader = self.a_reader()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="librarium.source"):
            current = reader._load_if_current(path, DOCS[self.ONLY])

        assert current is True
        assert any("reused cached index" in r.getMessage()
                   for r in caplog.records)
        assert len(reader.chunks) == len(lib.chunks) > 0

    async def test_the_policy_is_the_only_difference_between_the_two(
        self, tmp_path, monkeypatch
    ):
        # Why the refusal is attributable to `EMBED_SOURCE` and not to some
        # incidental difference between the two files: the same librarian
        # saved under the two policies differs *only* in the key. The chunks,
        # the vectors and the fingerprint on disk are identical, so there is
        # nothing else for `_load_if_current` to have objected to.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"),
                          order=(self.ONLY,))
        old, new = tmp_path / "old", tmp_path / "new"
        written_under("text", monkeypatch, lib, old)
        lib.save(new)

        before, after = meta_of(old), meta_of(new)
        # Exactly the two keys the policy decides, named as a set rather than
        # spot-checked: `content_hash` is the cache-currency half and
        # `embed_source` the `load()` half, and an earlier version of this
        # test asserted only the first — which would have gone quiet about
        # the second the moment it was added.
        assert {k for k in before if before[k] != after.get(k)} == {
            "content_hash", "embed_source"
        }
        assert (before["embed_source"], after["embed_source"]) == (
            "text", "prefixed"
        )
        # And the vectors themselves, which is what the helper's docstring
        # claims and what nothing here compared before: the same librarian
        # saved twice, so a differing `.npy` would mean the fixture is not
        # the controlled pair this class rests on.
        assert np.array_equal(
            np.load(old / "embeddings.npy"), np.load(new / "embeddings.npy")
        )


class TestLoadDiscardsADifferentVectorSpace:
    """The `load()` half, which `content_hash` never covered.

    `content_hash` decides whether a *cache* is current, and only
    `_load_if_current` asks it. A caller who reaches for the documented
    `load()` consults no hash at all, so before the `embed_source` key a
    pre-change index installed text-space vectors at `mode=hybrid` in
    silence — and the fingerprint cannot help, because it compares the
    *model* and the model is identical. That is the same blind spot the class
    above describes, reached by the other door.

    It compounded rather than staying still: `_refresh_embeddings` only
    encodes the chunks past the end of `chunk_embeddings`, so the next
    `add_document` left the loaded rows in text space and appended prefixed
    rows after them — one array, one cosine, never repaired. Reproduced
    against an index really built by the pre-change code (evidence:
    `cbc62e5`, the sha whose tree wrote the reproduction index; re-running it
    needs a real encoder, so no test here can).

    **Discarded, not refused, and `_dense_ok` is left alone.** These are
    honest vectors of the wrong text, so the recovery is the one the
    fingerprint-drift and count-mismatch discards already use: drop them and
    let the next `add_document` re-embed into the current space. Leaving
    `_dense_ok` True is what makes that happen, which is why it is asserted
    here rather than left implied — it is the easy thing to "tidy" to False,
    and doing so would turn a one-add recovery into a permanent BM25-only
    librarian.

    Catches, one test per way the branch can go wrong, and each verified to
    fail alone: the branch being dropped from `load()`; its default written
    as a truthiness test, which would let the *absent* key through; the
    branch discarding unconditionally; `_dense_ok` cleared alongside the
    vectors; and `strict` being made to raise here for symmetry with the
    drift discard.

    That list is a closed enumeration, which is the rot shape nothing can
    check — nothing in it is untrue and nobody has to edit it for it to go
    wrong; the set simply grows. It is kept as a list because naming the five
    mutations is clearer than any property covering them, so the obligation
    comes with it: **a sixth condition on this branch extends this clause in
    the same commit.** The fifth item is that rule working — it arrived with
    the test below it, not a round later.
    """

    @staticmethod
    def a_reader(*, strict: bool = False):
        """A librarian that would load vectors if it were offered any.

        `_dense_ok` is left at its constructed value on purpose: the branch
        under test is gated on it, so forcing it True would test a state the
        constructor may not produce, and forcing it False would make every
        assertion below vacuous.

        `strict` goes through the same constructor rather than a second one,
        so the strict and non-strict cases cannot drift into differing by
        anything else — which is the whole claim of
        :meth:`test_strict_makes_no_difference_to_this_discard`.
        """
        reader = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0,
                                 use_reranker=False, strict=strict)
        assert reader._dense_ok, (
            "this reader cannot load vectors at all, so nothing below is "
            "about the embed_source branch"
        )
        assert reader.strict is strict
        return reader

    @staticmethod
    async def an_index_with_vectors(tmp_path, monkeypatch, *, policy):
        """A saved index holding real vectors, written under *policy*."""
        lib = await build(embedder=WeightedEmbedder("nomic-2024"),
                          order=("policy.md",))
        path = tmp_path / f"written-{policy}"
        if policy == "prefixed":
            lib.save(path)                       # the current policy
        else:
            written_under(policy, monkeypatch, lib, path)
        assert (path / "embeddings.npy").exists(), "no vectors to discard"
        return lib, path

    async def test_a_text_policy_index_loses_its_vectors_and_warns(
        self, tmp_path, monkeypatch, caplog
    ):
        lib, path = await self.an_index_with_vectors(
            tmp_path, monkeypatch, policy="text",
        )
        assert meta_of(path)["embed_source"] == "text"

        reader = self.a_reader()
        found = load_capturing(reader, path, caplog, logging.WARNING)

        assert reader.chunk_embeddings is None
        assert reader.retrieval_mode == "bm25-only"
        # The whole point of discarding rather than refusing: the next add
        # re-embeds, and only an untouched `_dense_ok` lets it.
        assert reader._dense_ok is True
        # The chunks themselves arrived — this is a discard, not a failed load.
        assert len(reader.chunks) == len(lib.chunks) > 0

        messages = [r.getMessage() for r in found]
        assert any("'text'" in m and "'prefixed'" in m for m in messages), (
            f"the warning must name both policies so a reader can tell which "
            f"way round it is: {messages}"
        )
        assert any("discarding them" in m for m in messages), messages

    async def test_an_index_without_the_key_behaves_the_same_way(
        self, tmp_path, monkeypatch, caplog
    ):
        # **The real pre-change shape.** Every file written before the change
        # lacks the key entirely, and every one of them holds text-space
        # vectors — so absence has to read as `"text"` and not as "nothing to
        # check". That is a deliberate asymmetry with the `embed_model`
        # branch beside it, where `""` means "no vectors were saved" and is
        # skipped, and it is why the source reads
        # `meta.get("embed_source", "text")` rather than testing truthiness.
        # Without the default this test passes vectors straight through while
        # the one above still passes, which is what makes this not a
        # duplicate.
        lib, path = await self.an_index_with_vectors(
            tmp_path, monkeypatch, policy="text",
        )
        meta = meta_of(path)
        del meta["embed_source"]
        (path / "index.json").write_text(json.dumps(meta), encoding="utf-8")
        assert "embed_source" not in meta_of(path)

        reader = self.a_reader()
        found = load_capturing(reader, path, caplog, logging.WARNING)

        assert reader.chunk_embeddings is None
        assert reader.retrieval_mode == "bm25-only"
        assert reader._dense_ok is True
        assert len(reader.chunks) == len(lib.chunks) > 0
        assert any("'text'" in m.getMessage() for m in found), (
            f"an absent key must be reported as the text policy it stands "
            f"for: {[m.getMessage() for m in found]}"
        )

    async def test_the_next_add_re_embeds_into_the_current_space(
        self, tmp_path, monkeypatch, caplog
    ):
        # What "discarded, not refused" buys, end to end: after the discard a
        # single `add_document` restores dense retrieval, and every row is of
        # the current policy's text. The array is rebuilt rather than appended
        # to, which is the failure the unguarded `load()` produced — rows 0-1
        # of one space and row 2 of another in a single array.
        _, path = await self.an_index_with_vectors(
            tmp_path, monkeypatch, policy="text",
        )
        reader = self.a_reader()
        embedder = WeightedEmbedder("nomic-2024")
        load_capturing(reader, path, caplog, logging.WARNING)
        reader.embed_model = embedder
        assert reader.chunk_embeddings is None

        await reader.add_document(DOCS["dental.md"], doc_id="dental.md")

        assert reader.chunk_embeddings is not None
        assert len(reader.chunk_embeddings) == len(reader.chunks)
        assert reader.retrieval_mode == "hybrid"
        for i, chunk in enumerate(reader.chunks):
            expected = embedder.encode([embedded_text(chunk)])[0]
            assert np.allclose(reader.chunk_embeddings[i], expected), (
                f"row {i} is not of the current policy's text ({chunk.doc_id})"
            )

    async def test_a_current_policy_round_trip_keeps_its_vectors(
        self, tmp_path, monkeypatch, caplog
    ):
        # The assertion that stops the branch above being vacuous. A version
        # that discarded unconditionally would satisfy every test before this
        # one; this is the only thing that says the discard is *selective*.
        lib, path = await self.an_index_with_vectors(
            tmp_path, monkeypatch, policy="prefixed",
        )
        assert meta_of(path)["embed_source"] == "prefixed"

        reader = self.a_reader()
        found = load_capturing(reader, path, caplog, logging.WARNING)

        assert reader.chunk_embeddings is not None
        assert len(reader.chunk_embeddings) == len(lib.chunks)
        assert np.array_equal(reader.chunk_embeddings, lib.chunk_embeddings)
        assert reader.retrieval_mode == "hybrid"
        assert reader._dense_ok is True
        assert not [m for m in found if "different space" in m.getMessage()], (
            f"a current index must load its vectors in silence: "
            f"{[m.getMessage() for m in found]}"
        )

    async def test_strict_makes_no_difference_to_this_discard(
        self, tmp_path, monkeypatch, caplog
    ):
        """`strict` does not raise here, and that is a deliberate divergence.

        The fingerprint-drift discard *does* raise `StaleEmbeddingsError`
        under `strict` — `TestStrictRaises` pins that — so this pair is
        asymmetric on purpose, and until now only the raising half was
        tested. That is worse than neither being tested: somebody tidying the
        two into symmetry would find a test demanding the raise and none
        forbidding it, and the source says so in as many words ("somebody
        tidying the two into symmetry would break no test today"). A
        rationale that names the regression nothing catches is a rationale
        asking to become an assertion.

        The reason for the divergence, which this test does not judge and only
        holds in place: drift says something moved *underneath* the caller and
        re-embedding may not fix it, while a source mismatch says librarium
        was upgraded, the vectors are exactly what the same model made from
        less text, and one `add_document` repairs it completely. Raising would
        turn a migration into an outage for every `strict` caller with an
        index written before the upgrade.

        Asserted as *identity* with the non-strict case rather than as a bare
        "did not raise", because the flag must change nothing at all here —
        not the discard, not the warning, not `_dense_ok`.

        **One test per side, and the pair is what guards the divergence.**
        This one owns the non-raising half; `TestStrictRaises
        ::test_strict_stops_on_drift` owns the raising half against a real
        two-weight fixture. Tidying either branch into the other's shape
        fails one of them, so neither test needs to reach into the other's
        subject — and reproducing drift here would be a second copy of a
        fixture that class already owns.
        """
        lib, path = await self.an_index_with_vectors(
            tmp_path, monkeypatch, policy="text",
        )
        reader = self.a_reader(strict=True)

        # No `pytest.raises`: the load simply completes. A raise here would
        # surface as an error rather than a failed assertion, which is the
        # clearest possible signal for this particular regression.
        found = load_capturing(reader, path, caplog, logging.WARNING)

        assert reader.chunk_embeddings is None
        assert reader.retrieval_mode == "bm25-only"
        assert reader._dense_ok is True
        assert len(reader.chunks) == len(lib.chunks) > 0
        assert any("discarding them" in m.getMessage() for m in found)


class TestNothingOnDiskWasInvalidated:
    """The change adds an optional key and nothing else."""

    def test_the_format_version_was_not_bumped(self):
        # A bump makes `load()` raise and `_load_if_current` miss on every
        # index on disk — every user paying a full re-index, one LLM call
        # per chunk — to add a field that is absent-tolerant by
        # construction. Catches a reflexive bump.
        assert INDEX_FORMAT_VERSION == 1

    async def test_an_index_written_without_the_key_still_loads(
        self, tmp_path
    ):
        # Absent, not null: the literal shape of every index already on
        # disk. Catches the reader being written as `meta["..."]`, which
        # would turn every pre-upgrade load into a KeyError.
        lib = await build(embedder=WeightedEmbedder("nomic-2024"))
        lib.save(tmp_path / "idx")
        meta = meta_of(tmp_path / "idx")
        del meta["embed_fingerprint"]
        (tmp_path / "idx" / "index.json").write_text(json.dumps(meta),
                                                     encoding="utf-8")

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.load(tmp_path / "idx")
        assert len(victim.chunks) == len(lib.chunks)
        assert victim._vector_fingerprint is None

    async def test_the_fingerprint_is_not_folded_into_content_hash(
        self, tmp_path
    ):
        # `content_hash()` structurally cannot carry this — it runs before
        # any model loads — and the spec keeps it unchanged. Catches it
        # being folded in anyway: the hash would then differ between a
        # librarian that had probed its model and one that had not, and
        # every cache would miss.
        text = DOCS["policy.md"]
        with_vectors = SourceLibrarian(MockLLM(["A prefix."]),
                                       chunk_threshold=0, use_reranker=False)
        with_vectors.embed_model = WeightedEmbedder("nomic-2024")
        await with_vectors.index(text, cache=tmp_path / "idx")
        assert with_vectors._vector_fingerprint is not None

        # A librarian that has probed nothing hashes the same documents to
        # the same key as one that recorded a fingerprint over them.
        bare = SourceLibrarian(MockLLM(), chunk_threshold=0,
                               use_reranker=False)
        assert bare._vector_fingerprint is None
        assert (bare._content_hash(with_vectors.documents)
                == meta_of(tmp_path / "idx")["content_hash"])

        # And the consequence that matters: it is still a cache hit, so
        # nobody pays one LLM call per chunk for the new key.
        llm = MockLLM(["A prefix."])
        reader = SourceLibrarian(llm, chunk_threshold=0, use_reranker=False)
        await reader.index(text, cache=tmp_path / "idx")
        assert llm.calls == [], "a fingerprint must not move the cache key"
