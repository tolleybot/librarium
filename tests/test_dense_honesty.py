"""Dense retrieval: the fusion arithmetic, and load()'s honesty about it.

Three separable things are pinned here, and none of them loads a real
model — the suite has to stay runnable and *meaningful* on a core
install, where the ``embeddings`` extra is absent:

* **The fusion.** A query BM25 scores at zero everywhere still produces a
  full-length lexical ranking, in index order, and RRF cannot tell that
  filler from a ranking. Fused against a dense ranking that put the
  answer first, the filler wins on position. The bug is pure ranking
  arithmetic, so a content-derived fake embedder is enough; a real
  encoder would only make it slow and machine-dependent.
* **The take-back guard**, which restores that filler when the question
  and every phrasing scored nothing — pinned in both directions, because
  restoring it is right when the lexical side is the whole fusion and
  wrong when a dense ranking is there to carry the result.
* **``load()``'s warning** for an index built without vectors, including
  every condition that keeps it quiet. ``find_spec`` is monkeypatched in
  both directions so both branches run whatever this machine has
  installed.

:class:`ConceptEmbedder` is content-derived, and carries
one property a marker-word counter does not: it maps *synonyms* onto a
shared axis, so it bridges vocabulary the way a real embedding does. A
double that only counted literal words could not tell the dense half
apart from the lexical one, which is the entire subject of this file.
Every vector is derived from the text it was made from, so a vector
filed against the wrong chunk ranks differently and fails an assertion.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

from librarium.source import SourceLibrarian, embedding_backend_available
from tests.conftest import MockLLM

# ── The double ───────────────────────────────────────────────────

#: Each axis is one concept and the words that express it. A text embeds
#: as how many of each concept's words it contains, so "auto ... motor
#: vehicle ... deductible" and the query "car insurance excess" land on
#: the same axes without sharing a single token — which is exactly the
#: gap BM25 cannot cross and dense retrieval exists to cross.
CONCEPTS = {
    "motor": ("car", "auto", "vehicle", "collision", "motor", "driving"),
    "money": ("insurance", "policy", "deductible", "premium", "excess",
              "claim", "cover"),
    "teeth": ("dental", "dentist", "cleaning", "cavity", "hygienist",
              "molar"),
    "transit": ("train", "platform", "commute", "fare", "timetable"),
}


class ConceptEmbedder:
    """Deterministic stand-in for a sentence-transformer, synonym-aware."""

    def encode(self, texts, show_progress_bar: bool = False):
        rows = []
        for text in texts:
            words = [w.strip(".,;:!?") for w in text.lower().split()]
            rows.append([float(sum(w in synonyms for w in words))
                         for synonyms in CONCEPTS.values()])
        return np.array(rows, dtype=np.float32)


# ── The corpus ───────────────────────────────────────────────────

DOCS = {
    "dental.md": "Brightsmile dentist appointments. A cleaning is booked "
                 "every six months and the hygienist checks each molar.",
    "commute.md": "The express train leaves platform four. A monthly "
                  "commute fare beats the daily timetable price.",
    "policy.md": "Northwind Mutual auto cover 41-B. The motor vehicle "
                 "deductible is 500 per collision claim.",
    # Says "car" three times and is about nothing else, so it outscores
    # policy.md lexically on "car deductible" while the dense half — which
    # reads the whole concept mix — prefers policy.md. That disagreement
    # is what the tie-break test needs.
    "carpark.md": "The car park behind the depot. Car spaces are allocated "
                  "by seniority and the car park gate closes at seven.",
}

#: Not one of these three words appears in dental.md, commute.md or
#: policy.md, and none of their stems either: BM25 scores every chunk
#: zero, so its ranking is index order and nothing else.
PROBE = "car insurance excess"

#: The answer is **last**. This is the whole regression: with policy.md
#: at index 0 the broken fusion returns it too, for the wrong reason.
ANSWER_LAST = ("dental.md", "commute.md", "policy.md")


async def build(order, *, embedder=None, **kwargs) -> SourceLibrarian:
    """One chunk per document, in *order*, hybrid or BM25-only."""
    lib = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0, **kwargs)
    if embedder is None:
        lib._dense_ok = False  # as if the embeddings extra were absent
    else:
        lib.embed_model = embedder
    for doc_id in order:
        await lib.add_document(DOCS[doc_id], doc_id=doc_id)
    return lib


def ids(chunks) -> list[str]:
    return [c.doc_id for c in chunks]


def dense_order(lib: SourceLibrarian, question: str) -> list[str]:
    """The ranking the dense half would produce, by hand.

    Cosine over ``chunk_embeddings`` and the query vector, computed from
    the same double the librarian was given — so a test can state the
    dense ranking as a premise without asking the librarian for it.
    """
    q = ConceptEmbedder().encode([f"search_query: {question}"])[0]
    norms = np.linalg.norm(lib.chunk_embeddings, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1.0, norms)
    cosine = np.dot(lib.chunk_embeddings, q) / norms
    order = sorted(range(len(cosine)), key=lambda i: cosine[i], reverse=True)
    return [lib.chunks[i].doc_id for i in order]


# ── Part A: the fusion ───────────────────────────────────────────


class TestAZeroScoringLexicalHalfNoLongerOutvotesTheDenseOne:
    """The measured regression, reproduced without a real encoder.

    Probe scores zero on every chunk; the answer is indexed last; the
    dense half puts it first. Before the fix the lexical filler's
    index-order vote took it, by 0.000256 of RRF score.
    """

    async def test_the_answer_indexed_last_is_retrieved_first(self):
        # Catches the fix being reverted to `scored=bool(variants)`: with
        # no variants the lexical half votes for all three chunks in
        # index order, dental.md collects 1/60 + 1/61 against policy.md's
        # 1/62 + 1/60, and the answer loses.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())

        # The premise, stated rather than assumed — with policy.md first
        # this test passes however broken the fusion is.
        assert ids(lib.chunks) == list(ANSWER_LAST)
        assert lib.chunks[-1].doc_id == "policy.md", "the answer must be last"
        assert lib.retrieval_mode == "hybrid"

        assert ids(lib.retrieve(PROBE, top_k=1)) == ["policy.md"]

    async def test_bm25_alone_really_does_point_the_other_way(self):
        # The control that makes the test above discriminate: the same
        # corpus without the dense half returns index order, so the
        # answer is last. If this ever returned policy.md, the regression
        # test above would be passing on BM25's own ranking.
        lib = await build(ANSWER_LAST)
        assert lib.retrieval_mode == "bm25-only"
        assert ids(lib.retrieve(PROBE, top_k=3)) == list(ANSWER_LAST)

    async def test_the_dense_half_is_what_knows_the_answer(self):
        # Pins the double: the concept axes really do rank policy.md
        # first for a probe sharing no token with it. Without this a
        # future edit to CONCEPTS could make the test above pass because
        # the dense ranking degenerated to index order reversed.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        assert dense_order(lib, PROBE)[0] == "policy.md"


class TestBm25OnlyResultsAreUnchanged:
    """The line the fix may not cross: with no dense half and no extra
    phrasings, ``retrieve`` returns exactly what it always returned —
    including the chunks BM25 scored at zero, which are the only thing
    keeping a sole lexical ranking at ``top_k``.
    """

    async def test_a_query_only_one_chunk_scores_still_returns_top_k(self):
        # Catches the truncation being made unconditional: "hygienist"
        # scores dental.md and nothing else, so a truncated sole ranking
        # would hand back one chunk where three were asked for.
        lib = await build(ANSWER_LAST)
        got = ids(lib.retrieve("hygienist", top_k=3))
        assert got == ["dental.md", "commute.md", "policy.md"], got

    async def test_a_query_nothing_scores_still_returns_top_k(self):
        # The same guarantee at the extreme: nothing scores at all, and
        # BM25-only still owes the caller an arbitrary but non-empty
        # answer rather than the whole-documents fallback.
        lib = await build(ANSWER_LAST)
        assert ids(lib.retrieve(PROBE, top_k=2)) == ["dental.md", "commute.md"]

    async def test_a_lexical_ranking_no_dense_one_can_back_is_not_truncated(
        self,
    ):
        # Why the dense ranking is *computed* rather than forecast from
        # `chunk_embeddings is not None and _dense_ok`. Here both
        # preconditions hold and `_embedding_ranking` still returns None,
        # because the vectors no longer cover the chunks. A forecast
        # would truncate a ranking that turns out to be the only one in
        # the fusion — one chunk back instead of three.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        lib.chunk_embeddings = lib.chunk_embeddings[:-1]
        assert lib._dense_ok is True and lib.chunk_embeddings is not None

        got = ids(lib.retrieve("hygienist", top_k=3))
        assert got == ["dental.md", "commute.md", "policy.md"], got


class TestTheTakeBackGuardDefersToTheDenseHalf:
    """When the question and every phrasing score nothing, the
    untruncated lexical ranking is restored — but only when it is the
    whole fusion. Both directions, because the guard's value and its cost
    are the same behaviour seen from either side.
    """

    NOISE = ["zzz nonsense"]

    async def test_it_fires_when_the_lexical_side_is_the_whole_fusion(self):
        # Catches the guard being deleted along with its new condition:
        # without it the fusion is empty here and `retrieve` falls back
        # to whole documents, which ignores top_k and is what makes
        # `ask()` raise from check_context.
        lib = await build(ANSWER_LAST)
        got = lib.retrieve(PROBE, top_k=2, also=self.NOISE)
        # top_k=2 and the prefix together separate this from the
        # whole-documents fallback, which returns one chunk per document,
        # unprefixed, and ignores top_k.
        assert ids(got) == ["dental.md", "commute.md"]
        assert all(c.prefix == "A prefix." for c in got)

    async def test_it_stands_down_when_a_dense_ranking_exists(self):
        # Catches the `and cos_ranked is None` gating being dropped:
        # taking the untruncated ranking back would re-insert the exact
        # index-order filler the fix removes, and dental.md would beat
        # policy.md again — the same 0.0331-vs-0.0328 arithmetic, now
        # reached through the guard instead of through `scored`.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        got = ids(lib.retrieve(PROBE, top_k=3, also=self.NOISE))
        assert got == dense_order(lib, PROBE)
        assert got[0] == "policy.md", got

    async def test_the_hybrid_caller_still_gets_top_k(self):
        # Standing down costs nothing only because the dense ranking is
        # full-length. If that ever stopped being true the caller would
        # silently get fewer chunks than asked for.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        assert len(lib.retrieve(PROBE, top_k=3, also=self.NOISE)) == 3


class TestTheQuestionsRankingKeepsFirstPlaceInTheFusion:
    """RRF breaks ties by first appearance, so the order rankings are
    appended in is load-bearing — the dense ranking is computed before
    the lexical block and appended after it, and the second half is as
    important as the first.
    """

    #: "car" scores carpark.md and only carpark.md; "deductible" scores
    #: policy.md and only policy.md. Two chunks scored, in that order.
    QUESTION = "car deductible"
    ORDER = ("dental.md", "commute.md", "carpark.md", "policy.md")

    async def test_a_dead_heat_resolves_towards_the_lexical_ranking(self):
        # The two halves rank the same pair in opposite orders, so RRF
        # scores them identically (1/60 + 1/61 either way) and only
        # append order decides. Catches `rankings.append(cos_ranked)`
        # moving up to where `cos_ranked` is now computed: policy.md
        # would then win the tie and the question's own literal top hit
        # would have been displaced by the dense half.
        lib = await build(self.ORDER, embedder=ConceptEmbedder())

        # Premises: lexical prefers carpark.md, dense prefers policy.md.
        lexical = await build(self.ORDER)
        assert ids(lexical.retrieve(self.QUESTION, top_k=2)) == [
            "carpark.md", "policy.md"]
        assert dense_order(lib, self.QUESTION)[:2] == ["policy.md",
                                                       "carpark.md"]

        assert ids(lib.retrieve(self.QUESTION, top_k=1)) == ["carpark.md"]


class TestTheHybridPathStillBehaves:
    """The truncation is not allowed to make ordinary hybrid retrieval
    worse: full results, and literal hits still lead.
    """

    async def test_a_query_only_one_chunk_scores_still_returns_top_k(self):
        # The BM25-only version of this returns index-order filler; the
        # hybrid one has to reach top_k from the dense side instead.
        # Catches the truncation shrinking hybrid results.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        assert len(lib.retrieve("hygienist", top_k=3)) == 3

    async def test_the_literal_hit_still_leads(self):
        # The one failure that would make this change worse than the bug:
        # a question with an exact lexical match must not be displaced by
        # the dense half now that the lexical vote is shorter.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        assert ids(lib.retrieve("hygienist", top_k=3))[0] == "dental.md"


# ── Part C: the availability check ───────────────────────────────


def fake_find_spec(monkeypatch, *, present: bool) -> list[str]:
    """Point ``find_spec`` at a machine that does (not) have the extra.

    Both directions have to be forced: on a core install the "present"
    branch never runs, and on this machine the "absent" branch never
    does. Every other name is delegated to the real implementation, so
    nothing else that asks a question of the import system during the
    test gets a wrong answer.
    """
    real = importlib.util.find_spec
    asked: list[str] = []

    def spoofed(name, package=None):
        asked.append(name)
        if name == "sentence_transformers":
            return object() if present else None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", spoofed)
    return asked


class TestEmbeddingBackendAvailable:
    def test_it_is_true_when_the_package_is_findable(self, monkeypatch):
        # Catches the sense of the check being inverted.
        asked = fake_find_spec(monkeypatch, present=True)
        assert embedding_backend_available() is True
        assert "sentence_transformers" in asked

    def test_it_is_false_when_the_package_is_not(self, monkeypatch):
        # The branch a core install runs and this machine never would.
        fake_find_spec(monkeypatch, present=False)
        assert embedding_backend_available() is False

    def test_it_answers_without_importing_the_backend(self):
        # The reason it is find_spec and not `try: import`: importing
        # sentence_transformers pulls in torch — seconds of wall-clock and
        # hundreds of MB — for a question asked only to decide whether to
        # print a log line. A subprocess because much of this suite has
        # already imported torch by the time this test runs, so an
        # in-process sys.modules assertion would pass or fail on test
        # order rather than on the code.
        #
        # use_reranker=False deliberately: `load()` calls
        # `_load_reranker()`, which imports CrossEncoder when reranking is
        # on. That is existing intended behaviour and not what this is
        # about.
        result = subprocess.run(
            [sys.executable, "-c", NO_IMPORT_ON_LOAD],
            capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == \
            "bm25-only sentence_transformers=False torch=False", result.stdout


NO_IMPORT_ON_LOAD = """
import asyncio, sys, tempfile
from pathlib import Path

from librarium.source import SourceLibrarian


class LLM:
    async def complete(self, system, user, *, think=False, temperature=0.1):
        return "Context."


async def main():
    tmp = Path(tempfile.mkdtemp())
    built = SourceLibrarian(LLM(), chunk_threshold=0, use_reranker=False)
    built._dense_ok = False              # built on a machine without the extra
    await built.add_document("Northwind Mutual auto cover 41-B. The motor "
                             "vehicle deductible is 500.", doc_id="policy.md")
    assert built.use_chunking and built.chunks, "premise: a chunked index"
    built.save(tmp / "idx")

    victim = SourceLibrarian(LLM(), use_reranker=False)
    victim.load(tmp / "idx")
    print(victim.retrieval_mode,
          "sentence_transformers=%s" % ("sentence_transformers"
                                        in sys.modules),
          "torch=%s" % ("torch" in sys.modules))


asyncio.run(main())
"""


# ── Part C: the load-time warning ────────────────────────────────


MISSING_VECTORS = "was built without embeddings"


def warnings_from_load(lib: SourceLibrarian, path, caplog) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="librarium.source"):
        lib.load(path)
    return [r.getMessage() for r in caplog.records
            if r.name == "librarium.source" and r.levelno >= logging.WARNING]


class TestLoadSaysWhenAnIndexHasNoVectors:
    """An index built without vectors used to load as BM25-only in
    silence and cache as current, so installing the extra to get dense
    retrieval appeared to work and did nothing.
    """

    async def vectorless_index(self, tmp_path) -> Path:
        lib = await build(ANSWER_LAST)
        assert lib.chunk_embeddings is None and lib.use_chunking
        lib.save(tmp_path / "idx")
        assert not (tmp_path / "idx" / "embeddings.npy").exists()
        return tmp_path / "idx"

    async def test_it_warns_when_the_backend_is_there_and_vectors_are_not(
        self, tmp_path, caplog, monkeypatch
    ):
        # The whole of Part C: catches the warning being removed, and
        # catches it being logged below WARNING where a default logging
        # config would swallow it.
        path = await self.vectorless_index(tmp_path)
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        warned = warnings_from_load(victim, path, caplog)

        assert len(warned) == 1, warned
        assert MISSING_VECTORS in warned[0]
        assert "re-index" in warned[0]        # names the remedy
        assert str(path) in warned[0]         # names which index
        # Nothing about the load changed: still current, still BM25-only.
        assert victim.retrieval_mode == "bm25-only"
        assert victim._dense_ok is True

    async def test_it_is_silent_when_the_backend_is_absent(
        self, tmp_path, caplog, monkeypatch
    ):
        # BM25-only is the documented mode for a core install. Warning
        # there fires on every load for every user who never wanted dense
        # retrieval, which is how a real warning gets filtered out.
        # Catches the `embedding_backend_available()` condition being
        # dropped.
        path = await self.vectorless_index(tmp_path)
        fake_find_spec(monkeypatch, present=False)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        assert warnings_from_load(victim, path, caplog) == []

    async def test_it_is_silent_when_the_vectors_load(
        self, tmp_path, caplog, monkeypatch
    ):
        # Catches the condition being written without
        # `chunk_embeddings is None` — a warning on a perfectly good
        # hybrid index is worse than no warning at all.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        lib.save(tmp_path / "idx")
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = ConceptEmbedder()
        assert warnings_from_load(victim, tmp_path / "idx", caplog) == []
        assert victim.retrieval_mode == "hybrid"

    async def test_it_does_not_double_report_a_chunk_count_mismatch(
        self, tmp_path, caplog, monkeypatch
    ):
        # The discarding branch leaves `chunk_embeddings` None with
        # `_dense_ok` still true, so without the `discarded` flag the new
        # warning piles onto a path that has just said the same thing
        # better. Catches the flag being removed.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder())
        lib.save(tmp_path / "idx")
        meta = json.loads((tmp_path / "idx" / "index.json").read_text())
        meta["chunks"] = meta["chunks"][:-1]   # vectors now cover one too many
        (tmp_path / "idx" / "index.json").write_text(json.dumps(meta))
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        victim.embed_model = ConceptEmbedder()
        warned = warnings_from_load(victim, tmp_path / "idx", caplog)

        assert len(warned) == 1, warned
        assert "discarding" in warned[0]
        assert MISSING_VECTORS not in warned[0]

    async def test_it_does_not_double_report_a_model_name_mismatch(
        self, tmp_path, caplog, monkeypatch
    ):
        # That path sets `_dense_ok = False` and never loads the vectors,
        # so `chunk_embeddings` is None for a reason already reported.
        # Catches the `_dense_ok` condition being dropped.
        lib = await build(ANSWER_LAST, embedder=ConceptEmbedder(),
                          embed_model="acme/embed-v9")
        lib.save(tmp_path / "idx")
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), embed_model="other/embed-v1",
                                 use_reranker=False)
        warned = warnings_from_load(victim, tmp_path / "idx", caplog)

        assert len(warned) == 1, warned
        assert "not comparable" in warned[0]
        assert MISSING_VECTORS not in warned[0]

    async def test_it_is_silent_for_a_whole_document_index(
        self, tmp_path, caplog, monkeypatch
    ):
        # An index below the chunk threshold holds no vectors by design
        # and never consults the dense half — `retrieve` serves whole
        # documents. The warning's own text ("retrieval is BM25-only ...
        # re-index to use dense retrieval") would be false twice over:
        # the mode is whole-document, and re-indexing the same short text
        # produces no vectors either. Catches the `chunks`/`use_chunking`
        # conditions being dropped as redundant.
        short = SourceLibrarian(MockLLM(), use_reranker=False)
        short._dense_ok = False
        await short.add_document("The bakery sells 120 loaves.", doc_id="b.md")
        assert not short.use_chunking and not short.chunks
        short.save(tmp_path / "idx")
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        assert warnings_from_load(victim, tmp_path / "idx", caplog) == []
        assert victim.retrieval_mode == "whole-document"

    def test_it_is_silent_for_an_index_whose_chunking_is_off(
        self, tmp_path, caplog, monkeypatch
    ):
        # `use_chunking` and `chunks` are separately conditioned, and no
        # index this library writes has one without the other — so the
        # `use_chunking` term is pinned against a hand-written index, the
        # way the format-version tests are. It is the right term to keep:
        # `retrieve` serves whole documents whenever `use_chunking` is
        # false, whatever chunks are lying around, so the dense half is
        # unreachable and there is nothing to warn about.
        directory = tmp_path / "idx"
        directory.mkdir()
        (directory / "index.json").write_text(json.dumps({
            "format": 1,
            "content_hash": "irrelevant",
            "use_chunking": False,
            "embed_model": "",
            "documents": {"policy.md": DOCS["policy.md"]},
            "doc_summaries": {"policy.md": "A summary."},
            "doc_summary": "A summary.",
            "chunks": [{"text": DOCS["policy.md"], "doc_id": "policy.md",
                        "index": 0, "offset": 0, "prefix": "A prefix."}],
        }))
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        warned = warnings_from_load(victim, directory, caplog)

        assert warned == [], warned
        assert victim.chunks and victim.use_chunking is False

    def test_it_is_silent_for_an_empty_index(
        self, tmp_path, caplog, monkeypatch
    ):
        # The natural empty case: an unindexed librarian saved and
        # reloaded. Nothing to re-index, so nothing to say.
        empty = SourceLibrarian(MockLLM(), use_reranker=False)
        empty.save(tmp_path / "idx")
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        assert warnings_from_load(victim, tmp_path / "idx", caplog) == []

    def test_it_is_silent_for_a_chunkless_index_that_claims_chunking(
        self, tmp_path, caplog, monkeypatch
    ):
        # `use_chunking` true over no chunks is "a state nothing else
        # produces" — `_roll_back` normalises it away deliberately — so
        # the `self.chunks` term only ever fires against an index written
        # by something other than this library. Pinned against a
        # hand-written one, the way the format-version tests are, because
        # a redundant-looking condition is exactly what a later cleanup
        # deletes. It is the right term to keep: `retrieval_mode` reads
        # "whole-document" here, so the warning's text would be wrong,
        # and there are no chunks for a re-index to embed.
        directory = tmp_path / "idx"
        directory.mkdir()
        (directory / "index.json").write_text(json.dumps({
            "format": 1,
            "content_hash": "irrelevant",
            "use_chunking": True,
            "embed_model": "",
            "documents": {"policy.md": DOCS["policy.md"]},
            "doc_summaries": {"policy.md": "A summary."},
            "doc_summary": "A summary.",
            "chunks": [],
        }))
        fake_find_spec(monkeypatch, present=True)

        victim = SourceLibrarian(MockLLM(), use_reranker=False)
        warned = warnings_from_load(victim, directory, caplog)

        assert warned == [], warned
        assert victim.use_chunking is True and not victim.chunks

    async def test_the_warning_does_not_change_what_the_cache_does(
        self, tmp_path, monkeypatch
    ):
        # `_load_if_current` still judges a vector-less index current and
        # still serves it — the fix is a log line, not an invalidation.
        # Catches the rejected "fold vector-presence into content_hash"
        # or "re-index on load" alternatives arriving by the back door,
        # either of which would turn a cache hit into one LLM call per
        # chunk.
        first = SourceLibrarian(MockLLM(), use_reranker=False,
                                chunk_threshold=0)
        first._dense_ok = False
        text = DOCS["policy.md"]
        await first.index(text, cache=tmp_path / "idx")
        fake_find_spec(monkeypatch, present=True)

        llm = MockLLM()
        second = SourceLibrarian(llm, use_reranker=False, chunk_threshold=0)
        second.embed_model = ConceptEmbedder()   # never reached on a hit
        await second.index(text, cache=tmp_path / "idx")

        assert llm.calls == []                   # still a hit
        assert second.retrieval_mode == "bm25-only"
        assert second.chunk_embeddings is None


# ── Part B: the extra ────────────────────────────────────────────


class TestTheEmbeddingsExtraDeclaresWhatTheDefaultModelNeeds:
    """The default ``embed_model`` is loaded with ``trust_remote_code``
    and nomic's remote modeling file imports einops. Undeclared, a clean
    ``pip install -e '.[embeddings]'`` installs cleanly and yields
    BM25-only — the failure is in the declaration, so the declaration is
    what gets pinned.
    """

    PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

    @pytest.fixture
    def extras(self):
        if not self.PYPROJECT.exists():
            pytest.skip("running against an installed package, not the repo")
        data = tomllib.loads(self.PYPROJECT.read_text(encoding="utf-8"))
        return data["project"]["optional-dependencies"]

    def test_einops_is_declared_with_a_lower_bound(self, extras):
        # Catches the declaration being reverted, and catches it being
        # added as a bare name — every other extra in this file pins a
        # floor, and a bare name is how a resolver picks something from
        # 2019.
        embeddings = extras["embeddings"]
        einops = [d for d in embeddings
                  if d.split(">=")[0].strip() == "einops"]
        assert einops, embeddings
        assert ">=" in einops[0], einops

    def test_sentence_transformers_is_still_declared(self, extras):
        # The addition must not have replaced what was there.
        assert any(d.startswith("sentence-transformers")
                   for d in extras["embeddings"]), extras["embeddings"]
