"""The index and retrieval knobs — that they are settable, and that
setting them changes what the code actually does.

Every one of these was a module constant read at the point of use, so a
librarian tuned for one model could not coexist with a librarian tuned
for another in the same process. The tests here are less about the
individual numbers than about that: the value the constructor was given
is the value the running code reads, all the way down to the hash that
decides whether a saved index is reusable.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re

import numpy as np
import pytest

from librarium import Librarian, LibrarianRouter, SourceLibrarian
from librarium.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_spans
from librarium.source import (
    EXPAND_QUERY,
    EXPANSION_STYLE,
    EXPANSION_VARIANTS,
    MAX_CONTEXT_TOKENS,
    RRF_K,
    check_expansion_settings,
    check_index_settings,
    content_hash,
)
from tests.conftest import MockLLM, stub_models

# Long enough to chunk several times over at any geometry these tests
# use, and made of distinguishable paragraphs so a chunk boundary in the
# wrong place is visible in the text rather than only in a count.
DOC = "\n\n".join(
    f"Paragraph {i} concerns subject {i} and repeats itself. " * 8
    for i in range(24)
)

#: The three parameters `check_index_settings` takes and does not return,
#: per its own docstring: `top_k` and `use_reranker` are validated here and
#: stored elsewhere, and `warn` is not a setting at all. Named rather than
#: derived from the returned dict, because deriving it from the thing under
#: test would make the completeness assertion in
#: `TestModuleConstantsAreOnlyDefaults
#: ::test_check_index_settings_normalises_the_defaults` compare a set with
#: itself. A fourth such parameter fails that test loudly, which is where it
#: should be noticed.
READ_NOT_RETURNED = frozenset({"top_k", "use_reranker", "warn"})

#: Every index setting and its default, **derived from the signature that
#: declares them** rather than written out here.
#:
#: It was a hand-maintained dict of nine name/default pairs, and it went
#: stale the moment `rerank_pool` was added — a closed enumeration in the
#: sense that matters here: nothing in it was untrue,
#: nobody had to edit it for it to go wrong, and the set simply grew. It
#: failed loudly rather than silently, which is better than the enumerations
#: that rule was written for, but the fix is the one the rule names: point at
#: the code that enumerates the knobs. `check_index_settings`'s signature is
#: that code, so a knob added there now reaches all three tests below
#: without anybody remembering to extend a list.
#:
#: This strengthens the tests rather than relaxing them. A knob declared and
#: not returned fails the completeness test; one returned but never stored on
#: the librarian fails the defaults test; one the facade grows a *default*
#: for that drifts from `SourceLibrarian`'s fails the parity test. Under the
#: hand list all three passed silently until somebody added the string.
#:
#: What none of the three catches is a knob the facade declares and then
#: forgets to **forward** — the parity test compares two default-built
#: librarians, so an unforwarded knob reads the same default on both sides
#: and the comparison is blind to it. An earlier version of this paragraph
#: claimed that test covered forwarding; measured false, by deleting
#: `rerank_pool=rerank_pool` from all three layers and watching the suite
#: stay green at 930 while `Librarian(llm, rerank_pool=25).source.rerank_pool`
#: read `None`. `TestForwarding::test_the_facade_forwards_every_setting` is
#: what covers it, and that list is hand-maintained — see the note there.
ALL_SETTINGS = {
    name: parameter.default
    for name, parameter in
    inspect.signature(check_index_settings).parameters.items()
    if name not in READ_NOT_RETURNED
}


class StubReranker:
    """Cross-encoder stand-in that scores a chunk by the paragraph it
    came from.

    Content-derived on purpose, like every other double here: a constant
    score would let the reranker return a correctly-sized list of the
    wrong chunks. Scoring on the paragraph number rather than on, say,
    the chunk's length also spreads the scores out, which is what lets a
    threshold test cut the set somewhere real.
    """

    _SUBJECT_RE = re.compile(r"subject (\d+)")

    def predict(self, pairs):
        return np.array(
            [self.score(text) for _, text in pairs], dtype=np.float32
        )

    @classmethod
    def score(cls, text: str) -> float:
        # One-based, so paragraph 0 does not score 0.0 and get cut by the
        # default rerank_threshold in tests that are not about the
        # threshold at all.
        match = cls._SUBJECT_RE.search(text)
        return float(match.group(1)) + 1.0 if match else 0.0


def indexed(**settings) -> SourceLibrarian:
    """A librarian that has indexed DOC under *settings*."""
    lib = SourceLibrarian(MockLLM(), **settings)
    lib._dense_ok = False  # as if the embeddings extra were absent
    asyncio.run(lib.index(DOC))
    return lib


class TestChunkGeometry:
    def test_chunk_size_and_overlap_reach_the_chunker(self):
        # Compared against chunk_spans directly rather than against a
        # chunk count: the count would also match if the geometry arrived
        # half-applied — the size honoured and the overlap dropped.
        lib = indexed(chunk_size=700, chunk_overlap=100)
        assert [c.text for c in lib.chunks] == [
            body for body, _ in chunk_spans(DOC, 700, 100)
        ]

    def test_a_smaller_chunk_size_produces_more_chunks(self):
        assert len(indexed(chunk_size=600).chunks) > len(indexed().chunks)

    def test_every_chunk_stays_within_the_configured_size(self):
        assert all(len(c.text) <= 600 for c in indexed(chunk_size=600).chunks)

    def test_overlap_at_or_above_chunk_size_is_refused(self):
        # chunk_spans() clamps this rather than raising, which is right
        # for a library call handed a computed value and wrong for a
        # setting: the caller would get geometry they never asked for and
        # keep it for the life of the librarian.
        with pytest.raises(ValueError, match="chunk_overlap must be <"):
            SourceLibrarian(MockLLM(), chunk_size=500, chunk_overlap=500)


class TestChunkThreshold:
    def test_raising_it_holds_a_document_whole(self):
        lib = indexed(chunk_threshold=len(DOC) + 1)
        assert lib.retrieval_mode == "whole-document"
        assert lib.chunks == []

    def test_zero_chunks_even_a_short_document(self):
        lib = SourceLibrarian(MockLLM(), chunk_threshold=0)
        lib._dense_ok = False
        asyncio.run(lib.index("One short line, indexed anyway."))
        assert lib.use_chunking and lib.chunks

    def test_the_whole_document_path_still_costs_no_llm_calls(self):
        # The point of the threshold, not an incidental property of it.
        llm = MockLLM()
        lib = SourceLibrarian(llm, chunk_threshold=len(DOC) + 1)
        asyncio.run(lib.index(DOC))
        assert llm.calls == []


class TestContextEstimate:
    def test_chars_per_token_moves_the_budget(self):
        # 400 chars of prompt: ~100 tokens at 4 chars each, ~400 at one.
        prompt = "x" * 400
        lenient = SourceLibrarian(
            MockLLM(), max_context_tokens=200, reserved_output_tokens=0,
            chars_per_token=4,
        )
        lenient.check_context(prompt, "")  # fits

        strict = SourceLibrarian(
            MockLLM(), max_context_tokens=200, reserved_output_tokens=0,
            chars_per_token=1,
        )
        with pytest.raises(ValueError, match="exceeds max_context_tokens"):
            strict.check_context(prompt, "")

    def test_the_error_names_a_setting_that_exists(self):
        # It used to advise editing CHUNK_SIZE in chunking.py, which is
        # not something a caller of this library should have to do.
        lib = SourceLibrarian(MockLLM(), max_context_tokens=10)
        with pytest.raises(ValueError) as excinfo:
            lib.check_context("x" * 4000, "")
        assert "chunk_size" in str(excinfo.value)
        assert "chunking.py" not in str(excinfo.value)


class TestFusionAndRerank:
    def test_rrf_k_reaches_the_fusion(self, monkeypatch):
        seen = {}

        import librarium.source as source

        real = source.reciprocal_rank_fusion

        def spy(rankings, k=RRF_K):
            seen["k"] = k
            return real(rankings, k)

        monkeypatch.setattr(source, "reciprocal_rank_fusion", spy)

        lib = SourceLibrarian(MockLLM(), rrf_k=7)
        # Both halves must rank, or retrieve() never fuses.
        lib.embed_model = _MarkEmbedder()
        asyncio.run(lib.index(DOC))
        lib.retrieve("subject 3")

        assert seen["k"] == 7

    @staticmethod
    def reranking(monkeypatch, **settings) -> SourceLibrarian:
        """A librarian with a stub reranker and more chunks than *top_k*.

        The small chunk_size is what makes the cap observable: at the
        stock geometry DOC yields fewer chunks than these top_k values,
        so a reranker that capped nothing would still return a short list
        and every assertion below would pass for the wrong reason.

        The stub goes in *before* `indexed()` rather than being assigned
        over the top afterwards. `use_reranker=True` makes `index()` call
        `_load_reranker`, so the old ordering downloaded and built a real
        `BAAI/bge-reranker-v2-m3` on every one of these tests and then
        threw it away unused — five real models per run out of a suite
        documented as needing none. Supplying it through the loader's own
        import is the same seam `test_stale_vectors.stub_cross_encoder`
        uses, and it makes the load path part of the setup rather than
        something worked around: `StubReranker` is what ranks below
        either way.
        """
        stub_models(monkeypatch, reranker=StubReranker())
        lib = indexed(chunk_size=600, use_reranker=True, **settings)
        assert len(lib.chunks) > settings.get("top_k", 10)
        # The librarian ranks through the double, not through anything it
        # fetched: if the stub ever stopped reaching the loader this would
        # be a real CrossEncoder and the assertions below would be
        # measuring a model instead of the setting.
        assert isinstance(lib.reranker, StubReranker)
        return lib

    def test_rerank_top_caps_what_retrieval_returns(self, monkeypatch):
        lib = self.reranking(monkeypatch, top_k=8, rerank_top=3)
        assert len(lib.retrieve("subject 3")) == 3

    def test_raising_rerank_top_stops_the_silent_truncation(self, monkeypatch):
        # The documented trap: at the stock rerank_top a top_k above it
        # returns rerank_top chunks and says nothing. Raising it is the fix
        # — and it has to actually reach the reranker for that to be true.
        lib = self.reranking(monkeypatch, top_k=8, rerank_top=8)
        assert len(lib.retrieve("subject 3")) == 8

    def test_rerank_threshold_drops_weak_matches(self, monkeypatch):
        # The cut is taken from the scores the stub actually produces for
        # the chunks retrieval selects, so the test does not depend on
        # which chunks BM25 happens to like.
        lenient = self.reranking(monkeypatch, top_k=8, rerank_top=8,
                                 rerank_threshold=-1.0)
        kept = lenient.retrieve("subject 3")
        scores = [StubReranker.score(c.text) for c in kept]
        assert len(set(scores)) > 1, "the cut below would be all-or-nothing"

        strict = self.reranking(monkeypatch, top_k=8, rerank_top=8,
                                rerank_threshold=float(np.median(scores)))
        assert len(strict.retrieve("subject 3")) < len(kept)

    def test_a_threshold_nothing_clears_still_returns_chunks(self, monkeypatch):
        # Falling back to the top rerank_top is deliberate: an empty
        # retrieval leaves the model answering from nothing.
        lib = self.reranking(monkeypatch, top_k=8, rerank_top=4,
                             rerank_threshold=1e9)
        assert len(lib.retrieve("subject 3")) == 4


class TestRerankPool:
    """`rerank_pool` is the reranker's *input* width; `rerank_top` its output.

    The cross-encoder used to reorder a set it could not add to: the fusion
    cut to `top_k` and only then reranked, so a chunk RRF ranked eleventh was
    gone before the model best placed to judge it was asked. `rerank_pool`
    moves that cut.

    **Nothing here asserts that a wider pool retrieves better, because it does
    not.** Measured on one index with a flat control, `recall@10` went *down*
    (−0.5 at pool 30, −1.6 at pool 50) and facts found fell, while `@5` rose
    and mean rank improved — the cross-encoder orders better over a slightly
    worse set, and nothing measured recommends setting this (evidence:
    internal benchmark runs). The knob ships defaulting off,
    so what these tests defend is the *mechanism* and the default, not a
    benefit.

    Catches, one test per way this can go wrong, each verified to fail alone:
    the default ceasing to follow `top_k`; the pool being applied after the
    truncation instead of before, which would silently restore the old
    behaviour; widening on a path with no reranker, where the extra
    candidates would come back unranked; and the construction warning going
    quiet. That list is a closed enumeration, so a fifth condition on this
    knob extends it in the same commit that adds one.
    """

    class Recording(StubReranker):
        """A `StubReranker` that also keeps every batch it was handed.

        Subclassed rather than replaced so the scores stay content-derived:
        a test can say both *how many* candidates the cross-encoder saw and
        *which*, which is what makes the pool observable at all — the pool
        is an input width, and the output is capped by `rerank_top` on top
        of it.
        """

        def __init__(self):
            self.batches: list[list[list[str]]] = []

        def predict(self, pairs):
            self.batches.append([list(pair) for pair in pairs])
            return super().predict(pairs)

        def candidates(self) -> list[str]:
            """The chunk texts scored by the most recent call."""
            assert self.batches, "the reranker was never asked to score"
            return [text for _, text in self.batches[-1]]

    @staticmethod
    def pooled(monkeypatch, **settings):
        """A librarian whose reranker records what it is given."""
        recorder = TestRerankPool.Recording()
        stub_models(monkeypatch, reranker=recorder)
        # 300 and not the 600 the sibling class uses: DOC's paragraphs are
        # ~416 characters, so at 600 each is its own chunk and the corpus is
        # exactly 24 — fewer than the widest pool below, which would make a
        # pool that widened nothing indistinguishable from one that did.
        # Halving it splits each paragraph in two without changing which
        # subject a chunk names, so the reranker still scores on content.
        lib = indexed(chunk_size=300, use_reranker=True, **settings)
        assert len(lib.chunks) > 25, (
            f"only {len(lib.chunks)} chunks — a pool of 25 cannot be shown "
            f"to widen anything against a corpus this small"
        )
        assert lib.reranker is recorder
        return lib, recorder

    def test_the_default_hands_the_reranker_exactly_top_k(self, monkeypatch):
        # The behaviour-preservation claim, which is the whole basis on
        # which this shipped: an unset `rerank_pool` leaves the cut where it
        # has always been. Asserted on what reaches the cross-encoder rather
        # than on the result, because `rerank_top` caps the result and would
        # hide a wider pool.
        lib, recorder = self.pooled(monkeypatch, top_k=8, rerank_top=8)
        assert lib.rerank_pool is None, "this test is about the default"

        lib.retrieve("subject 3")

        assert len(recorder.candidates()) == 8

    def test_the_pool_widens_what_the_reranker_is_given(self, monkeypatch):
        lib, recorder = self.pooled(monkeypatch, top_k=8, rerank_top=8,
                                    rerank_pool=25)
        lib.retrieve("subject 3")
        assert len(recorder.candidates()) == 25

    def test_a_chunk_below_top_k_can_reach_the_result(self, monkeypatch):
        """The mechanism, and the one that fails if the pool is applied late.

        A pool applied *after* the truncation reorders the same eight chunks
        and can add nothing, so the wide run's result would be a permutation
        of the narrow run's candidates. Asserted as set membership rather
        than as an ordering, because the claim is that a chunk RRF ranked
        outside `top_k` *arrives*, not where it lands.

        `StubReranker` scores by paragraph number while BM25 ranks on the
        query text, so the two orderings genuinely disagree — which is what
        makes a wider pool able to surface something the narrow one never
        offered.
        """
        narrow, narrow_rec = self.pooled(monkeypatch, top_k=5, rerank_top=5)
        narrow_result = narrow.retrieve("subject 3")
        offered = set(narrow_rec.candidates())
        assert len(offered) == 5

        wide, wide_rec = self.pooled(monkeypatch, top_k=5, rerank_top=5,
                                     rerank_pool=25)
        wide_result = wide.retrieve("subject 3")
        assert len(wide_rec.candidates()) == 25

        # The chunks are the same corpus, so anything in the wide result that
        # the narrow run was never offered arrived through the wider pool.
        arrived = {c.text for c in wide_result} - offered
        assert arrived, (
            "the wider pool returned nothing the narrow one was not already "
            "given — the cut is being applied before the pool rather than "
            "after"
        )
        # And it beat the narrow winner on the cross-encoder's own scale,
        # which is why widening surfaced it rather than luck of the ordering.
        assert max(StubReranker.score(t) for t in arrived) > max(
            StubReranker.score(c.text) for c in narrow_result
        )

    def test_no_reranker_means_no_widening(self, monkeypatch):
        # Fusing 25 and returning 25 unranked is the waste the gate exists
        # to prevent: with nothing to rank them, the extra candidates would
        # come straight back as results.
        lib = indexed(chunk_size=600, use_reranker=False, top_k=5,
                      rerank_pool=25)
        assert lib.reranker is None
        assert len(lib.retrieve("subject 3")) == 5

    def test_a_pool_below_top_k_widens_nothing(self, monkeypatch):
        # `max(pool, top_k)` — the setting is a floor on the input width and
        # cannot narrow it below the answer width. The construction warning
        # below is what tells a caller their setting did nothing.
        lib, recorder = self.pooled(monkeypatch, top_k=8, rerank_top=8,
                                    rerank_pool=3)
        lib.retrieve("subject 3")
        assert len(recorder.candidates()) == 8


class TestRerankPoolWarning:
    """A pool below `top_k` widens nothing, so construction says so.

    The mirror of `TestRerankTopWarning`, and the likely cause is the same
    confusion: `rerank_pool` is how many candidates the cross-encoder is
    *given*, `rerank_top` how many it hands *back*. A pool under `top_k` is
    silently inert — `retrieve()` takes `max(pool, top_k)` — and silence is
    what makes it worth a warning.
    """

    def test_warns_when_the_pool_is_below_top_k(self, caplog):
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=True, top_k=10,
                            rerank_pool=4)
        assert "rerank_pool=4" in caplog.text
        assert "top_k=10" in caplog.text
        # It has to name the other knob, because confusing the two is the
        # reason a caller lands here.
        assert "rerank_top" in caplog.text

    def test_quiet_when_the_pool_widens(self, caplog):
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=True, top_k=10,
                            rerank_top=10, rerank_pool=25)
        assert "rerank_pool" not in caplog.text

    def test_quiet_at_the_default(self, caplog):
        # `None` means "follow top_k", which no caller chose, so there is
        # nothing to tell them about.
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=True, top_k=10,
                            rerank_top=10)
        assert "rerank_pool" not in caplog.text

    def test_quiet_when_the_reranker_is_off(self, caplog):
        # The pool is only read when a reranker is loaded, so the advice
        # would be about a code path this librarian never takes.
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=False, top_k=10,
                            rerank_pool=4)
        assert "rerank_pool" not in caplog.text


class TestRerankTopWarning:
    """`rerank_top` below `top_k` discards chunks that were retrieved and
    paid for. Configurable is not enough on its own — the default pair
    still does it, so it has to say so.
    """

    def test_warns_when_the_reranker_will_discard_retrieved_chunks(self, caplog):
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=True, top_k=10,
                            rerank_top=5)
        assert "rerank_top" in caplog.text
        assert "top_k=10" in caplog.text

    def test_quiet_when_the_reranker_keeps_everything(self, caplog):
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=True, top_k=5,
                            rerank_top=5)
        assert "rerank_top" not in caplog.text

    def test_quiet_when_the_reranker_is_off(self, caplog):
        # Nothing is discarded, so the advice would be about a code path
        # this librarian never takes.
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), top_k=10, rerank_top=5)
        assert "rerank_top" not in caplog.text

    def test_suppressible_for_a_librarian_whose_owner_already_warned(self, caplog):
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), use_reranker=True, top_k=10,
                            rerank_top=5, _warn_settings=False)
        assert "rerank_top" not in caplog.text

    def test_suppression_does_not_reach_the_validation(self):
        # The flag silences advice, never a check — a bad value passed by
        # a lazily building layer must still raise.
        with pytest.raises(ValueError, match="rerank_top"):
            SourceLibrarian(MockLLM(), rerank_top=0, _warn_settings=False)


class TestModelNames:
    def test_the_embedding_model_name_reaches_the_loader(self, monkeypatch):
        loaded = self._stub_sentence_transformers(monkeypatch)
        lib = SourceLibrarian(MockLLM(), embed_model="acme/embed-v9")
        lib._ensure_embed_model()
        assert loaded["SentenceTransformer"] == "acme/embed-v9"

    def test_the_reranker_model_name_reaches_the_loader(self, monkeypatch):
        loaded = self._stub_sentence_transformers(monkeypatch)
        lib = SourceLibrarian(MockLLM(), use_reranker=True,
                              rerank_model="acme/rerank-v9")
        lib._load_reranker()
        assert loaded["CrossEncoder"] == "acme/rerank-v9"

    def test_the_configured_name_survives_the_model_being_loaded(self, monkeypatch):
        # `embed_model` holds the loaded SentenceTransformer, so the name
        # has to live somewhere the load does not overwrite — otherwise
        # content_hash() and save() see a model object where a name
        # belongs, the moment the dense half comes up.
        self._stub_sentence_transformers(monkeypatch)
        lib = SourceLibrarian(MockLLM(), embed_model="acme/embed-v9")
        lib._ensure_embed_model()
        assert lib.embed_model_name == "acme/embed-v9"
        assert lib.embed_model is not None

    @staticmethod
    def _stub_sentence_transformers(monkeypatch) -> dict:
        """Record the model name each loader is called with."""
        import sys
        import types

        loaded: dict[str, str] = {}
        module = types.ModuleType("sentence_transformers")

        def make(kind):
            def factory(name, **kwargs):
                loaded[kind] = name
                return object()
            return factory

        module.SentenceTransformer = make("SentenceTransformer")
        module.CrossEncoder = make("CrossEncoder")
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return loaded


class TestContentHash:
    """The hash decides whether a saved index is reusable, so it has to
    follow the settings that shaped that index — per librarian, not per
    module. Two differently-tuned librarians over the same text otherwise
    load each other's chunks.
    """

    @pytest.mark.parametrize(
        "setting,value",
        [("chunk_size", 1234), ("chunk_overlap", 42),
         ("chunk_threshold", 4321), ("embed_model", "acme/embed-v9")],
    )
    def test_every_index_shaping_setting_changes_the_hash(self, setting, value):
        docs = {"d": DOC}
        assert content_hash(docs) != content_hash(docs, **{setting: value})

    def test_a_librarian_hashes_with_its_own_settings(self):
        lib = SourceLibrarian(MockLLM(), chunk_size=1234)
        lib.documents = {"d": DOC}
        assert lib.content_hash() == content_hash({"d": DOC}, chunk_size=1234)

    def test_a_cache_is_reused_at_the_same_geometry(self, tmp_path,
                                                    monkeypatch):
        # The dense half stays *on* here — `indexed()`'s `_dense_ok =
        # False` would take the embeddings out of the round trip these
        # save and load, which is half of what a cache holds. It is
        # supplied rather than downloaded: without the stub each of these
        # librarians built a real nomic embedder.
        stub_models(monkeypatch)
        first = MockLLM()
        asyncio.run(SourceLibrarian(first, chunk_size=700).index(
            DOC, cache=tmp_path))
        assert first.calls  # it did the indexing work

        second = MockLLM()
        asyncio.run(SourceLibrarian(second, chunk_size=700).index(
            DOC, cache=tmp_path))
        assert second.calls == []  # and the second run did not repeat it

    def test_a_cache_is_not_reused_at_a_different_geometry(self, tmp_path,
                                                           monkeypatch):
        stub_models(monkeypatch)
        asyncio.run(SourceLibrarian(MockLLM(), chunk_size=700).index(
            DOC, cache=tmp_path))

        llm = MockLLM()
        lib = SourceLibrarian(llm, chunk_size=1400)
        asyncio.run(lib.index(DOC, cache=tmp_path))

        # Re-indexed rather than loaded: the cached chunks were cut to a
        # geometry this librarian did not ask for.
        assert llm.calls
        assert [c.text for c in lib.chunks] == [
            body for body, _ in chunk_spans(DOC, 1400, CHUNK_OVERLAP)
        ]


class TestPersistedEmbedModel:
    def test_save_records_the_librarian_s_embedding_model(self, tmp_path):
        lib = SourceLibrarian(MockLLM(), embed_model="acme/embed-v9")
        lib.embed_model = _MarkEmbedder()
        asyncio.run(lib.index(DOC))
        lib.save(tmp_path)

        meta = json.loads((tmp_path / "index.json").read_text())
        assert meta["embed_model"] == "acme/embed-v9"

    def test_loading_into_a_differently_embedded_librarian_warns(
        self, tmp_path, caplog
    ):
        lib = SourceLibrarian(MockLLM(), embed_model="acme/embed-v9")
        lib.embed_model = _MarkEmbedder()
        asyncio.run(lib.index(DOC))
        lib.save(tmp_path)

        other = SourceLibrarian(MockLLM(), embed_model="other/embed-v1")
        with caplog.at_level("WARNING", logger="librarium.source"):
            other.load(tmp_path)

        assert "not comparable" in caplog.text
        # Vectors from another model are worse than none: the dense half
        # stays off rather than ranking on nonsense.
        assert other._dense_ok is False
        assert other.retrieval_mode == "bm25-only"


class TestValidation:
    """Every knob is checked at construction, because the alternative is
    a stack trace from inside numpy, a division by zero, or — worst —
    a cache that silently never hits.
    """

    @pytest.mark.parametrize("setting", [
        "chunk_size", "chunk_overlap", "chunk_threshold", "rrf_k",
        "rerank_top", "chars_per_token",
    ])
    @pytest.mark.parametrize("bad", ["8", 8.5, None, True, False])
    def test_a_non_int_setting_gets_an_error_naming_it(self, setting, bad):
        # bool is refused for the reason it is everywhere else here: it is
        # an int as far as Python is concerned, but True as a chunk size
        # is a typo, not a request for one character.
        with pytest.raises(TypeError, match=setting):
            SourceLibrarian(MockLLM(), **{setting: bad})

    @pytest.mark.parametrize("setting,bad", [
        ("chunk_size", 0), ("chunk_overlap", -1), ("chunk_threshold", -1),
        # rrf_k divides by (k + rank), so 0 is a ZeroDivisionError on the
        # first-ranked chunk; chars_per_token divides directly.
        ("rrf_k", 0), ("rerank_top", 0), ("chars_per_token", 0),
    ])
    def test_an_out_of_range_setting_gets_an_error_naming_it(self, setting, bad):
        with pytest.raises(ValueError, match=setting):
            SourceLibrarian(MockLLM(), **{setting: bad})

    @pytest.mark.parametrize("setting", ["embed_model", "rerank_model"])
    def test_a_non_string_model_name_is_refused(self, setting):
        with pytest.raises(TypeError, match=setting):
            SourceLibrarian(MockLLM(), **{setting: 42})

    @pytest.mark.parametrize("setting", ["embed_model", "rerank_model"])
    def test_an_empty_model_name_is_refused(self, setting):
        # It reads as "no model" but reaches the loader, which then fails
        # deep inside a download path instead of here.
        with pytest.raises(ValueError, match=setting):
            SourceLibrarian(MockLLM(), **{setting: "  "})

    def test_a_non_float_rerank_threshold_is_refused(self):
        with pytest.raises(TypeError, match="rerank_threshold"):
            SourceLibrarian(MockLLM(), rerank_threshold="0.5")

    @pytest.mark.parametrize("setting,value", [
        ("chunk_size", 900), ("chunk_overlap", 30), ("chunk_threshold", 1500),
        ("rrf_k", 7), ("rerank_top", 3), ("chars_per_token", 3),
    ])
    def test_numpy_integers_are_accepted_and_normalised(self, setting, value):
        # numpy is a hard dependency, so a setting computed from an array
        # arrives as np.int64 and is a reasonable thing to pass. It is
        # normalised at the boundary because it goes on to slice lists and
        # to be rendered into log lines.
        lib = SourceLibrarian(MockLLM(), **{setting: np.int64(value)})
        assert getattr(lib, setting) == value
        assert type(getattr(lib, setting)) is int

    def test_a_negative_rerank_threshold_is_allowed(self):
        # Cross-encoder scores are unbounded logits in some models, so a
        # negative cut-off is a real configuration, not a mistake.
        assert SourceLibrarian(MockLLM(), rerank_threshold=-2.5) \
            .rerank_threshold == -2.5


class TestContextBudget:
    """`max_context_tokens` and `reserved_output_tokens` decide whether an
    answer call is allowed at all, and the two layers that build a
    librarian lazily used to drop both — a router aimed at a large-context
    model capped every answer at the small-model default.
    """

    def test_the_shared_default_is_one_number(self):
        # Four constructors default to it. As a literal in each signature
        # three could agree with the fourth by coincidence and drift apart
        # on the next edit.
        assert SourceLibrarian(MockLLM()).max_context_tokens \
            == MAX_CONTEXT_TOKENS
        assert Librarian(MockLLM()).source.max_context_tokens \
            == MAX_CONTEXT_TOKENS

    @pytest.mark.parametrize("setting", [
        "max_context_tokens", "reserved_output_tokens",
    ])
    @pytest.mark.parametrize("bad", ["8", 8.5, None, True])
    def test_a_non_int_budget_is_refused(self, setting, bad):
        with pytest.raises(TypeError, match=setting):
            SourceLibrarian(MockLLM(), **{setting: bad})

    def test_a_zero_context_budget_is_refused(self):
        with pytest.raises(ValueError, match="max_context_tokens"):
            SourceLibrarian(MockLLM(), max_context_tokens=0)

    def test_a_negative_reserve_is_refused(self):
        with pytest.raises(ValueError, match="reserved_output_tokens"):
            SourceLibrarian(MockLLM(), reserved_output_tokens=-1)

    def test_a_reserve_that_fills_the_budget_warns_rather_than_raises(
        self, caplog
    ):
        # Refusing it would make check_context's own failure impossible to
        # reproduce through the public API, which is how several tests here
        # reach that error at all.
        with caplog.at_level("WARNING", logger="librarium.source"):
            lib = SourceLibrarian(MockLLM(), max_context_tokens=100,
                                  reserved_output_tokens=100)
        assert "no prompt can fit" in caplog.text
        with pytest.raises(ValueError, match="exceeds max_context_tokens"):
            lib.check_context("anything at all", "")


class TestForwarding:
    """A knob nobody can reach through the facade is not configurable.
    Each layer either forwards these or builds a librarian that ignores
    them.
    """

    #: A probe value per knob, each meaningfully different from that knob's
    #: default so a dropped forwarding line shows up as the default coming
    #: back rather than as a coincidence.
    #:
    #: **Hand-maintained, and knowingly so** — this is the same closed
    #: enumeration shape as the `ALL_SETTINGS` dict above, which was derived
    #: this cycle for exactly that reason, and it is left as a list because
    #: deriving it is a harder problem than that one was rather than because
    #: nobody looked. `ALL_SETTINGS` needs only the knob *names*, which the
    #: signature already holds; this needs a *valid* value per knob, and
    #: validity is per-knob knowledge the signature does not carry:
    #: `_check_int_setting` enforces a different minimum for each, `rerank_pool`
    #: defaults to `None` so its type comes from an annotation rather than
    #: from the default, and the first knob with an upper bound or a fixed
    #: set of allowed strings would make a "default + 1" rule produce
    #: invalid probes and fail this test for a reason that has nothing to do
    #: with forwarding.
    #:
    #: So the obligation travels with the list: **a knob added to
    #: `check_index_settings` is added here in the same commit.**
    #: `test_every_knob_is_declared_by_every_layer` is the derived half that
    #: fails if you forget, and it says what it cannot see.
    @pytest.mark.parametrize("setting,value", [
        ("chunk_size", 700), ("chunk_overlap", 50), ("chunk_threshold", 1500),
        ("rrf_k", 7), ("rerank_top", 9), ("rerank_threshold", 0.5),
        ("chars_per_token", 3), ("embed_model", "acme/embed-v9"),
        ("rerank_model", "acme/rerank-v9"), ("rerank_pool", 25),
    ])
    def test_the_facade_forwards_every_setting(self, setting, value):
        lib = Librarian(MockLLM(), **{setting: value})
        attr = "embed_model_name" if setting == "embed_model" else setting
        assert getattr(lib.source, attr) == value

    def test_every_knob_is_declared_by_every_layer(self):
        """Every index setting is a parameter on each layer that carries one.

        The derived half of the forwarding contract, and the half the probe
        list above cannot be: this needs only the knob *names*, which
        `check_index_settings`'s signature already holds, so a knob added
        there reaches this test without anybody extending anything.

        **What it does not catch is the bug that produced it.** A parameter
        can be declared and then never passed on — `rerank_pool` was, at all
        three layers, and the suite stayed green at 930 because the only test
        that would have noticed is the hand-maintained sweep above and the
        knob was missing from it. Declaration and forwarding are two claims;
        this covers the first, `test_the_facade_forwards_every_setting` the
        second, and neither substitutes for the other. What this does catch
        is somebody adding a knob to the validator and forgetting the facade
        entirely — at which point the probe list would need an entry too, and
        this fails first and says which knob.

        `WebLibrarian` is absent because importing it needs the `web` extra,
        which this suite runs without; `test_indexing.py`'s
        `TestWebLibrarianIndexSettings` covers that layer behind a module
        stub. Three layers of four, said rather than implied.
        """
        layers = (SourceLibrarian, Librarian, LibrarianRouter)
        missing = {
            layer.__name__: sorted(
                set(ALL_SETTINGS)
                - set(inspect.signature(layer.__init__).parameters)
            )
            for layer in layers
        }
        assert not any(missing.values()), (
            f"knobs `check_index_settings` validates that a layer does not "
            f"accept, so no caller can set them there: "
            f"{ {k: v for k, v in missing.items() if v} }"
        )

    def test_the_facade_defaults_match_the_source_librarian(self):
        facade = Librarian(MockLLM()).source
        direct = SourceLibrarian(MockLLM())
        for setting in ALL_SETTINGS:
            attr = "embed_model_name" if setting == "embed_model" else setting
            assert getattr(facade, attr) == getattr(direct, attr), setting

    def test_the_settings_cannot_be_passed_positionally(self):
        # They trail eight public arguments on Librarian. As
        # positional-or-keyword parameters a caller who miscounted would
        # land a stray value on one and quietly re-cut every chunk.
        with pytest.raises(TypeError, match="positional argument"):
            Librarian(MockLLM(), False, 6000, True, 512, False, False, 8,
                      10, 700)

    def test_the_router_forwards_to_the_librarian_it_builds(self, monkeypatch):
        # The librarian is built inside index(), so there is no handle to
        # switch the dense half off on — which is why this path was one
        # of the sixteen that loaded a real embedder. The stub keeps the
        # librarian fully assembled, exactly as the assertions below
        # assume, without the download.
        stub_models(monkeypatch)
        router = LibrarianRouter(MockLLM(), use_web=False, chunk_size=700,
                                 chars_per_token=3)
        assert router.source is None  # nothing injected: built on index()

        asyncio.run(router.index(DOC))

        assert router.source.chunk_size == 700
        assert router.source.chars_per_token == 3
        assert [c.text for c in router.source.chunks] == [
            body for body, _ in chunk_spans(DOC, 700, CHUNK_OVERLAP)
        ]

    def test_the_router_forwards_the_context_budget(self, monkeypatch):
        # The setting the router dropped that costs the most: every answer
        # it produced was measured against the small-model default.
        stub_models(monkeypatch)
        router = LibrarianRouter(MockLLM(), use_web=False,
                                 max_context_tokens=32_000,
                                 reserved_output_tokens=2048, strict=True)
        asyncio.run(router.index(DOC))

        assert router.source.max_context_tokens == 32_000
        assert router.source.reserved_output_tokens == 2048
        assert router.source.strict is True

    def test_the_router_validates_the_budget_before_it_builds_anything(self):
        with pytest.raises(TypeError, match="max_context_tokens"):
            LibrarianRouter(MockLLM(), use_web=False,
                            max_context_tokens="32000")

    def test_the_router_validates_before_it_builds_anything(self):
        # The librarian carrying the setting does not exist until the
        # first index(), so without a check here the mistake surfaces a
        # long way from the call that made it.
        with pytest.raises(ValueError, match="chunk_size"):
            LibrarianRouter(MockLLM(), use_web=False, chunk_size=0)

    def test_the_router_says_when_an_injected_librarian_ignores_them(
        self, caplog
    ):
        injected = SourceLibrarian(MockLLM(), chunk_size=1400)
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(MockLLM(), source_librarian=injected,
                            use_web=False, chunk_size=700)

        # Silence would leave the caller believing 700 is in force.
        assert "chunk_size" in caplog.text
        assert "is ignored" in caplog.text
        assert injected.chunk_size == 1400

    def test_the_router_names_every_setting_an_injection_makes_inert(
        self, caplog
    ):
        injected = SourceLibrarian(MockLLM())
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(MockLLM(), source_librarian=injected,
                            use_web=False, max_context_tokens=32_000,
                            strict=True, use_reranker=True, rrf_k=3)

        for setting in ("max_context_tokens", "strict", "use_reranker",
                        "rrf_k"):
            assert setting in caplog.text, setting

    def test_the_router_is_quiet_about_defaults_nobody_passed(self, caplog):
        injected = SourceLibrarian(MockLLM(), chunk_size=1400)
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(MockLLM(), source_librarian=injected,
                            use_web=False)
        assert "ignored" not in caplog.text


class TestModuleConstantsAreOnlyDefaults:
    def test_rebinding_a_constant_does_not_reach_a_built_librarian(
        self, monkeypatch
    ):
        # The failure this replaces: the running code read the module
        # constant, so a librarian's geometry was whatever the module said
        # at the moment a chunk was cut — settable from anywhere, by
        # anyone, for everyone.
        lib = SourceLibrarian(MockLLM(), chunk_size=700)
        monkeypatch.setattr("librarium.chunking.CHUNK_SIZE", 5000)
        monkeypatch.setattr("librarium.source.CHUNK_SIZE", 5000)
        lib._dense_ok = False
        asyncio.run(lib.index(DOC))

        assert lib.chunk_size == 700
        assert all(len(c.text) <= 700 for c in lib.chunks)

    def test_the_defaults_are_the_documented_constants(self):
        lib = SourceLibrarian(MockLLM())
        for setting, expected in ALL_SETTINGS.items():
            attr = "embed_model_name" if setting == "embed_model" else setting
            assert getattr(lib, attr) == expected, setting

    def test_check_index_settings_normalises_the_defaults(self):
        # The lazily building layers hold the returned dict and hand it
        # straight to a constructor, so it has to be complete and already
        # normalised.
        settings = check_index_settings(warn=False)
        # `ALL_SETTINGS` comes from the signature, so this asserts the
        # function returns every knob it declares — a knob accepted and not
        # returned reaches no constructor, because the lazy layers splat this
        # dict. Not a tautology: the two sets have different sources.
        assert set(settings) == set(ALL_SETTINGS)
        # And that the three it deliberately withholds are still those three.
        # Stated separately so a fourth read-not-returned parameter fails
        # here, naming itself, rather than surfacing as a confusing extra key
        # in the comparison above.
        declared = set(inspect.signature(check_index_settings).parameters)
        assert declared - set(settings) == READ_NOT_RETURNED
        assert settings["chunk_size"] == CHUNK_SIZE
        assert isinstance(settings["rerank_threshold"], float)


class TestExpansionSettings:
    """The two query-expansion knobs, threaded the way every other knob
    is: validated in a module function because the router and the web
    librarian carry them for a librarian they build lazily, and forwarded
    by all the layers because a knob nobody can reach is not configurable.

    They are deliberately *not* index settings: expansion happens at query
    time, so turning it on must not invalidate a saved index.
    """

    def test_check_expansion_settings_normalises_the_defaults(self):
        # The lazily building layers hold this dict and splat it into a
        # constructor, so it has to be complete and already normalised.
        settings = check_expansion_settings()
        assert settings == {"expand_query": EXPAND_QUERY,
                            "expansion_style": EXPANSION_STYLE,
                            "expansion_variants": EXPANSION_VARIANTS}

    def test_it_normalises_a_numpy_variant_count(self):
        settings = check_expansion_settings(expansion_variants=np.int64(4))
        assert settings["expansion_variants"] == 4
        assert type(settings["expansion_variants"]) is int

    @pytest.mark.parametrize("bad", [1, 0, "yes", None])
    def test_it_refuses_a_non_bool_expand_query(self, bad):
        with pytest.raises(TypeError, match="expand_query"):
            check_expansion_settings(expand_query=bad)

    @pytest.mark.parametrize("bad", [True, False, "2", 2.5])
    def test_it_refuses_a_non_int_variant_count(self, bad):
        with pytest.raises(TypeError, match="expansion_variants"):
            check_expansion_settings(expansion_variants=bad)

    def test_the_defaults_are_the_documented_constants(self):
        lib = SourceLibrarian(MockLLM())
        assert lib.expansion_enabled == EXPAND_QUERY
        assert lib.expansion_variants == EXPANSION_VARIANTS
        # The one that would be a bad surprise in a deployment: an LLM
        # call per question that nobody opted into.
        assert EXPAND_QUERY is False

    # ── Forwarding ───────────────────────────────────────────────

    def test_the_facade_forwards_both(self):
        lib = Librarian(MockLLM(), expand_query=True, expansion_variants=4)
        assert lib.source.expansion_enabled is True
        assert lib.source.expansion_variants == 4

    def test_the_facade_defaults_match_the_source_librarian(self):
        facade = Librarian(MockLLM()).source
        direct = SourceLibrarian(MockLLM())
        assert facade.expansion_enabled == direct.expansion_enabled
        assert facade.expansion_variants == direct.expansion_variants

    def test_the_router_forwards_to_the_librarian_it_builds(self, monkeypatch):
        stub_models(monkeypatch)
        router = LibrarianRouter(MockLLM(), use_web=False, expand_query=True,
                                 expansion_variants=4)
        assert router.source is None  # nothing injected: built on index()

        asyncio.run(router.index(DOC))

        assert router.source.expansion_enabled is True
        assert router.source.expansion_variants == 4

    def test_the_router_validates_before_it_builds_anything(self):
        # The librarian carrying the setting does not exist until the
        # first index(), so without a check here the mistake surfaces a
        # long way from the call that made it.
        with pytest.raises(TypeError, match="expand_query"):
            LibrarianRouter(MockLLM(), use_web=False, expand_query="yes")

    def test_the_router_validates_the_variant_count_too(self):
        with pytest.raises(ValueError, match="expansion_variants"):
            LibrarianRouter(MockLLM(), use_web=False, expansion_variants=0)

    def test_the_router_says_when_an_injected_librarian_ignores_them(
        self, caplog
    ):
        # The router's warning is a whole-set defaults comparison, so a
        # knob added without being added there goes silently inert.
        injected = SourceLibrarian(MockLLM())
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(MockLLM(), source_librarian=injected,
                            use_web=False, expand_query=True,
                            expansion_variants=4)
        assert "expand_query" in caplog.text
        assert "expansion_variants" in caplog.text
        assert injected.expansion_enabled is False

    def test_the_router_is_quiet_about_expansion_defaults(self, caplog):
        injected = SourceLibrarian(MockLLM())
        with caplog.at_level("WARNING", logger="librarium.router"):
            LibrarianRouter(MockLLM(), source_librarian=injected,
                            use_web=False)
        assert "expand_query" not in caplog.text

    # ── Not an index setting ─────────────────────────────────────

    def test_expansion_does_not_change_the_content_hash(self):
        # Folding it in would invalidate every saved index the first time
        # a deployment turned expansion on, re-paying the whole prefix
        # cost for a query-time setting.
        plain = SourceLibrarian(MockLLM())
        expanding = SourceLibrarian(MockLLM(), expand_query=True,
                                    expansion_variants=4)
        plain.documents = expanding.documents = {"d": DOC}
        assert plain.content_hash() == expanding.content_hash()

    def test_a_cache_is_reused_across_the_expansion_setting(self, tmp_path,
                                                            monkeypatch):
        stub_models(monkeypatch)
        asyncio.run(SourceLibrarian(MockLLM()).index(DOC, cache=tmp_path))

        llm = MockLLM()
        asyncio.run(SourceLibrarian(llm, expand_query=True).index(
            DOC, cache=tmp_path))
        assert llm.calls == []  # still a hit: nothing about the index moved


class _MarkEmbedder:
    """Vectors carrying the identity of the text they were made from.

    Enough for the persistence tests, which need the dense half to be
    *present* — plausible floats would let a row filed against the wrong
    chunk pass unnoticed.
    """

    def encode(self, texts, show_progress_bar: bool = False):
        return np.array(
            [[float(len(t) % 11), float(sum(map(ord, t[:8])) % 13)]
             for t in texts],
            dtype=np.float32,
        )
