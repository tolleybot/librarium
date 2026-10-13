"""Retrieval, persistence and provenance — no model, no network.

The embedding half is exercised with a deterministic stub so these run
identically whether or not the `embeddings` extra is installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from librarium import Chunk, Fact, Librarian, LibrarianAnswer, SourceLibrarian
from librarium.source import (
    CHUNK_JOINER,
    EMBED_MODEL,
    EMBED_PROBE,
    TOP_K,
    embed_prompts,
    parse_gap_query,
    reciprocal_rank_fusion,
)
from tests.conftest import MockLLM, stub_models

# Eight sections, not three or four: with a handful of chunks BM25's IDF
# term goes to zero for a word that appears in half of them, and every
# score ties. Real documents have enough chunks for IDF to mean something.
SECTIONS = ["alpha", "beta", "gamma", "delta",
            "epsilon", "zeta", "eta", "theta"]
LONG_DOC = "\n\n".join(
    f"The {name} division reported figures for {name}. " * 30
    for name in SECTIONS
)
SHORT_DOC = "The bakery sells 120 loaves at $4.50 each. It opened in 1987."

ONE_FACT = json.dumps({"items": [{"content": "a fact"}], "coverage": "complete"})


class StubEmbedder:
    """Deterministic stand-in for sentence-transformers.

    Embeds each text as the counts of a few marker words, so cosine
    similarity is predictable without loading a real model.
    """

    def encode(self, texts, show_progress_bar: bool = False):
        return np.array(
            [[float(t.lower().count(name)) for name in SECTIONS] for t in texts],
            dtype=np.float32,
        )


async def build(*, embedder=None, llm=None, doc: str = LONG_DOC) -> SourceLibrarian:
    """A chunked librarian, with or without the dense half."""
    lib = SourceLibrarian(llm or MockLLM())
    if embedder is None:
        lib._dense_ok = False  # as if the embeddings extra were absent
    else:
        lib.embed_model = embedder
    await lib.index(doc)
    return lib


class TestRRF:
    def test_single_ranking_passes_through(self):
        assert reciprocal_rank_fusion([[3, 1, 2]]) == [3, 1, 2]

    def test_agreement_wins(self):
        # 0 is liked by both rankers; 3 only by the second.
        assert reciprocal_rank_fusion([[0, 1, 2, 3], [3, 0, 1, 2]])[0] == 0

    def test_ties_keep_first_ranking_order(self):
        assert reciprocal_rank_fusion([[0, 1], [1, 0]]) == [0, 1]

    def test_covers_indices_from_every_ranking(self):
        assert sorted(reciprocal_rank_fusion([[0, 1], [2, 3]])) == [0, 1, 2, 3]

    def test_smaller_k_sharpens_the_top_rank(self):
        spread = reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]], k=1)
        assert spread[0] in (0, 2)  # the two rankers disagree on the winner
        assert spread[-1] == 1


class TestRetrievalPaths:
    async def test_bm25_only_when_embeddings_unavailable(self):
        lib = await build()
        assert lib.retrieval_mode == "bm25-only"
        assert lib.chunk_embeddings is None
        chunks = lib.retrieve("gamma", top_k=2)
        assert chunks and all(isinstance(c, Chunk) for c in chunks)
        assert "gamma" in chunks[0].text

    async def test_hybrid_when_embeddings_present(self):
        lib = await build(embedder=StubEmbedder())
        assert lib.retrieval_mode == "hybrid"
        assert lib.chunk_embeddings.shape == (len(lib.chunks), len(SECTIONS))
        assert "gamma" in lib.retrieve("gamma", top_k=2)[0].text

    async def test_top_k_is_honoured(self):
        lib = await build()
        assert len(lib.retrieve("alpha", top_k=2)) == 2

    async def test_zero_norm_embeddings_do_not_crash(self):
        lib = await build(embedder=StubEmbedder())
        # A query with no marker words embeds to the zero vector.
        assert lib.retrieve("nothing recognisable here", top_k=3)

    async def test_retrieval_mode_transitions(self):
        lib = SourceLibrarian(MockLLM())
        assert lib.retrieval_mode == "unindexed"
        await lib.index(SHORT_DOC)
        assert lib.retrieval_mode == "whole-document"

    async def test_whole_document_path_returns_chunks(self):
        lib = SourceLibrarian(MockLLM())
        await lib.index(SHORT_DOC, doc_id="bakery.md")
        chunks = lib.retrieve("loaves")
        assert [c.text for c in chunks] == [SHORT_DOC]
        assert chunks[0].doc_id == "bakery.md"

    async def test_chunks_carry_provenance(self):
        lib = await build()
        assert all(c.doc_id == "document" for c in lib.chunks)
        assert [c.index for c in lib.chunks] == list(range(len(lib.chunks)))
        for chunk in lib.chunks:
            assert LONG_DOC[chunk.offset : chunk.offset + 20] in chunk.text

    async def test_the_prefix_is_stored_apart_and_composed_into_prefixed(self):
        # Renamed from `test_prefixes_index_separately_from_raw_text`, which
        # became false: the prefix no longer indexes separately from anything.
        # Both retrieval halves read `c.prefixed` — BM25 through
        # `_build_bm25`, the encoder through `_refresh_embeddings` (see
        # `TestWhatReachesTheEncoder` in test_indexing.py). What is still
        # true, and what these assertions were always really about, is the
        # *storage*: the prefix is kept as its own field and composed on
        # demand, never spliced into `text`.
        #
        # The one consumer that does read `text` alone is the answering
        # model — `ask()` joins `c.text` for the prompt — which is why
        # `Chunk.text` is described as what the model reads when a chunk is
        # quoted. Three consumers, two texts, and the distinction is worth
        # keeping straight because a previous version of this comment said
        # only "the model sees the raw chunk" and left which model to guess.
        lib = await build(llm=MockLLM(["A prefix about the division."]))
        assert all(c.prefix == "A prefix about the division." for c in lib.chunks)
        assert all(c.prefixed.startswith("A prefix") for c in lib.chunks)
        assert not any(c.text.startswith("A prefix") for c in lib.chunks)


# LONG_DOC chunks into fewer pieces than TOP_K, so retrieving at the
# default there returns the whole document and proves nothing about the
# cap. Same shape, enough sections that the default really bites.
WIDE_DOC = "\n\n".join(
    f"The {name} division reported figures for {name}. " * 30
    for name in SECTIONS + [f"unit{i}" for i in range(16)]
)


class TestTopK:
    """`top_k` as a per-librarian default plus a per-call override.

    What `retrieve()` hands back was always tunable. What matters here
    is the two paths that call the model — shrinking the prompt is the
    only reason to reach for the argument at all — so these assert on
    what the LLM was actually given, not on the returned list.
    """

    @staticmethod
    def chunks_in(prompt: str) -> int:
        """How many retrieved chunks a recorded system prompt carried.

        Material reaches the model joined by CHUNK_JOINER, and neither
        ASK_SYSTEM nor ANSWER_SYSTEM contains that separator, so counting
        separators counts chunks.
        """
        return prompt.count(CHUNK_JOINER) + 1

    @staticmethod
    async def facade(query_llm: MockLLM, **kwargs) -> Librarian:
        """A chunked librarian whose recorded calls are query calls only.

        Indexing costs one LLM call per chunk, which would both bury the
        query at an unknown offset in `calls` and eat the canned responses
        meant for it. So index against a throwaway client and swap in the
        one the test inspects.
        """
        lib = Librarian(MockLLM(["A prefix."]), **kwargs)
        lib.source._dense_ok = False
        await lib.index(WIDE_DOC)
        lib.llm = lib.source.llm = query_llm
        return lib

    # ── The two paths that call the model ────────────────────────

    async def test_ask_sends_only_top_k_chunks_to_the_model(self):
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm)
        await lib.ask("alpha", top_k=3)
        assert self.chunks_in(llm.calls[0][0]) == 3

    async def test_ask_material_is_exactly_the_retrieved_chunks(self):
        # Counting separators would also pass if the prompt carried the
        # right number of the wrong chunks, so this checks identity: every
        # retrieved chunk present, and nothing that was not retrieved.
        #
        # No longer a single byte-equality against the joined texts,
        # because `_retrieve` now prefixes each chunk with a provenance
        # header (`cite_sources`). The invariant that mattered was never
        # the exact string — it was *which* chunks reach the model, and
        # that is what is asserted here.
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm)
        await lib.ask("alpha", top_k=3)
        prompt = llm.calls[0][0]
        got = lib.retrieve("alpha", top_k=3)
        for chunk in got:
            assert chunk.text in prompt
        missed = [c for c in lib.source.chunks if c not in got]
        assert not [c for c in missed if c.text in prompt], (
            "a chunk that was not retrieved reached the prompt"
        )

    async def test_ask_material_carries_each_chunk_provenance(self):
        """The header exists so the model can tell near-identical chunks
        apart — two policy years whose text opens identically are
        indistinguishable without it."""
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm)
        await lib.ask("alpha", top_k=3)
        prompt = llm.calls[0][0]
        for chunk in lib.retrieve("alpha", top_k=3):
            assert lib.source.chunk_header(chunk) in prompt

    async def test_cite_sources_off_sends_the_bare_chunks(self):
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm)
        lib.source.cite_sources = False
        await lib.ask("alpha", top_k=3)
        expected = CHUNK_JOINER.join(
            c.text for c in lib.retrieve("alpha", top_k=3)
        )
        assert expected in llm.calls[0][0]

    async def test_a_lower_top_k_shrinks_what_ask_sends(self):
        wide, narrow = MockLLM(["An answer."]), MockLLM(["An answer."])
        await (await self.facade(wide)).ask("alpha")
        await (await self.facade(narrow)).ask("alpha", top_k=3)
        assert self.chunks_in(wide.calls[0][0]) == TOP_K
        assert self.chunks_in(narrow.calls[0][0]) == 3
        assert len(narrow.calls[0][0]) < len(wide.calls[0][0])

    async def test_facts_sends_only_top_k_chunks_to_the_model(self):
        llm = MockLLM([ONE_FACT])
        lib = await self.facade(llm)
        await lib.facts("alpha", top_k=3)
        assert self.chunks_in(llm.calls[0][0]) == 3

    async def test_a_lower_top_k_shrinks_what_facts_sends(self):
        wide, narrow = MockLLM([ONE_FACT]), MockLLM([ONE_FACT])
        await (await self.facade(wide)).facts("alpha")
        await (await self.facade(narrow)).facts("alpha", top_k=3)
        assert self.chunks_in(wide.calls[0][0]) == TOP_K
        assert self.chunks_in(narrow.calls[0][0]) == 3
        assert len(narrow.calls[0][0]) < len(wide.calls[0][0])

    async def test_refine_rounds_retrieve_at_the_same_top_k(self):
        # The second pass reads the same context budget as the first, so
        # a top_k asked for once has to hold for the whole call.
        llm = MockLLM([ONE_FACT, "QUERY: beta figures",
                       json.dumps({"items": [{"content": "another fact"}],
                                   "coverage": "complete"})])
        lib = await self.facade(llm)
        await lib.facts("alpha", refine=True, top_k=3)
        answer_calls = [s for s, _ in llm.calls if CHUNK_JOINER in s]
        assert len(answer_calls) == 2  # first pass plus one refine round
        assert all(self.chunks_in(s) == 3 for s in answer_calls)

    # ── The constructor default ──────────────────────────────────

    async def test_default_is_the_module_constant(self):
        assert SourceLibrarian(MockLLM()).top_k == TOP_K
        assert Librarian(MockLLM()).source.top_k == TOP_K

    async def test_constructor_default_reaches_the_model(self):
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm, top_k=2)
        await lib.ask("alpha")
        assert self.chunks_in(llm.calls[0][0]) == 2

    async def test_constructor_default_reaches_retrieve(self):
        lib = SourceLibrarian(MockLLM(["A prefix."]), top_k=3)
        lib._dense_ok = False
        await lib.index(LONG_DOC)
        assert len(lib.retrieve("alpha")) == 3

    async def test_the_facade_passes_the_default_through(self):
        assert Librarian(MockLLM(), top_k=4).source.top_k == 4

    # ── Override beats default ───────────────────────────────────

    async def test_per_call_top_k_beats_the_constructor_default(self):
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm, top_k=2)
        await lib.ask("alpha", top_k=5)
        assert self.chunks_in(llm.calls[0][0]) == 5

    async def test_an_override_does_not_stick_to_the_librarian(self):
        # Per-call means per-call: the next question gets the default back.
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm, top_k=2)
        await lib.ask("alpha", top_k=5)
        await lib.ask("beta")
        assert self.chunks_in(llm.calls[1][0]) == 2
        assert lib.source.top_k == 2

    async def test_override_applies_to_retrieve_too(self):
        lib = SourceLibrarian(MockLLM(["A prefix."]), top_k=3)
        lib._dense_ok = False
        await lib.index(LONG_DOC)
        assert len(lib.retrieve("alpha", top_k=6)) == 6
        assert len(lib.retrieve("alpha")) == 3

    # ── Validation ───────────────────────────────────────────────

    def test_zero_is_rejected(self):
        # Unlike the prefix cap, 0 has no useful meaning here: it
        # retrieves nothing and the model answers from empty material
        # rather than raising, which reads as a bad answer, not a bug.
        with pytest.raises(ValueError, match="must be >= 1"):
            SourceLibrarian(MockLLM(), top_k=0)

    def test_negative_is_rejected(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            SourceLibrarian(MockLLM(), top_k=-3)

    @pytest.mark.parametrize("bad", ["5", 5.0, None, True])
    def test_a_non_int_top_k_gets_a_readable_error(self, bad):
        # A bare comparison would report "'<' not supported between
        # instances of 'str' and 'int'", naming neither the setting nor
        # the fix.
        with pytest.raises(TypeError, match="top_k"):
            SourceLibrarian(MockLLM(), top_k=bad)

    @pytest.mark.parametrize("bad", [0, -3, "5", 5.0])
    async def test_a_bad_per_call_top_k_is_rejected(self, bad):
        lib = SourceLibrarian(MockLLM())
        await lib.index(SHORT_DOC)
        with pytest.raises((TypeError, ValueError)):
            lib.retrieve("loaves", top_k=bad)

    async def test_a_bad_per_call_top_k_is_rejected_before_the_model(self):
        llm = MockLLM(["An answer."])
        lib = await self.facade(llm)
        with pytest.raises(ValueError, match="must be >= 1"):
            await lib.ask("alpha", top_k=0)
        assert llm.calls == []

    async def test_context_guard_advises_the_argument_not_the_constant(self):
        # The old advice was "Reduce TOP_K in source.py", which asked a
        # library user to edit installed source. There is an API now.
        lib = Librarian(MockLLM(), max_context_tokens=200,
                        reserved_output_tokens=50)
        await lib.index("word " * 400)
        with pytest.raises(ValueError) as excinfo:
            await lib.ask("What does it say?")
        assert "TOP_K in source.py" not in str(excinfo.value)
        assert "top_k" in str(excinfo.value)

    # ── numpy integers, same contract as the prefix cap ──────────

    @pytest.mark.parametrize("k", [np.int64(3), np.int32(3), np.uint8(3)])
    def test_a_numpy_top_k_is_accepted(self, k):
        # check_top_k documents itself as mirroring check_prefix_cap, and
        # that one takes numbers.Integral: numpy is a hard dependency, so
        # a top_k that fell out of a numpy computation — a context budget
        # divided by an average chunk size, a value read out of an array —
        # is an ordinary thing to pass.
        assert SourceLibrarian(MockLLM(), top_k=k).top_k == 3

    def test_a_numpy_top_k_is_stored_as_a_builtin_int(self):
        # Normalised at the boundary, like the cap: the stored value goes
        # on to slice result lists and to be formatted into the context
        # guard's message, and nothing downstream should have to know
        # where it came from.
        lib = SourceLibrarian(MockLLM(), top_k=np.int64(3))
        assert type(lib.top_k) is int

    def test_a_numpy_per_call_top_k_is_accepted_and_normalised(self):
        # The per-call override runs the same validator, so it widens with
        # it — retrieve() slices with whatever comes back.
        from librarium.source import check_top_k

        assert check_top_k(np.int64(5)) == 5
        assert type(check_top_k(np.int64(5))) is int

    @pytest.mark.parametrize("bad", [True, False])
    def test_widening_to_numpy_does_not_admit_bool(self, bad):
        # bool registers as numbers.Integral. True as a retrieval count is
        # a typo, not a request for one chunk — and False is 0, which the
        # range check already calls out as useless.
        with pytest.raises(TypeError, match="top_k"):
            SourceLibrarian(MockLLM(), top_k=bad)

    @pytest.mark.parametrize("bad", [np.int64(0), np.int64(-3)])
    def test_an_out_of_range_numpy_top_k_is_still_rejected(self, bad):
        # Widening the accepted types must not widen the accepted values.
        with pytest.raises(ValueError, match="must be >= 1"):
            SourceLibrarian(MockLLM(), top_k=bad)


class TestMultiDocument:
    async def test_add_document_keeps_earlier_documents(self):
        lib = SourceLibrarian(MockLLM(["A prefix."]))
        await lib.index("Auto policy: the deductible is $500.", doc_id="auto.md")
        await lib.add_document("Home policy: the deductible is $1000.",
                               doc_id="home.md")
        assert list(lib.documents) == ["auto.md", "home.md"]
        assert "$500" in lib.source and "$1000" in lib.source

    async def test_index_replaces_and_says_so(self, caplog):
        lib = SourceLibrarian(MockLLM())
        await lib.index("first document", doc_id="first.md")
        with caplog.at_level("WARNING"):
            await lib.index("second document", doc_id="second.md")
        assert list(lib.documents) == ["second.md"]
        assert "replacing" in caplog.text

    async def test_duplicate_doc_id_rejected(self):
        lib = SourceLibrarian(MockLLM())
        await lib.index("some text", doc_id="a.md")
        with pytest.raises(ValueError, match="already indexed"):
            await lib.add_document("other text", doc_id="a.md")

    async def test_chunks_from_several_documents_keep_their_doc_id(self):
        lib = SourceLibrarian(MockLLM(["A prefix."]))
        lib._dense_ok = False
        await lib.index(LONG_DOC, doc_id="one.md")
        await lib.add_document(LONG_DOC.replace("division", "unit"),
                               doc_id="two.md")
        doc_ids = {c.doc_id for c in lib.chunks}
        assert doc_ids == {"one.md", "two.md"}
        assert lib.bm25 is not None
        assert len(lib.chunks) > len(lib.retrieve("gamma", top_k=2))


class TestProvenance:
    async def test_fact_source_is_the_document_id(self):
        lib = SourceLibrarian(MockLLM([ONE_FACT]))
        await lib.index(SHORT_DOC, doc_id="bakery.md")
        result = await lib.answer("How many loaves?")
        assert result.items[0].source == "bakery.md"

    async def test_fact_source_lists_contributing_documents(self):
        lib = SourceLibrarian(MockLLM([ONE_FACT]))
        await lib.index("Auto policy: deductible $500.", doc_id="auto.md")
        await lib.add_document("Home policy: deductible $1000.", doc_id="home.md")
        result = await lib.answer("What is the deductible?")
        assert result.items[0].source == "auto.md,home.md"


class TestPersistence:
    async def test_save_load_roundtrip(self, tmp_path):
        lib = await build(embedder=StubEmbedder())
        lib.save(tmp_path / "idx")

        restored = SourceLibrarian(MockLLM())
        restored.embed_model = StubEmbedder()
        restored.load(tmp_path / "idx")

        assert restored.documents == lib.documents
        assert restored.prefixed_chunks == lib.prefixed_chunks
        assert restored.retrieval_mode == "hybrid"
        assert restored.bm25 is not None
        np.testing.assert_allclose(restored.chunk_embeddings, lib.chunk_embeddings)
        assert ([c.text for c in restored.retrieve("gamma")]
                == [c.text for c in lib.retrieve("gamma")])

    async def test_roundtrip_without_embeddings(self, tmp_path):
        lib = await build()
        lib.save(tmp_path / "idx")
        restored = SourceLibrarian(MockLLM())
        restored._dense_ok = False
        restored.load(tmp_path / "idx")
        assert restored.retrieval_mode == "bm25-only"
        assert restored.chunk_embeddings is None
        assert restored.prefixed_chunks == lib.prefixed_chunks

    async def test_cache_skips_reindexing(self, tmp_path):
        first_llm = MockLLM()
        lib = SourceLibrarian(first_llm)
        lib._dense_ok = False
        await lib.index(LONG_DOC, cache=tmp_path / "idx")
        assert len(first_llm.calls) == len(lib.chunks) + 1  # prefixes + summary

        second_llm = MockLLM()
        reloaded = SourceLibrarian(second_llm)
        reloaded._dense_ok = False
        await reloaded.index(LONG_DOC, cache=tmp_path / "idx")

        assert second_llm.calls == []  # the whole index cost was avoided
        assert reloaded.prefixed_chunks == lib.prefixed_chunks

    async def test_cache_invalidated_when_the_document_changes(self, tmp_path):
        lib = SourceLibrarian(MockLLM())
        lib._dense_ok = False
        await lib.index(LONG_DOC, cache=tmp_path / "idx")

        edited_llm = MockLLM()
        edited = SourceLibrarian(edited_llm)
        edited._dense_ok = False
        await edited.index(LONG_DOC + "\n\nA new closing section was added.",
                           cache=tmp_path / "idx")

        assert edited_llm.calls  # re-indexed rather than served stale
        assert "closing section" in edited.source

    async def test_cache_hit_takes_the_callers_doc_id(self, tmp_path):
        first = SourceLibrarian(MockLLM())
        first._dense_ok = False
        await first.index(LONG_DOC, doc_id="old-name.md", cache=tmp_path / "idx")

        llm = MockLLM()
        renamed = SourceLibrarian(llm)
        renamed._dense_ok = False
        await renamed.index(LONG_DOC, doc_id="new-name.md", cache=tmp_path / "idx")

        assert llm.calls == []  # still a cache hit
        assert list(renamed.documents) == ["new-name.md"]
        assert {c.doc_id for c in renamed.chunks} == {"new-name.md"}

    async def test_load_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SourceLibrarian(MockLLM()).load(tmp_path / "nothing")

    async def test_load_rejects_a_future_format(self, tmp_path):
        directory = tmp_path / "idx"
        directory.mkdir()
        (directory / "index.json").write_text(json.dumps({"format": 999}))
        with pytest.raises(ValueError, match="format"):
            SourceLibrarian(MockLLM()).load(directory)

    async def test_unreadable_cache_falls_back_to_indexing(self, tmp_path):
        directory = tmp_path / "idx"
        directory.mkdir()
        (directory / "index.json").write_text("{ not json")
        llm = MockLLM()
        lib = SourceLibrarian(llm)
        lib._dense_ok = False
        await lib.index(LONG_DOC, cache=directory)
        assert llm.calls
        assert lib.chunks

    async def test_facade_save_and_load(self, tmp_path):
        lib = Librarian(MockLLM())
        lib.source._dense_ok = False
        await lib.index(LONG_DOC)
        lib.save(tmp_path / "idx")

        restored = Librarian(MockLLM())
        restored.source._dense_ok = False
        restored.load(tmp_path / "idx")
        assert restored.retrieval_mode == lib.retrieval_mode
        assert restored.retrieve("gamma")[0].text == lib.retrieve("gamma")[0].text


class TestContextGuard:
    async def test_ask_raises_before_overflowing(self):
        lib = Librarian(MockLLM(), max_context_tokens=200,
                        reserved_output_tokens=50)
        await lib.index("word " * 400)
        with pytest.raises(ValueError, match="max_context_tokens"):
            await lib.ask("What does it say?")

    async def test_ask_counts_the_prompt_and_the_reply_not_just_material(self):
        # ~125 tokens of material alone would fit under 180; the system
        # prompt, the question and the reserved reply do not.
        material = "word " * 100
        lib = Librarian(MockLLM(), max_context_tokens=180,
                        reserved_output_tokens=50)
        await lib.index(material)
        assert len(material) // 4 < 180
        with pytest.raises(ValueError, match="reserved for the reply"):
            await lib.ask("What does it say?")

    async def test_ask_succeeds_within_budget(self):
        lib = Librarian(MockLLM(["An answer."]), max_context_tokens=6000)
        await lib.index(SHORT_DOC)
        assert await lib.ask("How many loaves?") == "An answer."


class TestGapParsing:
    @pytest.mark.parametrize("reply,expected", [
        ("QUERY: opening price", "opening price"),
        ("query: opening price", "opening price"),
        ("QUERY: opening price\nThat should cover it.", "opening price"),
        ("SUFFICIENT", None),
        ("  sufficient  ", None),
        ("opening price", "opening price"),
        ('"opening price"', "opening price"),
        ("- opening price", "opening price"),
        ("", None),
        ("   ", None),
        # The failure mode this guard exists for: a whole sentence fed
        # back in as a search query retrieves poorly.
        ("The facts do not mention the opening price, so search for: "
         "opening price", None),
        ("Not mentioned. Search for the opening price.", None),
    ])
    def test_parse_gap_query(self, reply, expected):
        assert parse_gap_query(reply) == expected

    async def test_refine_ignores_a_prose_gap_reply(self):
        answer = json.dumps({"items": [{"content": "opened in 1987"}],
                             "coverage": "partial"})
        prose = ("The facts above do not say what the price per loaf is, "
                 "so you should look for the price.")
        llm = MockLLM([answer, prose])
        lib = SourceLibrarian(llm)
        await lib.index(SHORT_DOC)
        result = await lib.answer("When did it open and at what price?",
                                  refine=True)
        assert [f.content for f in result.items] == ["opened in 1987"]
        assert len(llm.calls) == 2  # answer + gap check, no garbage re-query


class TestStrictGrounding:
    async def test_off_by_default(self):
        llm = MockLLM(["An answer."])
        lib = Librarian(llm)
        await lib.index(SHORT_DOC)
        await lib.ask("How many loaves?")
        assert "never guess" not in llm.calls[0][0]

    async def test_adds_a_decline_clause(self):
        llm = MockLLM(["An answer."])
        lib = Librarian(llm, strict_grounding=True)
        await lib.index(SHORT_DOC)
        await lib.ask("How many loaves?")
        system = llm.calls[0][0]
        assert "never guess" in system
        assert SHORT_DOC in system  # raw material still reaches the model

    async def test_applies_to_a_custom_system_prompt(self):
        llm = MockLLM(["An answer."])
        lib = Librarian(llm, strict_grounding=True)
        await lib.index(SHORT_DOC)
        await lib.ask("How many loaves?", system="Be terse.")
        assert llm.calls[0][0].startswith("Be terse. If the answer is not")


class TestRefineCoverage:
    """A gap check that says SUFFICIENT must promote coverage to complete.

    Not cosmetic: LibrarianRouter falls through to the web librarian whenever
    coverage != "complete", so a stale "partial" sends a fully-answered
    question out for a search + scrape + index.
    """

    @staticmethod
    def _scripted(first_coverage: str, gap_reply: str) -> MockLLM:
        return MockLLM([
            json.dumps({"items": [{"content": "opened 2021 at $18"}],
                        "coverage": first_coverage}),
            gap_reply,
        ])

    async def test_sufficient_promotes_coverage(self):
        lib = SourceLibrarian(self._scripted("partial", "SUFFICIENT"))
        await lib.add_document("The fund opened in 2021 at $18.", doc_id="d")
        answer = await lib.answer("When did it open and at what price?", refine=True)
        assert answer.coverage == "complete"

    async def test_no_new_facts_keeps_original_coverage(self):
        # Gap found, re-query returns nothing new -> the gap was never closed,
        # so the honest label is the one the first pass reported.
        llm = MockLLM([
            json.dumps({"items": [{"content": "opened 2021"}], "coverage": "partial"}),
            "QUERY: opening price",
            json.dumps({"items": [{"content": "opened 2021"}], "coverage": "partial"}),
        ])
        lib = SourceLibrarian(llm)
        await lib.add_document("The fund opened in 2021.", doc_id="d")
        answer = await lib.answer("When did it open and at what price?", refine=True)
        assert answer.coverage == "partial"

    async def test_router_skips_web_when_refine_completes(self):
        from librarium.router import LibrarianRouter

        class SpyWeb:
            def __init__(self):
                self.called = 0

            async def answer(self, question, context_facts=None):
                self.called += 1
                return LibrarianAnswer(items=[Fact(content="web")], coverage="complete")

        lib = SourceLibrarian(self._scripted("partial", "SUFFICIENT"))
        await lib.add_document("The fund opened in 2021 at $18.", doc_id="d")

        web = SpyWeb()
        router = LibrarianRouter(
            MockLLM(), source_librarian=lib, web_librarian=web,
        )
        await router.answer("When did it open and at what price?", refine=True)
        assert web.called == 0


class TestDenseDisabledSkipsEncode:
    """A loaded-but-disabled embedding model must not be paid for.

    _ensure_embed_model() returns the cached model whenever one is loaded, so
    it cannot carry this check: a model that loaded fine and was later
    disabled still comes back non-None.
    """

    class CountingEmbed:
        def __init__(self):
            self.calls = 0
            #: What each call was asked to embed, not only how many there
            #: were: a count alone cannot tell one chunk batch plus the
            #: weight-fingerprint probe from two chunk batches.
            self.batches = []

        def encode(self, docs, show_progress_bar=False):
            self.calls += 1
            self.batches.append(list(docs))
            return np.ones((len(docs), 8), dtype="float32")

    async def test_no_encode_when_dense_disabled(self, mock_llm):
        lib = SourceLibrarian(mock_llm)
        stub = self.CountingEmbed()
        lib.embed_model = stub          # loaded...
        lib._dense_ok = False           # ...then disabled by a later failure

        await lib.add_document("x" * 10_000, doc_id="doc1")

        assert stub.calls == 0
        assert lib.chunk_embeddings is None
        assert lib.retrieval_mode == "bm25-only"

    async def test_encode_still_runs_when_dense_enabled(self, mock_llm):
        lib = SourceLibrarian(mock_llm)
        stub = self.CountingEmbed()
        lib.embed_model = stub
        lib._dense_ok = True

        await lib.add_document("x" * 10_000, doc_id="doc1")

        # Two calls, and *which* two is the point. Indexing now also
        # fingerprints the weights — one encode of one constant string,
        # once per loaded model — so a bare `calls == 2` would be
        # satisfied by the chunk encode happening twice, which is the
        # regression this class exists to catch.
        assert [len(b) for b in stub.batches] == [len(lib.chunks), 1]
        assert stub.batches[1] == [EMBED_PROBE]
        _doc_prefix = embed_prompts(EMBED_MODEL)[0]
        assert all(t.startswith(_doc_prefix) and t != EMBED_PROBE
                   for t in stub.batches[0])
        assert stub.calls == 2
        assert lib.chunk_embeddings is not None


class TestGapQueryRequiresPrefix:
    """Unprefixed prose must never become a search query.

    The old lenient branch ("assume the first line is keywords") let a whole
    sentence through when the model ignored the format — the exact failure
    parse_gap_query exists to stop.
    """

    @pytest.mark.parametrize("reply", [
        "The facts do not mention the price, so search for: opening price",
        "Missing: the opening price of the fund on its first day",
        "I could not find the deductible amount anywhere in the documents",
    ])
    def test_prose_without_prefix_is_rejected(self, reply):
        assert parse_gap_query(reply) is None

    @pytest.mark.parametrize("reply,expected", [
        ("QUERY: opening price", "opening price"),
        ("QUERY: revenue vs. target", "revenue vs. target"),
        ("QUERY: Q3 rev. est.", "Q3 rev. est"),
        ("QUERY: Dr. Alvarez appointment", "Dr. Alvarez appointment"),
    ])
    def test_abbreviations_are_not_mistaken_for_prose(self, reply, expected):
        # The old sentence-mark regex rejected all of these as "prose".
        assert parse_gap_query(reply) == expected

    @pytest.mark.parametrize("reply,expected", [
        ("opening price", "opening price"),
        ("price per loaf", "price per loaf"),
    ])
    def test_bare_keywords_still_accepted(self, reply, expected):
        # Models that ignore the format but reply with actual keywords stay
        # supported — that leniency is deliberate, it just needs a tighter bar.
        assert parse_gap_query(reply) == expected

    def test_sufficient_still_stops_refinement(self):
        assert parse_gap_query("SUFFICIENT") is None

    def test_overlong_query_rejected(self):
        long = "QUERY: " + " ".join(f"word{i}" for i in range(30))
        assert parse_gap_query(long) is None

    def test_empty_reply(self):
        assert parse_gap_query("") is None
        assert parse_gap_query("QUERY:   ") is None


class TestPagingWithExclude:
    """`exclude` is the cheap half of "look again, I know it is in there".

    Worth having because on the benchmark corpus the caller is usually
    right: 70% of expected literals sit in ranks 1-10, another 12% in
    11-20 and 12% more in 21-50, and seven of the eight questions that
    find nothing on the first page are recovered by paging deeper. What it
    cannot fix is the 6% absent from the top 50 entirely.

    Every test here is model-free and makes no LLM call, which is the
    property that lets paging stay inside `retrieve()`'s synchronous
    contract rather than becoming another round trip.
    """

    def _librarian(self, monkeypatch, n: int = 30) -> SourceLibrarian:
        stub_models(monkeypatch)
        lib = SourceLibrarian(MockLLM(), top_k=5)
        lib.chunks = [
            Chunk(text=f"chunk {i} concerning premium and deductible",
                  doc_id=f"d{i // 5}", index=i)
            for i in range(n)
        ]
        lib.use_chunking = True
        lib._build_bm25()
        lib._refresh_embeddings()
        return lib

    def test_a_second_page_shares_nothing_with_the_first(self, monkeypatch):
        lib = self._librarian(monkeypatch)
        first = lib.retrieve("premium")
        seen = {c.index for c in first}
        second = lib.retrieve("premium", exclude=seen)
        assert len(second) == 5
        assert not (seen & {c.index for c in second})

    def test_paging_walks_the_ranking_rather_than_restarting_it(self, monkeypatch):
        """The second page must be the *next* best chunks, which is what
        makes this different from retrieving again and hoping."""
        lib = self._librarian(monkeypatch)
        deep = [c.index for c in lib.retrieve("premium", top_k=10)]
        first = [c.index for c in lib.retrieve("premium", top_k=5)]
        second = [c.index for c in lib.retrieve("premium", top_k=5,
                                                exclude=set(first))]
        assert first == deep[:5]
        assert second == deep[5:10]

    def test_excluding_nothing_is_exactly_the_unpaged_call(self, monkeypatch):
        """The default has to be inert, or every existing caller changes."""
        lib = self._librarian(monkeypatch)
        plain = [c.index for c in lib.retrieve("premium")]
        for empty in ((), set(), frozenset(), []):
            assert [c.index for c in lib.retrieve("premium", exclude=empty)] == plain

    def test_excluding_everything_returns_nothing_rather_than_raising(self, monkeypatch):
        lib = self._librarian(monkeypatch)
        assert lib.retrieve("premium", exclude=range(30)) == []

    def test_an_index_that_was_never_ranked_is_harmless(self, monkeypatch):
        lib = self._librarian(monkeypatch)
        plain = [c.index for c in lib.retrieve("premium")]
        assert [c.index for c in lib.retrieve("premium", exclude={9999})] == plain

    def test_it_composes_with_also(self, monkeypatch):
        """Paging and hinting are the two halves of "look again" and a
        caller reaching for one will reach for both."""
        lib = self._librarian(monkeypatch)
        first = lib.retrieve("premium", also=["deductible"])
        second = lib.retrieve("premium", also=["deductible"],
                              exclude={c.index for c in first})
        assert not ({c.index for c in first} & {c.index for c in second})

    def test_the_pool_is_cut_after_the_exclusion_not_before(self, monkeypatch):
        """The ordering that makes a second page worth asking for.

        With a reranker loaded the fusion is cut to `rerank_pool` before
        scoring. Filtering *after* that cut would hand the cross-encoder
        the very chunks the caller is paging past, and a second page would
        be a reranked view of the first — the failure this ordering
        exists to prevent.
        """
        scored = []

        class Reranker:
            def predict(self, pairs):
                scored.append([text for _, text in pairs])
                return [1.0 / (i + 1) for i in range(len(pairs))]

        lib = self._librarian(monkeypatch)
        lib.reranker = Reranker()
        lib.use_reranker = True
        lib.rerank_pool = 5
        lib.rerank_top = 5

        first = lib.retrieve("premium")
        seen = {c.index for c in first}
        scored.clear()
        lib.retrieve("premium", exclude=seen)
        assert scored, "the reranker should have been asked"
        excluded_text = {lib.chunks[i].text for i in seen}
        assert not (set(scored[-1]) & excluded_text), (
            "the reranker was handed chunks the caller excluded — the pool "
            "was cut before the exclusion")
