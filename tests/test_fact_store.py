"""The fact store — the parser, the cost, the mix, and what it must not break.

Five separable things are pinned here:

* :func:`parse_facts` is pure and tolerant. It never raises, because
  extraction must not fail a chunk: a reply that will not parse leaves the
  chunk indexed and retrievable exactly as it was.
* ``extract_facts`` costs **one LLM call per chunk** — the largest
  index-time cost any knob in this library adds — so it is off by default,
  and that is checked by *counting calls* rather than by reading the flag.
* The mix **replaces slots, never extends the budget**. A retrieval that
  returned ``top_k`` chunks plus facts on top would be a wider retrieval
  wearing a mix's name.
* ``content_hash`` is byte-identical to its pre-knob value while the knob
  is off, so **no index already on disk is invalidated** — checked against
  a hash computed without the argument, not asserted in prose.
* A rolled-back document takes its facts with it, or the store holds
  entries citing a ``doc_id`` the index no longer has.

The double is content-derived: :class:`FactsLLM` builds its
facts out of the chunk text it was actually handed, so a misrouted chunk
produces facts naming the wrong document rather than something plausible.
"""

from __future__ import annotations

import json

import pytest

from librarium.models import Chunk
from librarium.source import (
    FACTS_VERSION,
    MAX_FACT_VALUE_CHARS,
    MAX_FACTS_PER_CHUNK,
    SourceLibrarian,
    check_fact_mix,
    content_hash,
    fact_text,
    parse_facts,
)
from tests.conftest import stub_models

DOC = """# Auto Policy 2024
Collision deductible is $1,000.00 for each loss.
Bodily injury liability is $100,000 each person.
Annual premium is $1,275.00 payable in twelve instalments.
"""


class FactsLLM:
    """Returns facts derived from the text it was handed.

    Content-derived on purpose: the value is a token taken out of the
    chunk, so a chunk routed to the wrong call produces a fact that names
    the wrong document rather than a plausible-looking constant. A double
    returning a fixed record would let a misfiled fact pass unnoticed.
    """

    def __init__(self, per_chunk: int = 2):
        self.per_chunk = per_chunk
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, **kwargs) -> str:
        self.calls.append((system, user))
        if "fact" not in system.lower():
            # The summary and prefix calls, which this double is not for.
            return f"context for {user[:20]}"
        words = [w for w in user.split() if w.startswith("$")][: self.per_chunk]
        return json.dumps({"facts": [
            {"subject": user.split("\n")[0][:40] or "doc",
             "attribute": f"amount {i}", "value": word,
             "quote": f"...{word}..."}
            for i, word in enumerate(words)
        ]})

    def fact_calls(self) -> int:
        return sum(1 for system, _ in self.calls if "fact" in system.lower())


class TestParseFacts:
    """Pure, tolerant, and never raising."""

    def test_a_reply_that_is_not_json_yields_nothing_rather_than_raising(self):
        assert parse_facts("I could not find any facts, sorry.") == []

    def test_an_empty_reply_yields_nothing(self):
        assert parse_facts("") == []
        assert parse_facts("   ") == []

    def test_a_fenced_block_is_unwrapped(self):
        reply = '```json\n{"facts": [{"subject": "s", "attribute": "a", ' \
                '"value": "$5", "quote": "q"}]}\n```'
        assert [f["value"] for f in parse_facts(reply)] == ["$5"]

    def test_a_bare_list_is_accepted_as_well_as_the_wrapped_shape(self):
        bare = '[{"subject": "s", "attribute": "a", "value": "$5", "quote": "q"}]'
        assert [f["value"] for f in parse_facts(bare)] == ["$5"]

    def test_a_record_with_no_value_is_dropped_because_it_is_not_a_fact(self):
        reply = json.dumps({"facts": [
            {"subject": "s", "attribute": "a", "value": "", "quote": "q"},
            {"subject": "s", "attribute": "a", "value": "$5", "quote": "q"},
        ]})
        assert [f["value"] for f in parse_facts(reply)] == ["$5"]

    def test_an_empty_subject_or_attribute_is_kept_because_the_value_is_reachable(self):
        """A chunk whose header names the subject leaves the model little
        to say there, and dropping the record would lose a value the quote
        still makes retrievable."""
        reply = json.dumps({"facts": [
            {"subject": "", "attribute": "", "value": "$5", "quote": "paid $5"}]})
        kept = parse_facts(reply)
        assert len(kept) == 1 and kept[0]["value"] == "$5"

    def test_an_overlong_value_is_dropped_rather_than_truncated(self):
        """Truncating would store a *different* value and claim it came
        from the document; the store's whole advantage is short, faithful
        entries."""
        reply = json.dumps({"facts": [
            {"subject": "s", "attribute": "a",
             "value": "x" * (MAX_FACT_VALUE_CHARS + 1), "quote": "q"}]})
        assert parse_facts(reply) == []

    def test_the_per_chunk_cap_bounds_a_pathological_reply(self):
        reply = json.dumps({"facts": [
            {"subject": "s", "attribute": "a", "value": f"${i}", "quote": "q"}
            for i in range(MAX_FACTS_PER_CHUNK * 2)]})
        assert len(parse_facts(reply)) == MAX_FACTS_PER_CHUNK

    def test_non_dict_items_are_skipped_not_fatal(self):
        reply = json.dumps({"facts": [
            "not a record", 7,
            {"subject": "s", "attribute": "a", "value": "$5", "quote": "q"}]})
        assert [f["value"] for f in parse_facts(reply)] == ["$5"]


class TestFactText:
    """What BM25 and the encoder actually see."""

    def test_subject_and_attribute_lead_because_they_carry_the_question_words(self):
        text = fact_text({"subject": "auto policy", "attribute": "collision "
                          "deductible", "value": "$1,000.00", "quote": "q"})
        assert text.startswith("auto policy — collision deductible: $1,000.00")

    def test_a_fact_with_no_head_still_renders_its_value(self):
        assert fact_text({"value": "$5", "quote": "paid $5"}).startswith("$5")


class TestTheCacheKey:
    """Off, this knob invalidates nothing already written."""

    def test_off_is_byte_identical_to_not_having_the_argument(self):
        docs = {"a": "hello", "b": "world"}
        assert content_hash(docs, extract_facts=False) == content_hash(docs)

    def test_on_is_a_different_key_so_a_stale_store_is_not_served(self):
        docs = {"a": "hello"}
        assert content_hash(docs, extract_facts=True) != content_hash(docs)

    def test_the_version_is_derived_so_a_prompt_edit_cannot_go_unnoticed(self):
        """`DOC_FIELDS_VERSION`'s argument: a hand-written literal with a
        comment asking the next person to bump it is how a cache key goes
        stale in silence."""
        assert FACTS_VERSION.startswith("facts-1:")
        assert len(FACTS_VERSION.split(":")[1]) == 12


class TestTheMix:
    """Slots are replaced, never added."""

    def _librarian(self, monkeypatch, mix: float) -> SourceLibrarian:
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM(), top_k=4, fact_mix=mix)
        lib.chunks = [Chunk(text=f"chunk {i} about money", doc_id="d",
                            index=i, offset=i * 10) for i in range(6)]
        lib.use_chunking = True
        lib.facts = [{"subject": "s", "attribute": f"amount {i}",
                      "value": f"${i}", "quote": "money", "doc_id": "d",
                      "chunk": i} for i in range(6)]
        lib._build_bm25()
        lib._build_fact_index()
        return lib

    def test_the_total_is_still_top_k(self, monkeypatch):
        lib = self._librarian(monkeypatch, 0.5)
        assert len(lib.retrieve("money", top_k=4)) == 4

    def test_a_mix_of_zero_returns_chunks_only(self, monkeypatch):
        lib = self._librarian(monkeypatch, 0.0)
        assert all(c.index != -1 for c in lib.retrieve("money", top_k=4))

    def test_facts_are_marked_so_nothing_mistakes_one_for_a_position(self, monkeypatch):
        """A fact is not a position in `self.chunks`, and `index` is what a
        caller would use to ask for neighbours."""
        lib = self._librarian(monkeypatch, 0.5)
        facts = [c for c in lib.retrieve("money", top_k=4) if c.index == -1]
        assert facts, "expected some facts in the mix"
        assert all(c.doc_id == "d" for c in facts)

    def test_an_empty_store_leaves_retrieval_exactly_as_it_was(self, monkeypatch):
        lib = self._librarian(monkeypatch, 0.5)
        with_facts = [c.text for c in lib.retrieve("money", top_k=4)]
        lib.facts = []
        lib._build_fact_index()
        without = [c.text for c in lib.retrieve("money", top_k=4)]
        assert without != with_facts
        assert all("chunk" in t for t in without)


class TestTheMixValidator:
    def test_a_share_outside_zero_to_one_is_refused(self):
        with pytest.raises(ValueError):
            check_fact_mix(1.5)
        with pytest.raises(ValueError):
            check_fact_mix(-0.1)

    def test_bool_is_refused_because_true_would_silently_mean_all_facts(self):
        with pytest.raises(TypeError):
            check_fact_mix(True)

    def test_the_bounds_themselves_are_allowed(self):
        assert check_fact_mix(0.0) == 0.0
        assert check_fact_mix(1.0) == 1.0


class TestTheCost:
    """One call per chunk, and off by default — counted, not read."""

    @pytest.mark.asyncio
    async def test_off_by_default_spends_no_fact_call(self, monkeypatch):
        stub_models(monkeypatch)
        llm = FactsLLM()
        lib = SourceLibrarian(llm, chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100)
        await lib.add_document(DOC, doc_id="policy")
        assert llm.fact_calls() == 0
        assert lib.facts == []

    @pytest.mark.asyncio
    async def test_on_it_costs_one_call_per_chunk(self, monkeypatch):
        stub_models(monkeypatch)
        llm = FactsLLM()
        lib = SourceLibrarian(llm, chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True)
        await lib.add_document(DOC, doc_id="policy")
        assert llm.fact_calls() == len(lib.chunks)
        assert lib.facts, "expected facts from a document full of amounts"
        assert all(f["doc_id"] == "policy" for f in lib.facts)

    @pytest.mark.asyncio
    async def test_a_failing_extraction_still_leaves_the_document_indexed(
            self, monkeypatch):
        """Trading a chunk's retrievability for its facts would be the
        wrong way round."""
        stub_models(monkeypatch)

        class Broken(FactsLLM):
            async def complete(self, system, user, **kwargs):
                if "fact" in system.lower():
                    raise RuntimeError("extraction backend is down")
                return await super().complete(system, user, **kwargs)

        lib = SourceLibrarian(Broken(), chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True)
        await lib.add_document(DOC, doc_id="policy")
        assert lib.chunks, "the document must still be indexed"
        assert lib.facts == []
        assert lib.retrieve("collision", top_k=2)

    @pytest.mark.asyncio
    async def test_strict_turns_a_failed_extraction_into_an_error(self, monkeypatch):
        stub_models(monkeypatch)

        class Broken(FactsLLM):
            async def complete(self, system, user, **kwargs):
                if "fact" in system.lower():
                    raise RuntimeError("extraction backend is down")
                return await super().complete(system, user, **kwargs)

        lib = SourceLibrarian(Broken(), chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True,
                              strict=True)
        with pytest.raises(RuntimeError):
            await lib.add_document(DOC, doc_id="policy")


class TestPersistence:
    @pytest.mark.asyncio
    async def test_a_saved_store_survives_a_round_trip(self, monkeypatch, tmp_path):
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM(), chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True)
        await lib.add_document(DOC, doc_id="policy")
        before = [f["value"] for f in lib.facts]
        lib.save(str(tmp_path))

        other = SourceLibrarian(FactsLLM(), chunk_size=200, chunk_overlap=0,
                                chunk_threshold=100, extract_facts=True)
        other.load(str(tmp_path))
        assert [f["value"] for f in other.facts] == before

    @pytest.mark.asyncio
    async def test_a_store_from_another_version_is_discarded_not_filtered_on(
            self, monkeypatch, tmp_path, caplog):
        """`_restore_doc_fields`' argument: a direct `load()` consults no
        hash, so without this check a store extracted under an older
        prompt is installed silently."""
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM(), chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True)
        await lib.add_document(DOC, doc_id="policy")
        lib.save(str(tmp_path))

        path = tmp_path / "index.json"
        meta = json.loads(path.read_text())
        meta["facts_version"] = "facts-0:deadbeefcafe"
        path.write_text(json.dumps(meta))

        other = SourceLibrarian(FactsLLM(), chunk_size=200, chunk_overlap=0,
                                chunk_threshold=100, extract_facts=True)
        with caplog.at_level("WARNING"):
            other.load(str(tmp_path))
        assert other.facts == []
        assert any("facts from" in r.message for r in caplog.records)

    def test_an_index_predating_facts_loads_without_a_word(self, monkeypatch, tmp_path):
        """Optional by construction, so `INDEX_FORMAT_VERSION` was not
        bumped and every index on disk is spared a re-index."""
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM())
        lib.chunks = [Chunk(text="a chunk", doc_id="d", index=0)]
        lib.documents = {"d": "a chunk"}
        lib.use_chunking = True
        lib.save(str(tmp_path))
        path = tmp_path / "index.json"
        meta = json.loads(path.read_text())
        del meta["facts"], meta["facts_version"]
        path.write_text(json.dumps(meta))

        other = SourceLibrarian(FactsLLM())
        other.load(str(tmp_path))
        assert other.facts == []
        assert other.chunks


class TestRollback:
    """A rolled-back document takes its facts with it."""

    def _stocked(self, monkeypatch) -> SourceLibrarian:
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM())
        lib.documents = {"keep": "k", "drop": "d"}
        lib.chunks = [
            Chunk(text="keep one", doc_id="keep", index=0, offset=0),
            Chunk(text="drop one", doc_id="drop", index=1, offset=0),
            Chunk(text="keep two", doc_id="keep", index=2, offset=10),
        ]
        lib.use_chunking = True
        lib.facts = [
            {"subject": "s", "attribute": "a", "value": "$1", "quote": "q",
             "doc_id": "keep", "chunk": 0},
            {"subject": "s", "attribute": "a", "value": "$2", "quote": "q",
             "doc_id": "drop", "chunk": 1},
            {"subject": "s", "attribute": "a", "value": "$3", "quote": "q",
             "doc_id": "keep", "chunk": 2},
        ]
        lib._build_bm25()
        lib._build_fact_index()
        return lib

    def test_the_rolled_back_document_leaves_no_facts_behind(self, monkeypatch):
        """Left behind they would be retrievable entries citing a `doc_id`
        the index no longer holds."""
        lib = self._stocked(monkeypatch)
        lib._roll_back("drop")
        assert [f["value"] for f in lib.facts] == ["$1", "$3"]
        assert all(f["doc_id"] == "keep" for f in lib.facts)

    def test_the_survivors_are_repointed_at_the_chunks_that_moved(self, monkeypatch):
        """The chunk at position 2 slides to 1 when position 1 is removed.
        A fact still saying 2 would cite whatever landed there — and this
        is the bug the first draft of `_roll_back` shipped, because the
        remap was built after the renumbering and was an identity map."""
        lib = self._stocked(monkeypatch)
        lib._roll_back("drop")
        assert [c.index for c in lib.chunks] == [0, 1]
        by_value = {f["value"]: f["chunk"] for f in lib.facts}
        assert by_value == {"$1": 0, "$3": 1}
        for fact in lib.facts:
            source = lib.chunks[fact["chunk"]]
            assert source.doc_id == fact["doc_id"]

    def test_a_rollback_of_a_document_with_no_facts_disturbs_nothing(self, monkeypatch):
        lib = self._stocked(monkeypatch)
        lib.documents["absent"] = "x"
        before = [dict(f) for f in lib.facts]
        lib._roll_back("absent")
        assert lib.facts == before


class TestTheDenseHalfIsLive:
    """A fact store that silently ranks BM25-only is the failure this
    class exists to make audible.

    Every other test here passed for its whole first draft while
    `_build_fact_index` was handing `embed_prompts`' *tuple* a string key,
    catching the TypeError, logging it, and degrading to lexical — which
    is exactly the "a test built on one cannot tell a live dense half from
    a dead one" trap that conftest names for the embedder doubles.
    """

    def _stocked(self, monkeypatch) -> SourceLibrarian:
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM(), top_k=4, fact_mix=0.5)
        lib.facts = [{"subject": "policy", "attribute": f"amount {i}",
                      "value": f"${i}", "quote": "money", "doc_id": "d",
                      "chunk": 0} for i in range(6)]
        lib.chunks = [Chunk(text="a chunk about money", doc_id="d", index=0)]
        lib.use_chunking = True
        lib._build_bm25()
        lib._build_fact_index()
        return lib

    def test_building_the_index_produces_vectors(self, monkeypatch):
        lib = self._stocked(monkeypatch)
        assert lib.fact_embeddings is not None, (
            "the fact store fell back to BM25-only — check the encode call, "
            "not the ranking")
        assert len(lib.fact_embeddings) == len(lib.facts)

    def test_no_warning_is_logged_when_the_store_embeds_cleanly(
            self, monkeypatch, caplog):
        """The degraded path logs, so its absence is what says the dense
        half ran."""
        with caplog.at_level("WARNING"):
            self._stocked(monkeypatch)
        assert not [r for r in caplog.records
                    if "could not embed the fact store" in r.message]

    def test_a_ranking_still_comes_back_when_vectors_are_absent(self, monkeypatch):
        """Degrading is correct behaviour — it just must not be silent or
        the default."""
        lib = self._stocked(monkeypatch)
        lib.fact_embeddings = None
        assert lib._fact_ranking("money", 2), "lexical half should still rank"


class TestWhenTheCallsHappen:
    """Index-time, once. Retrieval is free, however often it runs.

    The question this answers is the one that decides whether the knob is
    affordable at all: one call per chunk is the largest index-time cost
    in this library, and it would be indefensible if it recurred per
    query. `retrieve()` makes no LLM call and contains no `await` — the
    fact half changes neither, because the facts were extracted before the
    question arrived.
    """

    @pytest.mark.asyncio
    async def test_retrieval_spends_nothing_however_many_times_it_runs(
            self, monkeypatch):
        stub_models(monkeypatch)
        llm = FactsLLM()
        lib = SourceLibrarian(llm, chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True,
                              top_k=4, fact_mix=0.5)
        await lib.add_document(DOC, doc_id="policy")
        after_indexing = len(llm.calls)
        assert llm.fact_calls() == len(lib.chunks)

        for _ in range(25):
            lib.retrieve("what is the collision deductible", top_k=4)
        assert len(llm.calls) == after_indexing, (
            "retrieval spent an LLM call — the fact store is an index-time "
            "cost and must stay one")

    @pytest.mark.asyncio
    async def test_a_second_document_costs_only_its_own_chunks(self, monkeypatch):
        """Incremental, so adding a document does not re-extract the
        corpus. There is no incremental path for a *changed prompt*,
        which is the cost worth knowing about."""
        stub_models(monkeypatch)
        llm = FactsLLM()
        lib = SourceLibrarian(llm, chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True)
        await lib.add_document(DOC, doc_id="first")
        first = llm.fact_calls()
        chunks_after_first = len(lib.chunks)

        await lib.add_document(DOC.replace("2024", "2025"), doc_id="second")
        added = len(lib.chunks) - chunks_after_first
        assert llm.fact_calls() - first == added

    @pytest.mark.asyncio
    async def test_a_loaded_store_costs_nothing_to_reuse(self, monkeypatch, tmp_path):
        """The whole point of persisting it: an index is extracted once
        and served many times."""
        stub_models(monkeypatch)
        lib = SourceLibrarian(FactsLLM(), chunk_size=200, chunk_overlap=0,
                              chunk_threshold=100, extract_facts=True)
        await lib.add_document(DOC, doc_id="policy")
        lib.save(str(tmp_path))

        fresh_llm = FactsLLM()
        other = SourceLibrarian(fresh_llm, chunk_size=200, chunk_overlap=0,
                                chunk_threshold=100, extract_facts=True,
                                top_k=4, fact_mix=0.5)
        other.load(str(tmp_path))
        for _ in range(10):
            other.retrieve("collision deductible", top_k=4)
        assert fresh_llm.calls == [], (
            "loading and querying a saved store spent an LLM call")
        assert other.facts
