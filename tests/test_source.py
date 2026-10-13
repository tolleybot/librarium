"""SourceLibrarian + Librarian facade tests (short-text path, no models)."""

from __future__ import annotations

import json

import pytest

from librarium import Librarian, SourceLibrarian
from tests.conftest import MockLLM

DOC = "The bakery sells 120 loaves at $4.50 each. It opened in 1987."


class TestShortTextIndex:
    async def test_short_text_skips_chunking(self, mock_llm):
        lib = SourceLibrarian(mock_llm)
        await lib.index(DOC)
        assert lib.use_chunking is False
        assert lib.source == DOC
        assert mock_llm.calls == []  # no index-time LLM calls for short text


class TestFacts:
    async def test_facts_parses_json(self):
        llm = MockLLM([json.dumps({
            "items": [{"content": "120 loaves at $4.50 each"}],
            "coverage": "complete",
        })])
        lib = SourceLibrarian(llm)
        await lib.index(DOC)
        result = await lib.answer("How many loaves?")
        assert result.coverage == "complete"
        assert result.items[0].content == "120 loaves at $4.50 each"

    async def test_facts_text_fallback_on_bad_json(self):
        llm = MockLLM(["not json at all", "still not json",
                       "- 120 loaves at $4.50"])
        lib = SourceLibrarian(llm)
        await lib.index(DOC)
        result = await lib.answer("How many loaves?")
        assert result.coverage == "uncertain"
        assert any("120" in f.content for f in result.items)

    async def test_refine_requeries_on_gap(self):
        first = json.dumps({"items": [{"content": "opened in 1987"}],
                            "coverage": "partial"})
        gap = "price per loaf"
        second = json.dumps({"items": [{"content": "$4.50 each"}],
                             "coverage": "complete"})
        llm = MockLLM([first, gap, second, "SUFFICIENT"])
        lib = SourceLibrarian(llm)
        await lib.index(DOC)
        result = await lib.answer("When did it open and at what price?",
                                  refine=True)
        contents = [f.content for f in result.items]
        assert "opened in 1987" in contents
        assert "$4.50 each" in contents

    async def test_refine_stops_when_sufficient(self):
        first = json.dumps({"items": [{"content": "opened in 1987"}],
                            "coverage": "complete"})
        llm = MockLLM([first, "SUFFICIENT"])
        lib = SourceLibrarian(llm)
        await lib.index(DOC)
        result = await lib.answer("When did it open?", refine=True)
        assert len(result.items) == 1
        assert len(llm.calls) == 2  # answer + one gap check, no re-query


class TestAsk:
    async def test_ask_passes_raw_material(self):
        llm = MockLLM(["The bakery sells 120 loaves."])
        lib = Librarian(llm)
        await lib.index(DOC)
        answer = await lib.ask("How many loaves?")
        assert answer == "The bakery sells 120 loaves."
        system, user = llm.calls[0]
        assert DOC in system          # raw text reaches the model
        assert "How many loaves?" in user

    async def test_ask_without_index_raises(self, mock_llm):
        lib = Librarian(mock_llm)
        with pytest.raises(RuntimeError):
            await lib.ask("Anything?")


class TestContextLimit:
    async def test_overflow_raises_actionable_error(self):
        llm = MockLLM([json.dumps({"items": [], "coverage": "complete"})])
        lib = SourceLibrarian(llm, max_context_tokens=10)
        await lib.index(DOC)  # short text, kept whole
        with pytest.raises(ValueError, match="max_context_tokens"):
            await lib.answer("How many loaves?")
