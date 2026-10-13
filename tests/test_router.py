"""Router priority, dedup, and refine-passthrough tests."""

from __future__ import annotations

import pytest

from librarium import Fact, LibrarianAnswer, LibrarianRouter, SourceLibrarian
from librarium.source import TOP_K


class FakeSource:
    """Stands in for SourceLibrarian — old two-arg answer() signature."""

    def __init__(self, facts, coverage="complete"):
        self._facts = facts
        self._coverage = coverage
        self.use_chunking = True
        self.source = "fake"
        self.refine_seen = None
        self.top_k_seen = None

    async def answer(self, question, context_facts=None, **kwargs):
        self.refine_seen = kwargs.get("refine")
        self.top_k_seen = kwargs.get("top_k")
        return LibrarianAnswer(
            items=[Fact(content=f) for f in self._facts],
            coverage=self._coverage,
        )


class FakeWeb:
    def __init__(self, facts):
        self._facts = facts
        self.called = False
        self.top_k_seen = None

    async def answer(self, question, context_facts=None, **kwargs):
        self.called = True
        self.top_k_seen = kwargs.get("top_k")
        return LibrarianAnswer(
            items=[Fact(content=f) for f in self._facts],
            coverage="partial",
        )


class OldSignatureWeb(FakeWeb):
    """A web librarian written before either optional kwarg existed."""

    async def answer(self, question, context_facts=None):
        self.called = True
        return LibrarianAnswer(
            items=[Fact(content=f) for f in self._facts],
            coverage="partial",
        )


class TestPriority:
    async def test_complete_source_skips_web(self, mock_llm):
        web = FakeWeb(["web fact"])
        router = LibrarianRouter(
            mock_llm, source_librarian=FakeSource(["doc fact"]),
            web_librarian=web,
        )
        result = await router.answer("q")
        assert [f.content for f in result.items] == ["doc fact"]
        assert web.called is False

    async def test_partial_source_triggers_web(self, mock_llm):
        web = FakeWeb(["web fact"])
        router = LibrarianRouter(
            mock_llm,
            source_librarian=FakeSource(["doc fact"], coverage="partial"),
            web_librarian=web,
        )
        result = await router.answer("q")
        assert web.called is True
        contents = [f.content for f in result.items]
        assert "doc fact" in contents and "web fact" in contents

    async def test_web_disabled(self, mock_llm):
        web = FakeWeb(["web fact"])
        router = LibrarianRouter(
            mock_llm,
            source_librarian=FakeSource([], coverage="uncertain"),
            web_librarian=web, use_web=False,
        )
        await router.answer("q")
        assert web.called is False


class TestDedupAndRefine:
    async def test_dedup_across_sources(self, mock_llm):
        router = LibrarianRouter(
            mock_llm,
            source_librarian=FakeSource(["same fact"], coverage="partial"),
            web_librarian=FakeWeb(["same fact"]),
        )
        result = await router.answer("q")
        assert len(result.items) == 1

    async def test_refine_false_not_passed(self, mock_llm):
        """Old-signature librarians must keep working when refine is off."""
        src = FakeSource(["f"])
        router = LibrarianRouter(mock_llm, source_librarian=src)
        await router.answer("q", refine=False)
        assert src.refine_seen is None  # kwarg omitted entirely

    async def test_refine_true_passed(self, mock_llm):
        src = FakeSource(["f"])
        router = LibrarianRouter(mock_llm, source_librarian=src)
        await router.answer("q", refine=True)
        assert src.refine_seen is True


class TestTopKPassthrough:
    """The model behind the router has one context window, so a `top_k`
    the router was built with — or asked for — has to reach whichever
    librarian ends up answering.
    """

    async def test_default_matches_the_source_librarian(self, mock_llm):
        assert LibrarianRouter(mock_llm, use_web=False).top_k == TOP_K

    async def test_the_default_reaches_the_lazily_built_librarian(self, mock_llm):
        router = LibrarianRouter(mock_llm, use_web=False, top_k=3)
        assert router.source is None      # nothing injected: built on index()
        await router.index("some short text")
        assert router.source.top_k == 3

    async def test_an_injected_librarian_keeps_its_own_default(self, mock_llm):
        injected = SourceLibrarian(mock_llm, top_k=7)
        router = LibrarianRouter(mock_llm, source_librarian=injected, top_k=3)
        assert router.source.top_k == 7

    async def test_per_call_top_k_reaches_the_source(self, mock_llm):
        src = FakeSource(["f"])
        router = LibrarianRouter(mock_llm, source_librarian=src)
        await router.answer("q", top_k=3)
        assert src.top_k_seen == 3

    async def test_no_top_k_omits_the_kwarg(self, mock_llm):
        """Old-signature librarians must keep working when nobody asks."""
        src = FakeSource(["f"])
        router = LibrarianRouter(mock_llm, source_librarian=src)
        await router.answer("q")
        assert src.top_k_seen is None  # kwarg omitted entirely

    async def test_per_call_top_k_reaches_the_web_fallback(self, mock_llm):
        # The web half answers from a crawl against the same model, so a
        # budget tight enough to ask for top_k applies there too.
        web = FakeWeb(["web fact"])
        router = LibrarianRouter(
            mock_llm,
            source_librarian=FakeSource(["doc fact"], coverage="partial"),
            web_librarian=web,
        )
        await router.answer("q", top_k=3)
        assert web.top_k_seen == 3

    async def test_an_old_signature_web_librarian_still_works(self, mock_llm):
        web = OldSignatureWeb(["web fact"])
        router = LibrarianRouter(
            mock_llm,
            source_librarian=FakeSource(["doc fact"], coverage="partial"),
            web_librarian=web,
        )
        await router.answer("q")
        assert web.called is True

    async def test_a_bad_top_k_is_rejected_at_construction(self, mock_llm):
        # The librarian that would validate it is not built until index(),
        # a long way from the call that made the mistake.
        with pytest.raises(ValueError, match="must be >= 1"):
            LibrarianRouter(mock_llm, use_web=False, top_k=0)

    async def test_a_non_int_top_k_is_rejected_at_construction(self, mock_llm):
        with pytest.raises(TypeError, match="top_k"):
            LibrarianRouter(mock_llm, use_web=False, top_k="5")
