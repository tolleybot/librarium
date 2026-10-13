"""K-line store + cache-wrapper tests (SQLite in tmp, no embeddings)."""

from __future__ import annotations

import pytest

from librarium import Fact, KLineLibrarian, KLineStore, LibrarianAnswer


class FakeLibrarian:
    def __init__(self, facts):
        self._facts = facts
        self.calls = 0
        self.kwargs_seen = None

    async def answer(self, question, context_facts=None, **kwargs):
        self.calls += 1
        self.kwargs_seen = kwargs
        return LibrarianAnswer(
            items=[Fact(content=f) for f in self._facts],
            coverage="complete",
        )


@pytest.fixture
def store(tmp_path):
    return KLineStore(db_path=str(tmp_path / "klines.db"))


class TestCacheFlow:
    async def test_unverified_not_cached(self, store):
        inner = FakeLibrarian(["Tokyo pop is 14M"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        await lib.answer("Tokyo population?")
        await lib.answer("Tokyo population?")
        assert inner.calls == 2  # unverified answers are never reused

    async def test_verified_hit_skips_librarian(self, store):
        inner = FakeLibrarian(["Tokyo pop is 14M"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        await lib.answer("Tokyo population?")
        lib.verify("Tokyo population?", confidence=0.95)
        result = await lib.answer("Tokyo population?")
        assert inner.calls == 1  # second call served from cache
        assert any("14M" in f.content for f in result.items)

    async def test_force_refresh_bypasses_cache(self, store):
        inner = FakeLibrarian(["Tokyo pop is 14M"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        await lib.answer("Tokyo population?")
        lib.verify("Tokyo population?")
        await lib.answer("Tokyo population?", force_refresh=True)
        assert inner.calls == 2

    async def test_negative_kline_short_circuits(self, store):
        inner = FakeLibrarian(["irrelevant"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        lib.mark_failed("paywalled thing?", reason="all results paywalled")
        result = await lib.answer("paywalled thing?")
        assert inner.calls == 0
        assert "Previously failed" in result.items[0].content


class TestTopKPassthrough:
    async def test_top_k_forwarded_on_a_cache_miss(self, store):
        inner = FakeLibrarian(["Tokyo pop is 14M"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        await lib.answer("Tokyo population?", top_k=3)
        assert inner.kwargs_seen == {"top_k": 3}

    async def test_no_top_k_omits_the_kwarg(self, store):
        # Librarians (and mocks) that predate the kwarg must keep working,
        # and the librarian's own default must stay in charge.
        inner = FakeLibrarian(["Tokyo pop is 14M"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        await lib.answer("Tokyo population?")
        assert inner.kwargs_seen == {}

    async def test_a_cache_hit_is_served_as_stored(self, store):
        # The facts in a verified K-line were checked, not re-retrieved,
        # so top_k has nothing to act on and must not force a re-query.
        inner = FakeLibrarian(["Tokyo pop is 14M"])
        lib = KLineLibrarian(librarian=inner, kline_store=store)
        await lib.answer("Tokyo population?")
        lib.verify("Tokyo population?", confidence=0.95)
        result = await lib.answer("Tokyo population?", top_k=3)
        assert inner.calls == 1
        assert any("14M" in f.content for f in result.items)
