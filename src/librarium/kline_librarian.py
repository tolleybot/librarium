"""K-Line Librarian — wraps other librarians with K-line memory.

Sits above Source/Blackboard/Web librarians. On each query:
  1. Check K-line store for a verified match → return immediately
  2. Check negative K-lines → skip known-bad patterns
  3. Query the underlying librarian
  4. Store the result as an unverified K-line
  5. Caller can later mark it verified via verify()

This keeps the K-line logic separate from the retrieval logic.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Protocol

from librarium.klines import KLine, KLineStore, NegativeKLine
from librarium.models import Fact, LibrarianAnswer

logger = logging.getLogger(__name__)


class Answerable(Protocol):
    """Any librarian that can answer questions."""
    async def answer(
        self, question: str, context_facts: list[Fact] | None = None
    ) -> LibrarianAnswer: ...


def _pending_key(question: str) -> str:
    """Hash the full question to avoid truncation collisions."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]


class KLineLibrarian:
    """Wraps a librarian with K-line memory.

    Checks the K-line store before querying the underlying librarian.
    Stores results after querying. Caller verifies via verify().
    """

    def __init__(
        self,
        librarian: Answerable,
        kline_store: KLineStore,
        source: str = "document",
    ):
        self.librarian = librarian
        self.store = kline_store
        self.source = source
        self._pending: dict[str, int] = {}  # hash(question) → kline_id

    async def answer(
        self,
        question: str,
        context_facts: list[Fact] | None = None,
        force_refresh: bool = False,
        refine: bool = False,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Answer with K-line cache. Checks memory before querying.

        Args:
            force_refresh: skip the cache and query the librarian
                directly. Use when re-verifying stale data.
            refine: forwarded to the underlying librarian on cache
                miss (iterative retrieval refinement).
            top_k: forwarded to the underlying librarian on cache miss.
                A cache hit is served as stored — the facts it holds were
                verified, not re-retrieved.
        """
        # 1. Check K-line store for verified match (unless refreshing)
        if not force_refresh:
            cached = await self.store.lookup(question, verified_only=True)
            if cached:
                logger.info(
                    "  KLINE LIBRARIAN: cache hit for '%s' → %d facts",
                    question[:60], len(cached.facts),
                )
                return LibrarianAnswer(
                    items=[Fact(content=f, source=f"kline:{cached.id}") for f in cached.facts],
                    coverage="complete",
                )

        # 2. Check negative K-lines (skip on refresh too)
        negative = self.store.lookup_negative(question) if not force_refresh else None
        if negative:
            logger.info(
                "  KLINE LIBRARIAN: negative hit for '%s' → %s",
                question[:60], negative.reason[:60],
            )
            return LibrarianAnswer(
                items=[Fact(content=f"Previously failed: {negative.reason}", source="kline:negative")],
                coverage="uncertain",
            )

        # 3. Query the underlying librarian.  Pass refine and top_k only
        # when set — keeps compatibility with librarians that predate
        # either kwarg (test mocks), and leaves the librarian's own
        # default in charge when the caller did not ask for something else.
        extra: dict[str, object] = {"refine": True} if refine else {}
        if top_k is not None:
            extra["top_k"] = top_k
        result = await self.librarian.answer(question, context_facts, **extra)

        # 4. Store as unverified K-line (caller verifies later)
        if result.items:
            kline = KLine(
                query=question,
                facts=[f.content for f in result.items],
                answer=result.items[0].content if result.items else "",
                verified=False,
                confidence=0.0,
                source=self.source,
                timestamp=time.time(),
            )
            kline_id = self.store.store(kline)
            self._pending[_pending_key(question)] = kline_id
            logger.info(
                "  KLINE LIBRARIAN: stored pending id=%d for '%s'",
                kline_id, question[:60],
            )

        return result

    def verify(self, question: str, confidence: float = 1.0) -> None:
        """Mark a pending K-line as verified (critic approved the answer).

        If an older verified K-line exists with a different answer,
        it gets replaced — the new verified answer wins.
        """
        key = _pending_key(question)
        kline_id = self._pending.pop(key, None)
        if kline_id is None:
            logger.debug("  KLINE LIBRARIAN: no pending kline for '%s'", question[:60])
            return

        self.store.verify_and_replace(kline_id, confidence)

    def mark_failed(self, question: str, reason: str, urls: list[str] | None = None) -> None:
        """Record a failed query pattern as a negative K-line."""
        key = _pending_key(question)
        kline_id = self._pending.pop(key, None)
        if kline_id:
            self.store._delete(kline_id)

        self.store.store_negative(NegativeKLine(
            query=question,
            reason=reason,
            urls=urls or [],
            timestamp=time.time(),
        ))
        logger.info(
            "  KLINE LIBRARIAN: marked failed for '%s': %s",
            question[:60], reason[:60],
        )
