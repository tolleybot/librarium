"""K-Line Store — memory traces of successful retrieval patterns.

When a query is answered and verified, the entire chain is stored:
  query → facts → answer → verification status

On future queries, the store checks for similar past queries via
embedding similarity. If a verified match is found, the stored
facts are returned directly — skipping retrieval entirely.

Negative K-lines track failed patterns (paywall sites, dead ends)
so the system doesn't repeat mistakes.

Storage: SQLite (structured data) + numpy (query embeddings).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default K-line matching thresholds. Override per store via the
# match_high / match_low constructor arguments on KLineStore.
MATCH_THRESHOLD_HIGH = 0.75   # above this → automatic HIT
MATCH_THRESHOLD_LOW = 0.50    # below this → automatic MISS
                               # between → LLM confirms "same question?"

# NLTK corpora needed for named-entity extraction. Names changed in
# NLTK 3.9, so both spellings are tried. None of this is on the
# retrieval path — it is fetched lazily, on first ambiguous lookup.
_ENTITY_CORPORA = (
    ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"),
    ("maxent_ne_chunker_tab", "maxent_ne_chunker"),
    ("words",),
)
_entity_support: bool | None = None  # None = untried, False = unavailable


def _fetch_entity_corpora() -> bool:
    """Download the tagger/chunker corpora. False if any is unavailable."""
    from librarium.chunking import _ensure_nltk

    return all(
        any(_ensure_nltk(name) for name in alternatives)
        for alternatives in _ENTITY_CORPORA
    )


class KLine(BaseModel):
    """A memory trace of a query → answer chain."""
    id: int = 0
    query: str = ""
    facts: list[str] = Field(default_factory=list)
    answer: str = ""
    verified: bool = False
    confidence: float = 0.0
    source: str = ""           # "document", "web", "blackboard"
    urls: list[str] = Field(default_factory=list)
    timestamp: float = 0.0
    access_count: int = 0


class NegativeKLine(BaseModel):
    """A memory trace of a failed pattern — don't repeat this."""
    id: int = 0
    query: str = ""
    reason: str = ""
    urls: list[str] = Field(default_factory=list)
    timestamp: float = 0.0


_MONEY_RE = re.compile(
    r"\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:trillion|billion|million|thousand|[kmbt]\b))?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*%")
_AMOUNT_RE = re.compile(r"^[$\d]")


def _extract_amounts(text: str) -> set[str]:
    """Pull money and percentage figures out of raw text.

    Scanning tokens does not work: word_tokenize splits "$4.9" into "$"
    and "4.9", so a token-level check only ever yields a bare "$" — an
    entity every money question shares, which makes the compatibility
    check accept pairs it exists to reject.
    """
    found: set[str] = set()
    for pattern in (_MONEY_RE, _PERCENT_RE):
        for match in pattern.finditer(text):
            amount = match.group(0).lower().rstrip(" ,")
            amount = re.sub(r"\s+", " ", amount)
            amount = amount.replace(",", "")
            amount = re.sub(r"^\$\s+", "$", amount)
            amount = re.sub(r"\s+%", "%", amount)
            if amount not in {"$", "%"}:
                found.add(amount)
    return found


def _is_amount(entity: str) -> bool:
    """True for money/percentage entities, which must match exactly.

    Substring matching is right for names — "Rao" belongs to "Deepa Rao"
    — but wrong for figures, where it would make "$50" match "$500".
    """
    return bool(_AMOUNT_RE.match(entity))


def _normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison.

    Strips formatting so "$4.9 trillion" and "$4.9T" and
    "4900000000000" don't trigger false replacements.
    """
    s = answer.lower().strip()
    # Remove common formatting
    s = s.replace(",", "").replace("$", "").replace("%", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    # Normalize number suffixes — only match if preceded by a digit
    for suffix, mult in [("trillion", "e12"), ("billion", "e9"), ("million", "e6"),
                          ("t", "e12"), ("b", "e9"), ("m", "e6")]:
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            # Only convert if the prefix is actually a number
            # ("4.9t" → convert, "result" → skip)
            try:
                val = float(num)
                s = str(val * float(f"1{mult}"))
            except ValueError:
                continue  # not a number prefix, try next suffix
            break
    return s


class KLineStore:
    """SQLite + embedding store for K-lines.

    Thread-safe: uses a lock around all mutations to the in-memory
    embedding lists.
    """

    def __init__(
        self,
        db_path: str = "klines.db",
        embed_model=None,
        llm=None,
        match_high: float = MATCH_THRESHOLD_HIGH,
        match_low: float = MATCH_THRESHOLD_LOW,
    ):
        self.db_path = db_path
        self.embed_model = embed_model
        self.llm = llm  # LLMClient, for confirmation in the ambiguous zone
        self.match_high = match_high
        self.match_low = match_low
        self._lock = threading.RLock()  # reentrant — allows nested acquire
        self._embeddings: list[np.ndarray] = []
        self._kline_ids: list[int] = []
        self._neg_embeddings: list[np.ndarray] = []
        self._neg_ids: list[int] = []
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        self._load_embeddings()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                facts TEXT NOT NULL,
                answer TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0.0,
                source TEXT NOT NULL DEFAULT '',
                urls TEXT NOT NULL DEFAULT '[]',
                embedding BLOB,
                timestamp REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS negative_klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                reason TEXT NOT NULL,
                urls TEXT NOT NULL DEFAULT '[]',
                embedding BLOB,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.commit()

    def _load_embeddings(self) -> None:
        """Load all embeddings into memory for fast similarity search."""
        with self._lock:
            # Positive K-lines
            cursor = self._conn.execute(
                "SELECT id, embedding FROM klines WHERE embedding IS NOT NULL"
            )
            self._embeddings = []
            self._kline_ids = []
            for row in cursor:
                if row[1]:
                    self._embeddings.append(
                        np.frombuffer(row[1], dtype=np.float32).copy()
                    )
                    self._kline_ids.append(row[0])

            # Negative K-lines
            cursor = self._conn.execute(
                "SELECT id, embedding FROM negative_klines WHERE embedding IS NOT NULL"
            )
            self._neg_embeddings = []
            self._neg_ids = []
            for row in cursor:
                if row[1]:
                    self._neg_embeddings.append(
                        np.frombuffer(row[1], dtype=np.float32).copy()
                    )
                    self._neg_ids.append(row[0])

        logger.info(
            "  KLINE loaded %d embeddings + %d negative from store",
            len(self._embeddings), len(self._neg_embeddings),
        )

    # ── Embedding helper ─────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray | None:
        """Embed a query string. Returns None if no model."""
        if not self.embed_model:
            return None
        return self.embed_model.encode(
            [f"search_query: {text}"], show_progress_bar=False
        )[0].astype(np.float32)

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract named entities from text using NLTK.

        Returns a set of entity strings (lowercased).
        Catches company names, person names, locations, numbers.
        """
        global _entity_support
        if _entity_support is False:
            return set()

        import nltk
        for retry in (False, True):
            try:
                tokens = nltk.word_tokenize(text)
                tagged = nltk.pos_tag(tokens)
                tree = nltk.ne_chunk(tagged)
                entities = set()
                for subtree in tree:
                    if hasattr(subtree, "label"):
                        entity = " ".join(w for w, _ in subtree.leaves()).lower()
                        entities.add(entity)
                entities |= _extract_amounts(text)
                _entity_support = True
                return entities
            except LookupError:
                # The tagger and chunker corpora are not part of the
                # retrieval path, so they are fetched only if a K-line
                # lookup actually reaches the ambiguous zone.
                if retry or not _fetch_entity_corpora():
                    _entity_support = False
                    logger.warning(
                        "  KLINE named-entity corpora unavailable — the "
                        "entity compatibility check is disabled and "
                        "ambiguous matches will rest on cosine + LLM alone"
                    )
                    return set()
            except Exception as e:
                logger.warning("  KLINE entity extraction failed (%s: %s)",
                               type(e).__name__, e)
                return set()
        return set()

    def _entities_compatible(self, query_a: str, query_b: str) -> bool:
        """Check if two queries have compatible named entities.

        Uses fuzzy matching: "Rao" matches "Deepa Rao" (substring).
        Figures are compared exactly instead, so "$500" does not match
        "$1500" the way a substring test would.
        If both have entities, at least one must overlap.
        If only one has entities, reject (suspicious mismatch).
        If neither has entities, accept (generic questions).
        """
        ents_a = self._extract_entities(query_a)
        ents_b = self._extract_entities(query_b)

        if not ents_a and not ents_b:
            return True  # neither has entities — cosine decides
        if not ents_a or not ents_b:
            return False  # one has entities, other doesn't — suspicious

        # Fuzzy match: "rao" matches "deepa rao", "lumen" matches "lumen tech"
        for ea in ents_a:
            for eb in ents_b:
                if _is_amount(ea) or _is_amount(eb):
                    if ea == eb:
                        return True
                elif ea in eb or eb in ea:
                    return True

        logger.info(
            "  KLINE entity mismatch: %s vs %s",
            ents_a, ents_b,
        )
        return False

    async def _confirm_ambiguous(self, query_a: str, query_b: str) -> bool:
        """Confirm a match in the ambiguous zone.

        Uses entity matching first (fast, no LLM). If entities are
        incompatible, rejects immediately. If compatible, asks LLM.
        """
        # Entity check first — fast, catches "Lumen employees" vs "Deepa equity"
        if not self._entities_compatible(query_a, query_b):
            return False

        # LLM confirmation
        if not self.llm:
            return True  # no LLM → accept on entity match alone

        try:
            text = await self.llm.complete(
                "Are these two questions asking for the same information? "
                "Answer with exactly one word: yes or no.",
                f"Question 1: {query_a}\nQuestion 2: {query_b}",
                think=False, temperature=0.1,
            )
            return text.strip().lower().startswith("yes")
        except Exception as e:
            logger.warning("  KLINE LLM confirm failed: %s", e)
            return True  # fail open

    def _cosine_best(
        self, query_emb: np.ndarray, embeddings: list[np.ndarray]
    ) -> tuple[int, float]:
        """Find best cosine match. Returns (index, score)."""
        if not embeddings:
            return -1, 0.0
        emb_matrix = np.stack(embeddings)
        norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb)
        # Guard against zero norms
        norms = np.where(norms == 0, 1.0, norms)
        cos_scores = np.dot(emb_matrix, query_emb) / norms
        best_idx = int(np.argmax(cos_scores))
        return best_idx, float(cos_scores[best_idx])

    # ── Store ────────────────────────────────────────────────────

    def store(self, kline: KLine) -> int:
        """Store a K-line. If a similar verified K-line exists with a
        different answer, delete it first (replace, not accumulate)."""
        emb_array = self._embed(kline.query)

        # Check for existing K-line to replace
        if kline.verified and emb_array is not None:
            existing = self._find_similar(emb_array)
            if existing and not self._answers_match(existing.answer, kline.answer):
                logger.info(
                    "  KLINE replacing id=%d: old='%s' new='%s'",
                    existing.id, existing.answer[:60], kline.answer[:60],
                )
                self._delete(existing.id)

        # Insert into DB
        embedding_bytes = emb_array.tobytes() if emb_array is not None else None
        cursor = self._conn.execute(
            """INSERT INTO klines (query, facts, answer, verified, confidence,
               source, urls, embedding, timestamp, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                kline.query,
                json.dumps(kline.facts),
                kline.answer,
                int(kline.verified),
                kline.confidence,
                kline.source,
                json.dumps(kline.urls),
                embedding_bytes,
                kline.timestamp or time.time(),
                0,
            ),
        )
        self._conn.commit()
        kline_id = cursor.lastrowid

        # Update in-memory index AFTER successful insert
        if emb_array is not None:
            with self._lock:
                self._embeddings.append(emb_array)
                self._kline_ids.append(kline_id)

        logger.info(
            "  KLINE stored: id=%d verified=%s query='%s'",
            kline_id, kline.verified, kline.query[:60],
        )
        return kline_id

    def store_negative(self, neg: NegativeKLine) -> int:
        """Store a negative K-line (failed pattern). Embeds at insert time."""
        emb_array = self._embed(neg.query)
        embedding_bytes = emb_array.tobytes() if emb_array is not None else None

        cursor = self._conn.execute(
            """INSERT INTO negative_klines (query, reason, urls, embedding, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                neg.query,
                neg.reason,
                json.dumps(neg.urls),
                embedding_bytes,
                neg.timestamp or time.time(),
            ),
        )
        self._conn.commit()
        neg_id = cursor.lastrowid

        if emb_array is not None:
            with self._lock:
                self._neg_embeddings.append(emb_array)
                self._neg_ids.append(neg_id)

        logger.info(
            "  KLINE stored negative: query='%s' reason='%s'",
            neg.query[:60], neg.reason[:60],
        )
        return neg_id

    # ── Lookup ───────────────────────────────────────────────────

    async def lookup(self, query: str, verified_only: bool = True) -> KLine | None:
        """Find a matching K-line using three-tier similarity check.

        score > match_high  → automatic HIT
        score match_low..match_high → LLM confirms "same question?"
        score < match_low   → automatic MISS

        Without an embedding model, falls back to exact match on the
        normalized query — the cache still works, just without fuzzy
        similarity.
        """
        q_emb = self._embed(query)
        if q_emb is None:
            return self._lookup_exact(query, verified_only)

        with self._lock:
            best_idx, best_score = self._cosine_best(q_emb, self._embeddings)

        if best_idx < 0 or best_score < self.match_low:
            logger.info(
                "  KLINE lookup: best=%.3f (below low %.2f) for '%s'",
                best_score, self.match_low, query[:60],
            )
            return None

        kline_id = self._kline_ids[best_idx]
        kline = self._get_by_id(kline_id)
        if not kline:
            return None

        if verified_only and not kline.verified:
            logger.info(
                "  KLINE lookup: found id=%d (%.3f) but NOT verified",
                kline_id, best_score,
            )
            return None

        # Ambiguous zone: entity check + LLM confirm
        if best_score < self.match_high:
            if not await self._confirm_ambiguous(query, kline.query):
                logger.info(
                    "  KLINE lookup: score=%.3f in ambiguous zone, REJECTED for '%s'",
                    best_score, query[:60],
                )
                return None
            logger.info(
                "  KLINE lookup: score=%.3f in ambiguous zone, CONFIRMED for '%s'",
                best_score, query[:60],
            )

        if kline:
            self._conn.execute(
                "UPDATE klines SET access_count = access_count + 1 WHERE id = ?",
                (kline_id,),
            )
            self._conn.commit()
            logger.info(
                "  KLINE HIT: id=%d score=%.3f verified=%s query='%s'",
                kline_id, best_score, kline.verified, kline.query[:60],
            )

        return kline

    def _lookup_exact(self, query: str, verified_only: bool) -> KLine | None:
        """Embedding-free fallback: exact match on normalized query."""
        norm = re.sub(r"\s+", " ", query.strip().lower())
        cursor = self._conn.execute(
            "SELECT id, query, verified FROM klines ORDER BY timestamp DESC"
        )
        for kid, q, verified in cursor:
            if re.sub(r"\s+", " ", q.strip().lower()) == norm:
                if verified_only and not verified:
                    continue
                kline = self._get_by_id(kid)
                if kline:
                    self._conn.execute(
                        "UPDATE klines SET access_count = access_count + 1 "
                        "WHERE id = ?", (kid,),
                    )
                    self._conn.commit()
                    logger.info("  KLINE HIT (exact): id=%d query='%s'",
                                kid, q[:60])
                return kline
        return None

    def _lookup_negative_exact(self, query: str) -> NegativeKLine | None:
        """Embedding-free fallback: exact match on normalized query."""
        norm = re.sub(r"\s+", " ", query.strip().lower())
        cursor = self._conn.execute(
            "SELECT id, query, reason, urls, timestamp FROM negative_klines "
            "ORDER BY timestamp DESC"
        )
        for row in cursor:
            if re.sub(r"\s+", " ", row[1].strip().lower()) == norm:
                return NegativeKLine(
                    id=row[0], query=row[1], reason=row[2],
                    urls=json.loads(row[3]), timestamp=row[4],
                )
        return None

    def lookup_negative(self, query: str) -> NegativeKLine | None:
        """Check if this query pattern previously failed."""
        q_emb = self._embed(query)
        if q_emb is None:
            return self._lookup_negative_exact(query)

        with self._lock:
            best_idx, best_score = self._cosine_best(q_emb, self._neg_embeddings)

        if best_idx < 0 or best_score < self.match_high:
            return None

        neg_id = self._neg_ids[best_idx]
        cursor = self._conn.execute(
            "SELECT id, query, reason, urls, timestamp FROM negative_klines WHERE id = ?",
            (neg_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        logger.info(
            "  KLINE NEGATIVE HIT (%.3f): '%s' matches '%s'",
            best_score, query[:60], row[1][:60],
        )
        return NegativeKLine(
            id=row[0], query=row[1], reason=row[2],
            urls=json.loads(row[3]), timestamp=row[4],
        )

    # ── Query ────────────────────────────────────────────────────

    def get_all(self, verified_only: bool = False) -> list[KLine]:
        """Get all K-lines."""
        where = "WHERE verified = 1" if verified_only else ""
        cursor = self._conn.execute(
            f"SELECT id, query, facts, answer, verified, confidence, "
            f"source, urls, timestamp, access_count FROM klines {where} "
            f"ORDER BY timestamp DESC"
        )
        return [self._row_to_kline(row) for row in cursor]

    def get_stats(self) -> dict:
        """Get store statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM klines").fetchone()[0]
        verified = self._conn.execute(
            "SELECT COUNT(*) FROM klines WHERE verified = 1"
        ).fetchone()[0]
        negative = self._conn.execute(
            "SELECT COUNT(*) FROM negative_klines"
        ).fetchone()[0]
        total_accesses = self._conn.execute(
            "SELECT SUM(access_count) FROM klines"
        ).fetchone()[0] or 0
        return {
            "total_klines": total,
            "verified": verified,
            "unverified": total - verified,
            "negative_klines": negative,
            "total_reuses": total_accesses,
        }

    # ── Verify ───────────────────────────────────────────────────

    def verify_and_replace(self, kline_id: int, confidence: float = 1.0) -> bool:
        """Mark a K-line as verified. If an older verified K-line exists
        for the same query pattern with a different answer, delete it.

        Returns True if the K-line was found and verified.
        """
        new_kline = self._get_by_id(kline_id)
        if not new_kline:
            return False

        # Check for existing verified K-line with different answer
        emb = self._embed(new_kline.query)
        if emb is not None:
            with self._lock:
                if self._embeddings:
                    emb_matrix = np.stack(self._embeddings)
                    norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(emb)
                    norms = np.where(norms == 0, 1.0, norms)
                    cos_scores = np.dot(emb_matrix, emb) / norms

                    for idx in range(len(cos_scores)):
                        other_id = self._kline_ids[idx]
                        if other_id == kline_id:
                            continue
                        if cos_scores[idx] < self.match_high:
                            continue
                        existing = self._get_by_id(other_id)
                        if (
                            existing
                            and existing.verified
                            and not self._answers_match(existing.answer, new_kline.answer)
                        ):
                            logger.info(
                                "  KLINE replacing id=%d with id=%d",
                                existing.id, kline_id,
                            )
                            self._delete(existing.id)
                            break

        self._conn.execute(
            "UPDATE klines SET verified = 1, confidence = ? WHERE id = ?",
            (confidence, kline_id),
        )
        self._conn.commit()
        logger.info(
            "  KLINE verified id=%d confidence=%.2f query='%s'",
            kline_id, confidence, new_kline.query[:60],
        )
        return True

    # ── Clear ────────────────────────────────────────────────────

    def clear(self) -> None:
        """Delete all K-lines (positive and negative). For testing."""
        self._conn.execute("DELETE FROM klines")
        self._conn.execute("DELETE FROM negative_klines")
        self._conn.commit()
        with self._lock:
            self._embeddings.clear()
            self._kline_ids.clear()
            self._neg_embeddings.clear()
            self._neg_ids.clear()
        logger.info("  KLINE store cleared")

    def clear_negative(self) -> None:
        """Delete all negative K-lines only."""
        self._conn.execute("DELETE FROM negative_klines")
        self._conn.commit()
        with self._lock:
            self._neg_embeddings.clear()
            self._neg_ids.clear()
        logger.info("  KLINE negative K-lines cleared")

    # ── Internal ─────────────────────────────────────────────────

    def _find_similar(self, query_emb: np.ndarray) -> KLine | None:
        """Find an existing K-line similar to this query embedding."""
        with self._lock:
            best_idx, best_score = self._cosine_best(query_emb, self._embeddings)
        if best_idx < 0 or best_score < self.match_high:
            return None
        return self._get_by_id(self._kline_ids[best_idx])

    def _delete(self, kline_id: int) -> None:
        """Delete a K-line from DB and in-memory embeddings."""
        self._conn.execute("DELETE FROM klines WHERE id = ?", (kline_id,))
        self._conn.commit()
        with self._lock:
            if kline_id in self._kline_ids:
                idx = self._kline_ids.index(kline_id)
                self._kline_ids.pop(idx)
                self._embeddings.pop(idx)
        logger.info("  KLINE deleted id=%d", kline_id)

    @staticmethod
    def _answers_match(a: str, b: str) -> bool:
        """Check if two answers are semantically the same.

        Handles formatting differences like "$4.9 trillion" vs "$4.9T".
        """
        if a == b:
            return True
        na, nb = _normalize_answer(a), _normalize_answer(b)
        if na == nb:
            return True
        # Try numeric comparison
        try:
            return abs(float(na) - float(nb)) < 0.01 * max(abs(float(na)), 1)
        except ValueError:
            return False

    def _get_by_id(self, kline_id: int) -> KLine | None:
        cursor = self._conn.execute(
            "SELECT id, query, facts, answer, verified, confidence, "
            "source, urls, timestamp, access_count FROM klines WHERE id = ?",
            (kline_id,),
        )
        row = cursor.fetchone()
        return self._row_to_kline(row) if row else None

    @staticmethod
    def _row_to_kline(row) -> KLine:
        return KLine(
            id=row[0], query=row[1], facts=json.loads(row[2]),
            answer=row[3], verified=bool(row[4]), confidence=row[5],
            source=row[6], urls=json.loads(row[7]),
            timestamp=row[8], access_count=row[9],
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
