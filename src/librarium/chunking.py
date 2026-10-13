"""Text chunking and tokenization utilities.

Chunking guarantees every returned chunk is at most ``chunk_size``
characters.  Boundaries are preferred in this order: paragraph, then
sentence, then whitespace, then a hard cut.  Documents that are mostly
tabular — markdown tables, CSV dumps, code blocks, minified text — have
paragraphs that exceed the limit on their own, and those get split
rather than emitted oversized.

NLTK corpora are fetched lazily, on first use, not at import time.  Set
``LIBRARIUM_OFFLINE=1`` to forbid the download entirely: tokenization
then falls back to regex splitting, which needs no corpora and no
network.
"""

from __future__ import annotations

import logging
import os
import re

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)

_STEMMER = PorterStemmer()

# Defaults
CHUNK_SIZE = 2000    # chars (~500 tokens)
CHUNK_OVERLAP = 200  # chars

SEPARATOR = "\n\n"   # joins paragraphs inside a chunk

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_FALLBACK_RE = re.compile(r"\w+", re.UNICODE)

_downloaded: set[str] = set()
_stopword_cache: set[str] | None = None
_warned_offline = False


# ── NLTK data, fetched on demand ─────────────────────────────────


def _offline() -> bool:
    return os.environ.get("LIBRARIUM_OFFLINE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ensure_nltk(pkg: str) -> bool:
    """Download an NLTK corpus on demand. True if it is usable afterwards."""
    global _warned_offline
    if pkg in _downloaded:
        return True
    if _offline():
        if not _warned_offline:
            logger.warning(
                "LIBRARIUM_OFFLINE is set — skipping NLTK download of %r, "
                "falling back to regex tokenization", pkg,
            )
            _warned_offline = True
        return False
    try:
        ok = bool(nltk.download(pkg, quiet=True))
    except Exception as e:  # no network, read-only home, proxy refusal
        logger.warning("NLTK download of %r failed (%s) — using regex fallback", pkg, e)
        return False
    if ok:
        _downloaded.add(pkg)
    return ok


def _stopwords() -> set[str]:
    """Load the stopword list, falling back to none when it is unavailable.

    The retry lives inside the ``try`` for the same reason it does in
    ``_words``: ``_ensure_nltk`` reports whether the *download*
    succeeded, not whether the corpus is now findable on
    ``nltk.data.path``.  The two disagree whenever the download
    directory and the search path are configured independently — a
    writable ``NLTK_DATA`` nltk.download does not write to, a truncated
    archive that still returned 200 — and a retry outside the ``try``
    would raise ``LookupError`` out of ``tokenize()`` and kill indexing
    rather than degrade it.  An empty set costs slightly worse BM25
    tokenization; that is the documented fallback.
    """
    global _stopword_cache
    if _stopword_cache is None:
        for retry in (False, True):
            try:
                _stopword_cache = set(nltk_stopwords.words("english"))
                break
            except LookupError:
                if retry or not _ensure_nltk("stopwords"):
                    _stopword_cache = set()
                    break
    return _stopword_cache


def _words(text: str) -> list[str]:
    """Word-tokenize, falling back to regex when punkt is unavailable."""
    for retry in (False, True):
        try:
            return word_tokenize(text)
        except LookupError:
            if retry or not _ensure_nltk("punkt_tab"):
                break
    return _WORD_FALLBACK_RE.findall(text)


def _sentences(text: str) -> list[str]:
    """Sentence-split, falling back to regex when punkt is unavailable."""
    for retry in (False, True):
        try:
            return sent_tokenize(text)
        except LookupError:
            if retry or not _ensure_nltk("punkt_tab"):
                break
    return [s for s in _SENTENCE_FALLBACK_RE.split(text) if s.strip()]


# ── Chunking ─────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks, breaking at paragraph boundaries.

    Every returned chunk is at most *chunk_size* characters — including
    the carried-over overlap, and including paragraphs that exceed the
    limit on their own.
    """
    return [chunk for chunk, _ in chunk_spans(text, chunk_size, overlap)]


def chunk_spans(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int]]:
    """Like :func:`chunk_text`, but each chunk is paired with its offset.

    The offset is the character position in *text* where the chunk's
    first unit of *new* material starts.  A chunk may repeat up to
    *overlap* characters from immediately before that point, so the
    offset marks where the chunk starts contributing, not the first
    character it contains.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    spans: list[tuple[str, int]] = []
    current = ""
    current_offset = 0
    cursor = 0  # how far into `text` we have located material

    for para in _PARAGRAPH_RE.split(text):
        para = para.strip()
        if not para:
            continue
        for piece in _split_to_size(para, chunk_size):
            offset = _locate(text, piece, cursor)
            cursor = offset + len(piece)
            if current and len(current) + len(SEPARATOR) + len(piece) > chunk_size:
                spans.append((current, current_offset))
                current = _carry_overlap(current, piece, chunk_size, overlap)
                current_offset = offset
            elif current:
                current = current + SEPARATOR + piece
            else:
                current = piece
                current_offset = offset

    if current:
        spans.append((current, current_offset))
    return spans


def _locate(text: str, piece: str, cursor: int) -> int:
    """Best-effort offset of *piece* in *text*, searching forward from *cursor*.

    Pieces are stripped and may have had internal whitespace normalized
    by sentence re-packing, so an exact match is not guaranteed; the
    probe uses a prefix and falls back to the cursor.
    """
    probe = piece[:60]
    found = text.find(probe, max(0, cursor - 1))
    return found if found != -1 else min(cursor, len(text))


def _carry_overlap(prev: str, piece: str, chunk_size: int, overlap: int) -> str:
    """Start a new chunk with a tail of the previous one, staying in budget."""
    room = chunk_size - len(piece) - len(SEPARATOR)
    tail = min(overlap, len(prev), room)
    if tail <= 0:
        return piece
    return prev[-tail:] + SEPARATOR + piece


def _split_to_size(text: str, size: int) -> list[str]:
    """Break *text* into pieces of at most *size* chars, largest-first.

    Sentence boundaries are preferred; a single sentence longer than
    *size* is split on whitespace, and failing that mid-word.
    """
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    for sentence in _sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= size:
            _pack(pieces, sentence, size)
        else:
            for fragment in _hard_split(sentence, size):
                _pack(pieces, fragment, size)
    return pieces or _hard_split(text, size)


def _pack(pieces: list[str], part: str, size: int, sep: str = " ") -> None:
    """Append *part*, merging it into the last piece when it still fits."""
    if pieces and len(pieces[-1]) + len(sep) + len(part) <= size:
        pieces[-1] = pieces[-1] + sep + part
    else:
        pieces.append(part)


def _hard_split(text: str, size: int) -> list[str]:
    """Split with no regard for sentences — whitespace if possible, else mid-word."""
    out: list[str] = []
    start = 0
    while start < len(text):
        if start + size >= len(text):
            out.append(text[start:].strip())
            break
        window = text[start : start + size]
        cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        # A break point in the first half wastes more than it buys.
        if cut < size // 2:
            cut = size
        out.append(text[start : start + cut].strip())
        start += cut
        while start < len(text) and text[start].isspace():
            start += 1
    return [piece for piece in out if piece]


# ── Tokenization ─────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Tokenize for BM25: lowercase, drop stopwords and punctuation, stem."""
    stops = _stopwords()
    return [
        _STEMMER.stem(token)
        for token in _words(text.lower())
        if token.isalnum() and token not in stops
    ]
