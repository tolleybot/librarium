"""Data models for the Librarian system."""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    """A retrievable span of a document, with provenance.

    ``text`` is the raw span — that is what the *model* reads when a chunk
    is quoted into a prompt. ``prefix`` is the LLM-written context prefix
    that situates the span in its parent document, and **both retrieval
    halves index ``prefixed``**: BM25 through ``_build_bm25``, the encoder
    through ``_refresh_embeddings``.

    This docstring used to say the prefix was "used for keyword search
    only; it would pollute dense vectors, so the two are kept apart" —
    the only recorded reason the split ever had, and an assertion rather
    than a result. Measured the other way, three times, each on one index
    with the prefixes held constant so only the encoder input varied and
    each against a flat ``bm25-only`` control: recall at 10 rose in all
    three runs and more of the expected literals were found in all three.
    Pollution is the effect that did **not** show up; see
    ``benchmarks/README.md`` for the runs, the per-suite split and the one
    suite that lost ground, and ``EMBED_SOURCE`` in ``source.py`` for what
    puts the choice in the cache key. Recorded rather than quietly
    deleted, because a plausible claim that turned out backwards is worth
    more to the next reader than a clean sentence.
    """

    text: str
    doc_id: str = ""     # which document this came from
    offset: int = 0      # character offset of the chunk's new material
    index: int = 0       # position in the librarian's chunk list
    prefix: str = ""     # context prefix for BM25; "" when not generated

    @property
    def prefixed(self) -> str:
        """The chunk as both halves index it: context prefix, then the text."""
        return f"{self.prefix}\n\n{self.text}" if self.prefix else self.text


class Fact(BaseModel):
    """A single extracted fact from source material."""
    content: str
    source: str = "librarian"   # doc_id(s) the fact was retrieved from


class LibrarianAnswer(BaseModel):
    """Response from a Librarian query."""
    items: list[Fact]
    coverage: str = "complete"  # complete | partial | uncertain
