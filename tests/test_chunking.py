"""chunk_text / chunk_spans / tokenize — no model, no network."""

from __future__ import annotations

import pytest

from librarium.chunking import CHUNK_SIZE, chunk_spans, chunk_text, tokenize

TABLE_ROW = "| " + " | ".join(f"col{i}" for i in range(20)) + " |"

# Documents that have historically produced oversized chunks. Normal
# prose never trips it; anything tabular or unbroken does.
PATHOLOGICAL = {
    "unbroken blob": "x" * 10_000,
    "normal prose": "\n\n".join(["word " * 60] * 40),
    "markdown table": "\n".join([TABLE_ROW] * 400),
    "two long paragraphs": "word " * 500 + "\n\n" + "other " * 500,
    "long sentences": "The bakery opened in 1987. " * 400,
    "no trailing break": "a" * 2500 + "\n\n" + "b" * 100,
}


class TestChunkSize:
    @pytest.mark.parametrize("name", sorted(PATHOLOGICAL))
    def test_no_chunk_exceeds_size(self, name):
        chunks = chunk_text(PATHOLOGICAL[name])
        assert chunks
        assert all(len(c) <= CHUNK_SIZE for c in chunks), (
            f"{name}: largest chunk was {max(len(c) for c in chunks)}"
        )

    def test_custom_size_respected(self):
        chunks = chunk_text("word " * 2000, chunk_size=300, overlap=50)
        assert chunks
        assert all(len(c) <= 300 for c in chunks)

    def test_oversized_paragraph_loses_nothing(self):
        doc = "z" * 10_000
        chunks = chunk_text(doc)
        assert len(chunks) > 1
        assert "".join(chunks) == doc

    def test_overlap_repeats_tail_of_previous_chunk(self):
        doc = ("alpha " * 250).strip() + "\n\n" + ("beta " * 250).strip()
        chunks = chunk_text(doc, chunk_size=2000, overlap=200)
        assert len(chunks) == 2
        assert chunks[1].startswith(chunks[0][-200:])

    def test_overlap_never_pushes_a_chunk_over_budget(self):
        # A paragraph longer than chunk_size - overlap used to produce a
        # chunk of size chunk_size + overlap.
        doc = "\n\n".join(["q" * 1900] * 6)
        chunks = chunk_text(doc, chunk_size=2000, overlap=200)
        assert all(len(c) <= 2000 for c in chunks)

    def test_sentences_are_preferred_break_points(self):
        chunks = chunk_text("The bakery opened in 1987. " * 200,
                            chunk_size=1000, overlap=0)
        assert len(chunks) > 1
        assert all(c.endswith(".") for c in chunks)

    def test_empty_and_whitespace_documents(self):
        assert chunk_text("") == []
        assert chunk_text("\n\n   \n\n") == []

    def test_rejects_nonpositive_size(self):
        with pytest.raises(ValueError):
            chunk_text("anything", chunk_size=0)


class TestChunkSpans:
    DOC = "\n\n".join(f"Section {i}. " + "filler words here. " * 40
                      for i in range(12))

    def test_offset_points_at_the_chunks_new_material(self):
        spans = chunk_spans(self.DOC)
        assert spans
        for chunk, offset in spans:
            assert self.DOC[offset : offset + 30] in chunk

    def test_offsets_advance(self):
        offsets = [offset for _, offset in chunk_spans(self.DOC)]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0

    def test_matches_chunk_text(self):
        assert [c for c, _ in chunk_spans(self.DOC)] == chunk_text(self.DOC)


class TestTokenize:
    def test_drops_stopwords(self):
        tokens = tokenize("The bakery is on the corner and it was open")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "wa" not in tokens  # "was" stemmed, had it survived

    def test_drops_punctuation(self):
        assert all(t.isalnum() for t in tokenize("Revenue: $4.50, up 12%!"))

    def test_stemming_matches_inflections(self):
        # This is what lets a BM25 query for "bakery" hit "bakeries".
        assert set(tokenize("Bakeries opened")) & set(tokenize("bakery open"))

    def test_case_insensitive(self):
        assert tokenize("NORTHWIND Dental") == tokenize("northwind dental")

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("!!! ??? ...") == []
