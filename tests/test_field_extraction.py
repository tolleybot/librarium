"""Index-time field extraction — the parser, the critic, and the re-ask.

Four separable things are pinned here:

* :func:`parse_fields` is pure and **does not see the document**. It reads
  the ``{"value": ..., "quote": ...}`` shape and applies the bars that
  need no corpus — placeholders above all, because a stored ``"N/A"`` is
  not an absent field but a value a filter matches and no document equals.
* :func:`verify_fields` is the **mechanical critic**: deterministic string
  work against the document, one check per failure mode measured in
  ``fields-1``. It is not a model, and that is the point — this repo has
  already measured what an LLM critic does when asked whether an answer
  is good enough.
* The **re-ask** hands the model its own failed checks and asks again,
  once. It is a repair request, never "are you sure".
* ``extract_fields`` costs LLM calls and is **off by default**, checked by
  counting calls rather than reading the flag.

Nothing here filters on the fields, because nothing can yet.

The double is content-derived: :class:`FieldsLLM` quotes out
of the document it was actually handed, so a misrouted sample produces a
quote that fails verification rather than something plausible.
"""

from __future__ import annotations

import json
import logging

import pytest

import librarium.source as source_module
from librarium.source import (
    DOC_FIELDS_VERSION,
    DOC_TYPES,
    MAX_FIELD_CHARS,
    MAX_QUOTE_CHARS,
    SourceLibrarian,
    content_hash,
    parse_fields,
    verify_fields,
)
from tests.conftest import stub_models

# ── Fixtures ─────────────────────────────────────────────────────

LETTER = """# {issuer}
**Date:** March 4, 2024
**To:** A Person, 1 Somewhere Street

Effective April 1, 2024, the annual percentage yield on Everyday
Checking accounts changes from 0.45% to 0.35%. Account {account}.

> **Date:** January 9, 2024
> an older message being quoted in this thread
"""


def a_document(issuer: str, account: str = "AP-9912", pad: int = 2500) -> str:
    """A document that reads like one, so quotes can be taken from it."""
    return LETTER.format(issuer=issuer, account=account) + (
        "\n\nclause text. " * (pad // 13)
    )


def vals(fields: dict[str, dict]) -> dict[str, str]:
    """Just the values, for assertions that are not about provenance."""
    return {name: rec["value"] for name, rec in fields.items()}


def a_reply(**overrides) -> str:
    """A well-formed extraction reply, quoting the fixture above."""
    base = {
        "doc_type": {"value": "letter", "quote": "**To:** A Person"},
        "issuer": {"value": "Fabled Rock", "quote": "# Fabled Rock"},
        "issued_date": {"value": "2024-03-04",
                        "quote": "**Date:** March 4, 2024"},
        "effective_date": {"value": "2024-04-01",
                           "quote": "Effective April 1, 2024"},
        "account": {"value": "AP-9912", "quote": "Account AP-9912."},
    }
    base.update(overrides)
    return json.dumps({k: v for k, v in base.items() if v is not None})


# ── Doubles ──────────────────────────────────────────────────────


class FieldsLLM:
    """Summaries, prefixes and field records, all derived from the text.

    The field reply quotes the issuer line out of the document it was
    actually sent, so a librarian that hands the extraction call the wrong
    sample produces a quote that is not in that document and fails
    verification — the identity-carrying property the suite asks of every
    double here.
    """

    def __init__(self, fields_reply: str | None = None,
                 replies: list[str] | None = None):
        self.fields_reply = fields_reply
        self.replies = list(replies) if replies else None
        self.systems: list[str] = []

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        self.systems.append(system)
        if "identifying fields" in system:
            if self.replies:
                n = self.field_calls - 1
                return self.replies[min(n, len(self.replies) - 1)]
            if self.fields_reply is not None:
                return self.fields_reply
            issuer = "unknown"
            for line in user.splitlines():
                if line.startswith("# "):
                    issuer = line[2:].strip()
            return a_reply(
                issuer={"value": issuer, "quote": f"# {issuer}"}
            )
        if "Summarize" in system:
            return f"summary of {len(user)} chars"
        return "context prefix"

    @property
    def field_calls(self) -> int:
        return sum("identifying fields" in s for s in self.systems)

    @property
    def reask_calls(self) -> int:
        return sum("Your previous answer had these problems" in s
                   for s in self.systems)


class ExplodingFieldsLLM(FieldsLLM):
    """The extraction call itself raises; everything else works."""

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        if "identifying fields" in system:
            self.systems.append(system)
            raise RuntimeError("backend said no")
        return await super().complete(
            system, user, think=think, temperature=temperature
        )


def a_librarian(llm, **kw) -> SourceLibrarian:
    kw.setdefault("chunk_size", 800)
    kw.setdefault("chunk_threshold", 0)
    return SourceLibrarian(llm, **kw)


# ── The parser ───────────────────────────────────────────────────


class TestParseFields:
    """What survives :func:`parse_fields`. Pure; the document is not here."""

    def test_it_reads_the_value_and_the_quote(self):
        fields = parse_fields(a_reply())
        assert vals(fields)["issued_date"] == "2024-03-04"
        assert fields["issued_date"]["quote"] == "**Date:** March 4, 2024"

    def test_a_bare_scalar_is_a_value_with_no_quote(self):
        """The fields-1 shape, and a model that ignored the schema.

        Kept rather than rejected so the two failures stay distinct:
        "ignored the schema" and "could not support its answer" want
        different repair messages, and the second is the critic's job.
        """
        fields = parse_fields('{"issuer": "Fabled Rock"}')
        assert fields["issuer"] == {"value": "Fabled Rock", "quote": None}

    def test_json_null_becomes_an_absent_key(self):
        fields = parse_fields(
            '{"doc_type": {"value": "receipt", "quote": "x"}, '
            '"effective_date": {"value": null, "quote": null}}'
        )
        assert "effective_date" not in fields

    @pytest.mark.parametrize(
        "placeholder",
        ["N/A", "n/a", "none", "None", "unknown", "not specified",
         "undated", "-", "TBD", ""],
    )
    def test_a_placeholder_is_dropped_rather_than_stored(self, placeholder):
        """The bar the whole filter rests on.

        An absent key reads as "no evidence, do not exclude". A stored
        ``"N/A"`` is a value a filter compares against and no real
        document ever equals — so the document is excluded from every
        filtered query on the strength of a field it never had.
        """
        fields = parse_fields(json.dumps({
            "doc_type": {"value": "receipt", "quote": "x"},
            "effective_date": {"value": placeholder, "quote": "x"},
        }))
        assert list(fields) == ["doc_type"]

    def test_a_placeholder_word_inside_a_real_value_survives(self):
        """Matched against the whole value, never as a substring."""
        fields = parse_fields(
            '{"issuer": {"value": "None Trust Bank", "quote": "x"}}'
        )
        assert vals(fields) == {"issuer": "None Trust Bank"}

    def test_an_unknown_key_is_dropped(self):
        fields = parse_fields(
            '{"doc_type": {"value": "receipt", "quote": "x"}, '
            '"favourite_colour": {"value": "blue", "quote": "x"}}'
        )
        assert list(fields) == ["doc_type"]

    def test_a_number_is_coerced_but_a_bool_is_refused(self):
        """``bool`` is an ``int`` subclass, so admitting numbers admits it
        by accident — and ``true`` is not an identifier."""
        fields = parse_fields(
            '{"account": {"value": 900123, "quote": "x"}, '
            '"doc_type": {"value": true, "quote": "x"}}'
        )
        assert vals(fields) == {"account": "900123"}

    def test_a_nested_value_is_refused_rather_than_stringified(self):
        fields = parse_fields(
            '{"issuer": {"value": {"name": "C"}, "quote": "x"}, '
            '"period": {"value": "2024-01", "quote": "x"}}'
        )
        assert vals(fields) == {"period": "2024-01"}

    def test_a_value_too_long_to_be_an_identifier_is_dropped(self):
        fields = parse_fields(json.dumps({
            "issuer": {"value": "x" * (MAX_FIELD_CHARS + 1), "quote": "q"},
            "doc_type": {"value": "receipt", "quote": "q"},
        }))
        assert list(fields) == ["doc_type"]

    def test_an_over_long_quote_costs_the_quote_not_the_field(self):
        """Dropping the field over its footnote would throw away a good
        answer; it falls through to the critic as unquoted instead."""
        fields = parse_fields(json.dumps({
            "issuer": {"value": "Fabled Rock", "quote": "x" * (MAX_QUOTE_CHARS + 1)},
        }))
        assert fields["issuer"] == {"value": "Fabled Rock", "quote": None}

    def test_a_reply_with_no_json_yields_nothing(self):
        assert parse_fields("I could not find any fields in this.") == {}

    def test_a_list_payload_yields_nothing(self):
        assert parse_fields('[{"doc_type": "receipt"}]') == {}

    def test_a_fenced_block_and_surrounding_prose_are_tolerated(self):
        fields = parse_fields(
            'Here you go:\n```json\n{"doc_type": {"value": "receipt", '
            '"quote": "x"}}\n```\nHope that helps!'
        )
        assert vals(fields) == {"doc_type": "receipt"}

    def test_a_think_block_does_not_reach_the_parser(self):
        fields = parse_fields(
            '<think>the issuer looks like {"a": 1}</think>'
            '{"doc_type": {"value": "receipt", "quote": "x"}}'
        )
        assert vals(fields) == {"doc_type": "receipt"}


# ── The critic ───────────────────────────────────────────────────


class TestVerifyFields:
    """One check per failure mode ``fields-1`` actually produced.

    Deterministic string work, not a model. This repo measured what an
    LLM critic does when asked whether an answer is good enough:
    ``_find_gap``'s ``SUFFICIENT`` promotes questions to complete
    coverage, 5 of 9 on replies the parser threw away. A string either
    contains another string or does not.
    """

    DOC = a_document("Fabled Rock")

    def test_a_well_supported_reply_survives_with_no_flags(self):
        kept, complaints = verify_fields(parse_fields(a_reply()), self.DOC)
        assert not complaints
        assert set(kept) == {"doc_type", "issuer", "issued_date",
                             "effective_date", "account"}
        assert all(rec["flags"] == [] for rec in kept.values())

    def test_the_two_dates_separate(self):
        """The definitional fix. ``fields-1`` asked for one field meaning
        "takes effect **or** was issued" and got whichever the model
        preferred; the documents label them differently and now so does
        the schema."""
        kept, _ = verify_fields(parse_fields(a_reply()), self.DOC)
        assert kept["issued_date"]["value"] == "2024-03-04"
        assert kept["effective_date"]["value"] == "2024-04-01"

    def test_a_quote_absent_from_the_document_is_rejected(self):
        """Two of nine wrong dates were values the model assembled rather
        than read. A value can be fabricated silently; a quote cannot."""
        kept, complaints = verify_fields(
            parse_fields(a_reply(issued_date={
                "value": "2024-08-12", "quote": "Notice filed August 12, 2024"
            })), self.DOC,
        )
        assert "issued_date" not in kept
        assert any("does not appear in the document" in c for c in complaints)

    def test_a_quote_that_does_not_support_its_value_is_rejected(self):
        kept, complaints = verify_fields(
            parse_fields(a_reply(account={
                "value": "XYZ-9", "quote": "Effective April 1, 2024"
            })), self.DOC,
        )
        assert "account" not in kept
        assert any("does not support" in c for c in complaints)

    def test_a_date_is_checked_for_derivability_not_identity(self):
        """``DOC_FIELDS_SYSTEM`` asks for ISO and documents write prose, so
        the value is never a substring of its own evidence. A substring
        test here rejected every correctly-quoted date."""
        kept, complaints = verify_fields(
            parse_fields(a_reply()), self.DOC
        )
        assert kept["issued_date"]["value"] == "2024-03-04"
        assert not complaints

    def test_a_date_from_the_wrong_line_is_still_rejected(self):
        kept, _ = verify_fields(
            parse_fields(a_reply(issued_date={
                "value": "2024-07-22", "quote": "**Date:** March 4, 2024"
            })), self.DOC,
        )
        assert "issued_date" not in kept

    def test_a_quote_from_quoted_material_is_flagged_not_dropped(self):
        """Four of nine came from a ``Date:`` header inside a ``>``-prefixed
        reply. Flagged rather than dropped: a thread whose every line is
        quoted has no other evidence, and a flag a filter can decline to
        act on beats deleting the only answer."""
        kept, _ = verify_fields(
            parse_fields(a_reply(issued_date={
                "value": "2024-01-09", "quote": "**Date:** January 9, 2024"
            })), self.DOC,
        )
        assert kept["issued_date"]["flags"] == ["quoted_material"]

    def test_a_quote_reflowed_across_a_line_break_still_matches(self):
        """A model re-wraps what it copies, so whitespace is normalised on
        both sides. Everything else is compared exactly."""
        kept, _ = verify_fields(
            parse_fields(a_reply(effective_date={
                "value": "2024-04-01",
                "quote": "Effective April 1, 2024, the annual percentage "
                         "yield on Everyday Checking accounts",
            })), self.DOC,
        )
        assert "effective_date" in kept

    @pytest.mark.parametrize("bogus", ["tenant policy", "gallery notes", "bill"])
    def test_a_doc_type_outside_the_vocabulary_is_dropped(self, bogus):
        """Free text measured at 69 labels over 176 documents is not a key.
        Dropped rather than flagged: a filter matching an unusable key
        silently returns nothing, which is worse than having no key."""
        kept, complaints = verify_fields(
            parse_fields(a_reply(doc_type={"value": bogus, "quote": "**To:**"})),
            self.DOC,
        )
        assert "doc_type" not in kept
        assert any("not one of the allowed values" in c for c in complaints)

    def test_every_allowed_doc_type_survives(self):
        for allowed in DOC_TYPES:
            kept, _ = verify_fields(
                parse_fields(a_reply(
                    doc_type={"value": allowed, "quote": "**To:** A Person"}
                )), self.DOC,
            )
            assert kept["doc_type"]["value"] == allowed

    def test_an_unquoted_value_is_flagged_not_dropped(self):
        """``fields-1`` measured issuer at 158/161 present verbatim, so
        unquoted values are mostly right and dropping them would cost the
        strongest field this extracts."""
        kept, complaints = verify_fields(
            {"issuer": {"value": "Fabled Rock", "quote": None}}, self.DOC
        )
        assert kept["issuer"]["flags"] == ["unquoted"]
        assert any("no quote" in c for c in complaints)


# ── The re-ask ───────────────────────────────────────────────────


class TestTheReAsk:

    @pytest.mark.asyncio
    async def test_a_clean_reply_is_not_re_asked(self, monkeypatch):
        stub_models(monkeypatch)
        llm = FieldsLLM()
        lib = a_librarian(llm, extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        assert llm.field_calls == 1
        assert llm.reask_calls == 0

    @pytest.mark.asyncio
    async def test_a_failed_check_is_handed_back_and_fixed(self, monkeypatch):
        """The loop the whole design rests on: the model is shown the
        specific defect, not asked whether it is sure."""
        stub_models(monkeypatch)
        llm = FieldsLLM(replies=[
            a_reply(issuer={"value": "Elsewhere Bank", "quote": "# Elsewhere Bank"}),
            a_reply(),
        ])
        lib = a_librarian(llm, extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")

        assert llm.reask_calls == 1
        # The extraction prompts specifically: prefix calls come after
        # these, so systems[-1] is one of those.
        asked = [s for s in llm.systems if "identifying fields" in s]
        assert "does not appear in the document" in asked[-1]
        assert lib.doc_fields["d"]["issuer"]["value"] == "Fabled Rock"

    @pytest.mark.asyncio
    async def test_the_re_ask_is_bounded(self, monkeypatch):
        """A model that ignores a named defect twice will ignore it a
        third time, and the call is not free."""
        stub_models(monkeypatch)
        bad = a_reply(issuer={"value": "Elsewhere", "quote": "# Elsewhere"})
        llm = FieldsLLM(replies=[bad, bad, bad, bad])
        lib = a_librarian(llm, extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")

        assert llm.field_calls == 2
        assert "issuer" not in lib.doc_fields.get("d", {})

    @pytest.mark.asyncio
    async def test_a_worse_second_reply_is_discarded(self, monkeypatch):
        """Nothing guarantees a retry improves, and silently taking the
        newer answer would make this loop able to lose information."""
        stub_models(monkeypatch)
        llm = FieldsLLM(replies=[
            # good on everything but account
            a_reply(account={"value": "XYZ", "quote": "Effective April 1, 2024"}),
            # the retry drops most of it
            json.dumps({"doc_type": {"value": "letter", "quote": "**To:**"}}),
        ])
        lib = a_librarian(llm, extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")

        assert set(lib.doc_fields["d"]) >= {"issuer", "issued_date",
                                            "effective_date"}


# ── Opt-in, and never failing a document ─────────────────────────


class TestExtractionIsOptIn:

    @pytest.mark.asyncio
    async def test_off_by_default_makes_no_call_and_records_nothing(
        self, monkeypatch
    ):
        stub_models(monkeypatch)
        llm = FieldsLLM()
        lib = a_librarian(llm)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        assert llm.field_calls == 0
        assert lib.doc_fields == {}
        assert lib.chunks

    @pytest.mark.asyncio
    async def test_on_it_records_fields_from_the_document_it_was_given(
        self, monkeypatch
    ):
        stub_models(monkeypatch)
        llm = FieldsLLM()
        lib = a_librarian(llm, extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="a")
        await lib.add_document(a_document("Northgate Trust"), doc_id="b")
        assert lib.doc_fields["a"]["issuer"]["value"] == "Fabled Rock"
        assert lib.doc_fields["b"]["issuer"]["value"] == "Northgate Trust"


class TestExtractionNeverFailsADocument:
    """A field record is an optimisation for filtering; losing an index to
    a malformed JSON reply would trade the document for a filter."""

    @pytest.mark.asyncio
    async def test_an_unparseable_reply_still_indexes_the_document(
        self, monkeypatch, caplog
    ):
        stub_models(monkeypatch)
        llm = FieldsLLM(fields_reply="Sorry, I can't tell what this is.")
        lib = a_librarian(llm, extract_fields=True)
        with caplog.at_level(logging.WARNING):
            await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        assert lib.doc_fields == {}
        assert lib.retrieve("clause text")
        assert "no verified fields" in caplog.text

    @pytest.mark.asyncio
    async def test_a_failing_call_still_indexes_the_document(
        self, monkeypatch, caplog
    ):
        stub_models(monkeypatch)
        lib = a_librarian(ExplodingFieldsLLM(), extract_fields=True)
        with caplog.at_level(logging.WARNING):
            await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        assert lib.doc_fields == {}
        assert lib.chunks
        assert "field extraction failed" in caplog.text

    @pytest.mark.asyncio
    async def test_a_reply_yielding_nothing_records_no_empty_dict(
        self, monkeypatch
    ):
        """Absent, never present-and-empty. Downstream, absence means
        "no evidence, do not filter this out"."""
        stub_models(monkeypatch)
        llm = FieldsLLM(fields_reply='{"doc_type": {"value": "N/A"}}')
        lib = a_librarian(llm, extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        assert "d" not in lib.doc_fields

    @pytest.mark.asyncio
    async def test_strict_raises_on_an_unparseable_reply(self, monkeypatch):
        stub_models(monkeypatch)
        llm = FieldsLLM(fields_reply="no idea")
        lib = a_librarian(llm, extract_fields=True, strict=True)
        with pytest.raises(ValueError, match="no verified fields"):
            await lib.add_document(a_document("Fabled Rock"), doc_id="d")

    @pytest.mark.asyncio
    async def test_strict_raises_when_the_call_fails(self, monkeypatch):
        stub_models(monkeypatch)
        lib = a_librarian(
            ExplodingFieldsLLM(), extract_fields=True, strict=True
        )
        with pytest.raises(RuntimeError, match="backend said no"):
            await lib.add_document(a_document("Fabled Rock"), doc_id="d")


# ── The cache key ────────────────────────────────────────────────


class TestTheCacheKey:

    def test_extraction_off_hashes_exactly_as_it_did_before_the_knob(self):
        """Every index already on disk was written by a librarian with no
        such knob. If ``extract_fields=False`` contributed anything, an
        upgrade would miss every one of those caches and charge a full
        re-index for a feature nobody turned on."""
        docs = {"a": "hello world"}
        assert content_hash(docs) == content_hash(docs, extract_fields=False)

    def test_extraction_on_is_a_different_index(self):
        docs = {"a": "hello world"}
        assert content_hash(docs, extract_fields=True) != content_hash(docs)


# ── Persistence ──────────────────────────────────────────────────


class TestTheVersionCannotDriftFromWhatItDescribes:
    """The suffix of ``DOC_FIELDS_VERSION`` is derived, not remembered.

    It used to be the literal ``"fields-3"`` with a comment asking the
    next person to bump it when the vocabulary or the parser changed.
    Editing ``DOC_TYPES`` then left ``content_hash`` untouched, so an
    index extracted under the old vocabulary was served as current and its
    records mixed with new ones — the hole ``embed_source`` exists to
    close, rebuilt one field over.
    """

    def _fingerprint_with(self, monkeypatch, **overrides) -> str:
        for name, value in overrides.items():
            monkeypatch.setattr(source_module, name, value)
        return source_module._fields_fingerprint()

    def test_the_baseline_is_stable(self, monkeypatch):
        assert (source_module._fields_fingerprint()
                == source_module._fields_fingerprint())
        assert DOC_FIELDS_VERSION.endswith(source_module._fields_fingerprint())

    def test_a_new_doc_type_moves_it(self, monkeypatch):
        """The case that motivated this. DOC_TYPES is interpolated into
        the prompt, so the prompt is what the digest actually sees."""
        before = source_module._fields_fingerprint()
        after = self._fingerprint_with(
            monkeypatch,
            DOC_FIELDS_SYSTEM=source_module.DOC_FIELDS_SYSTEM.replace(
                "insurance policy", "insurance policy, warranty card",
            ),
        )
        assert after != before

    def test_a_new_field_moves_it(self, monkeypatch):
        before = source_module._fields_fingerprint()
        after = self._fingerprint_with(
            monkeypatch, DOC_FIELD_NAMES=DOC_FIELD_NAMES_PLUS,
        )
        assert after != before

    def test_a_changed_placeholder_bar_moves_it(self, monkeypatch):
        """The bar decides whether "N/A" is stored as a value, which
        changes what every record means."""
        before = source_module._fields_fingerprint()
        after = self._fingerprint_with(
            monkeypatch,
            DOC_FIELD_PLACEHOLDERS=frozenset(
                source_module.DOC_FIELD_PLACEHOLDERS | {"pending"}
            ),
        )
        assert after != before

    def test_a_changed_length_cap_moves_it(self, monkeypatch):
        before = source_module._fields_fingerprint()
        assert self._fingerprint_with(monkeypatch, MAX_FIELD_CHARS=121) != before

    def test_placeholder_order_does_not_move_it(self, monkeypatch):
        """Sorted before hashing: a frozenset's iteration order is not a
        change to what the bar admits, and a version that moved with it
        would invalidate every index on an interpreter restart."""
        before = source_module._fields_fingerprint()
        shuffled = frozenset(reversed(sorted(
            source_module.DOC_FIELD_PLACEHOLDERS
        )))
        assert self._fingerprint_with(
            monkeypatch, DOC_FIELD_PLACEHOLDERS=shuffled) == before


DOC_FIELD_NAMES_PLUS = (
    "doc_type", "issuer", "issued_date", "effective_date", "period",
    "account", "recipient",
)


class TestPersistence:

    @pytest.mark.asyncio
    async def test_fields_and_their_provenance_survive_a_round_trip(
        self, monkeypatch, tmp_path
    ):
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)

        restored = a_librarian(FieldsLLM(), extract_fields=True)
        restored.load(tmp_path)
        assert restored.doc_fields == lib.doc_fields
        assert restored.doc_fields["d"]["issued_date"]["quote"]

    @pytest.mark.asyncio
    async def test_the_version_that_produced_them_is_recorded(
        self, monkeypatch, tmp_path
    ):
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)
        meta = json.loads((tmp_path / "index.json").read_text())
        assert meta["doc_fields_version"] == DOC_FIELDS_VERSION

    @pytest.mark.asyncio
    async def test_an_index_with_no_fields_records_no_version(
        self, monkeypatch, tmp_path
    ):
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM())
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)
        meta = json.loads((tmp_path / "index.json").read_text())
        assert meta["doc_fields_version"] == ""

    @pytest.mark.asyncio
    async def test_fields_from_another_version_are_discarded_and_reported(
        self, monkeypatch, tmp_path, caplog
    ):
        """``fields-1`` records are exactly this case: same keys, different
        meaning, and a filter applied to them would exclude documents on a
        comparison nobody checked."""
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)

        meta = json.loads((tmp_path / "index.json").read_text())
        meta["doc_fields_version"] = "fields-1"
        (tmp_path / "index.json").write_text(json.dumps(meta))

        restored = a_librarian(FieldsLLM(), extract_fields=True)
        with caplog.at_level(logging.WARNING):
            restored.load(tmp_path)
        assert restored.doc_fields == {}
        assert "discarding them" in caplog.text
        assert restored.chunks

    @pytest.mark.asyncio
    async def test_an_index_predating_the_feature_loads_silently(
        self, monkeypatch, tmp_path, caplog
    ):
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM())
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)

        meta = json.loads((tmp_path / "index.json").read_text())
        del meta["doc_fields"], meta["doc_fields_version"]
        (tmp_path / "index.json").write_text(json.dumps(meta))

        restored = a_librarian(FieldsLLM(), extract_fields=True)
        with caplog.at_level(logging.WARNING):
            restored.load(tmp_path)
        assert restored.doc_fields == {}
        assert "discarding" not in caplog.text

    @pytest.mark.asyncio
    async def test_a_placeholder_edited_into_the_file_is_re_filtered(
        self, monkeypatch, tmp_path
    ):
        """The file is editable, so the parser's bars are re-applied on the
        way back in rather than trusted."""
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)

        meta = json.loads((tmp_path / "index.json").read_text())
        meta["doc_fields"]["d"]["issuer"]["value"] = "N/A"
        (tmp_path / "index.json").write_text(json.dumps(meta))

        restored = a_librarian(FieldsLLM(), extract_fields=True)
        restored.load(tmp_path)
        assert "issuer" not in restored.doc_fields["d"]

    @pytest.mark.asyncio
    async def test_a_hand_written_record_reads_as_unverified(
        self, monkeypatch, tmp_path
    ):
        """A record with no recorded flags has unknown provenance, which is
        what a filter should be cautious about — not an empty list, which
        means "checked and clean"."""
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        lib.save(tmp_path)

        meta = json.loads((tmp_path / "index.json").read_text())
        meta["doc_fields"]["d"]["issuer"].pop("flags", None)
        (tmp_path / "index.json").write_text(json.dumps(meta))

        restored = a_librarian(FieldsLLM(), extract_fields=True)
        restored.load(tmp_path)
        assert restored.doc_fields["d"]["issuer"]["flags"] == ["unverified"]


# ── Index mutations ──────────────────────────────────────────────


class TestFieldsFollowTheirDocument:

    @pytest.mark.asyncio
    async def test_a_rolled_back_document_takes_its_fields_with_it(
        self, monkeypatch
    ):
        stub_models(monkeypatch)

        class FailsOnThePrefixes(FieldsLLM):
            async def complete(self, system, user, *, think=False,
                               temperature=0.1):
                if "context prefix" in system and "boom" in user:
                    raise RuntimeError("boom")
                return await super().complete(
                    system, user, think=think, temperature=temperature
                )

        lib = a_librarian(FailsOnThePrefixes(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="good")
        with pytest.raises(RuntimeError):
            await lib.add_document(
                "boom " + a_document("Northgate Trust"), doc_id="bad"
            )
        assert "bad" not in lib.doc_fields
        assert "bad" not in lib.documents
        assert lib.doc_fields["good"]["issuer"]["value"] == "Fabled Rock"

    @pytest.mark.asyncio
    async def test_a_reset_drops_them(self, monkeypatch):
        stub_models(monkeypatch)
        lib = a_librarian(FieldsLLM(), extract_fields=True)
        await lib.add_document(a_document("Fabled Rock"), doc_id="d")
        assert lib.doc_fields
        await lib.index(a_document("Northgate Trust"), doc_id="fresh")
        assert list(lib.doc_fields) == ["fresh"]
