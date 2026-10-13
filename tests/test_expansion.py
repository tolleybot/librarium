"""Query expansion — the fusion, the call, and the failure behaviour.

Three things are being pinned here, and they are deliberately separable
because the API is:

* ``retrieve(..., also=[...])`` fuses extra phrasings and makes **no LLM
  call**, so most of this file runs against a client that raises if it is
  touched at all.
* ``expand_query()`` makes the one call and never fails a question,
  unless ``strict``.
* ``retrieve_expanded()`` is the composition of the two and nothing else.

The doubles are content-derived: :class:`ExpansionLLM`
rewrites the question *it was actually given* through a vocabulary table
and takes its line count from the prompt it was actually sent, so a
misrouted question or an ignored ``expansion_variants`` fails an
assertion rather than quietly returning something plausible.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from librarium import Chunk, Librarian, SourceLibrarian
from librarium.source import (
    HYDE_SYSTEM,
    EXPANSION_TEMPERATURE,
    MAX_EXPANSION_WORDS,
    distinct_variants,
    parse_expansion,
)
from tests.conftest import MockLLM

# ── Corpus ───────────────────────────────────────────────────────
#
# Eight short documents on unrelated subjects. The last one is the
# paraphrase probe's target: it is about motor insurance and contains
# not one word of the vocabulary a person would search with — no "car",
# no "insurance", no "excess", and none of them in the doc_id either.
# The others are distractors that DO carry some of that vocabulary, so a
# question asked literally ranks them above the document that actually
# answers it.
PROBE_DOCS = {
    "bakery.md": "The bakery on Mill Street sells 120 sourdough loaves each "
                 "morning at $4.50 each and opened in 1987.",
    "commute.md": "Staff who drive share the car park behind the depot; "
                  "spaces are allocated by seniority and there is no excess "
                  "capacity on Fridays.",
    "health.md": "The staff health plan is administered by Cascade Benefits "
                 "and covers dental and optical claims up to $2,000.",
    "travel.md": "Travel bookings go through the agency portal. Rail fares "
                 "are reimbursed in full; taxi fares need a receipt.",
    "payroll.md": "Payroll runs on the 28th. Overtime is paid the following "
                  "month and pension contributions are deducted at source.",
    "office.md": "The Mill Street office is open 8am to 6pm. Reception "
                 "takes deliveries and holds keys for the meeting rooms.",
    "security.md": "Badges must be worn at all times. Lost badges are "
                   "replaced by the facilities desk within one working day.",
    "northwind.md": "Northwind Mutual auto policy 41-B covers the vehicle "
                    "against collision. The deductible is $500 and "
                    "comprehensive coverage adds $250.",
}

# The probe. Every word of it is absent from every document.
PROBE_QUESTION = "car insurance excess"


# ── Doubles ──────────────────────────────────────────────────────


class ExpansionLLM:
    """Expansion client whose variants are derived from what it was sent.

    Two content-derived properties, both load-bearing:

    * the reply is a rewrite of the question in the *user* turn, so a
      misrouted question (the document summary prompt, an empty string,
      the previous question) produces variants that retrieve the wrong
      chunks and fail the caller's assertion. A constant-string double
      would fuse a ranking and return chunks either way.
    * the number of lines is read out of the *system* prompt, so an
      ``expansion_variants`` that never reached ``EXPANSION_SYSTEM``
      shows up as the wrong number of variants.

    It also refuses to be used for anything but expansion, so a test that
    accidentally leaves it in place over indexing fails loudly instead of
    filling the index with paraphrases.
    """

    #: One synonym per requested variant, so variant *i* is a consistent
    #: vocabulary swap rather than a random one.
    VOCABULARY = {
        "car": ["auto", "vehicle", "motor"],
        "insurance": ["policy", "collision", "cover"],
        "excess": ["deductible", "coverage", "franchise"],
        "bill": ["invoice", "statement", "account"],
        "office": ["practice", "premises", "surgery"],
    }

    _N_RE = re.compile(r"Output ONLY (\d+) rewritten")

    def __init__(self, reply: str | None = None):
        #: Set to return a fixed reply instead — for the malformed cases,
        #: where the point is that the model did *not* answer the prompt.
        self.reply = reply
        self.calls: list[tuple[str, str, float]] = []

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        self.calls.append((system, user, temperature))
        assert "rewrite one search question" in system, (
            "this double only answers the expansion prompt; it was sent "
            f"something else: {system[:80]!r}"
        )
        if self.reply is not None:
            return self.reply
        assert user.startswith("QUESTION: "), user[:80]
        question = user[len("QUESTION: "):]
        n = int(self._N_RE.search(system).group(1))
        return "\n".join(self.variant(question, i) for i in range(n))

    @classmethod
    def variant(cls, question: str, i: int) -> str:
        """The *i*-th vocabulary swap of *question*, word by word.

        Unknown words are echoed, so a variant is always recognisably
        the paraphrase of one particular question.
        """
        return " ".join(
            cls.VOCABULARY.get(word, [word])[i % len(cls.VOCABULARY.get(word, [word]))]
            for word in question.split()
        )


class ExplodingLLM:
    """A client that fails the test if anything calls it.

    ``retrieve()`` is documented as making zero LLM calls and that has to
    stay true with ``also=`` in play; ``expand_query=False`` is
    documented as not calling the model at all. Both are assertions about
    a call that does *not* happen, which only a client like this can make.
    """

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        raise AssertionError(
            "the model was called on a path that must not call it"
        )


# ── Fixtures ─────────────────────────────────────────────────────


async def probe_librarian(*, llm=None, **settings) -> SourceLibrarian:
    """PROBE_DOCS indexed one chunk per document, BM25-only.

    ``chunk_threshold=0`` so the short documents chunk at all; the dense
    half is off so every assertion below is about the lexical rankings
    expansion actually contributes to.

    Defaults to ``expansion_style="paraphrase"``. Every caller here
    predates the hyde style and asserts against phrasings — the parser's
    bars, the variant count, the refusal handling — none of which a
    passage goes through. A caller wanting hyde passes it explicitly, so
    the pin is visible at the one place it is decided rather than
    repeated at twelve call sites.
    """
    settings.setdefault("expansion_style", "paraphrase")
    lib = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0, **settings)
    lib._dense_ok = False
    for doc_id, text in PROBE_DOCS.items():
        await lib.add_document(text, doc_id=doc_id)
    if llm is not None:
        lib.llm = llm  # indexing is done; only expansion may call now
    return lib


def doc_ids(chunks) -> list[str]:
    return [c.doc_id for c in chunks]


# ── The fusion, with no model anywhere ───────────────────────────


class TestFusionWithoutAModel:
    """``also=`` is the whole point of the split API: the fusion path has
    to be reachable with no model, no mock client and no ``await``.
    """

    async def test_a_variant_ranking_changes_what_is_retrieved(self):
        # Catches: `also=` accepted and ignored — a signature that takes
        # the phrasings and never builds a ranking from them. The
        # question alone cannot reach northwind.md; the variant can.
        lib = await probe_librarian(llm=ExplodingLLM())
        plain = doc_ids(lib.retrieve(PROBE_QUESTION, top_k=3))
        fused = doc_ids(lib.retrieve(PROBE_QUESTION, top_k=3,
                                     also=["auto policy deductible",
                                           "vehicle collision coverage"]))
        assert "northwind.md" not in plain
        assert "northwind.md" in fused

    async def test_retrieve_with_also_makes_no_llm_call(self):
        # Catches: an implementation that quietly expands inside
        # retrieve(), which would make the documented "no LLM call,
        # synchronous, lock-free" contract false and break the consumer
        # that calls retrieve() from a thread.
        lib = await probe_librarian(llm=ExplodingLLM())
        assert lib.retrieve(PROBE_QUESTION, top_k=2, also=["auto policy"])

    async def test_retrieve_with_also_is_still_synchronous(self):
        # Catches: retrieve() turning into a coroutine. The existing
        # consumers call it from sync code; a coroutine would come back
        # un-awaited and silently retrieve nothing.
        lib = await probe_librarian(llm=ExplodingLLM())
        result = lib.retrieve(PROBE_QUESTION, also=["auto policy"])
        assert isinstance(result, list)

    async def test_a_bare_string_also_is_refused(self):
        # Catches the sharpest edge of Sequence[str]: a plain string
        # iterates into characters, so `also="auto policy"` would build
        # one BM25 ranking per letter and retrieve noise without any
        # error at all.
        lib = await probe_librarian(llm=ExplodingLLM())
        with pytest.raises(TypeError, match="not a single string"):
            lib.retrieve(PROBE_QUESTION, also="auto policy")

    async def test_a_non_string_variant_is_refused(self):
        # Catches a variant list assembled from Chunks or dicts by a
        # consumer: tokenize() would raise from somewhere much deeper.
        lib = await probe_librarian(llm=ExplodingLLM())
        with pytest.raises(TypeError, match="also must contain strings"):
            lib.retrieve(PROBE_QUESTION, also=["auto policy", 7])

    async def test_the_whole_document_path_ignores_also(self):
        # Catches an `also` that reaches the degraded path and raises
        # from an index that is not there: a short document is served
        # whole, where no ranking — and so no variant — applies.
        lib = SourceLibrarian(MockLLM())
        lib.llm = ExplodingLLM()
        await lib.index("One short line about an auto policy.", doc_id="d.md")
        assert lib.retrieval_mode == "whole-document"
        assert (doc_ids(lib.retrieve("car insurance", also=["auto policy"]))
                == doc_ids(lib.retrieve("car insurance")))


class TestExistingCallersAreUntouched:
    """``retrieve(q)`` and ``retrieve(q, top_k=n)`` behave exactly as
    before. The rest of the suite pins the values; these pin that the new
    argument's *default* is what produces them.
    """

    async def test_the_default_also_is_the_no_variant_path(self):
        # Catches a default that is not empty — `also=None` handled as
        # "expand", or a mutable default that accumulates.
        lib = await probe_librarian(llm=ExplodingLLM())
        assert (doc_ids(lib.retrieve("Northwind Mutual"))
                == doc_ids(lib.retrieve("Northwind Mutual", also=[])))

    async def test_top_k_still_caps_the_fused_result(self):
        # Catches a fusion that slices before it fuses, or not at all:
        # more rankings must not mean more chunks.
        lib = await probe_librarian(llm=ExplodingLLM())
        assert len(lib.retrieve(PROBE_QUESTION, top_k=2,
                                also=["auto policy", "vehicle collision"])) == 2

    async def test_also_cannot_be_passed_positionally(self):
        # Catches `also` sliding into third position, where a caller who
        # miscounted would land a value on it. Keyword-only per the spec.
        lib = await probe_librarian(llm=ExplodingLLM())
        with pytest.raises(TypeError, match="positional argument"):
            lib.retrieve(PROBE_QUESTION, 2, ["auto policy"])

    async def test_retrieval_mode_keeps_its_four_strings(self):
        # Catches "hybrid+expanded" and friends: anything already
        # matching on the four documented strings must keep working with
        # expansion switched on.
        lib = await probe_librarian(expand_query=True, expansion_style="paraphrase")
        assert lib.retrieval_mode == "bm25-only"
        blank = SourceLibrarian(MockLLM(), expand_query=True, expansion_style="paraphrase")
        assert blank.retrieval_mode == "unindexed"
        whole = SourceLibrarian(MockLLM(), expand_query=True, expansion_style="paraphrase")
        await whole.index("A short line.")
        assert whole.retrieval_mode == "whole-document"


# ── The one failure mode that would make retrieval worse ─────────


class TestTheQuestionSurvivesTheExpansion:
    """A drifting expansion must never displace a correct literal match.

    Exact-entity lookup is the case BM25 is best at and the case a
    paraphrase is most likely to wreck, so this is the criterion that
    decides whether the feature is safe to turn on at all.
    """

    async def test_a_useless_variant_does_not_displace_a_literal_hit(self):
        # Catches a fusion in which a variant that matches nothing still
        # casts a full-length vote. BM25 scores every chunk zero for
        # "zzz nonsense", and a ranking of all-equal scores is just index
        # order — so an expansion that found nothing quietly votes for
        # whichever chunk happens to be first.
        lib = await probe_librarian(llm=ExplodingLLM())
        literal = doc_ids(lib.retrieve("Northwind Mutual", top_k=3))
        assert literal[0] == "northwind.md", "premise: the literal hit wins"

        with_noise = doc_ids(lib.retrieve("Northwind Mutual", top_k=3,
                                          also=["zzz nonsense"]))
        assert with_noise[0] == "northwind.md", (
            f"a useless expansion displaced the literal hit: {with_noise}"
        )

    async def test_a_useless_variant_does_not_evict_the_hit_entirely(self):
        # The same failure one step worse: not merely reordered, but
        # pushed out of the returned set, so the model answers from
        # material that never mentioned what was asked about.
        lib = await probe_librarian(llm=ExplodingLLM())
        with_noise = doc_ids(lib.retrieve("Northwind Mutual", top_k=3,
                                          also=["zzz nonsense"]))
        assert "northwind.md" in with_noise

    async def test_the_question_outranks_a_variant_on_a_tie(self):
        # Catches a rankings list built variant-first. RRF breaks ties by
        # first appearance, so the order the list is assembled in is what
        # makes "the question is always present" concrete.
        lib = await probe_librarian(llm=ExplodingLLM())
        both = doc_ids(lib.retrieve("bakery loaves", top_k=1,
                                    also=["badges facilities desk"]))
        assert both == ["bakery.md"]


class TestScoredTruncation:
    """``_bm25_ranking(..., scored=True)`` and the two rules around it.

    A query that matches nothing still produces a full ``sorted()``
    ranking, because every score is equal and ``sorted`` falls back to
    index order. As the only ranking that is harmless filler; as one vote
    among several it beats real hits. The truncation that fixes it is
    switched on by ``also=`` being non-empty and applies to the
    question's own ranking as well as the variants', which is a visible
    behaviour change for callers that pass ``also=`` and no change at all
    for callers that do not. Both halves of that are pinned here.
    """

    async def test_without_also_the_ranking_keeps_its_unscored_filler(self):
        # The byte-identity guarantee, stated as the property that makes
        # it true. "Northwind Mutual" scores exactly one chunk, and every
        # existing caller of retrieve() gets top_k back regardless —
        # four of them index-order filler nothing matched.
        #
        # Catches two different breaks. `scored=` made unconditional (the
        # tidier rule the implementer deliberately did not take) cuts
        # this to one chunk, silently shrinking every existing caller's
        # result. And `also=` removed or renamed fails the equality,
        # which is the byte-for-byte criterion spelled out.
        lib = await probe_librarian(llm=ExplodingLLM())
        plain = doc_ids(lib.retrieve("Northwind Mutual", top_k=5))
        assert len(plain) == 5, plain
        assert plain[0] == "northwind.md"
        assert doc_ids(lib.retrieve("Northwind Mutual", top_k=5, also=[])) == plain

    async def test_with_also_the_result_is_not_padded_to_top_k(self):
        # The disclosed consequence, pinned so it cannot change silently
        # in either direction. "Northwind Mutual" scores one chunk and
        # "zzz nonsense" scores none, so one chunk is the whole of what
        # the phrasings between them asked for. Catches padding added
        # back (index-order filler returning as though it were ranked)
        # and catches the truncation being dropped again.
        lib = await probe_librarian(llm=ExplodingLLM())
        with_noise = doc_ids(lib.retrieve("Northwind Mutual", top_k=5,
                                          also=["zzz nonsense"]))
        assert with_noise == ["northwind.md"], with_noise
        assert len(doc_ids(lib.retrieve("Northwind Mutual", top_k=5))) == 5

    async def test_the_questions_own_ranking_is_truncated_too(self):
        # The symmetry the implementer chose: truncating the variants
        # alone would leave the question voting index order for chunks it
        # never scored. Two phrasings, each with exactly one real hit, so
        # a surviving filler vote on either side shows up as extra chunks
        # nothing matched.
        lib = await probe_librarian(llm=ExplodingLLM())
        both = doc_ids(lib.retrieve("Northwind Mutual", top_k=5,
                                    also=["bakery loaves"]))
        assert both == ["northwind.md", "bakery.md"], both

    async def test_when_nothing_matches_at_all_the_ranking_is_taken_back(self):
        # The take-back guard. Question and variant both score zero, so
        # the truncated lexical ranking is empty — and an empty lexical
        # ranking with no dense half leaves the fusion nothing, which
        # falls through to _whole_documents() and hands back all eight
        # documents whole, ignoring top_k. Catches the guard being
        # deleted: expansion is not allowed to leave a question with less
        # than it started with.
        lib = await probe_librarian(llm=ExplodingLLM())
        alone = doc_ids(lib.retrieve("zzz qqq", top_k=5))
        with_useless = doc_ids(lib.retrieve("zzz qqq", top_k=5,
                                            also=["www vvv"]))
        assert with_useless == alone
        assert len(with_useless) == 5, with_useless
        # Whole documents would be all eight, ignoring top_k entirely.
        assert len(lib._whole_documents()) == 8

    async def test_a_bm25_index_behind_the_chunks_skips_the_variants_too(self):
        # The staleness guard, which is live rather than defensive,
        # crossed with the new argument. Catches `also=` changing what
        # the degraded path returns — a future "the lexical half is
        # stale, so at least rank the variants" would hand back chunks
        # here instead of whole documents, and the documented
        # consequence (ask() raising from check_context in this window)
        # would quietly stop being true.
        #
        # Stated honestly: this does NOT catch the variant loop being
        # lifted out of the `bm25_ranked is not None` arm. _bm25_ranking
        # returns None before it can index past the short score array, so
        # every variant returns None too and the result is unchanged.
        # What it pins is the degraded path's output, not the guard.
        lib = await probe_librarian(llm=ExplodingLLM())
        # A chunk bm25 was never rebuilt over — the same window a sweep
        # opens between extending `chunks` and rebuilding the index.
        lib.chunks.append(Chunk(text="An unscored latecomer about auto "
                                     "policy cover.", doc_id="late.md",
                                offset=0, index=len(lib.chunks)))
        assert lib.bm25.corpus_size < len(lib.chunks), "premise: stale index"

        chunks = lib.retrieve("Northwind Mutual", top_k=3,
                              also=["auto policy", "vehicle collision"])

        # Same degradation `retrieve()` alone gives here: whole
        # documents, top_k not applied, and `also` making no difference.
        assert doc_ids(chunks) == doc_ids(lib.retrieve("Northwind Mutual",
                                                       top_k=3))
        assert doc_ids(chunks) == list(lib.documents)

    async def test_the_take_back_does_not_fire_when_a_variant_scored(self):
        # Catches the guard written against the question's ranking alone
        # (`if not bm25_ranked`) rather than against every lexical
        # ranking: the question's index-order filler would come back and
        # drown the one chunk a phrasing genuinely found.
        lib = await probe_librarian(llm=ExplodingLLM())
        found = doc_ids(lib.retrieve("zzz qqq", top_k=5,
                                     also=["auto policy"]))
        assert found == ["northwind.md"], found


class TestTheShortfallIsLexicalOnly:
    """The residual asymmetry, recorded as behaviour rather than as prose.

    ``_embedding_ranking`` ranks every chunk whatever the query, so a
    hybrid index has a full-length vote in the fusion and the lexical
    shortfall never reaches the caller. Only BM25-only indexes return
    fewer than ``top_k``.
    """

    #: Content-derived: each text embeds as the counts of
    #: marker words it actually contains, spread across the corpus, so a
    #: chunk embedded from the wrong text scores differently.
    MARKERS = ("bakery", "car", "policy", "badge", "payroll", "travel",
               "dental")

    class MarkerEmbedder:
        def encode(self, texts, show_progress_bar: bool = False):
            return np.array(
                [[float(t.lower().count(m))
                  for m in TestTheShortfallIsLexicalOnly.MARKERS]
                 for t in texts],
                dtype=np.float32,
            )

    async def hybrid(self) -> SourceLibrarian:
        lib = SourceLibrarian(MockLLM(["A prefix."]), chunk_threshold=0)
        lib.embed_model = self.MarkerEmbedder()
        for doc_id, text in PROBE_DOCS.items():
            await lib.add_document(text, doc_id=doc_id)
        assert lib.retrieval_mode == "hybrid", "premise: the dense half is on"
        return lib

    async def test_the_dense_half_fills_what_the_lexical_side_dropped(self):
        # The same query that returns one chunk BM25-only returns a full
        # top_k here. Catches the truncation spreading to the dense
        # ranking, which would shrink every hybrid result that used
        # `also=` without anyone asking for it.
        lib = await self.hybrid()
        with_noise = doc_ids(lib.retrieve("Northwind Mutual", top_k=5,
                                          also=["zzz nonsense"]))
        assert len(with_noise) == 5, with_noise
        assert "northwind.md" in with_noise

    async def test_the_literal_hit_still_leads_on_the_hybrid_path(self):
        # The dense half votes for chunks nothing lexical asked for, so
        # the guarantee that a useless expansion cannot displace a
        # literal match has to hold with it in play too.
        lib = await self.hybrid()
        with_noise = doc_ids(lib.retrieve("Northwind Mutual", top_k=5,
                                          also=["zzz nonsense"]))
        assert with_noise[0] == "northwind.md", with_noise


class TestVariantHygiene:
    """Empty, repeated and question-identical phrasings are dropped
    before fusion. A duplicate is not free: RRF sums 1/(k+rank) over the
    rankings it is given, so the same phrasing twice doubles its vote.
    """

    async def test_a_variant_equal_to_the_question_changes_nothing(self):
        # Catches the doubled vote directly: if the repeat survived, the
        # question's own ranking would be counted twice and the result
        # would differ from the single-ranking one.
        lib = await probe_librarian(llm=ExplodingLLM())
        assert (doc_ids(lib.retrieve("Northwind Mutual", top_k=3,
                                     also=["Northwind Mutual"]))
                == doc_ids(lib.retrieve("Northwind Mutual", top_k=3)))

    async def test_case_and_whitespace_do_not_smuggle_a_duplicate_past(self):
        # Catches an exact-string dedup: "Auto  Policy" and "auto policy"
        # tokenize identically, so they are the same ranking twice.
        assert distinct_variants("car insurance",
                                 ["auto policy", "Auto  Policy"]) \
            == ["auto policy"]

    def test_empty_and_blank_variants_are_dropped(self):
        assert distinct_variants("car insurance", ["", "   ", "auto policy"]) \
            == ["auto policy"]

    def test_the_question_itself_is_dropped(self):
        assert distinct_variants("car insurance",
                                 ["Car Insurance", "auto policy"]) \
            == ["auto policy"]

    def test_order_is_preserved(self):
        # Fusion is order-deterministic, which requires the variant list
        # to keep the order the model produced.
        assert distinct_variants("q", ["c", "a", "b"]) == ["c", "a", "b"]


class TestReplyParsing:
    """``parse_expansion`` drops what must not become a search query.

    Same bargain as ``parse_gap_query``: a line that is not a query is
    dropped rather than fed to the index.
    """

    QUESTION = "car insurance excess"

    @pytest.mark.parametrize("reply,expected", [
        ("auto policy\nvehicle cover", ["auto policy", "vehicle cover"]),
        # Models number a list however they like, whatever was asked.
        ("1. auto policy\n2) vehicle cover", ["auto policy", "vehicle cover"]),
        ("- auto policy\n* vehicle cover", ["auto policy", "vehicle cover"]),
        ("• auto policy", ["auto policy"]),
        ('"auto policy"', ["auto policy"]),
        # A trailing colon is a model announcing a list, never a query.
        ("Here are two phrasings:\nauto policy", ["auto policy"]),
        ("\n\nauto policy\n\n", ["auto policy"]),
        # Models double-space a list as often as they don't; a blank line
        # between two queries must not end up as a third, empty one.
        ("auto policy\n\nvehicle cover", ["auto policy", "vehicle cover"]),
    ])
    def test_a_line_is_cleaned_not_rejected(self, reply, expected):
        # Catches a parser that rejects any line carrying a list marker,
        # which is most replies from most models.
        assert parse_expansion(reply, self.QUESTION, 2) == expected

    def test_a_think_block_never_becomes_a_query(self):
        # Catches the one path where an unstripped <think> block does not
        # fail loudly: reasoning is line-oriented prose and would parse
        # straight into the ranking list.
        reply = ("<think>The user means motor cover, so I should say "
                 "auto.</think>\nauto policy")
        assert parse_expansion(reply, self.QUESTION, 2) == ["auto policy"]

    def test_an_overlong_line_is_dropped(self):
        # Catches a sentence of explanation used as a query: BM25 would
        # retrieve whatever its non-stopwords happen to hit.
        essay = " ".join(f"word{i}" for i in range(MAX_EXPANSION_WORDS + 1))
        assert parse_expansion(f"auto policy\n{essay}", self.QUESTION, 2) \
            == ["auto policy"]

    def test_an_overlong_line_says_so(self, caplog):
        essay = " ".join(f"word{i}" for i in range(MAX_EXPANSION_WORDS + 1))
        with caplog.at_level("WARNING", logger="librarium.source"):
            parse_expansion(essay, self.QUESTION, 2)
        assert "too long to be a paraphrase" in caplog.text

    @pytest.mark.parametrize("reply", ["", "   \n\n  ", "Here are some:"])
    def test_an_unusable_reply_yields_nothing(self, reply):
        assert parse_expansion(reply, self.QUESTION, 2) == []

    def test_the_limit_caps_what_comes_back(self):
        # Catches a limit applied before dedup, which would return fewer
        # variants than asked for whenever the model repeated itself.
        reply = "auto policy\ncar insurance excess\nvehicle cover\nmotor plan"
        assert parse_expansion(reply, self.QUESTION, 2) \
            == ["auto policy", "vehicle cover"]


class TestARefusalIsNotAQuery:
    """A refusal parsed as a query is worse than no expansion: it
    retrieves on *cannot*, *help* and *request*, and those rankings fuse
    with the question's.

    **Two independent bars stop it, and the tests here keep them
    separable.** The first is sentence punctuation — comma, colon,
    semicolon — the bar ``parse_gap_query`` already holds an unprefixed
    reply to. The second is ``_REFUSAL_RE``, because the punctuation bar
    leaves "Sorry I cannot do that": no comma, five words, well inside a
    40-word cap sized for paraphrases.

    Every case below is written so that exactly one bar can be
    responsible for it — the punctuated cases carry no refusal marker
    where that is the point, and the refusal cases carry no comma, colon
    or semicolon at all. A regression in either bar therefore fails its
    own tests rather than hiding behind the other.
    """
    # Pinned to the paraphrase style: this class is about
    # EXPANSION_SYSTEM's prompt, parse_expansion's bars, or the refusal
    # handling that only a phrasing goes through. The default style is
    # hyde, whose reply is a passage and bypasses the parser entirely, so
    # without the pin these would be asserting paraphrase behaviour
    # against a path that never runs it.

    QUESTION = "car insurance excess"

    # ── Bar one: sentence punctuation ────────────────────────────

    @pytest.mark.parametrize("reply", [
        "I cannot help with that request, sorry.",
        "I am sorry, but I cannot assist with that.",
        "I can't do that; ask me something else.",
        "Sorry: I have no documents to work from.",
    ])
    def test_a_punctuated_refusal_yields_nothing(self, reply):
        # Catches a refusal reaching the fusion as a phrasing. These four
        # trip *both* bars, so they no longer isolate the punctuation one
        # — the marker-free cases below do that.
        assert parse_expansion(reply, self.QUESTION, 2) == []

    @pytest.mark.parametrize("reply", [
        "auto policy, motor cover",
        "which deductible applies; comprehensive or collision",
        "vehicle cover: the excess",
    ])
    def test_punctuated_prose_goes_even_with_no_refusal_marker(self, reply):
        # Isolates bar one. Not one of these carries "sorry", "I cannot"
        # or "as an AI", so _REFUSAL_RE cannot be what drops them: delete
        # _PROSE_PUNCT_RE and this test fails while every refusal test
        # below still passes.
        assert parse_expansion(reply, self.QUESTION, 2) == []

    # ── Bar two: the refusal marker ──────────────────────────────

    @pytest.mark.parametrize("reply", [
        "Sorry I cannot do that",
        "As an AI language model I do not have access to your documents",
        "I am unable to help with this request",
        "I'm unable to assist with that",
        "I can't do that",
        "Cannot comply with this request",
        "Unable to assist",
        "I apologize but this is not something I can do",
    ])
    def test_an_unpunctuated_refusal_yields_nothing(self, reply):
        # Isolates bar two, and is the test the whole _REFUSAL_RE change
        # exists for. Every one of these is comma-, colon- and
        # semicolon-free and well under MAX_EXPANSION_WORDS, so before
        # the marker bar landed each parsed straight into a phrasing and
        # BM25 retrieved on *sorry*, *cannot* and *do*. Delete the
        # _REFUSAL_RE check and all eight fail.
        assert not any(p in reply for p in ",:;"), reply  # bar one is out
        assert parse_expansion(reply, self.QUESTION, 2) == []

    @pytest.mark.parametrize("reply", [
        "the agent said sorry about the delay",
        "the agent will apologize for the delay",
        "the agent apologised for the delay",
        # The plural *noun*, not the verb — "Apologies, I cannot ..." is
        # the second-most common way a model opens a refusal after
        # "Sorry", and the verb-only arm did not reach it.
        "the agent sent apologies for the delay",
        "written as an AI assistant would write it",
        "answering as a language model would",
    ])
    def test_an_apology_or_self_identification_counts_anywhere(self, reply):
        # One case per unanchored arm of _REFUSAL_RE, placed mid-line on
        # purpose. Catches an arm dropped from the alternation, and
        # catches the whole pattern being anchored with ^, which would
        # readmit "As an AI language model I cannot ..." the moment a
        # model prefixed it with anything at all. The "apologies" case
        # additionally catches the apology arm narrowing back to the
        # verb inflections (apologise/apologize/apologised/apologized),
        # which is what the docs already promised it did not.
        assert parse_expansion(reply, self.QUESTION, 2) == []

    @pytest.mark.parametrize("refusal,phrasing", [
        # "I ..." / "I'm ..." / "I am ..." — how an assistant refuses.
        ("I cannot renew the policy online",
         "customer cannot renew the policy online"),
        ("I can't find the deductible",
         "why can't a driver find the deductible"),
        ("I am unable to open the portal",
         "the policyholder is unable to open the portal"),
        ("I'm unable to open the portal",
         "renewal is unable to complete online"),
        ("I am not able to open the portal",
         "the driver is not able to open the portal"),
        ("I do not have the policy number",
         "the claim does not have a policy number"),
    ])
    def test_a_modal_counts_only_against_a_first_person_subject(
        self, refusal, phrasing
    ):
        # Both halves are load-bearing and they fail in opposite
        # directions. The first fails if the modal arms are dropped from
        # _REFUSAL_RE. The second fails if the bar is widened to the bare
        # word list ["cannot", "can't", "unable", ...] — "cannot" and
        # "unable" are ordinary words in a support corpus, and a
        # word-list bar would eat these legitimate rewrites.
        assert parse_expansion(refusal, self.QUESTION, 2) == []
        assert parse_expansion(phrasing, self.QUESTION, 2) == [phrasing]

    @pytest.mark.parametrize("refusal,phrasing", [
        ("Cannot comply with this request",
         "renewal cannot be completed online"),
        ("Can not comply with this request",
         "the form can not be submitted twice"),
        ("Unable to assist", "the driver is unable to assist"),
        ("Not able to assist", "the driver is not able to assist"),
    ])
    def test_a_modal_also_counts_at_the_very_start_of_a_line(
        self, refusal, phrasing
    ):
        # The other anchor: a rewrite does not begin "Cannot ...". Same
        # two-way bracket — the first half fails if the ^ arm goes, the
        # second if the arm loses its anchor and becomes a word list.
        assert parse_expansion(refusal, self.QUESTION, 2) == []
        assert parse_expansion(phrasing, self.QUESTION, 2) == [phrasing]

    @pytest.mark.parametrize("phrasing", [
        "auto policy",
        "vehicle deductible",
        "motor cover franchise",
        "Northwind Mutual auto policy 41-B",
    ])
    def test_an_ordinary_phrasing_still_survives_both_bars(self, phrasing):
        # The bars are only worth having if the paraphrases the feature
        # exists to produce get through. Catches a bar widened until it
        # swallows the vocabulary swaps themselves — including the
        # entity-preserving one, which carries a digit-and-letter product
        # code the marker regex must not touch.
        assert parse_expansion(phrasing, self.QUESTION, 2) == [phrasing]

    @pytest.mark.parametrize("reply,cost", [
        # The line-start clause, which is the loosest of the three and
        # the one to revisit first if this bites in practice.
        ("unable to connect error message", "line-start modal"),
        # The "I" here is the *user*, not the model — a perfectly good
        # rewrite of a support question, dropped anyway.
        ("I cannot access my policy documents", "first-person modal"),
        # "sorry" counts anywhere, so a phrasing about the word itself
        # goes too.
        ("sorry we are closed sign", "bare apology"),
    ])
    def test_the_refusal_bar_costs_these_legitimate_phrasings(self, reply, cost):
        # Disclosed false positives, pinned so they are a decision on
        # record rather than folklore. Each one is a phrasing a real
        # corpus would ask for, and each is dropped. Catches both
        # directions: the bar disappearing (they would come back), and
        # anyone "fixing" one of these three without deciding what the
        # bar is for.
        assert parse_expansion(reply, self.QUESTION, 2) == [], cost

    @pytest.mark.parametrize("reply", [
        "Here are two phrasings:",
        "Options:",
        "Alternative queries you could try:",
    ])
    def test_an_announcement_line_is_still_dropped(self, reply):
        # The trailing-colon check was removed as redundant once the
        # punctuation rule landed. Catches its removal having been wrong
        # — a preamble line becoming the only "phrasing" the model
        # supplied, which then retrieves on "here", "are" and "two".
        assert parse_expansion(reply, self.QUESTION, 2) == []

    def test_the_bar_costs_a_comma_bearing_phrasing(self):
        # The deliberate trade, pinned so it is a decision and not a
        # surprise: a legitimate phrasing that wanted a comma goes too.
        # Catches the rule being narrowed back to line-final colons,
        # which would readmit every refusal above.
        reply = "auto policy, motor cover\nvehicle deductible"
        assert parse_expansion(reply, self.QUESTION, 2) == ["vehicle deductible"]

    def test_one_reply_shows_both_bars_and_the_survivor(self):
        # The two bars and the pass-through in a single reply, which is
        # what a hedging model actually sends. Line one trips
        # punctuation only, line two trips the refusal marker only, line
        # three trips neither. Catches either bar going missing — remove
        # one and its line joins the result.
        reply = ("auto policy, motor cover\n"
                 "Sorry I cannot rewrite that\n"
                 "vehicle deductible")
        assert parse_expansion(reply, self.QUESTION, 3) == ["vehicle deductible"]

    def test_a_refusal_drops_silently_so_the_whole_reply_is_logged_first(
        self, caplog
    ):
        # Per-line prose is dropped without a record on purpose: the
        # WARNING that shows what the model actually said has to be the
        # first one a reader sees. Catches the refusal surviving the
        # parse, and catches a per-line warning added here, which would
        # arrive ahead of the whole-reply one and bury it.
        with caplog.at_level("WARNING", logger="librarium.source"):
            dropped = parse_expansion("I cannot help with that, sorry.",
                                      self.QUESTION, 2)
        assert dropped == []
        assert caplog.records == [], caplog.text

    def test_an_unpunctuated_refusal_drops_just_as_silently(self, caplog):
        # Same contract for the marker bar: no per-line record, so
        # expand_query's whole-reply WARNING stays the first thing a
        # reader sees. Catches a logger.warning added to the _REFUSAL_RE
        # branch, which would bury the record that shows what the model
        # actually said — and catches the refusal surviving the parse.
        with caplog.at_level("WARNING", logger="librarium.source"):
            dropped = parse_expansion("Sorry I cannot do that",
                                      self.QUESTION, 2)
        assert dropped == []
        assert caplog.records == [], caplog.text

    # ── End to end, on a reply with no punctuation to fall back on ──

    UNPUNCTUATED = "Sorry I cannot do that"

    async def test_an_unpunctuated_refusal_degrades_and_warns(self, caplog):
        # The consequence the marker bar exists to produce. Without
        # _REFUSAL_RE this reply parses into one phrasing, expand_query
        # returns it, no WARNING is emitted at all, and BM25 fuses a
        # ranking built from *sorry*, *cannot* and *do*.
        lib = await probe_librarian(llm=ExpansionLLM(reply=self.UNPUNCTUATED),
                                    expand_query=True)
        with caplog.at_level("WARNING", logger="librarium.source"):
            assert await lib.expand_query(PROBE_QUESTION) == []
        records = [r for r in caplog.records
                   if r.name == "librarium.source"
                   and r.levelname in ("WARNING", "ERROR")
                   and "expansion" in r.getMessage()]
        assert records, caplog.text
        assert "nothing usable" in records[0].getMessage()

    async def test_an_unpunctuated_refusal_raises_under_strict(self):
        # strict is the deployment that wants to hear about it. Catches
        # the refusal parsing: strict would accept it as a phrasing and
        # raise nothing.
        lib = await probe_librarian(llm=ExpansionLLM(reply=self.UNPUNCTUATED),
                                    expand_query=True, strict=True)
        with pytest.raises(ValueError, match="no usable phrasings"):
            await lib.expand_query(PROBE_QUESTION)

    async def test_an_unpunctuated_refusal_retrieves_for_the_question_alone(
        self,
    ):
        # The degradation itself. "excess" appears in commute.md, so a
        # refusal fused as a phrasing moves the ranking rather than
        # leaving it alone — this is an assertion with teeth, not an
        # identity that holds either way.
        lib = await probe_librarian(llm=ExpansionLLM(reply=self.UNPUNCTUATED),
                                    expand_query=True)
        assert (doc_ids(await lib.retrieve_expanded(PROBE_QUESTION, top_k=3))
                == doc_ids(lib.retrieve(PROBE_QUESTION, top_k=3)))

    async def test_a_refused_reply_degrades_and_warns(self, caplog):
        # End to end: the criterion is "malformed, empty, **or refused**".
        # Catches a refusal that parses into a variant — expand_query
        # would return it, no WARNING would be emitted, and the
        # degradation the spec asks for would never happen.
        refusal = "I cannot help with that request, sorry."
        lib = await probe_librarian(llm=ExpansionLLM(reply=refusal),
                                    expand_query=True)
        with caplog.at_level("WARNING", logger="librarium.source"):
            assert await lib.expand_query(PROBE_QUESTION) == []
        records = [r for r in caplog.records
                   if r.name == "librarium.source"
                   and r.levelname in ("WARNING", "ERROR")
                   and "expansion" in r.getMessage()]
        assert records, caplog.text
        assert "nothing usable" in records[0].getMessage()

    async def test_a_refused_reply_raises_under_strict(self):
        # The other half of the convention. Catches a refusal that
        # parses: strict would silently accept it as a phrasing.
        refusal = "I cannot help with that request, sorry."
        lib = await probe_librarian(llm=ExpansionLLM(reply=refusal),
                                    expand_query=True, strict=True)
        with pytest.raises(ValueError, match="no usable phrasings"):
            await lib.expand_query(PROBE_QUESTION)

    async def test_a_refused_reply_retrieves_for_the_question_alone(self):
        # The degradation itself, not just the return value: the answer
        # must be the one the unexpanded question would have produced.
        refusal = "I cannot help with that request, sorry."
        lib = await probe_librarian(llm=ExpansionLLM(reply=refusal),
                                    expand_query=True)
        assert (doc_ids(await lib.retrieve_expanded("Northwind Mutual", top_k=3))
                == doc_ids(lib.retrieve("Northwind Mutual", top_k=3)))


# ── The call ─────────────────────────────────────────────────────


class TestExpansionIsOptIn:
    async def test_off_by_default(self):
        # Catches a default flipped to True, which would start spending a
        # call per question on a deployment that never asked for one.
        assert SourceLibrarian(MockLLM()).expansion_enabled is False
        assert Librarian(MockLLM()).source.expansion_enabled is False

    async def test_expand_query_off_makes_no_llm_call(self):
        # The criterion stated as a call that must not happen, asserted
        # against a client that raises if it is.
        lib = await probe_librarian(llm=ExplodingLLM())
        assert await lib.expand_query(PROBE_QUESTION) == []

    async def test_retrieve_expanded_off_makes_no_llm_call(self):
        lib = await probe_librarian(llm=ExplodingLLM())
        assert (doc_ids(await lib.retrieve_expanded("Northwind Mutual", top_k=2))
                == doc_ids(lib.retrieve("Northwind Mutual", top_k=2)))

    async def test_the_flag_does_not_shadow_the_method(self):
        # Catches binding the constructor argument to `self.expand_query`:
        # the coroutine the setting exists to enable would become a bool
        # and `await lib.expand_query(q)` would raise "not callable".
        lib = SourceLibrarian(MockLLM(), expand_query=True, expansion_style="paraphrase")
        assert callable(lib.expand_query)


class TestTheExpansionCall:
    # Pinned to the paraphrase style: this class is about
    # EXPANSION_SYSTEM's prompt, parse_expansion's bars, or the refusal
    # handling that only a phrasing goes through. The default style is
    # hyde, whose reply is a passage and bypasses the parser entirely, so
    # without the pin these would be asserting paraphrase behaviour
    # against a path that never runs it.
    async def test_one_call_returns_the_models_phrasings(self):
        # Catches a misrouted question: the double derives its reply from
        # the user turn it was sent, so variants of anything else fail here.
        llm = ExpansionLLM()
        lib = await probe_librarian(llm=llm, expand_query=True)
        assert await lib.expand_query(PROBE_QUESTION) == [
            "auto policy deductible", "vehicle collision coverage",
        ]
        assert len(llm.calls) == 1

    async def test_expansion_variants_reaches_the_prompt(self):
        # Catches a setting stored and never used: the double reads the
        # count out of EXPANSION_SYSTEM, so a prompt still asking for the
        # default returns two variants where three were configured.
        llm = ExpansionLLM()
        lib = await probe_librarian(llm=llm, expand_query=True,
                                    expansion_variants=3)
        assert len(await lib.expand_query("car insurance excess")) == 3

    async def test_the_prompt_warns_about_the_punctuation_it_will_drop(self):
        # The parser drops a comma-bearing line *silently*, so a model
        # that lists its vocabulary swaps inline ("auto policy, motor
        # cover") loses every one of them and never learns why. The
        # prompt has to say so — GAP_SYSTEM sets that precedent with
        # "Never explain. Never write a sentence." Asserted against the
        # prompt the librarian actually sent, together with the parser
        # behaviour it describes, so the two cannot drift apart: catches
        # the clause being dropped from EXPANSION_SYSTEM while the
        # comma rule stays.
        llm = ExpansionLLM()
        lib = await probe_librarian(llm=llm, expand_query=True)
        await lib.expand_query(PROBE_QUESTION)
        system = llm.calls[0][0]
        assert "comma" in system.lower(), system
        # The rule the sentence is promising, still enforced.
        assert parse_expansion("auto policy, motor cover",
                               PROBE_QUESTION, 2) == []

    async def test_the_call_uses_the_expansion_temperature(self):
        # Catches an inherited 0.1, which the spec explicitly refuses to
        # accept as a decision: at 0.1 the model returns the question back
        # and the variants add nothing to the fusion.
        llm = ExpansionLLM()
        lib = await probe_librarian(llm=llm, expand_query=True)
        await lib.expand_query(PROBE_QUESTION)
        assert llm.calls[0][2] == EXPANSION_TEMPERATURE

    async def test_indexing_still_calls_at_the_low_temperature(self):
        # Catches EXPANSION_TEMPERATURE leaking into _llm_call's default,
        # which would loosen every prefix and summary in the library.
        class Recording(MockLLM):
            def __init__(self):
                super().__init__(["A prefix."])
                self.temperatures: list[float] = []

            async def complete(self, system, user, *, think=False,
                               temperature=0.1):
                self.temperatures.append(temperature)
                return await super().complete(system, user, think=think,
                                              temperature=temperature)

        llm = Recording()
        lib = SourceLibrarian(llm, chunk_threshold=0)
        lib._dense_ok = False
        await lib.add_document(PROBE_DOCS["bakery.md"], doc_id="bakery.md")
        assert llm.temperatures and set(llm.temperatures) == {0.1}

    async def test_an_empty_question_is_not_sent_to_the_model(self):
        # Catches a call spent on nothing: there is no vocabulary in a
        # blank question to vary.
        lib = await probe_librarian(llm=ExplodingLLM(), expand_query=True)
        assert await lib.expand_query("   ") == []


class TestExpansionNeverFailsAQuestion:
    """A malformed, empty or failed expansion degrades to single-query
    retrieval — and says so. Silent degradation is the specific thing
    being designed against: a quiet failure is indistinguishable from an
    expansion that did not help, which would make the feature impossible
    to evaluate.
    """
    # Pinned to the paraphrase style: this class is about
    # EXPANSION_SYSTEM's prompt, parse_expansion's bars, or the refusal
    # handling that only a phrasing goes through. The default style is
    # hyde, whose reply is a passage and bypasses the parser entirely, so
    # without the pin these would be asserting paraphrase behaviour
    # against a path that never runs it.

    MALFORMED = "Sure! Here are the alternative phrasings you asked for:"

    async def test_a_malformed_reply_degrades_to_the_question_alone(self):
        # Catches a parser that lets a preamble line through as a query,
        # and an expand_query that raises on an unusable reply.
        llm = ExpansionLLM(reply=self.MALFORMED)
        lib = await probe_librarian(llm=llm, expand_query=True)
        assert await lib.expand_query(PROBE_QUESTION) == []

    async def test_a_malformed_reply_retrieves_exactly_as_before(self):
        llm = ExpansionLLM(reply=self.MALFORMED)
        lib = await probe_librarian(llm=llm, expand_query=True)
        assert (doc_ids(await lib.retrieve_expanded("Northwind Mutual", top_k=3))
                == doc_ids(lib.retrieve("Northwind Mutual", top_k=3)))

    async def test_a_malformed_reply_is_logged(self, caplog):
        # The criterion: not merely degraded, but visibly degraded. The
        # record is asserted on, not just the text, so a DEBUG line
        # nobody's handler would show does not satisfy it.
        llm = ExpansionLLM(reply=self.MALFORMED)
        lib = await probe_librarian(llm=llm, expand_query=True)
        with caplog.at_level("WARNING", logger="librarium.source"):
            await lib.expand_query(PROBE_QUESTION)
        records = [r for r in caplog.records
                   if r.name == "librarium.source"
                   and r.levelname in ("WARNING", "ERROR")
                   and "expansion" in r.getMessage()]
        assert records, caplog.text
        assert "nothing usable" in records[0].getMessage()

    async def test_a_failing_client_degrades_and_logs(self, caplog):
        # Catches an expansion that takes the whole question down with
        # it: an optional recall aid must not be able to fail an answer.
        class Failing:
            async def complete(self, system, user, *, think=False,
                               temperature=0.1):
                raise RuntimeError("backend down")

        lib = await probe_librarian(llm=Failing(), expand_query=True)
        with caplog.at_level("ERROR", logger="librarium.source"):
            assert await lib.expand_query(PROBE_QUESTION) == []
        assert "backend down" in caplog.text
        assert "original question alone" in caplog.text

    async def test_the_call_is_budgeted(self, caplog):
        # Catches the one call that skips check_context. It is small, but
        # "small" is how a call ends up unchecked — and the check has to
        # happen before the client is touched, not after.
        llm = ExpansionLLM()
        lib = await probe_librarian(llm=llm, expand_query=True,
                                    max_context_tokens=1,
                                    reserved_output_tokens=0)
        with caplog.at_level("ERROR", logger="librarium.source"):
            assert await lib.expand_query(PROBE_QUESTION) == []
        assert llm.calls == []
        assert "max_context_tokens" in caplog.text


class TestStrict:
    """``strict`` raises where the default degrades — the same convention
    the embed failures already follow.
    """
    # Pinned to the paraphrase style: this class is about
    # EXPANSION_SYSTEM's prompt, parse_expansion's bars, or the refusal
    # handling that only a phrasing goes through. The default style is
    # hyde, whose reply is a passage and bypasses the parser entirely, so
    # without the pin these would be asserting paraphrase behaviour
    # against a path that never runs it.

    async def test_strict_raises_on_an_unusable_reply(self):
        llm = ExpansionLLM(reply="Here are the phrasings:")
        lib = await probe_librarian(llm=llm, expand_query=True, strict=True)
        with pytest.raises(ValueError, match="no usable phrasings"):
            await lib.expand_query(PROBE_QUESTION)

    async def test_strict_re_raises_a_client_failure(self):
        class Failing:
            async def complete(self, system, user, *, think=False,
                               temperature=0.1):
                raise RuntimeError("backend down")

        lib = await probe_librarian(llm=Failing(), expand_query=True,
                                    strict=True)
        with pytest.raises(RuntimeError, match="backend down"):
            await lib.expand_query(PROBE_QUESTION)

    async def test_the_default_degrades_instead(self):
        # The pair: without strict the same reply is a no-op, not an error.
        llm = ExpansionLLM(reply="Here are the phrasings:")
        lib = await probe_librarian(llm=llm, expand_query=True)
        assert await lib.expand_query(PROBE_QUESTION) == []


class TestRetrieveExpanded:
    """A thin wrapper with no logic of its own, so that the two halves
    used directly give an identical result.
    """

    async def test_it_is_exactly_the_composition_of_the_halves(self):
        # Catches logic added to the wrapper — a different top_k, a
        # second fusion, a filter — which the split path would skip.
        lib = await probe_librarian(llm=ExpansionLLM(), expand_query=True)
        variants = await lib.expand_query(PROBE_QUESTION)
        assert (doc_ids(await lib.retrieve_expanded(PROBE_QUESTION, top_k=3))
                == doc_ids(lib.retrieve(PROBE_QUESTION, top_k=3, also=variants)))

    async def test_top_k_reaches_the_retrieval(self):
        lib = await probe_librarian(llm=ExpansionLLM(), expand_query=True)
        assert len(await lib.retrieve_expanded(PROBE_QUESTION, top_k=2)) == 2

    async def test_it_makes_exactly_one_llm_call(self):
        # Catches expansion growing rounds: it is one scoped call before
        # one retrieval, explicitly not a refine loop.
        llm = ExpansionLLM()
        lib = await probe_librarian(llm=llm, expand_query=True)
        await lib.retrieve_expanded(PROBE_QUESTION)
        assert len(llm.calls) == 1


# ── The probe that proves the feature works ──────────────────────


class TestParaphraseProbe:
    """A document findable only through a semantic neighbour.

    ``northwind.md`` answers "car insurance excess" and contains none of
    those words — not in its body, not in its doc_id. The distractors do
    carry them. Everything else in this file proves expansion is wired
    correctly; this proves it does the thing it exists for.
    """
    # Pinned to the paraphrase style: this class is about
    # EXPANSION_SYSTEM's prompt, parse_expansion's bars, or the refusal
    # handling that only a phrasing goes through. The default style is
    # hyde, whose reply is a passage and bypasses the parser entirely, so
    # without the pin these would be asserting paraphrase behaviour
    # against a path that never runs it.

    def test_the_probe_vocabulary_really_is_absent(self):
        # The premise, asserted rather than assumed: if a later edit put
        # "insurance" into the target document the probe below would
        # still pass, and would no longer mean anything.
        target = PROBE_DOCS["northwind.md"].lower()
        for word in PROBE_QUESTION.split():
            assert word not in target, word
            assert word not in "northwind.md"

    async def test_the_question_alone_cannot_find_it(self):
        # The half that establishes the miss. Without it "found with
        # expansion" would pass against a corpus the question already
        # retrieved fine.
        lib = await probe_librarian(llm=ExplodingLLM())
        assert "northwind.md" not in doc_ids(
            lib.retrieve(PROBE_QUESTION, top_k=3)
        )

    async def test_expansion_finds_it(self):
        # Catches every way the feature could be inert end to end: the
        # flag not read, the variants not fused, the prompt not sent, the
        # reply not parsed. One call, through the public API.
        lib = await probe_librarian(llm=ExpansionLLM(), expand_query=True)
        found = doc_ids(await lib.retrieve_expanded(PROBE_QUESTION, top_k=3))
        assert "northwind.md" in found

    async def test_it_is_the_expansion_that_finds_it_not_the_truncation(self):
        # The probe's own control, and the reason it still means what it
        # says after the ranking truncation landed. `also=` switches the
        # truncation on whatever the phrasings are, so a nonsense variant
        # strips exactly the same competitors a useful one does — and
        # northwind.md still does not appear. It appears only when a
        # phrasing actually carries the document's vocabulary.
        #
        # Catches the probe degenerating into a test of the truncation:
        # without this, "found with expansion" would keep passing if the
        # variants stopped reaching the fusion altogether.
        lib = await probe_librarian(llm=ExplodingLLM())
        truncated_only = doc_ids(lib.retrieve(PROBE_QUESTION, top_k=3,
                                              also=["zzz nonsense"]))
        assert "northwind.md" not in truncated_only, truncated_only

        expanded = doc_ids(lib.retrieve(
            PROBE_QUESTION, top_k=3,
            also=[ExpansionLLM.variant(PROBE_QUESTION, 0),
                  ExpansionLLM.variant(PROBE_QUESTION, 1)],
        ))
        assert "northwind.md" in expanded, expanded

    async def test_the_facade_finds_it_too(self):
        # The facade is where most users would reach it, and a method
        # that exists on SourceLibrarian and not on Librarian is a
        # feature most of them cannot use.
        facade = Librarian(MockLLM(["A prefix."]), chunk_threshold=0,
                           expansion_style="paraphrase",
                           expand_query=True)
        facade.source._dense_ok = False
        for doc_id, text in PROBE_DOCS.items():
            await facade.source.add_document(text, doc_id=doc_id)
        facade.llm = facade.source.llm = ExpansionLLM()
        found = doc_ids(await facade.retrieve_expanded(PROBE_QUESTION, top_k=3))
        assert "northwind.md" in found

    async def test_the_facade_exposes_the_call_on_its_own(self):
        # Catches a facade that surfaces only the convenience wrapper: the
        # split API exists so a caller can batch, cache or precompute the
        # phrasings, and that is impossible without the half that makes
        # the call.
        facade = Librarian(MockLLM(), expand_query=True,
                           expansion_style="paraphrase")
        facade.llm = facade.source.llm = ExpansionLLM()
        assert await facade.expand_query(PROBE_QUESTION) == [
            "auto policy deductible", "vehicle collision coverage",
        ]

    async def test_the_facade_also_argument_reaches_the_fusion(self):
        # The model-free half of the same journey through the facade.
        facade = Librarian(MockLLM(["A prefix."]), chunk_threshold=0)
        facade.source._dense_ok = False
        for doc_id, text in PROBE_DOCS.items():
            await facade.source.add_document(text, doc_id=doc_id)
        facade.llm = facade.source.llm = ExplodingLLM()
        found = doc_ids(facade.retrieve(PROBE_QUESTION, top_k=3,
                                        also=["auto policy deductible",
                                              "vehicle collision coverage"]))
        assert "northwind.md" in found


# ── expansion_mode ───────────────────────────────────────────────


class TestExpansionMode:
    """A property of its own rather than four more ``retrieval_mode``
    strings, so anything already matching on those keeps working.
    """

    async def test_off_when_the_flag_is_off(self):
        lib = await probe_librarian()
        assert lib.expansion_mode == "off"

    async def test_on_when_the_index_can_use_it(self):
        lib = await probe_librarian(expand_query=True, expansion_style="paraphrase")
        assert lib.expansion_mode == "on"

    async def test_inert_on_the_whole_document_path(self):
        # Catches a mode that reports "on" where the call would be paid
        # for and change nothing: whole documents come back regardless of
        # any ranking.
        lib = SourceLibrarian(MockLLM(), expand_query=True, expansion_style="paraphrase")
        await lib.index("One short line.")
        assert lib.retrieval_mode == "whole-document"
        assert lib.expansion_mode == "inert"

    def test_inert_before_anything_is_indexed(self):
        assert SourceLibrarian(MockLLM(), expand_query=True, expansion_style="paraphrase").expansion_mode \
            == "inert"

    async def test_the_facade_exposes_it(self):
        assert Librarian(MockLLM(), expand_query=True).expansion_mode \
            == "inert"

    def test_the_three_strings_are_the_documented_ones(self):
        modes = {SourceLibrarian(MockLLM()).expansion_mode,
                 SourceLibrarian(MockLLM(), expand_query=True, expansion_style="paraphrase").expansion_mode}
        assert modes <= {"off", "on", "inert"}


# ── Validation ───────────────────────────────────────────────────


class TestExpansionValidation:
    """Validated in a module-level ``check_*``, like every other knob,
    because the router and the web librarian carry these for a librarian
    they do not build until much later.
    """

    @pytest.mark.parametrize("bad", [1, 0, "yes", None, 1.0])
    def test_a_non_bool_expand_query_is_refused(self, bad):
        # Catches "anything truthy": expand_query=1 is a miscounted
        # positional argument far more often than a considered choice.
        with pytest.raises(TypeError, match="expand_query"):
            SourceLibrarian(MockLLM(), expand_query=bad)

    @pytest.mark.parametrize("bad", ["2", 2.5, None, True, False])
    def test_a_non_int_variant_count_is_refused(self, bad):
        with pytest.raises(TypeError, match="expansion_variants"):
            SourceLibrarian(MockLLM(), expansion_variants=bad)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_an_out_of_range_variant_count_is_refused(self, bad):
        # 0 is "expansion on, nothing to expand into" — a call whose
        # result is discarded. Off is spelled expand_query=False.
        with pytest.raises(ValueError, match="expansion_variants"):
            SourceLibrarian(MockLLM(), expansion_variants=bad)

    def test_a_numpy_variant_count_is_accepted_and_normalised(self):
        # numpy is a hard dependency, so a count computed from an array
        # arrives as np.int64; it goes on to be formatted into a prompt
        # and to slice a list, so it is normalised at the boundary.
        lib = SourceLibrarian(MockLLM(), expansion_variants=np.int64(3))
        assert lib.expansion_variants == 3
        assert type(lib.expansion_variants) is int

    def test_the_settings_cannot_be_passed_positionally(self):
        with pytest.raises(TypeError, match="positional argument"):
            SourceLibrarian(MockLLM(), False, 6000, True, 512, False, False,
                            8, 10, True)

class TestExpansionStyle:
    """`expand_query` writes one of two things, and they need different
    handling on the way back.

    `paraphrase` produces phrasings and goes through `parse_expansion`.
    `hyde` produces a passage and **cannot**: every line of prose carries
    a comma, so the prose bar would discard the entire reply. That is not
    a tuning question — the bars exist to keep a refusal or an
    explanation out of a *phrasing*, and a passage is neither.
    """

    PASSAGE = ("Transportation Expenses. The Company will reimburse the "
               "Insured $42.50 per day for a temporary substitute auto, "
               "up to $1,275.00 for any one Loss.")

    async def _expand(self, reply, **kw):
        llm = MockLLM([reply])
        lib = SourceLibrarian(llm, expand_query=True, **kw)
        return await lib.expand_query("what do they pay for a hire car"), llm

    async def test_hyde_is_the_default_style(self):
        _, llm = await self._expand(self.PASSAGE)
        assert llm.calls[0][0] == HYDE_SYSTEM

    async def test_a_passage_reaches_also_whole(self):
        got, _ = await self._expand(self.PASSAGE)
        assert got == [" ".join(self.PASSAGE.split())]

    async def test_the_prose_bar_would_have_eaten_it(self):
        """The reason the parser is skipped, pinned rather than asserted
        in a comment. If `parse_expansion` ever starts admitting prose,
        this fails and the skip can be revisited."""
        assert parse_expansion(self.PASSAGE, "q", 2) == []

    async def test_paraphrase_still_goes_through_the_parser(self):
        got, llm = await self._expand("What does the policy pay for a hire car",
                                      expansion_style="paraphrase")
        assert "rewrite one search question" in llm.calls[0][0]
        assert got == ["What does the policy pay for a hire car"]

    async def test_an_empty_passage_yields_nothing(self):
        got, _ = await self._expand("   \n  ")
        assert got == []

    async def test_a_bad_style_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="expansion_style"):
            SourceLibrarian(MockLLM(), expansion_style="hyde2")

    async def test_style_does_not_reach_the_cache_key(self):
        """Query-time, like every expansion setting: toggling it must not
        invalidate a saved index."""
        a = SourceLibrarian(MockLLM(), expansion_style="hyde")
        b = SourceLibrarian(MockLLM(), expansion_style="paraphrase")
        a.documents = b.documents = {"d": "text"}
        assert a.content_hash() == b.content_hash()
