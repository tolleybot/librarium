"""Source Librarian — indexes and retrieves from source documents.

Dual-index retrieval:
  BM25 on contextualized chunks (LLM-generated prefix per chunk)
  Embeddings on raw chunks (nomic-embed-text-v1.5)
  Merged via Reciprocal Rank Fusion (RRF)
  Optional cross-encoder reranking (bge-reranker-v2-m3)

Indexing costs one LLM call for the document summary plus one per
chunk, so it is worth keeping: see :meth:`SourceLibrarian.save` and the
``cache=`` argument to :meth:`SourceLibrarian.index`.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import numbers
import re
from collections.abc import AsyncIterator, Collection, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    nullcontext,
)
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from librarium.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_spans, tokenize
from librarium.llm import LLMClient, parse_thinking
from librarium.models import Chunk, Fact, LibrarianAnswer

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────
#
# Defaults, not settings: every one of these is a constructor argument
# on SourceLibrarian (and forwarded by Librarian, LibrarianRouter and
# WebLibrarian), and the running code reads the instance attribute, never
# the constant. Rebinding one at runtime therefore changes what a
# librarian built *after* that point defaults to and nothing else — set
# the argument instead.

CHUNK_THRESHOLD = 3000  # chars — above this, chunk and search
TOP_K = 10

# Approximate chars-per-token ratio for context estimation.
# Most models use ~4 chars per token. Used to warn users before
# hitting the model's context limit.
CHARS_PER_TOKEN = 4
RESERVED_OUTPUT_TOKENS = 512  # headroom kept for the model's reply

# Sized for a small local model — 8K of context with room to spare, not
# whatever a hosted model technically accepts. A constant rather than a
# literal in each signature because four constructors default to it, and
# three of them agreeing with the fourth by coincidence is how they drift.
MAX_CONTEXT_TOKENS = 6000
RRF_K = 60
RERANK_TOP = 5
RERANK_THRESHOLD = 0.0001

# How many fused candidates the cross-encoder scores — the *input* width,
# where `rerank_top` is the output width. `None` is not "off": it means
# "as wide as the query asks for", i.e. whatever `top_k` resolves to for
# that call, which is what `retrieve()` has always sliced to. So the
# default leaves every existing librarian ranking exactly the chunks it
# ranked before, and this constant is the one place that choice is
# recorded rather than four signatures agreeing by coincidence.
#
# It cannot be normalised to a number here the way the other defaults are:
# `top_k` is overridable per call, so "follow top_k" is a decision only
# `retrieve()` is in a position to make.
RERANK_POOL: int | None = None

# Measured on the 194-document corpus, grafted onto one finished index so
# chunks and LLM-written prefixes were byte-identical and only the encoder
# moved: nomic 69.4%, bge-m3 **75.6%** — +6.2 points, and past the 74.6%
# RAGLite scores on the same corpus. Each model was given its OWN prompt
# convention; measuring bge-m3 under nomic's prefixes reads 71.8%, so 3.8
# of that gap is prompting rather than weights (evidence:
# `benchmarks/README.md`).
#
# bge-m3 needs no `trust_remote_code` and imports no einops, so the
# `embeddings` extra's einops dependency is now only for a librarian that
# points `embed_model` back at nomic. It is 1024 dimensions against
# nomic's 768, so vectors on disk grow by a third.
#
# **Changing this invalidates every saved index**, by design:
# `content_hash` folds `embed_model` in, so a cache built under nomic
# misses rather than loading vectors that are not comparable.
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Pinned revisions of the two default models ───────────────────
#
# Unlike everything above, these are NOT constructor defaults. Nothing
# takes them as an argument and no librarian stores them; the loaders
# read them at load time. They are facts about the two repos named
# directly above, not settings, which is why they are not knobs: the
# only correct value for `EMBED_REVISION` is a commit of *that* repo,
# and against anyone else's model a revision is at best unknown and at
# worst a same-named branch of something entirely different. So the pin
# is keyed by model name and simply does not exist for a user-supplied
# `embed_model` / `rerank_model` — see `pinned_revision`.
#
# The trade: pinning stops silent upstream fixes as well as silent
# upstream breakage. Bumping either sha is a deliberate act, and it
# re-opens the question these pins answer — whether the `embeddings`
# extra still declares everything the new revision imports.
#
# BUMPING `EMBED_REVISION` DOES NOT INVALIDATE SAVED INDEXES, and
# `content_hash()` is deliberately not the place that fixes it. It folds
# in `embed_model` but not its revision: the default model's weights are
# fetched revision-blind (see the block below), so hashing the sha would
# invalidate every cache on a deliberate bump while still missing the
# upstream weight change that needs no commit here — a partial guarantee
# living inside a function documented as a total one. It could not carry
# this anyway: `_load_if_current` runs the hash *before* any embedding
# model is loaded, so a weight identity there would mean loading the
# model at cache-check time or asking the network.
#
# What catches it instead is `embed_fingerprint` below — one encode of a
# constant probe, recorded beside the vectors and compared when the model
# actually loads, which is the same shape `_check_code_revision` uses for
# the executed code. It covers both ways the weights move: a bump of the
# sha above, and upstream changing what that revision-blind fetch returns
# with no commit here at all. On drift the vectors are discarded and
# retrieval degrades to BM25-only, so the old advice — remember to delete
# saved indexes by hand — is no longer the only control.
# `RERANK_REVISION` needs none of that: nothing the reranker produces is
# stored, it only reorders chunks at query time, so a bump takes effect on
# the next query and cannot stale an index.
# bge-m3, pinned to the revision the 75.6% was measured on rather than to
# whatever its default branch holds later (evidence:
# `benchmarks/README.md`). Written as one paragraph with the block above
# it, which is why the citation sits here.
# Leaving the default model
# unpinned was the first version of this change and it was wrong: the pin
# is what makes a saved index reproducible, and an unpinned default means
# a re-index months from now can silently produce vectors from different
# weights while `content_hash` — which folds in the model *name*, not its
# weights — calls the cache current. The fingerprint check is the backstop
# for that, and a backstop is not a reason to remove the guard in front of
# it.
EMBED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

# EXACTLY WHAT `EMBED_REVISION` COVERS, measured rather than assumed.
# Traced by spying on every `hf_hub_download` of a load pinned to an
# older revision and reading back which snapshot each file landed in.
# Every claim in this block is re-runnable, and has to be, because a
# sentence-transformers release or one commit to nomic's modeling file
# turns any of them over without touching this repo (evidence:
# benchmarks/pin_evidence.py — needs network and the `embeddings` extra,
# which is why it is a script and not a test; tests/test_pinning.py holds
# the offline half, that the revisions reach the loader at all):
#
#   config.json, tokenizer*, modules.json, sentence_bert_config.json
#       pinned. This is not nothing — config.json is where `auto_map`
#       names the code repo below, so pinning it fixes *which* foreign
#       repo and class can be reached at all.
#   model.safetensors / pytorch_model.bin
#       NOT pinned; fetched from the default branch. Not a
#       sentence-transformers bug: `revision` does reach
#       `AutoModel.from_pretrained`, but nomic's remote code overrides
#       `from_pretrained` and resolves weights itself with
#       `cached_file(model_name, WEIGHTS_NAME)`, passing no revision and
#       forwarding none from its caller. Loading the same model through
#       plain `AutoModel.from_pretrained(..., revision=...)` reproduces
#       it, and a stock model pins its weights correctly — which is why
#       RERANK_REVISION is whole and this one is not.
#   the executed modeling code
#       NOT pinned, and not pinnable. auto_map is
#       `nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel`
#       — the `--` means a *different* repo, named with no revision.
#       transformers has a separate `code_revision` for exactly this,
#       but sentence-transformers 5.7.0 (current latest) pops it out of
#       `model_kwargs` in `_load_module_class_from_ref` to resolve the
#       module class named in modules.json, so it never reaches the
#       model loader. Measured: with `model_kwargs={"code_revision":
#       <older sha>}` the config class loads from that sha and the model
#       class still loads from the code repo's default branch — the
#       `code_revision` check in the script above, and
#       `TestTheRevisionReachesTheLoader::test_no_code_revision_is_passed`
#       for the decision that follows from it. Passing
#       it would be a pin that appears to work and does not, which is
#       worse than none — so it is not passed.
#
# So the default embedding model's weights and code both float, and no
# argument this library can pass will stop them. That is the standing
# argument for eventually choosing a default that needs no
# trust_remote_code; until then, the sha below is the code the
# `embeddings` extra was verified against and `_check_code_revision`
# warns after loading when something else ran. Its only unguarded
# third-party imports are einops, numpy, torch, transformers and
# safetensors — einops is the one this extra has to declare itself
# (megablocks is behind a try/except and unused by this non-MoE model).
EMBED_CODE_REPO = "nomic-ai/nomic-bert-2048"
EMBED_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"

# The reranker is deliberately absent from all of that: bge-reranker-v2-m3
# declares `XLMRobertaForSequenceClassification` and has no auto_map at
# all, so it runs transformers' own loader and `RERANK_REVISION` pins
# every file including the weights — verified the same way. The
# asymmetry is the model's, not an oversight, and `_load_reranker`
# correspondingly passes no trust_remote_code.

# WHICH TEXT THE ENCODER IS GIVEN, and it is in `content_hash` because a
# change to it silently invalidates every vector on disk.
#
# `_refresh_embeddings` encodes `c.prefixed` — the contextual prefix plus
# the chunk, the same text `_build_bm25` indexes. It used to encode
# `c.text`, so the dense half worked from strictly less text than the
# lexical half and a chunk reading "the rate increased 3% year over year"
# was embedded with no idea which rate, which is the exact failure the
# prefix is generated to fix.
#
# Measured three times, each on one index with the prefixes held constant so
# only the encoder input varied, each against a `bm25-only` control flat at
# every cut-off. **What reproduced is the sign at `@10` — +2.2, +1.1, +3.8 —
# and the number of literals found, up in all three.** `@1`, `@3` and `@5`
# each changed sign between runs, so the +5.5 at `@1` that the first run
# showed is not the effect and is not quoted as one. `@10` is the figure
# worth having anyway: it is the ceiling reranking provably cannot move,
# measured at +0.0 there. The magnitude varies because each run generates
# fresh prefixes — a flat control proves the two arms are comparable to each
# other, not that one run is comparable to the next
# (evidence: `benchmarks/retrieval_arms.py --embed-source text prefixed`
# re-runs it, and `benchmarks/README.md` carries all three runs, the
# per-suite split and why an uncontrolled run is void).
#
# **A constant and not a per-librarian setting**, unlike the geometry
# `content_hash` takes as arguments. Those are arguments because two
# librarians can be configured differently over the same text and must not
# share a cache key; nothing configures this, so there is nothing to
# forward through the four layers and no `check_*` to extend. It is a fact
# about what this version of the code embeds, which is why `content_hash`
# reads it here rather than accepting it — the same reason
# `INDEX_FORMAT_VERSION` is read rather than passed. Changing this string
# is how a future change to the encoder's input announces itself to every
# cache written before it; changing it for any other reason costs every
# index on disk a re-index for nothing.
EMBED_SOURCE = "prefixed"

# ── Fingerprinting the weights the stored vectors came from ──────
#
# Not settings either, and for the same reason as the revisions above:
# the probe and its tolerance are facts about how the check works, not
# something a deployment tunes. A knob here would let a user "fix" a
# drift warning by widening the tolerance until it stopped firing, which
# is the one outcome the check exists to prevent.
#
# The probe carries nomic's `search_document:` task prefix so the
# fingerprint goes through exactly the path the stored chunk vectors went
# through — a model that changed only how it handles that prefix would
# otherwise pass. Its wording is arbitrary but frozen: it is compared
# against probes recorded by earlier processes, so changing it makes
# every recorded fingerprint incomparable (handled — see
# `fingerprint_drift` — but it means a silent loss of coverage, not a
# false alarm).
EMBED_PROBE = "search_document: librarium embedding weight fingerprint probe"

# Rounded before storage only to keep index.json readable. float32 holds
# ~7 significant decimal digits, so 6 discards nothing a cosine at the
# tolerance below could notice: on a unit vector of 768 dims the rounding
# perturbs the direction by ~1e-5 relative, i.e. ~1e-10 of cosine. That
# unit vector is a worst case, not a description — measured, the default
# model's probe vector has norm ~20.5 (it loads with no normalising
# step), which puts the real perturbation ~20x lower still at ~7e-7
# relative, ~2e-13 of cosine. See benchmarks/fingerprint_real_model.py.
EMBED_FINGERPRINT_DIGITS = 6

# Cosine *distance* (1 - cos) above which two probe vectors are called
# different weights. A distance threshold rather than a digest of the
# rounded vector, because a digest turns every rounding boundary into a
# cliff: with 768 components and float noise of ~1e-5, the chance that at
# least one component sits on the wrong side of a 1e-3 bin edge is
# effectively 1, so the digest would report drift on a device change
# every time while claiming to be exact.
#
# Cosine is the forgiving direction, which is what makes this workable,
# and the size of that forgiveness is now measured rather than argued —
# benchmarks/fingerprint_tolerance.py, the default model on 2026-08-10,
# macOS arm64 / torch 2.13, one load per device and dtype and every
# pairing through `fingerprint_drift`:
#
#   cpu vs mps, both fp32      7.5e-13   (1.3e8x inside)
#   the same config, reloaded  0.0       (identical to the stored digit)
#   fp16 against fp32          ~1e-6     (~100x inside)
#   bf16 against anything      5.1e-5 .. 8.2e-5   (only ~1.2x inside)
#
# So the case this constant exists to survive — the same weights on a
# different device — clears it by eight orders of magnitude, and the
# analytic ~d²/2 estimate that sized it was right about fp32 and
# pessimistic about fp16 (evidence: benchmarks/fingerprint_tolerance.py,
# the run tabulated above). `cuda` was not measurable on that machine and
# nothing here has been observed about it. The thin one is bfloat16,
# inside by a factor of 1.2 rather than by orders: unreachable today
# because `_ensure_embed_model` selects no dtype, but a dtype knob would
# put a supported deployment within noise of a spurious re-index, and
# this number has to be revisited before one lands.
#
# Anything that actually changed the weights moves it orders of magnitude
# further: an unrelated model is ~1, and even a light fine-tune of the
# same base lands in the 1e-3..1e-1 range.
#
# So the design errs towards the false positive: a device or dtype change
# large enough to clear 1e-4 is reported as drift, and the warning says
# so in as many words. That is the survivable failure — it costs a
# re-index of vectors that are, by then, measurably not the ones a query
# would produce (evidence: benchmarks/fingerprint_tolerance.py again —
# "measurably" here means at the scale tabulated there, not merely in
# principle). The false negative is the one this whole change exists
# to prevent, and it has a floor: a difference too small to clear 1e-4 is
# by construction too small to move a cosine ranking either, because it
# is the same metric retrieval scores with.
EMBED_FINGERPRINT_TOLERANCE = 1e-4

DEFAULT_DOC_ID = "document"
CHUNK_JOINER = "\n\n---\n\n"

# Simultaneous prefix calls during indexing. Deliberately conservative:
# the setting that is wrong everywhere is "all of them at once", and 8
# keeps a 30-chunk document as fast as it was while a 2MB one no longer
# opens a thousand sockets. Raise it for a local vLLM, drop it to 1 or 2
# for Ollama, which serves serially by default.
MAX_CONCURRENT_PREFIXES = 8

# Bump when the on-disk layout changes in a way an existing index cannot
# be read under. Deliberately NOT bumped for `embed_fingerprint`: it is
# an optional key that an older reader ignores (`meta.get`) and a newer
# one treats as "not recorded", which is exactly the pre-upgrade state.
# A bump would make `load()` raise and `_load_if_current` miss on every
# index on disk — every user paying a full re-index, one LLM call per
# chunk — to add a field that is absent-tolerant by construction. It is
# also folded into `content_hash()`, so a bump invalidates the caches
# twice over.
INDEX_FORMAT_VERSION = 1

# A gap query is meant to be search terms, not a sentence.
MAX_GAP_WORDS = 12       # cap for a reply that used the QUERY: format
MAX_BARE_GAP_WORDS = 6   # tighter cap for a reply that ignored it

# Query expansion is opt-in: retrieve() makes zero LLM calls and that is a
# documented property of the library, so nothing may start making one
# because a default changed under an existing deployment. It also ships
# unmeasured — the paraphrase benchmark that would justify a different
# default does not exist yet.
EXPAND_QUERY = False
# Two paraphrases is one fused ranking per phrasing plus the question's
# own. More phrasings dilute: RRF weights every ranking equally, so each
# additional variant is another vote against the question's own ranking.
EXPANSION_VARIANTS = 2

# Index-time field extraction is opt-in, for the reason above it: it costs
# one LLM call per document, and nothing may start making one because a
# default changed under an existing deployment. Off also keeps
# `content_hash` byte-identical to what it was before this knob existed,
# so no index already on disk is invalidated by upgrading — see
# :func:`content_hash`.
EXTRACT_FIELDS = False

# Prefix each retrieved chunk with where it came from, in the prompt that
# `ask()` and `answer()` build. Default **on**: the model was previously
# handed bare text with no way to tell which of three near-identical
# policy years it was reading, and a header cannot remove an answer the
# way a filter can — the worst case is spent tokens. It is a knob rather
# than unconditional because it changes what every answer call sends, and
# a caller with a tight context budget may want the room back.
#
# Query-time, so it is NOT folded into `content_hash`: toggling it must
# not invalidate a saved index. Same argument as the expansion settings.
CITE_SOURCES = True

# ── Prompts ──────────────────────────────────────────────────────

ANSWER_SYSTEM = """\
You are a librarian. Return ONLY facts that appear in the source material \
below. Copy the relevant phrases — do NOT compute, calculate, or draw \
conclusions. If the source says "120 loaves at $4.50 each", return that \
exact fact. Do NOT return "revenue is $540".

Respond with JSON:
{{"items": [{{"content": "fact 1"}}, {{"content": "fact 2"}}], \
"coverage": "complete|partial|uncertain"}}

SOURCE MATERIAL:
{source}"""

CONTEXT_PREFIX_SYSTEM = (
    "You are indexing a document for search. Given a document summary "
    "and a chunk of text, write a SHORT context prefix (2-3 sentences) "
    "that helps a keyword search engine find this chunk. Include:\n"
    "- What entity/topic this chunk is about\n"
    "- Key names, locations, numbers mentioned\n"
    "- Synonyms and related terms a searcher might use instead of "
    "the exact words in the chunk\n"
    "- How this relates to the broader document\n\n"
    "Output ONLY the prefix, nothing else.\n/no_think"
)

DOC_SUMMARY_SYSTEM = (
    "Summarize this document in 3-5 sentences. Cover: what it's about, "
    "key entities/people, key topics, and the main questions or issues "
    "discussed. Output ONLY the summary.\n/no_think"
)

#: The closed vocabulary `doc_type` must choose from.
#:
#: Free text was measured and it is not a filter key: `fields-1` produced
#: **69 distinct labels over 176 documents, 50 of them used once**, so a
#: renters policy came back as "tenant policy" and a query for "renters
#: policy" matched nothing. High coverage hid it completely (evidence:
#: `benchmarks/field_extraction.py`, `benchmarks/README.md`).
#:
#: Written from what personal paperwork *is*, deliberately not derived
#: from the benchmark corpus — a vocabulary read off the fixture would
#: score well on the fixture and generalise to nothing. Some corpus
#: documents therefore land in "other", which is the honest outcome and
#: not a defect to tune away.
DOC_TYPES = (
    "insurance policy", "bank statement", "brokerage statement",
    "utility bill", "medical bill", "explanation of benefits", "invoice",
    "receipt", "pay stub", "tax return", "letter", "notice",
    "email thread", "manual", "contract", "deed or title", "certificate",
    "report", "schedule", "other",
)

DOC_FIELDS_SYSTEM = (
    "You extract identifying fields from a document, for filtering.\n"
    "\n"
    "For EVERY field you must give both the value and the exact text you "
    "took it from. A value with no quote, or a quote you did not copy "
    "character-for-character out of the document, is worse than no "
    "answer — it will be discarded.\n"
    "\n"
    "Reply with ONLY a JSON object of this shape:\n"
    '{"doc_type": {"value": ..., "quote": ...}, '
    '"issuer": {"value": ..., "quote": ...}, '
    '"issued_date": {"value": ..., "quote": ...}, '
    '"effective_date": {"value": ..., "quote": ...}, '
    '"period": {"value": ..., "quote": ...}, '
    '"account": {"value": ..., "quote": ...}}\n'
    "\n"
    "- doc_type: choose EXACTLY ONE of: " + ", ".join(DOC_TYPES) + ". "
    "Use \"other\" if none fit. Do not invent a label. This one is your "
    "judgement rather than a quotation, so its quote may be null.\n"
    "- issuer: the organisation the document is FROM, not the recipient\n"
    # The distinction fields-1 could not make, because it asked for one
    # field meaning both. The documents themselves label the two — a
    # "Date:" header against the word "effective" — so this is a
    # question the source can actually answer.
    "- issued_date: the date the document itself is dated or was sent, "
    "usually beside \"Date:\" in a header\n"
    "- effective_date: the date something the document announces takes "
    "effect, usually beside the word \"effective\". This is often NOT "
    "the same as issued_date. Leave it null if the document announces "
    "no such date.\n"
    "- period: the span the document covers, if it states one\n"
    "- account: the account, policy or reference number\n"
    "\n"
    "Dates as YYYY-MM-DD when the day is stated, else YYYY-MM or YYYY.\n"
    "\n"
    # Measured cause of four of nine wrong dates in fields-1: the model
    # read a `**Date:**` header out of a nested reply inside an email
    # thread and returned it as the document's own (evidence:
    # `benchmarks/README.md`).
    "In an email thread, use only the thread's own newest message. "
    "Lines beginning with \">\" are older messages being quoted — never "
    "take a value from them.\n"
    # The one instruction the whole filter rests on. A model that writes
    # "N/A" where it means "absent" hands the filter a value to match on,
    # and a document filtered out on an invented date is the silent
    # failure this feature is most able to cause. parse_fields refuses
    # the common placeholder spellings as well, because saying it once
    # is not the same as enforcing it.
    "Use null for BOTH value and quote on any field the document does "
    "not state. Never guess, never write \"N/A\" or \"unknown\" — a "
    "wrong value is worse than a missing one.\n"
    "Output ONLY the JSON object.\n/no_think"
)

#: How many times `_extract_doc_fields` may hand the model its own failed
#: checks and ask again. One re-ask, so a document costs at most two
#: calls: the checks are cheap and deterministic, but the call is not,
#: and a model that ignores a named defect twice will ignore it a third
#: time. The re-ask is a repair request, never a "are you sure" — that
#: question is what `_find_gap` asks, and it is measured answering
#: SUFFICIENT (evidence: `benchmarks/refine_loop.py`,
#: `benchmarks/README.md`).
DOC_FIELDS_REASK = 1

# The keys `parse_fields` keeps. Anything else the model volunteers is
# dropped: a record that grows a key because a model felt like it is one
# a filter cannot rely on, and a typo'd key silently becomes a field
# nothing ever matches.
DOC_FIELD_NAMES = (
    "doc_type", "issuer", "issued_date", "effective_date", "period",
    "account",
)

# Values a model writes when it means "absent". Refused rather than
# stored, because the whole safety argument for filtering is that a
# document with no extracted date is never excluded — and a literal
# "unknown" sitting in `effective_date` is not no date, it is a date that
# matches nothing and excludes the document from every filtered query.
#
# Matched case-insensitively against the whole stripped value, never as a
# substring: "None Trust Bank" is an issuer.
DOC_FIELD_PLACEHOLDERS = frozenset({
    "n/a", "na", "n.a.", "none", "null", "nil", "unknown", "unspecified",
    "not specified", "not stated", "not applicable", "not available",
    "undated", "no date", "-", "--", "?", "tbd",
})

# A field is an identifier, not a sentence. A model that writes a clause
# into `issuer` has stopped extracting and started describing, and the
# value would be matched against literally by every filter downstream.
MAX_FIELD_CHARS = 120

#: A quote is evidence rather than an identifier, so it gets room for the
#: sentence around the value — but not room for the document. Past this
#: the model has pasted a section instead of pointing at a line, and a
#: "quote" that large makes the groundedness check vacuous: quote enough
#: and everything is contained in it.
MAX_QUOTE_CHARS = 400


def _fields_fingerprint() -> str:
    """Digest of everything that decides what a ``doc_fields`` record means.

    The prompt (which carries :data:`DOC_TYPES` interpolated into it), the
    key set, the placeholder bar and the two length caps. Change any of
    them and this changes, so :data:`DOC_FIELDS_VERSION` changes, so
    ``content_hash`` changes and ``load()`` discards records built under
    the old meaning.

    **This exists because the hand-written version could not track the
    thing it claimed to.** ``DOC_FIELDS_VERSION`` used to be the literal
    ``"fields-3"`` with a comment asking the next person to bump it when
    the vocabulary or the parser changed. Editing ``DOC_TYPES`` therefore
    left the cache key untouched: an index extracted under the old
    vocabulary was served as current and its records mixed with new ones,
    silently. That is exactly the hole ``embed_source`` exists to close,
    rebuilt one field over — and a convention that relies on remembering
    is not a mechanism.
    """
    material = "\0".join([
        DOC_FIELDS_SYSTEM,
        "\0".join(DOC_FIELD_NAMES),
        "\0".join(sorted(DOC_FIELD_PLACEHOLDERS)),
        str(MAX_FIELD_CHARS),
        str(MAX_QUOTE_CHARS),
    ])
    return hashlib.sha256(material.encode()).hexdigest()[:8]


#: What produced a `doc_fields` record. Read where the fields are derived
#: and folded into `content_hash` — never a per-librarian knob, the way
#: `EMBED_SOURCE` is not: it describes this version of the prompt and the
#: field set, not a deployment.
#:
#: Two halves, and the split is the point. The **suffix** is derived, so
#: every input listed in :func:`_fields_fingerprint` invalidates the cache
#: by itself and no one has to remember. The **prefix** is hand-written
#: and is the part a human still owns: it covers changes to the *logic* of
#: `parse_fields` and `verify_fields`, which no digest of their constants
#: can see. Bump it when what those functions *do* changes rather than
#: what they are configured with.
#:
#: So the residual is narrow and worth stating rather than implying it is
#: closed: rewrite a check inside `verify_fields` without touching a
#: constant, and this string does not move.
DOC_FIELDS_VERSION = f"fields-4:{_fields_fingerprint()}"

#: What one chunk's facts are extracted *for*. Every atomic assertion the
#: chunk makes, as subject/attribute/value with the wording it came from.
#:
#: The measured reason this exists: an expected answer on the benchmark
#: corpus is **7.6 characters** and the unit retrieval ranks is a
#: 1,378-character chunk, a signal-to-container ratio of 0.0055 that no
#: ranking knob touches. A fact is ~154 characters, so the same answer
#: arrives in a container it dominates rather than one it hides in
#: (evidence: `benchmarks/fact_retrieval.py`, `benchmarks/README.md`).
#:
#: The *attribute* is the half that earns the call. A plain regex over the
#: same chunks captures 208 of 209 expected literals, so extracting a
#: **value** needs no model at all; what a model supplies is the label that
#: lets a question match a fact sharing none of its words.
FACT_SYSTEM = """You extract facts from documents. Return every factual \
assertion the text makes, as records.

Each record has:
  subject   — what the fact is about, named as the document names it
  attribute — the property being stated, in plain words a reader would use
  value     — the value, copied EXACTLY as written
  quote     — the sentence or table row it came from, verbatim

Rules:
- Copy values character for character. Never round, reformat or convert.
- Include every amount, percentage, date, deadline, identifier, account \
number, limit, deductible and named quantity.
- One record per assertion. Do not merge two numbers into one record.
- The attribute should read like something a person would ask about, not \
like a column header. Prefer "collision deductible" over "Part D".
- Do not summarize, interpret, or add anything the text does not state."""

#: The keys a fact record carries. `quote` is kept because it is the only
#: thing that lets a reader check the value against the document; nothing
#: verifies it yet.
FACT_KEYS = ("subject", "attribute", "value", "quote")

#: Longest value and quote kept. A model that echoes half a table into one
#: field has not extracted a fact, and the long record would then dominate
#: a store whose whole advantage is that its entries are short.
MAX_FACT_VALUE_CHARS = 200
MAX_FACT_QUOTE_CHARS = 600

#: Most facts kept from one chunk. Measured at 13.5 per chunk on the
#: benchmark corpus with a maximum in the thirties, so this bounds a
#: pathological reply rather than trimming a normal one (evidence:
#: `benchmarks/fact_store.py`, `benchmarks/README.md`).
MAX_FACTS_PER_CHUNK = 60


def _facts_fingerprint() -> str:
    """Digest of everything that decides what a fact record means.

    `DOC_FIELDS_VERSION`'s argument, applied to the second extractor: a
    hand-written literal with a comment asking the next person to bump it
    is how `embed_source`'s hole was built, and editing the prompt would
    otherwise leave the cache key untouched and serve a store extracted
    under the old instructions as current.
    """
    material = "|".join((
        FACT_SYSTEM,
        ",".join(FACT_KEYS),
        str(MAX_FACT_VALUE_CHARS),
        str(MAX_FACT_QUOTE_CHARS),
        str(MAX_FACTS_PER_CHUNK),
    ))
    return hashlib.sha256(material.encode()).hexdigest()[:12]


#: What produced a fact store. The hand-written prefix covers changes to
#: the *logic* of `parse_facts`, which no digest of constants can see; the
#: derived suffix covers the prompt and the caps.
FACTS_VERSION = f"facts-1:{_facts_fingerprint()}"

#: Index-time fact extraction, default off. One LLM call per *chunk* —
#: more than any other knob here costs — so it is opt-in and stays opt-in.
EXTRACT_FACTS = False

#: What share of a retrieval's slots go to facts when a fact store exists.
#: 0.5 is what measured best; 0.0 disables the mix without discarding the
#: store (evidence: `benchmarks/fact_retrieval.py`).
FACT_MIX = 0.5


GAP_SYSTEM = (
    "You check whether a set of facts answers every part of a question.\n"
    "If every part is answered, reply with exactly: SUFFICIENT\n"
    "Otherwise reply with exactly one line, in this form:\n"
    "QUERY: <3-8 search keywords for the missing part>\n"
    "Never explain. Never write a sentence."
)

EXPANSION_SYSTEM = (
    "You rewrite one search question into alternative phrasings for a "
    "keyword search engine.\n"
    "Change the WORDS, not the sentence: swap in the vocabulary a "
    "document would use instead — \"auto policy\" for \"car insurance\", "
    "\"invoice\" for \"bill\", \"practice\" for \"office\". A rewrite "
    "that only reorders the same words is useless.\n"
    "Keep every name, place, product, date and number exactly as "
    "written — those are what keyword search is best at, and rewriting "
    "them destroys the match.\n"
    "Never answer the question. Never mention the documents.\n"
    # Says out loud what parse_expansion silently enforces: a line
    # carrying a comma, colon or semicolon is dropped as prose, so a
    # model that lists its swaps inline loses them. GAP_SYSTEM sets the
    # precedent of telling the model the shape the parser accepts
    # ("Never explain. Never write a sentence.").
    "Write each rewrite as one plain question. No commas, no lists.\n"
    "Output ONLY {n} rewritten question(s), one per line, nothing "
    "else.\n/no_think"
)

# The one call that does not want _llm_call's 0.1. Paraphrasing needs the
# model to reach past its first-choice wording, and at 0.1 it returns the
# question back with a synonym or two; the variants are then near-copies
# whose rankings add nothing to the fusion. It is deliberately not 0.7
# either: an expansion feeds a retrieval index rather than a reader, so a
# variant that wanders off the question costs recall directly.
#
# Unlike the constants at the top of this module, which are constructor
# defaults nothing reads at runtime, this one IS read at the call site
# (see :meth:`SourceLibrarian.expand_query`). It describes the prompt, not
# the deployment, so it is not a per-librarian knob.
#: What `expand_query` writes. Two styles, and the measured gap between
#: them is large enough that the default is the newer one.
#:
#: `"paraphrase"` rewrites the question into alternative phrasings and is
#: what this feature shipped with. Measured three times on the
#: 194-document corpus and negative every time — most recently **-1.2
#: headline points**, flat on three categories, and reaching only **23 of
#: 51 questions** because `parse_expansion`'s bars drop most of what an
#: 8B model writes. The cause is upstream of the fusion: a paraphrase
#: stays in the asker's register, so it produces a second consumer
#: phrasing of a question the document already does not match.
#:
#: `"hyde"` writes a short passage that would *answer* the question, in
#: the register of the answering document, and searches with that. The
#: text is never shown to anyone — it is a retrieval key, and its
#: invented specifics do not matter. Measured on two independent indexes:
#: **+8.2 headline both times**, +31.2 on `vocab_mismatch` both times,
#: +10 of 32 literals both times, treating 51/51 questions.
#:
#: **The mechanism is register, not length**, which is the one thing that
#: was open after the first run. A control arm restating the question at
#: hyde's length in the asker's own words gained **exactly nothing** —
#: 43.8% against off's 43.8% — while hyde reached 75.0%. More words in
#: the query buys nothing; the document's vocabulary buys everything
#: (evidence: `benchmarks/query_expansion.py`, `benchmarks/README.md`).
#:
#: What it does not fix is `near_duplicate` (+0.0 and -2.6 across the two
#: runs) or `versioning` (negative in both). Those need document
#: identity, not better phrasing.
EXPANSION_STYLE = "hyde"

#: A pseudo-answer goes to `also=` whole, and it **cannot** go through
#: `parse_expansion`: a passage is prose and every line of it carries a
#: comma, so the prose bar would eat the entire reply. That is why the
#: parser is skipped for this mode rather than tuned for it — the bars
#: exist to keep a refusal or an explanation out of a *phrasing*, and a
#: passage is neither.
HYDE_SYSTEM = (
    "You write a short passage that would answer a question, as though "
    "it were an excerpt from the document that answers it.\n"
    "Write in the register of the source: a policy, a statement, a "
    "contract, a bill. Use the formal terms of art such a document would "
    "use, not the everyday words of the question.\n"
    "Two or three sentences. Invent plausible specifics where you must. "
    "Never say that you are inventing anything, and never address the "
    "reader.\n"
    "Output ONLY the passage.\n/no_think"
)

EXPANSION_TEMPERATURE = 0.3

_GAP_QUERY_RE = re.compile(r"QUERY:\s*(.+)", re.IGNORECASE)
# Sentence structure an unprefixed reply should not have. Deliberately NOT
# terminal punctuation — a period cannot be told from an abbreviation.
_PROSE_PUNCT_RE = re.compile(r"[,:;]")

# A refusal the model wrote where a paraphrase was asked for. The prose
# punctuation bar above catches most of them ("I cannot help with that,
# sorry.") but not the unpunctuated ones — "Sorry I cannot do that" is
# five words with no comma, so it clears both that bar and
# MAX_EXPANSION_WORDS and would be fed to BM25 as a query, retrieving on
# *sorry*, *cannot* and *do*.
#
# Deliberately narrower than a bare word list. "cannot", "can't" and
# "unable" are ordinary words in a support corpus — "why can't I log in",
# "unable to connect error" are exactly the questions this library gets
# asked — so the modals only count against a first-person subject ("I
# cannot", "I'm unable"), which is how an assistant refuses and not how a
# question is phrased, or at the very start of the line, where a rewrite
# would not begin. An apology or a self-identification ("as an AI") is
# conclusive enough on its own.
#
# Same bargain as the punctuation bar: a dropped phrasing costs recall on
# one query, a refusal used as a query costs precision on the query that
# mattered.
_REFUSAL_RE = re.compile(
    r"\b(?:sorry|apologi(?:es|[sz]e[sd]?))\b"
    r"|\bas\s+an\s+ai\b|\bas\s+a\s+language\s+model\b"
    r"|\bi\b(?:\s*['’]m\b|\s+am\b)?\s+"
    r"(?:can\s*not|can['’]?t|unable|not\s+able"
    r"|do\s*(?:not|n['’]?t)\s+have)\b"
    r"|^\s*(?:cannot|can\s+not|unable|not\s+able)\b",
    re.IGNORECASE,
)

# "1." / "2)" / "-" / "*" / "•" at the start of a line — models number a
# list however they like whatever the prompt asked for, and the number is
# not part of the query.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)])\s*")
# A paraphrase of a question is about as long as the question. Well past
# that the line is the model explaining itself, and feeding a sentence of
# explanation to BM25 retrieves whatever its stopwords happen to hit.
MAX_EXPANSION_WORDS = 40


# ── Pure helpers (unit-testable without a model) ─────────────────


def estimate_tokens(
    *parts: str, chars_per_token: int = CHARS_PER_TOKEN
) -> int:
    """Rough token count for one or more prompt fragments."""
    return sum(len(part) for part in parts if part) // chars_per_token


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]], k: int = RRF_K
) -> list[int]:
    """Fuse several ranked index lists into one. Highest RRF score first.

    Ties keep the order of the first ranking they appeared in, which
    makes the result deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: -scores[i])


def parse_gap_query(response: str) -> str | None:
    """Pull a search query out of a gap-check reply.

    The model is asked for ``QUERY: <keywords>`` or ``SUFFICIENT``.  A reply
    that is neither — typically a sentence explaining what is missing — is
    rejected rather than fed back in as a search query, which retrieves
    poorly.

    Two tiers, because a reply that used the requested format has earned more
    trust than one that ignored it:

    * **Prefixed** (``QUERY: ...``) — allowed up to ``MAX_GAP_WORDS``.
    * **Unprefixed** — allowed only if it still looks like bare keywords:
      at most ``MAX_BARE_GAP_WORDS`` and free of the comma/colon punctuation
      that signals sentence structure.

    The tiering exists because a single flat check cannot do both jobs.  A
    sentence-mark regex (``[.!?](\\s|$)``) was tried and removed: it cannot
    tell an abbreviation from a sentence end, so it rejected legitimate
    queries — "revenue vs. target", "Q3 rev. est.", "Dr. Alvarez appointment"
    — while missing

        "The facts do not mention the price, so search for: opening price"

    which is twelve words with no terminal punctuation and so cleared both
    the word cap and the regex, going straight back in as a search query.

    Skipping a refine round costs one missed follow-up search.  Feeding a
    sentence back in poisons that round's retrieval *and* burns an LLM call.
    The safe failure is the quiet one.
    """
    text = response.strip()
    if not text:
        return None

    match = _GAP_QUERY_RE.search(text)
    if match:
        query = match.group(1).strip().splitlines()[0]
        limit, prefixed = MAX_GAP_WORDS, True
    elif "SUFFICIENT" in text.upper():
        return None
    else:
        # Tolerate a bare keyword line from models that ignore the format,
        # but hold it to a keyword-shaped bar rather than the full one.
        query = text.splitlines()[0].strip()
        limit, prefixed = MAX_BARE_GAP_WORDS, False

    query = query.strip("\"'-— .")
    if not query:
        return None

    if not prefixed and _PROSE_PUNCT_RE.search(query):
        logger.warning(
            "  LIBRARIAN gap reply was prose, not a query — ignoring: %r",
            query[:120],
        )
        return None

    if len(query.split()) > limit:
        logger.warning(
            "  LIBRARIAN gap reply was too long to be keywords — ignoring: %r",
            query[:120],
        )
        return None
    return query


def _normalize_variant(text: str) -> str:
    """Key a phrasing is deduplicated under.

    Case and whitespace only. Anything cleverer — stemming, stopword
    removal — would collapse "car insurance" and "insurance for cars",
    which are exactly the pair expansion exists to produce.
    """
    return " ".join(text.split()).casefold()


def distinct_variants(question: str, also: Sequence[str]) -> list[str]:
    """Drop the phrasings that must not reach the fusion, order kept.

    Empty strings, repeats of *question*, and repeats of each other all
    go. A duplicate is not free: RRF sums 1/(k + rank) over the rankings
    it is given, so the same phrasing twice doubles its vote and quietly
    outranks the question it was derived from.

    Applied to caller-supplied ``also=`` as well as to generated
    variants, because ``retrieve()`` is a public entry point and a
    consumer assembling variants from its own client has no reason to
    have deduplicated them first.
    """
    # A bare string is a Sequence[str] that iterates into characters, so
    # `also="auto policy"` would silently become one ranking per letter.
    # Refuse it rather than retrieve nonsense.
    if isinstance(also, str):
        raise TypeError(
            "also must be a sequence of phrasings, not a single string — "
            "pass [phrasing] rather than phrasing"
        )
    seen = {_normalize_variant(question)}
    kept: list[str] = []
    for variant in also:
        if not isinstance(variant, str):
            raise TypeError(
                f"also must contain strings, got {type(variant).__name__}"
            )
        key = _normalize_variant(variant)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(variant.strip())
    return kept


def parse_expansion(response: str, question: str, limit: int) -> list[str]:
    """Pull alternative phrasings out of an expansion reply.

    Follows :func:`parse_gap_query`'s bargain rather than inventing a
    third reply shape: the model was asked for one query per line, a
    reply that is something else is *dropped* rather than fed to the
    index, and a quietly skipped expansion costs recall while a sentence
    of preamble used as a query costs precision on the query that
    mattered.

    ``parse_thinking`` runs first because this is the one path where an
    unstripped ``<think>`` block does not fail loudly: the protocol asks
    clients to strip it, but a reasoning block is line-oriented prose and
    would parse straight into the ranking list as though it were queries.

    Dropped per line: list markers and surrounding quotes (stripped, not
    rejected — the line is still a query), blank lines, anything carrying
    the comma/colon/semicolon punctuation of sentence structure
    (:data:`_PROSE_PUNCT_RE`, the same bar ``parse_gap_query`` holds an
    unprefixed reply to), anything wearing a refusal marker
    (:data:`_REFUSAL_RE`), and anything past ``MAX_EXPANSION_WORDS``.
    Survivors are deduplicated against *question* and each other, and the
    first *limit* are returned.

    Those two bars are what keep a refusal out of the ranking list, and
    it takes both. A refusal parsed as a query is worse than no
    expansion: "I cannot help with that request, sorry." retrieves on
    *cannot*, *help* and *request*, and that ranking fuses with the
    question's. The punctuation bar catches it — but not "Sorry I cannot
    do that", which carries no comma and is five words, well inside a
    cap sized for paraphrases (``MAX_EXPANSION_WORDS`` is 40, where
    ``parse_gap_query``'s unprefixed tier is 6, so the word cap cannot
    do this job here the way it does there).

    Both bars cost the occasional legitimate phrasing — one that wanted a
    comma, or one that says "I cannot" about the user rather than the
    model. That is the trade ``parse_gap_query`` already made, in the
    direction it made it: the safe failure is the quiet one, because a
    dropped phrasing costs recall on one query while a refusal used as a
    query costs precision on the query that mattered.
    """
    cleaned, _ = parse_thinking(response)
    candidates: list[str] = []
    for line in cleaned.splitlines():
        line = _LIST_MARKER_RE.sub("", line).strip().strip("\"'`")
        if not line:
            continue
        # Also catches the trailing colon a model announces a list with
        # ("Here are two phrasings:"), which is never how it phrases a
        # question. Dropped silently, unlike the overlong line below: when
        # a reply is nothing *but* prose, `expand_query` logs it whole at
        # WARNING, and a per-line record would arrive first and bury the
        # one that shows what the model actually said.
        if _PROSE_PUNCT_RE.search(line):
            continue
        # Dropped silently for the same reason as the line above: a
        # refused reply is a refusal on every line, so the whole-reply
        # WARNING in `expand_query` is the record worth reading.
        if _REFUSAL_RE.search(line):
            continue
        if len(line.split()) > MAX_EXPANSION_WORDS:
            logger.warning(
                "  LIBRARIAN expansion reply had a line too long to be a "
                "paraphrase — ignoring it: %r", line[:120],
            )
            continue
        candidates.append(line)
    return distinct_variants(question, candidates)[:limit]


def _find_json(text: str) -> dict | list:
    """Pull the first JSON object or array out of model output.

    Raises ValueError when there is none. Extracted from
    :meth:`SourceLibrarian._extract_json`, which now delegates here, so
    :func:`parse_fields` can reuse it without reaching into the class —
    the parsers are module-level by convention, being the half of this
    library that is testable without a model.
    """
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    for i, c in enumerate(text):
        if c in "{[":
            try:
                return json.loads(text[i:])
            except json.JSONDecodeError:
                depth = 0
                close = "}" if c == "{" else "]"
                for j, ch in enumerate(text[i:]):
                    if ch == c:
                        depth += 1
                    elif ch == close:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[i : i + j + 1])
                            except json.JSONDecodeError:
                                break
                continue
    raise ValueError(f"No JSON found in: {text[:300]}")


def _clean_scalar(value: object) -> str | None:
    """A model-supplied scalar, or None if it is not a usable one.

    Shared by the value and the quote so the two cannot drift. Numbers
    are coerced — an ``account`` of ``123456`` is a reference, not a
    quantity — while ``bool`` is refused rather than coerced, because
    ``true`` is not an identifier and ``bool`` is an ``int`` subclass, so
    admitting numbers admits it by accident.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    elif not isinstance(value, str):
        # dict or list: a nested structure is not a field value, and
        # str() of one would store "{'a': 1}" as an issuer.
        return None
    value = value.strip().strip("\"'")
    if not value or value.lower() in DOC_FIELD_PLACEHOLDERS:
        return None
    return value


def parse_fields(response: str) -> dict[str, dict]:
    """Pull a document's fields, each with the span it came from.

    Returns ``{name: {"value": str, "quote": str | None}}``, and ``{}``
    for anything it cannot read — a reply with no JSON, JSON that is not
    an object, or an object with no usable value under any known key. The
    caller logs; nothing raises. Same bargain as :func:`parse_expansion`
    and :func:`parse_gap_query`, and here the case is stronger than
    either: a document with no fields is still retrievable by ranking, so
    a dropped extraction costs a filter and never an answer.

    **Pure, and deliberately does not see the document.** Whether a quote
    is real is :func:`verify_fields`' question, and separating them is
    what lets this be unit-tested on a string and that one be the
    mechanical critic. A parser that also verified would have to be
    handed the corpus to test at all.

    ``fields-1`` accepted a bare ``{"doc_type": "receipt"}``; this reads
    the ``{"value": ..., "quote": ...}`` shape and also accepts a bare
    scalar as a value with no quote, which then fails the groundedness
    check downstream rather than being rejected here. That keeps the two
    failures distinct: "the model ignored the schema" and "the model
    could not support its answer" want different repair messages.

    Dropped per key: anything outside :data:`DOC_FIELD_NAMES`, values
    that are not scalars, values longer than :data:`MAX_FIELD_CHARS`, and
    the placeholder spellings in :data:`DOC_FIELD_PLACEHOLDERS`.

    **The placeholder bar is the load-bearing one.** ``DOC_FIELDS_SYSTEM``
    asks for ``null`` on an absent field, and a model obliges by writing
    ``"N/A"``. JSON null becomes an absent key, which every filter reads
    as "no evidence, do not exclude"; the string ``"N/A"`` becomes a
    *value*, which a filter matches against and no real document ever
    equals — so the document is excluded from every filtered query on the
    strength of a field it never had. Absent and unknown must arrive as
    the same thing here, because downstream they are not.

    A quote is held to :data:`MAX_QUOTE_CHARS` rather than
    ``MAX_FIELD_CHARS``: it is evidence, not an identifier, and needs room
    for the sentence around the value.
    """
    cleaned, _ = parse_thinking(response)
    try:
        payload = _find_json(cleaned)
    except (ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        # A list of field objects is a shape this prompt never asks for,
        # and guessing which element is the document's would be inventing
        # a value — the one thing the placeholder bar above exists to stop.
        return {}

    fields: dict[str, dict] = {}
    for name in DOC_FIELD_NAMES:
        entry = payload.get(name)
        if isinstance(entry, dict):
            value = _clean_scalar(entry.get("value"))
            quote = _clean_scalar(entry.get("quote"))
        else:
            # The fields-1 shape, or a model that ignored the schema.
            value, quote = _clean_scalar(entry), None
        if value is None:
            continue
        if len(value) > MAX_FIELD_CHARS:
            logger.warning(
                "  LIBRARIAN field %s too long to be an identifier — "
                "ignoring it: %r", name, value[:120],
            )
            continue
        if quote is not None and len(quote) > MAX_QUOTE_CHARS:
            # The value still stands; only its evidence is unusable, and
            # dropping the field for an over-long quote would throw away
            # a good answer over its footnote.
            quote = None
        fields[name] = {"value": value, "quote": quote}
    return fields


#: Markdown emphasis and heading punctuation, which a model strips when
#: it quotes. Removed from *both* sides before comparing, so the check
#: stays symmetric and an identifier containing one of these still
#: matches itself.
_MARKUP_RE = re.compile(r"[*_`~#]+")


def _normalize_ws(text: str) -> str:
    """Reduce text to what it *says*, for comparing a quote to a document.

    Whitespace collapses because a model re-wraps what it copies, so a
    quote spanning a line break comes back as one line.

    Markdown emphasis goes for the same reason and it is the more
    important half: **measured, requiring it byte-identical cost more
    than half of every field this extracts** (evidence:
    ``benchmarks/README.md``). These documents write
    ``**Issued By:** Pinnacle Meridian`` and the model faithfully quotes
    ``Issued By: Pinnacle Meridian``; the first version of this check
    called that absent from the document and rejected it, taking
    ``doc_type`` from 176 records to 30 and ``effective_date`` from 145
    to 12. The model was right and the critic was wrong.

    The general form of that mistake has now appeared three times in this
    feature — ISO against prose dates in the scorer, a value against its
    own normalised quote, and markup here. **Any comparison between text
    a model wrote and text a human wrote needs an explicit decision about
    what counts as the same**, and the default of byte-identity is
    reliably the wrong one (evidence: ``benchmarks/README.md``).
    """
    return " ".join(_MARKUP_RE.sub("", text).split())


def verify_fields(
    fields: dict[str, dict], document: str
) -> tuple[dict[str, dict], list[str]]:
    """The mechanical critic. Check each field against the document.

    Returns the surviving fields, each carrying a ``flags`` list, and the
    complaints to hand back to the model on a re-ask.

    **Deterministic, free, and unable to be talked out of its answer**,
    which is the whole reason it is not a model. This project already
    measured what an LLM critic does when asked whether an answer is good
    enough: ``_find_gap`` issues a query on 63.4% of gap checks and its
    ``SUFFICIENT`` promotes 9 questions to complete coverage, 5 of them
    on replies the parser threw away. Asking a model to judge gets
    agreement; asking a string whether it is present in another string
    gets a fact (evidence: ``benchmarks/refine_loop.py``,
    ``benchmarks/README.md``).

    Four checks, and each maps to a failure mode measured in ``fields-1``
    (evidence: ``benchmarks/field_extraction.py``,
    ``benchmarks/README.md``):

    * **The quote must be in the document.** Two of nine wrong dates were
      values the model assembled rather than read — ``2024-08-12`` from
      the prose "Notice filed August 12", which states no year. A value
      can be fabricated silently; a quote supporting it cannot (evidence:
      ``benchmarks/README.md``).
    * **The value must be inside its own quote.** Catches the quote that
      is real but does not contain what it is offered as evidence for —
      ``2019`` lifted out of "a board guideline adopted in 2019", where
      the span is genuine and is not a document date.
    * **A quote from quoted material is flagged, not dropped.** Four of
      nine came from a ``Date:`` header inside a ``>``-prefixed reply in
      an email thread. Flagged rather than dropped because a thread whose
      every line is quoted has no other evidence to offer, and a flag a
      filter can decline to act on beats deleting the only answer.
    * **``doc_type`` must be in :data:`DOC_TYPES`.** Free text measured
      at 69 labels over 176 documents is not a key. Out-of-vocabulary is
      dropped rather than flagged: an unusable key is worse than none,
      because a filter matching it silently returns nothing.

    An **unquoted** field survives with ``"unquoted"``. The model was
    asked for evidence and gave none, which is weaker than evidence that
    failed — but ``fields-1`` measured issuer at 158/161 present verbatim,
    so unquoted values are mostly right and dropping them would cost the
    strongest field this extracts (evidence: ``benchmarks/README.md``).
    """
    haystack = _normalize_ws(document)
    kept: dict[str, dict] = {}
    complaints: list[str] = []

    for name, record in fields.items():
        value, quote = record["value"], record.get("quote")
        flags: list[str] = []

        if name == "doc_type":
            # A classification, not an extraction: no span in the
            # document says "insurance policy", so membership of
            # DOC_TYPES *is* its verification and a quote is neither
            # required nor checked. Demanding one made the model quote
            # its own answer back — "quote": "insurance policy" — which
            # then failed the groundedness check, and that single
            # mistake cost 146 of 176 doc_type records.
            if value.lower() not in DOC_TYPES:
                complaints.append(
                    f"{name}: {value!r} is not one of the allowed values. "
                    f"Choose exactly one of: {', '.join(DOC_TYPES)}."
                )
                continue
            kept[name] = {"value": value, "quote": quote, "flags": []}
            continue

        if quote is None:
            flags.append("unquoted")
            complaints.append(
                f"{name}: you gave a value with no quote. Copy the exact "
                f"text from the document that states it."
            )
        else:
            needle = _normalize_ws(quote)
            if needle not in haystack:
                complaints.append(
                    f"{name}: your quote {quote[:80]!r} does not appear "
                    f"in the document. Copy the text exactly as written."
                )
                continue
            if not _value_supported_by(name, value, quote):
                complaints.append(
                    f"{name}: your quote {quote[:80]!r} does not support "
                    f"the value {value!r}. Quote the text that states it."
                )
                continue
            if _quoted_material(document, quote):
                flags.append("quoted_material")

        kept[name] = {"value": value, "quote": quote, "flags": flags}

    return kept, complaints


#: Fields whose value is a *normalisation* of the quote rather than a
#: copy of it, so a substring test is the wrong question — see
#: `_value_supported_by`.
_DATE_FIELDS = ("issued_date", "effective_date")

_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)


def _date_supported_by(value: str, quote: str) -> bool:
    """Could *value*, an ISO date, have been read off *quote*?

    Derivability, not identity. ``DOC_FIELDS_SYSTEM`` asks for
    ``YYYY-MM-DD`` while a document writes ``March 4, 2024``, so the value
    is never a substring of its own evidence and a substring test would
    reject every correctly-quoted date — which is exactly what it did on
    the first draft of this check.

    Each component the value states must be findable: the year always,
    the month as a name or a number when stated, the day as a number when
    stated. Tolerant on purpose — this is asking "is this value
    consistent with its evidence", and the stricter question of whether
    the right span was chosen is not one string matching can answer.
    """
    low = quote.lower()
    parts = value.split("-")
    if not parts or parts[0] not in quote:
        return False
    if len(parts) > 1:
        month = int(parts[1])
        name = _MONTH_NAMES[month - 1]
        if not (name in low or name[:3] in low
                or re.search(rf"\b0?{month}\b", quote)):
            return False
    if len(parts) > 2:
        if not re.search(rf"\b0?{int(parts[2])}\b", quote):
            return False
    return True


def _value_supported_by(name: str, value: str, quote: str) -> bool:
    """Is *value* consistent with the span the model says it came from?

    Three kinds of field, three questions, because "is the value in the
    quote" is only the right one for a third of them:

    * ``doc_type`` is chosen from :data:`DOC_TYPES` and appears in no
      document verbatim — "letter" is a classification of the text, not a
      phrase in it. Membership of the vocabulary is the whole check, and
      it has already run by the time this is called.
    * Dates are normalised to ISO, so see :func:`_date_supported_by`.
    * Everything else — issuer, account, period — is asked for as the
      document writes it, so a substring test is exactly right. Case
      folded, because a model title-cases an ALL-CAPS letterhead.
    """
    if name == "doc_type":
        return True
    if name in _DATE_FIELDS:
        return _date_supported_by(value, quote)
    return _normalize_ws(value).lower() in _normalize_ws(quote).lower()


def _quoted_material(document: str, quote: str) -> bool:
    """Did *quote* come from a ``>``-prefixed part of *document*?

    Answered against the document rather than the quote, because a model
    strips the markers when it copies: the quote comes back clean and the
    line it was taken from is not. Matched on the quote's first few words,
    which is what survives re-wrapping.
    """
    head = _normalize_ws(quote)[:40]
    if not head:
        return False
    for line in document.splitlines():
        if head in _normalize_ws(line):
            return line.lstrip().startswith(">")
    return False


def parse_facts(reply: str) -> list[dict[str, str]]:
    """Facts from one extraction reply. Tolerant, and never raises.

    Extraction must not fail a chunk: a chunk whose facts cannot be parsed
    is still indexed and still retrievable exactly as it was, which is
    `_extract_doc_fields`'s rule applied to the second extractor and for
    the same reason — trading a chunk's retrievability for its facts would
    be the wrong way round.

    A record is kept only when it carries a **value**, since a fact with no
    value is not a fact, and only when the value is short enough to be one
    (`MAX_FACT_VALUE_CHARS`). The subject and attribute may be empty: a
    chunk whose header names the subject leaves the model little to say
    there, and dropping the record would lose a value that retrieval can
    still reach through its quote.
    """
    text = (reply or "").strip()
    if not text:
        return []
    # A fenced block is what a model writes when asked for JSON in prose,
    # and `response_format` is not available on every backend this library
    # talks to.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        loaded = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(loaded, dict):
        loaded = loaded.get("facts", [])
    if not isinstance(loaded, list):
        return []

    out: list[dict[str, str]] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        value = _clean_scalar(item.get("value"))
        if not value or len(value) > MAX_FACT_VALUE_CHARS:
            continue
        record = {
            "subject": (_clean_scalar(item.get("subject")) or "")[:MAX_FACT_VALUE_CHARS],
            "attribute": (_clean_scalar(item.get("attribute")) or "")[:MAX_FACT_VALUE_CHARS],
            "value": value,
            "quote": (_clean_scalar(item.get("quote")) or "")[:MAX_FACT_QUOTE_CHARS],
        }
        out.append(record)
        if len(out) >= MAX_FACTS_PER_CHUNK:
            break
    return out


def fact_text(fact: dict[str, str]) -> str:
    """How a fact is indexed, and so what BM25 and the encoder see.

    Subject and attribute carry the vocabulary a question would use; the
    value is the payload; the quote keeps the document's own wording so a
    lexical match on the original phrasing still lands. This exact string
    is what was measured — one field rather than several — so it is the
    shape the recorded numbers describe (evidence:
    `benchmarks/fact_retrieval.py`, `benchmarks/README.md`).
    """
    head = " — ".join(p for p in (fact.get("subject"), fact.get("attribute")) if p)
    line = f"{head}: {fact.get('value', '')}" if head else fact.get("value", "")
    quote = fact.get("quote") or ""
    return f"{line}\n{quote}".strip()


def check_fact_mix(fact_mix: float) -> float:
    """The share of a retrieval's slots spent on facts. 0.0 to 1.0.

    A module-level `check_*` rather than a method, for the reason the
    others are: the lazy layers carry settings for a `SourceLibrarian`
    they build only when a question arrives, and a check deferred to
    construction fires a long way from the mistake.

    `bool` is rejected the way the integer checks reject it — `True` is a
    valid float in Python and a silent 1.0 here would spend every slot on
    facts.
    """
    if isinstance(fact_mix, bool):
        raise TypeError(f"fact_mix must be a number, got {fact_mix!r}")
    if not isinstance(fact_mix, numbers.Real):
        raise TypeError(
            f"fact_mix must be a number, got {type(fact_mix).__name__}")
    value = float(fact_mix)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"fact_mix must be between 0.0 and 1.0, got {value!r} — it is "
            f"the share of `top_k` spent on facts, not a count")
    return value


def _facts_hash_part(extract_facts: bool) -> str:
    """`_fields_hash_part`'s argument, for the fact store.

    Off, it contributes nothing at all, so the hash of a librarian that
    never asked for facts is byte-identical to its pre-knob value and **no
    index already on disk is invalidated**.
    """
    return f"|{FACTS_VERSION}" if extract_facts else ""


def _fields_hash_part(extract_fields: bool) -> str:
    """What field extraction contributes to :func:`content_hash`.

    **The empty string when extraction is off**, which is the whole
    design: the hash is then byte-identical to its pre-knob value, so
    upgrading to a version that has this feature invalidates no index
    already on disk. Only a librarian that opts in pays a re-index, and
    it pays it once.

    On, it contributes :data:`DOC_FIELDS_VERSION`, so a prompt change
    retires the caches carrying fields of the old shape.

    Why both halves are needed: ``extract_fields`` is per-librarian, and
    two librarians over the same text — one extracting, one not — build
    genuinely different indexes, so they must not share a key. That is
    the geometry's argument, which is why it arrives as an argument. The
    *version* is not per-librarian and nothing configures it, which is
    ``EMBED_SOURCE``'s argument, which is why it is read from the module
    here rather than passed in. The two rules are different and this is
    the one place both apply.
    """
    return f"|{DOC_FIELDS_VERSION}" if extract_fields else ""


def content_hash(
    documents: dict[str, str],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_threshold: int = CHUNK_THRESHOLD,
    embed_model: str = EMBED_MODEL,
    extract_fields: bool = EXTRACT_FIELDS,
    extract_facts: bool = EXTRACT_FACTS,
) -> str:
    """Hash indexed text plus the settings that shaped the index.

    Chunk geometry and the embedding model are folded in, so changing
    ``chunk_size`` or ``embed_model`` invalidates a saved index
    automatically.

    They are arguments rather than module constants read from here
    because they are per-librarian settings: two librarians configured
    differently over the same text produce different chunks and
    non-comparable vectors, and a hash that ignored that would let each
    load the other's cache. Callers with a librarian in hand should
    prefer :meth:`SourceLibrarian.content_hash`, which fills these in
    from the instance.

    ``EMBED_SOURCE`` is folded in too, and is read here rather than taken
    as an argument because nothing configures it — see the constant. It
    closes a stale-vector path the fingerprint check is structurally
    unable to see: :meth:`_check_embed_fingerprint` compares the *model*
    that produced the vectors, so a cache holding vectors of a different
    *text* under identical weights passes it and is loaded into a
    librarian that queries as though the two matched. Same documents,
    same geometry, same model, different embedded input is a different
    index, and this is where that becomes true of the cache key.
    """
    digest = hashlib.sha256()
    digest.update(
        f"{INDEX_FORMAT_VERSION}|{chunk_size}|{chunk_overlap}|"
        f"{chunk_threshold}|{embed_model}|{EMBED_SOURCE}"
        f"{_fields_hash_part(extract_fields)}"
        f"{_facts_hash_part(extract_facts)}".encode()
    )
    for doc_id, text in documents.items():
        digest.update(f"\0{doc_id}\0".encode())
        digest.update(text.encode())
    return digest.hexdigest()


def embedding_backend_available() -> bool:
    """Is the ``embeddings`` extra importable? Asked without importing it.

    ``find_spec`` rather than a ``try: import`` — importing
    ``sentence_transformers`` drags in torch, seconds of wall-clock and
    hundreds of MB of RSS, which is an absurd price for a question asked
    only to decide whether a log line is worth printing. A librarian that
    never retrieves must never import torch.

    Answers "installed", not "works": the model can still fail to
    download, refuse ``trust_remote_code`` or miss a transitive
    dependency. :meth:`SourceLibrarian._ensure_embed_model` is the one
    that finds out, and it is the one that says so at ERROR. This is for
    the paths that need a cheap, honest "would dense retrieval even be on
    the table here".
    """
    return importlib.util.find_spec("sentence_transformers") is not None


#: Revision to load each pinned model at, keyed by repo id. A dict
#: rather than two ``if name == DEFAULT`` branches so that the guard the
#: pin needs — never fire for a model we did not pin — is the lookup
#: itself and cannot be forgotten at one of the two call sites.
#: How each embedder wants its inputs marked. A model trained
#: asymmetrically needs a task prefix on documents and a different one on
#: queries; a symmetric model needs none, and giving it one is noise.
#:
#: **This existed as two hardcoded f-strings and it was a real bug.**
#: `_refresh_embeddings` wrote `search_document:` onto every chunk and
#: `_embedding_ranking` wrote `search_query:` onto every question —
#: nomic's convention, applied to whatever model `embed_model` named. So
#: the knob was not usable: pointing it at anything else silently fed
#: that model prompts it does not use. Measured on the 194-document
#: corpus, bge-m3 scored 71.8% under nomic's prefixes and **75.6% under
#: its own** — 3.8 points, larger than the gain from changing model at
#: all (evidence: `benchmarks/README.md`).
#:
#: A name that is not here gets no prefix, which is what most models
#: want and is the safe default: an unwanted prefix costs recall, a
#: missing one costs less. Same reasoning as `pinned_revision` returning
#: None for an unknown name.
EMBED_PROMPTS: dict[str, tuple[str, str]] = {
    # (document prefix, query prefix)
    "nomic-ai/nomic-embed-text-v1.5": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1": ("search_document: ", "search_query: "),
    "BAAI/bge-m3": ("", ""),
    "Qwen/Qwen3-Embedding-0.6B": (
        "", "Instruct: Given a web search query, retrieve relevant "
            "passages that answer the query\nQuery: "),
    "Qwen/Qwen3-Embedding-4B": (
        "", "Instruct: Given a web search query, retrieve relevant "
            "passages that answer the query\nQuery: "),
    "Qwen/Qwen3-Embedding-8B": (
        "", "Instruct: Given a web search query, retrieve relevant "
            "passages that answer the query\nQuery: "),
}


def embed_prompts(model_name: str) -> tuple[str, str]:
    """The (document, query) prefixes *model_name* wants. ``("", "")`` if
    it is not one this module knows — see :data:`EMBED_PROMPTS`."""
    return EMBED_PROMPTS.get(model_name, ("", ""))


_PINNED_REVISIONS = {
    EMBED_MODEL: EMBED_REVISION,
    RERANK_MODEL: RERANK_REVISION,
}


def pinned_revision(model_name: str) -> str | None:
    """The revision *model_name* is pinned to, or ``None`` if it is not.

    ``None`` is what both loaders default to, so an unpinned name is
    loaded exactly as it was before pinning existed: a user who points
    ``embed_model`` at their own model still tracks its default branch,
    which is the only thing that could possibly be meant. Handing them
    nomic's sha would resolve to nothing on their repo, or — worse — to
    a same-named branch of something unrelated.
    """
    return _PINNED_REVISIONS.get(model_name)


def _check_code_revision(
    model: object,
    expected: str,
    *,
    model_name: str,
    code_repo: str,
) -> None:
    """Warn when a model is running remote code we have not verified.

    Detection, not prevention: by the time this runs the code has been
    imported and the class constructed. It exists because the code
    *cannot* be pinned through sentence-transformers' public API (see
    ``EMBED_CODE_REVISION``), so the alternative is no signal at all —
    the default model's remote imports define the ``embeddings`` extra's
    real dependency set, and they can change with no commit to this
    repo. Silence there is how a clean install quietly degrades to
    BM25-only.

    transformers caches a dynamically-loaded module under
    ``transformers_modules/<org>/<repo>/<commit sha>/``, so the sha that
    actually ran is recoverable from the class's module path. A model
    running no remote code at all yields no sha and is not reported:
    that is either a local directory or a model that stopped needing
    ``trust_remote_code``, and neither is drift worth a warning.

    ``model_name`` and ``code_repo`` are parameters rather than the
    module constants of those names (``EMBED_MODEL`` and
    ``EMBED_CODE_REPO``, both defined above), because a sha alone is not
    an identity: the message has to name the repo the reader should go
    and diff. Hard-coded, they made the signature promise a generality
    the body did not deliver —
    a second caller passing its own ``expected`` would have emitted a
    warning naming nomic while reporting someone else's sha, sending the
    reader to the wrong repo, which is worse than saying nothing. They
    carry no defaults for that same reason — a default would keep that
    mismatch one omitted argument away.
    """
    try:
        auto = model[0].auto_model  # the Transformer module wraps it
        module = type(auto).__module__
    except Exception:  # not the module layout we know — nothing to say
        return
    found = re.search(r"\.([0-9a-f]{40})\.", module)
    if found and found.group(1) != expected:
        logger.warning(
            "  LIBRARIAN %s is executing remote code from %s at %s, not the "
            "verified %s — upstream moved. Dense retrieval still works, but "
            "the `embeddings` extra was checked against the verified "
            "revision's imports and may no longer cover this one",
            model_name, code_repo, found.group(1), expected,
        )


class StaleEmbeddingsError(RuntimeError):
    """Stored vectors were not produced by the embedding weights now loaded.

    Raised only under ``strict``; without it the librarian discards the
    vectors, warns, and serves BM25-only.

    A ``RuntimeError`` rather than the ``ValueError`` the rest of this
    module raises, and that is the whole reason it exists as a type:
    ``_load_if_current`` catches ``ValueError`` around ``load()`` to
    treat a corrupt cache as a cache miss and re-index. Drift is not
    corruption — under ``strict`` the caller asked to be stopped, and a
    ValueError here would be swallowed into a silent, and expensive,
    re-index of the whole corpus.
    """


class IndexBusyError(RuntimeError):
    """A whole-index operation was refused because indexing is in flight.

    Raised by :meth:`SourceLibrarian.load` and :meth:`SourceLibrarian.save`
    when another task is indexing this librarian — on this thread's loop
    or on a live loop belonging to another thread, which is the same
    damage seen from a different call site
    (:meth:`SourceLibrarian._observe_replace_guard`).
    Both replace or serialize the *entire* index, which is what
    ``index()`` takes ``_replace_guard`` for; they are synchronous, so
    they cannot take it, and they observe it instead — both halves of it,
    the in-flight adds (``_adds_in_flight_here``) and an ``index()``
    holding the exclusive hold (``_replacing_elsewhere``).

    ``load`` refuses on either. ``save`` refuses on a replace, and on an
    add warns instead unless ``strict``: the two produce different files
    and only one of them is recoverable. Each method documents its own
    measurement (evidence: ``TestLoadRefusesMidAdd``,
    ``TestLoadRefusesMidIndex``, ``TestSaveMidAdd``,
    ``TestSaveRefusesMidIndex`` — one class per case, in that order).

    Four ways out were available for ``load``, and this is the one that
    was taken:

    * **Refuse** (this). Cheap, synchronous, no signature change, and it
      converts a silent corruption into a loud error at the call that
      caused it. The cost is real — a call that looks legal now raises —
      but the alternative it replaces is not "works": see :meth:`load`
      for what it actually did.
    * **An async ``aload()`` sibling** taking ``_replacing()`` properly.
      Rejected because the unguarded method stays the obvious one to
      call: the corruption below would remain reachable, by the name in
      every existing example, and the fix would only be available to
      callers who already knew they needed it.
    * **Making ``load()`` async.** Correct and complete, and it breaks
      every caller — including ``Librarian.load`` and, from inside
      ``index()``, ``_load_if_current``. That is a large API break to
      close a window a refusal closes for free.
    * **Documenting it and moving on.** Ruled out by measurement rather
      than taste: the race reproduces on the first attempt through the
      public API, and its damage survives to disk and past the cache
      check (evidence: ``TestTheCorruptionTheRefusalPrevents``).

    A ``RuntimeError`` and not the ``ValueError`` most of this module
    raises, for exactly the reason :class:`StaleEmbeddingsError` is one:
    ``_load_if_current`` catches ``ValueError`` around ``load()`` and
    would swallow this into a silent, expensive re-index.

    ``strict`` gates it on ``save``-mid-add alone, because that flag
    chooses between degrading and raising and only that case has a
    degraded outcome to offer: a partial file that loads cleanly and
    re-chunks its missing document on the next add, so warning and
    writing it is a real option. ``load``'s degraded outcome *is* the
    corruption. ``save``-mid-``index()``'s is a file recording one
    un-chunked document and nothing else, which ``_load_if_current``
    calls current and serves whole — not a partial index anything
    repairs. Neither leaves the flag anything to select between. See
    both methods for the measurements behind that.
    """


class LoopHandoverError(IndexBusyError):
    """Indexing was refused because the *previous* event loop still holds work.

    A librarian outlives the loop it was used on, and
    :meth:`SourceLibrarian._replace_guard` supports that: meeting a loop
    it has not seen, it rebuilds the asyncio primitives (which bind to
    the loop they are first awaited on) and forgets the bookkeeping of
    loops that have closed. What it cannot do is *move* work: an
    ``index()`` here waits out the adds registered on this loop and no
    others, because an ``asyncio.Event`` can only signal the loop it is
    bound to (:meth:`SourceLibrarian._replacing`). A loop abandoned
    **open** with an ``add_document`` or an ``index()`` parked on it is
    therefore holding work this loop can neither wait for nor see finish:
    the parked task is suspended, not dead, and the next turn of that loop
    commits it into whatever index is here by then. This is raised instead
    of adopting the new loop.

    That makes it **permanent rather than scaffolding**, and worth saying
    because it was once both. It was introduced to keep a live loop away
    from three resets that destroyed its bookkeeping; those resets are
    gone — the count and the owner are keyed by loop — and this is still
    the thing that makes "wait only for your own loop" a rule rather than
    a hole. What changed is that it is no longer the *only* barrier
    between a caller and a corrupted index.

    **Why a subclass of :class:`IndexBusyError` rather than that class
    itself, and rather than an unrelated name.** Same family: a call was
    refused because indexing is in flight on this librarian and
    proceeding would corrupt it, so a caller who already writes ``except
    IndexBusyError`` around a shared librarian keeps catching it, and the
    ``RuntimeError``-not-``ValueError`` choice made there (so
    ``_load_if_current``'s ``except ValueError`` cannot swallow it into a
    silent, expensive re-index) is inherited rather than re-argued. Its
    own name because two things about it differ, and both are things a
    caller acts on:

    * **The remedy is not "wait".** An ``IndexBusyError`` from
      ``save``/``load`` clears itself when the adds it names finish —
      which, for adds on the caller's own loop, they will; this one does
      not clear until the *other* loop is closed or the work parked on it
      is awaited, and nothing on this loop makes that happen. A retry loop
      that is right for the base class over an add of its own spins
      forever on this one.

      That qualification is load-bearing rather than pedantic, because the
      two overlap. The base error also fires over an add on a loop
      abandoned **open** — :meth:`_observe_replace_guard` reads liveness,
      not identity — and *there* waiting does not clear it either: the
      caller cannot await a task belonging to a loop it is not running —
      which is why :meth:`_busy_with_adds`'s remedy names awaiting the work
      *on its own loop*, or closing that loop, rather than offering
      ``gather`` for tasks this loop does not own. So the clean statement is
      not that one type self-clears and the other does not. What is always
      true of *this* one is that the state which has to change is never on
      this loop, and that is what makes a retry here a spin rather than a
      wait.
    * **It surfaces from different methods.** The base class comes out of
      the synchronous ``save``/``load`` pair; this comes out of ``await
      add_document(...)`` / ``await index(...)``, the ordinary indexing
      calls, which is where the illegal handover is actually being
      attempted.

    **Refusing rather than warning-and-proceeding** is the decision the
    damage forces, not a taste for strictness. Proceeding is not a
    degraded outcome that a caller might accept: it is the corruption
    itself — the parked add commits into the index that replaced it,
    leaving chunks whose ``doc_id`` is in neither ``documents`` nor
    ``source``, ranked first by ``retrieve`` and written into a file
    ``_load_if_current`` then calls current. That
    is ``load``'s own reason for ignoring ``strict``, re-checked here
    rather than inherited: a flag can only choose between raising and
    *degrading*, and there is no degraded outcome on offer to select.

    Both halves of that are measured. The one-loop mechanism is
    ``TestTheCorruptionTheRefusalPrevents::test_shape_1_load_strands_the_in_flight_documents_chunks``,
    and ``mutation_registry.py --only remove-refusal`` says this refusal is
    load-bearing (16 tests). The **cross-loop** instance — an ``index()``
    on the new loop resetting while the old loop's add is still parked, and
    then that add resuming — is ``TestWhatProceedingWouldCost``: 24 chunks
    for a document ``documents`` no longer names, ``ODD_MARK`` gone from
    ``source``, ``retrieve`` handing them back first, and no exception
    anywhere.

    ``TestWhatProceedingWouldCost`` replaced an unverified marker one round
    after it was written, and the pair is worth keeping in that order. The
    claim used to rest on the one-loop measurement plus the observation that
    ``_replacing``'s reset is the same reset ``load`` performs — sound, and
    an inference rather than a measurement, which is what the marker said.
    It is also the direction a marker rots in: the evidence arrived and
    nothing about the sentence would have noticed. Only the marker naming
    what would discharge it made that a one-line edit.

    It fires **before** any of it. Nothing is corrupted at the transition
    — the harm lands later, when the parked task resumes — so the refusal
    is preventive, and the librarian is left exactly as it was: the guard
    keeps the old loop, and the same call succeeds once that loop is
    closed or drained.
    """


def _report_unfingerprintable(model: object, reason: str) -> None:
    """Say that a model's weights cannot be identified, and what it costs.

    WARNING rather than INFO, and the choice is not close. The thing being
    reported is a protection that is *absent* while everything else looks
    fine — the same shape as `_dense_ok`'s quiet flip, which is the
    failure this module keeps having to design against. The library
    installs no handler, so under Python's last-resort handler an INFO is
    invisible to every user who has not configured logging: reporting this
    at INFO would be indistinguishable from the silence it exists to end.

    The argument the other way is that a stub or an exotic encoder is
    something the user chose deliberately, so this fires on a path that is
    not a mistake. That is real, but it is bounded — the probe is taken
    once per model object (`_current_fingerprint` memoises on the object),
    so the cost of being wrong is one line per model per process, while
    the cost of being quiet is a stale-vector check that nobody knows is
    off. The message says retrieval is unaffected so the line cannot be
    read as a retrieval failure.

    Deliberately not raised even under ``strict``: nothing here is
    broken. Dense retrieval works, the vectors are usable, and the only
    thing lost is the ability to *notice* later that they stopped being.
    """
    logger.warning(
        "  LIBRARIAN cannot fingerprint the embedding weights of %s: %s. "
        "Retrieval is unaffected, but stored vectors carry no record of "
        "which weights made them, so this librarian cannot tell later "
        "that the weights changed under them — the stale-vector check is "
        "off for this model, and re-indexing will not turn it on",
        type(model).__name__, reason,
    )


def embed_fingerprint(model: object) -> dict | None:
    """Identify embedding weights by what they do, in one encode.

    Returns ``{"probe": ..., "vector": [...]}`` — the model's own
    embedding of :data:`EMBED_PROBE`, rounded — or ``None`` when no
    usable answer came back.

    **Why behaviour rather than provenance.** The alternative was the
    resolved snapshot the weights were loaded from, which is exact and
    free when it is obtainable. It is not obtainable here: the default
    model's weights are fetched revision-blind by its own remote code
    (``cached_file(model_name, WEIGHTS_NAME)``, measured — see
    ``EMBED_REVISION``, and benchmarks/pin_evidence.py for the run behind
    it), so the snapshot the HF cache could report is the
    *config's*, and it stays put across exactly the upstream weight
    change this is meant to catch. Reaching into cache internals for an
    answer about the wrong file is worse than not asking. A fingerprint
    also answers the question actually being asked — "are the stored
    vectors comparable to the ones a query will produce now?" — needs no
    Hub at all, and works for a local directory or any object with
    ``encode``.

    **What it does not catch.** One probe is one point of the function:
    a change that leaves this string's direction intact while moving
    others is invisible, and so is anything that only affects inputs the
    probe does not exercise — a longer ``max_seq_length``, a tokenizer
    change that touches no token in the probe. It cannot distinguish
    "different weights" from "same weights, different device or dtype";
    that is what :data:`EMBED_FINGERPRINT_TOLERANCE` is sized against. And
    it says nothing about a model that never loads — a BM25-only
    librarian has no vectors to stale.

    Costs one encode of a constant string, once per loaded model. It
    deliberately does not read the weight files or hash parameter
    tensors: definitive, and a full pass over ~500 MB on every index.

    **Every ``None`` but one says so**, at WARNING, through
    :func:`_report_unfingerprintable` — see there for why that level.
    "Once" is the caller's doing rather than this function's: the probe
    is taken once per model object because `_current_fingerprint`
    memoises on the object, so a per-document or per-query repeat is not
    reachable through the librarian. Calling this directly in a loop logs
    per call, which is the honest answer for a function whose contract is
    "probe this model now".
    """
    try:
        vector = np.asarray(
            model.encode([EMBED_PROBE], show_progress_bar=False)[0],
            dtype=float,
        )
    except Exception:
        # The one silent path, and it stays silent — same argument as
        # `_check_code_revision`'s unknown layout: an object that cannot
        # encode a probe is about to fail at the real work, loudly, and a
        # second complaint from here would only arrive first and bury it.
        # The branches below are different in kind: there the encode
        # *worked*, so nothing else is going to complain and the loss
        # would go unmentioned.
        return None
    if vector.ndim != 1 or vector.size == 0:
        _report_unfingerprintable(
            model, f"the probe encoded to shape {vector.shape}, not one vector"
        )
        return None
    if not np.all(np.isfinite(vector)):
        _report_unfingerprintable(
            model, "the probe's vector holds non-finite values (nan or inf)"
        )
        return None
    if not np.linalg.norm(vector):
        # A zero vector has no direction, so cosine cannot compare it to
        # anything — recording it would be recording an identity that
        # matches every other zero vector and nothing else.
        _report_unfingerprintable(
            model,
            f"the probe encoded to {vector.size} zeros, a vector with no "
            "direction, which cosine cannot tell from any other model's zeros",
        )
        return None
    return {
        "probe": EMBED_PROBE,
        "vector": [round(float(x), EMBED_FINGERPRINT_DIGITS) for x in vector],
    }


def fingerprint_drift(saved: object, current: object) -> float | None:
    """Cosine distance between two fingerprints, or ``None`` if incomparable.

    ``None`` means "cannot tell", and every caller treats it the way
    :func:`_check_code_revision` treats a model with no sha: change
    nothing, say nothing. It covers a missing or malformed fingerprint
    and — the case worth naming — one recorded against a *different*
    probe string, which is what an index written by a librarium whose
    :data:`EMBED_PROBE` differed looks like. Comparing two vectors of
    different strings would report drift on every load.

    A different dimensionality is not incomparable, it is conclusive:
    nothing that changes the width of the embedding space leaves the old
    vectors usable, so it returns the maximum a same-direction check can
    report rather than declining to answer.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return None
    if saved.get("probe") != current.get("probe"):
        return None
    left = np.asarray(saved.get("vector") or [], dtype=float)
    right = np.asarray(current.get("vector") or [], dtype=float)
    if left.ndim != 1 or right.ndim != 1 or not left.size or not right.size:
        return None
    if left.size != right.size:
        return 1.0
    norms = np.linalg.norm(left) * np.linalg.norm(right)
    if not norms or not np.isfinite(norms):
        return None
    return float(1.0 - np.dot(left, right) / norms)


def check_prefix_cap(
    llm: object, max_concurrent_prefixes: int, *, warn: bool = True
) -> int:
    """Validate a ``max_concurrent_prefixes`` value and return it.

    Shared by every class that takes the cap — SourceLibrarian, and the
    two that only forward it to a SourceLibrarian they build lazily
    (LibrarianRouter, WebLibrarian). Those two are the reason this is a
    function rather than a method: their librarian is not constructed
    until the first index() or the first question that falls through to
    the web, so a check deferred to it fires a long way from the call
    that made the mistake. The same argument applies to the pool warning,
    which is why it lives here too.

    Pass ``warn=False`` when the caller has already run this check with
    the same client and cap — the lazily built librarian would otherwise
    repeat its owner's warning verbatim.
    """
    # numbers.Integral rather than int: numpy is a hard dependency here, so
    # a cap that came out of a numpy computation — a sizing heuristic, a
    # value read out of an array — arrives as np.int64 and is a perfectly
    # sensible thing to pass. bool is still refused: it registers as
    # Integral, but True as a fan-out cap is a mistake, not a request for
    # one slot. (operator.index() would accept it, which is why it is not
    # used here.)
    if (not isinstance(max_concurrent_prefixes, numbers.Integral)
            or isinstance(max_concurrent_prefixes, bool)):
        raise TypeError(
            f"max_concurrent_prefixes must be an int, got "
            f"{type(max_concurrent_prefixes).__name__} — pass 0 for "
            f"unbounded fan-out"
        )
    if max_concurrent_prefixes < 0:
        raise ValueError(
            f"max_concurrent_prefixes must be >= 0, got "
            f"{max_concurrent_prefixes} — pass 0 for unbounded fan-out"
        )
    # Normalise at the boundary so nothing downstream has to care where the
    # cap came from: it goes on to size an asyncio.Semaphore, to compare
    # against the pool, and into log lines whose rendering of a numpy
    # scalar has changed between numpy releases.
    max_concurrent_prefixes = int(max_concurrent_prefixes)
    if warn:
        _warn_if_pool_is_narrower(llm, max_concurrent_prefixes)
    return max_concurrent_prefixes


def check_top_k(top_k: int) -> int:
    """Validate a ``top_k`` value and return it.

    A function rather than a method for the same reason as
    :func:`check_prefix_cap`: LibrarianRouter and WebLibrarian carry a
    default for a SourceLibrarian they do not build until the first
    index() or the first question that falls through to the web, so a
    check deferred to it fires a long way from the call that made the
    mistake.

    It also guards the per-call override, where a bad value is worse
    than late: ``top_k=0`` retrieves nothing at all and the model then
    answers from empty material instead of raising.
    """
    # numbers.Integral rather than int, exactly as in check_prefix_cap:
    # numpy is a hard dependency here, so a top_k that came out of a numpy
    # computation — a sizing heuristic, a value read out of an array —
    # arrives as np.int64 and is a perfectly sensible thing to pass. bool
    # is still refused: it registers as Integral, but True as a retrieval
    # count is a mistake, not a request for one chunk.
    if not isinstance(top_k, numbers.Integral) or isinstance(top_k, bool):
        raise TypeError(
            f"top_k must be an int, got {type(top_k).__name__}"
        )
    if top_k < 1:
        raise ValueError(
            f"top_k must be >= 1, got {top_k} — a query that retrieves "
            f"nothing leaves the model with no material to answer from"
        )
    # Normalise at the boundary so nothing downstream has to care where the
    # value came from: it goes on to slice result lists, to be compared
    # against len(), and into log lines whose rendering of a numpy scalar
    # has changed between numpy releases.
    return int(top_k)


def _check_int_setting(value: object, name: str, minimum: int) -> int:
    """Validate one integer knob and normalise it to a builtin ``int``.

    The Integral/bool/normalise reasoning is
    :func:`check_prefix_cap`'s, applied to the rest of the knobs rather
    than restated eight times: numpy is a hard dependency, so a value
    computed from an array arrives as ``np.int64`` and is a sensible
    thing to pass; ``bool`` registers as Integral but ``True`` as a
    chunk size is a mistake; and the builtin is what the value has to be
    by the time it reaches an f-string, a slice, or a ``len()``
    comparison.
    """
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def _check_choice(value: object, name: str, allowed: tuple[str, ...]) -> str:
    """Validate one string knob against a closed set of values."""
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(allowed)}, got {value!r}"
        )
    return str(value)


def _check_bool_setting(value: object, name: str) -> bool:
    """Validate one boolean knob.

    ``bool`` exactly, not "anything truthy". A flag passed as ``1`` is
    more likely a typo than a request, and the knobs this guards each
    turn an LLM call on — silently accepting a stray truthy value buys a
    per-document or per-question call nobody asked for.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def check_context_budget(
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    reserved_output_tokens: int = RESERVED_OUTPUT_TOKENS,
    *,
    warn: bool = True,
) -> tuple[int, int]:
    """Validate the two halves of the answer-call budget and return them.

    A function for the same reason as :func:`check_index_settings`:
    LibrarianRouter and WebLibrarian carry a budget for a SourceLibrarian
    they do not build until the first index() or the first question that
    falls through to the web.

    The relationship between the two is warned about, not refused. A
    reserve at or above the whole budget leaves no room for any prompt,
    so every ask() raises — but "budget smaller than the reserve" is also
    exactly how you deliberately force that error while testing the
    guard, and refusing it would make the failure impossible to
    reproduce through the public API (evidence:
    ``TestContextBudget::test_a_reserve_that_fills_the_budget_warns_rather_than_raises``,
    which takes that route in one test).
    """
    max_context_tokens = _check_int_setting(
        max_context_tokens, "max_context_tokens", 1
    )
    reserved_output_tokens = _check_int_setting(
        reserved_output_tokens, "reserved_output_tokens", 0
    )
    if warn and reserved_output_tokens >= max_context_tokens:
        logger.warning(
            "  LIBRARIAN reserved_output_tokens=%d is not smaller than "
            "max_context_tokens=%d — no prompt can fit in what is left, so "
            "every ask() and facts() will raise from check_context.",
            reserved_output_tokens, max_context_tokens,
        )
    return max_context_tokens, reserved_output_tokens


def check_expansion_settings(
    *,
    expand_query: bool = EXPAND_QUERY,
    expansion_variants: int = EXPANSION_VARIANTS,
    expansion_style: str = EXPANSION_STYLE,
) -> dict[str, object]:
    """Validate the query-expansion knobs and return them normalised.

    Keyed by parameter name, like :func:`check_index_settings`, so the
    two lazy layers can hold the dict and splat it into the
    SourceLibrarian they build.

    A function rather than a method for the reason every check here is
    one: LibrarianRouter and WebLibrarian carry these settings for a
    librarian that is not constructed until the first index() or the
    first question that falls through to the web, so a check deferred to
    construction fires a long way from the mistake.

    Separate from :func:`check_index_settings` because neither knob
    shapes the index: expansion happens at query time and is not folded
    into :func:`content_hash`, so turning it on must not invalidate a
    saved index.
    """
    return {
        # bool exactly, not "anything truthy" — the reasoning moved to
        # _check_bool_setting when extract_fields needed the same check.
        "expand_query": _check_bool_setting(expand_query, "expand_query"),
        # 0 would be "expansion on, nothing to expand into" — an LLM call
        # whose result is discarded. Off is spelled expand_query=False.
        "expansion_variants": _check_int_setting(
            expansion_variants, "expansion_variants", 1
        ),
        # Rejected rather than defaulted: a typo silently falling back to
        # a style measured negative is the shape of bug this module keeps
        # finding (evidence: `benchmarks/README.md`).
        "expansion_style": _check_choice(
            expansion_style, "expansion_style", ("hyde", "paraphrase")
        ),
    }


def check_index_settings(
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunk_threshold: int = CHUNK_THRESHOLD,
    rrf_k: int = RRF_K,
    rerank_top: int = RERANK_TOP,
    rerank_pool: int | None = RERANK_POOL,
    rerank_threshold: float = RERANK_THRESHOLD,
    chars_per_token: int = CHARS_PER_TOKEN,
    embed_model: str = EMBED_MODEL,
    rerank_model: str = RERANK_MODEL,
    extract_fields: bool = EXTRACT_FIELDS,
    extract_facts: bool = EXTRACT_FACTS,
    top_k: int = TOP_K,
    use_reranker: bool = False,
    warn: bool = True,
) -> dict[str, object]:
    """Validate the index/retrieval knobs and return them normalised.

    Keyed by parameter name, so a caller writes
    ``self.chunk_size = settings["chunk_size"]`` and nothing has to
    agree on a tuple order.

    One function for the whole set rather than one per knob, for the
    reason :func:`check_prefix_cap` is a function at all: LibrarianRouter
    and WebLibrarian carry these settings for a SourceLibrarian they do
    not build until the first index() or the first question that falls
    through to the web, so each has to run the checks itself or a typo
    surfaces a long way from the call that made it. A call per knob in
    each of three constructors is a list to keep in sync three times; one
    call is not — and the set grows, which is the other half of why no
    count is written here.

    ``top_k`` and ``use_reranker`` are read, not returned — they are
    validated and stored elsewhere, and appear here only because the
    rerank cap cannot be judged without them.

    Pass ``warn=False`` when the caller has already run this check with
    the same values, so a lazily built librarian does not repeat its
    owner's warning verbatim.
    """
    chunk_size = _check_int_setting(chunk_size, "chunk_size", 1)
    chunk_overlap = _check_int_setting(chunk_overlap, "chunk_overlap", 0)
    # chunk_spans() clamps an over-large overlap to chunk_size - 1 rather
    # than raising, which is right for a library function handed a
    # computed value mid-chunk but wrong for a setting: the caller asked
    # for geometry that cannot exist and would get a silently different
    # one for the lifetime of the librarian.
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be < chunk_size, got "
            f"chunk_overlap={chunk_overlap} chunk_size={chunk_size} — an "
            f"overlap that fills the chunk leaves no room for new material"
        )

    settings: dict[str, object] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # 0 is meaningful: chunk everything, however short.
        "chunk_threshold": _check_int_setting(
            chunk_threshold, "chunk_threshold", 0
        ),
        # 1/(k + rank) — k=0 divides by zero on the first-ranked chunk.
        "rrf_k": _check_int_setting(rrf_k, "rrf_k", 1),
        "rerank_top": _check_int_setting(rerank_top, "rerank_top", 1),
        # None is a value here, not an absence: "follow top_k", which is
        # the default and the only setting that leaves retrieval byte
        # identical to what it did before this knob existed. It stays
        # None rather than being resolved to a number because top_k is
        # overridable per call, so the resolution belongs to retrieve().
        "rerank_pool": (
            None if rerank_pool is None
            else _check_int_setting(rerank_pool, "rerank_pool", 1)
        ),
        # len(text) // chars_per_token — 0 divides by zero, and the whole
        # context budget is estimated through it.
        "chars_per_token": _check_int_setting(
            chars_per_token, "chars_per_token", 1
        ),
        "embed_model": _check_model_name(embed_model, "embed_model"),
        "rerank_model": _check_model_name(rerank_model, "rerank_model"),
        # bool exactly, for check_expansion_settings' reason: extract_fields=1
        # is more likely a typo than a request.
        "extract_fields": _check_bool_setting(extract_fields, "extract_fields"),
        # Index-time like `extract_fields`, and folded into
        # `content_hash` only when on for the same reason: leaving it
        # off must invalidate nothing already written.
        "extract_facts": _check_bool_setting(extract_facts, "extract_facts"),
    }

    if not isinstance(rerank_threshold, numbers.Real) or isinstance(
        rerank_threshold, bool
    ):
        raise TypeError(
            f"rerank_threshold must be a float, got "
            f"{type(rerank_threshold).__name__}"
        )
    settings["rerank_threshold"] = float(rerank_threshold)

    # The trap this warning exists for: the reranker keeps rerank_top
    # chunks out of the top_k that retrieval selected, so with the stock
    # 5 against a top_k of 10 half of what was asked for is discarded and
    # the only symptom is a shorter list. Unlike the router's ignored-cap
    # warning this fires on the defaults too, because the defaults are
    # where it bites: nobody raising top_k for recall expects the
    # reranker to undo it.
    if warn and use_reranker and top_k > settings["rerank_top"]:
        logger.warning(
            "  LIBRARIAN use_reranker=True with rerank_top=%d caps retrieval "
            "at %d chunks even though top_k=%d — the reranker keeps only "
            "rerank_top of the chunks retrieval selected. Raise rerank_top to "
            "match top_k, or lower top_k to stop paying for chunks that are "
            "discarded.",
            settings["rerank_top"], settings["rerank_top"], top_k,
        )
    # The mirror-image mistake to the one above, and the likely cause is
    # confusing the two knobs: rerank_pool sets how many candidates the
    # cross-encoder is *given*, rerank_top how many it hands *back*. A
    # pool under top_k asks for a candidate set narrower than the answer,
    # which retrieve() will not do — it takes max(pool, top_k) — so the
    # setting silently does nothing at all rather than doing something
    # bad, and silence is exactly what makes it worth saying. Nothing to
    # warn about when it is None: that is the default, meaning follow
    # top_k, which no caller chose.
    pool = settings["rerank_pool"]
    if warn and use_reranker and pool is not None and pool < top_k:
        logger.warning(
            "  LIBRARIAN rerank_pool=%d is below top_k=%d, so it widens "
            "nothing — the cross-encoder is already given every chunk "
            "retrieval selected, and the pool is raised to top_k. "
            "rerank_pool sets how many fused candidates are scored; "
            "rerank_top=%d is what sets how many survive. Raise "
            "rerank_pool above top_k to give the reranker more to choose "
            "from, or set rerank_top if you meant the size of the result.",
            pool, top_k, settings["rerank_top"],
        )
    return settings


def _check_model_name(value: object, name: str) -> str:
    """Validate an embedding/reranker model identifier.

    Only that it is a non-empty string: whether the name resolves is
    sentence-transformers' business, and it reports a missing model far
    better than a guess here could. An empty string is worth catching
    because it reads as "no model" while in fact reaching the loader,
    which then fails deep inside a download path.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _warn_if_pool_is_narrower(llm: object, limit: int) -> None:
    """Warn when the HTTP pool, not the cap, is the real limit.

    The two settings are documented as needing to stay aligned; when
    they drift the symptom is requests queueing invisibly inside the
    client, which is the exact failure the cap exists to prevent.

    0 means unbounded. An unbounded *pool* narrows nothing, so it is
    always quiet; an unbounded *cap* against a finite pool is the
    loudest narrowing there is — a 2MB document fires a request per
    chunk into however many slots the pool has — so it warns too.
    """
    # Any client may advertise a pool, not just OpenAICompatClient, so this
    # reads whatever it finds — and accepts numpy integers for the same
    # reason the cap does, rather than silently going quiet on a client
    # whose pool size came out of a computation.
    pool = getattr(llm, "max_connections", None)
    if (not isinstance(pool, numbers.Integral) or isinstance(pool, bool)
            or pool <= 0):
        return
    if limit == 0:
        logger.warning(
            "  LIBRARIAN max_concurrent_prefixes=0 (unbounded fan-out) "
            "against the client's max_connections=%d — the HTTP pool is "
            "the real limit and every call past the %dth will queue "
            "there. Set max_concurrent_prefixes, or max_connections=0.",
            pool, pool,
        )
    elif pool < limit:
        logger.warning(
            "  LIBRARIAN max_concurrent_prefixes=%d exceeds the client's "
            "max_connections=%d — the HTTP pool is the real limit and the "
            "extra calls will queue there. Raise max_connections to match.",
            limit, pool,
        )


def _loop_is_live(loop: asyncio.AbstractEventLoop | None) -> bool:
    """Can a task parked on *loop* still run and commit into this index?

    The one question the replace guard's bookkeeping is read and pruned
    against, in one place so the readings cannot drift: it is asked of the
    guard's own loop by
    :meth:`SourceLibrarian._observe_replace_guard` and of every other
    loop the bookkeeping is keyed by in
    :meth:`SourceLibrarian._forget_closed_loops`.

    ``is_closed()`` and deliberately **not** ``is_running()``: a loop
    between ``run_until_complete`` calls is not executing, but a task
    parked on it is suspended rather than dead and the next turn of that
    loop commits it. See :meth:`SourceLibrarian._observe_replace_guard`
    for what erring the other way costs — that argument is the reason
    this is one line and not two.
    """
    return loop is not None and not loop.is_closed()


def _provenance(chunks: Sequence[Chunk]) -> str:
    """A ``Fact.source`` label naming the documents a fact came from."""
    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks if c.doc_id))
    if not doc_ids:
        return "librarian"
    if len(doc_ids) <= 3:
        return ",".join(doc_ids)
    return ",".join(doc_ids[:3]) + f",+{len(doc_ids) - 3} more"


class SourceLibrarian:
    """Indexes source text and answers questions via dual retrieval.

    Index time (once per document):
      1. Chunk the text
      2. Generate document summary (1 LLM call)
      3. Generate contextual prefix per chunk (parallel, /no_think)
      4. Build BM25 index on prefixed chunks
      5. Embed raw chunks with nomic-embed
      6. Optionally load cross-encoder reranker

    Query time (per question):
      1. BM25 + embedding search → RRF fusion → top-K
      2. Optional reranking
      3. LLM answers against retrieved raw chunks
    """

    def __init__(
        self,
        llm: LLMClient,
        use_reranker: bool = False,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        reserved_output_tokens: int = RESERVED_OUTPUT_TOKENS,
        strict: bool = False,
        max_concurrent_prefixes: int = MAX_CONCURRENT_PREFIXES,
        *,
        top_k: int = TOP_K,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        chunk_threshold: int = CHUNK_THRESHOLD,
        rrf_k: int = RRF_K,
        rerank_top: int = RERANK_TOP,
        rerank_pool: int | None = RERANK_POOL,
        rerank_threshold: float = RERANK_THRESHOLD,
        chars_per_token: int = CHARS_PER_TOKEN,
        embed_model: str = EMBED_MODEL,
        rerank_model: str = RERANK_MODEL,
        extract_fields: bool = EXTRACT_FIELDS,
        extract_facts: bool = EXTRACT_FACTS,
        fact_mix: float = FACT_MIX,
        cite_sources: bool = CITE_SOURCES,
        expand_query: bool = EXPAND_QUERY,
        expansion_variants: int = EXPANSION_VARIANTS,
        expansion_style: str = EXPANSION_STYLE,
        _warn_settings: bool = True,
    ):
        """
        Args:
            llm: any LLMClient (see librarium.llm) — e.g. OpenAICompatClient
            use_reranker: enable cross-encoder reranking (slower, more precise)
            max_context_tokens: total budget for an answer LLM call — prompt
                *and* reply. Set it at or below your model's context limit
                (e.g. 6000 for an 8K model).
            reserved_output_tokens: how much of that budget to hold back for
                the model's reply.
            strict: raise instead of degrading when the embedding model or
                reranker is installed but fails to load. The default (False)
                falls back to BM25-only retrieval and logs an error.
            max_concurrent_prefixes: how many index-time prefix calls may be
                in flight at once. This is a property of your deployment, not
                of the document — vLLM's --max-num-seqs, Ollama's serial
                default, a hosted rate limit. 0 restores unbounded fan-out.
                If you raise it, raise OpenAICompatClient's max_connections
                to match, or the HTTP pool becomes the real limit.
            top_k: keyword-only. How many chunks a query retrieves, for
                every path that does not say otherwise. The main lever on
                the context/precision tradeoff: lower it for a tight
                context window, raise it when recall matters more than the
                token bill. Every query method takes a ``top_k=`` of its
                own to override this for one call.
            chunk_size: keyword-only. Maximum characters per chunk,
                overlap included. Trades retrieval precision against how
                much surrounding context each retrieved chunk carries,
                and with ``top_k`` sets how much material an answer call
                sends: ``top_k`` × ``chunk_size`` chars is the ceiling.
            chunk_overlap: keyword-only. Characters of the previous chunk
                carried into the next, so a fact spanning a boundary is
                still whole in one of them. Must be < ``chunk_size``.
            chunk_threshold: keyword-only. Total indexed characters at or
                below which the corpus is served whole — no chunking, no
                index-time LLM calls, ``top_k`` unused. Raise it to skip
                indexing cost on larger inputs your context window can
                take; 0 chunks everything.
            rrf_k: keyword-only. The constant in reciprocal rank fusion's
                1/(k + rank). Lower makes the top of each ranking count
                for more, higher flattens the two rankings towards equal
                weight.
            rerank_top: keyword-only. How many chunks survive reranking,
                out of the ``top_k`` retrieval selected. Only read when
                ``use_reranker`` is on. The default is below the default
                ``top_k`` on purpose — reranking is a precision funnel —
                so raise it alongside ``top_k`` if you want every
                retrieved chunk kept.
            rerank_pool: keyword-only. How many fused candidates the
                cross-encoder scores — the input width, where
                ``rerank_top`` is the output width. ``None``, the
                default, means "as many as the query asks for", so
                retrieval is unchanged for anyone who does not set it.
                Raise it to let reranking reach chunks the two cheap
                rankers put outside ``top_k``: fusion, reranking and the
                cut to ``rerank_top`` happen in that order, so without
                this the cross-encoder can only reorder a set it cannot
                add to. Only read when a reranker is actually loaded, and
                never below ``top_k`` — a value under it widens nothing
                and warns at construction. The cost is one cross-encoder
                forward pass per extra candidate, paid inside the
                synchronous :meth:`retrieve`. What comes back is still
                whatever ``rerank_top`` keeps, which is the width that
                bounds an answer: with ``rerank_top`` set *above*
                ``top_k`` — where it used to be a cap that never bound,
                since nothing longer than ``top_k`` reached it — a raised
                pool can now return more than ``top_k`` chunks, and the
                context budget is spent accordingly.
            rerank_threshold: keyword-only. Cross-encoder score a chunk
                must beat to be kept. If nothing clears it the top
                ``rerank_top`` are kept anyway, so this trims weak
                matches rather than risking an empty result. With
                ``rerank_pool`` raised it judges a wider and therefore
                weaker-tailed population.
            chars_per_token: keyword-only. The ratio used to estimate a
                prompt's token count against ``max_context_tokens``. ~4
                suits most tokenizers; lower it for text they split
                harder (CJK, dense code) so the estimate stays
                conservative.
            embed_model: keyword-only. sentence-transformers model for
                the dense half of retrieval. Folded into
                :func:`content_hash`, so changing it invalidates a saved
                index rather than loading vectors that are not comparable.
            rerank_model: keyword-only. Cross-encoder model used when
                ``use_reranker`` is on.
            extract_fields: keyword-only. Spend one extra LLM call per
                document at index time pulling out its identifying
                fields — type, issuer, date, period, account — recorded
                on the instance as ``doc_fields`` and saved with the
                index. Off by default for ``expand_query``'s reason: it
                costs a call nobody asked for, and off keeps
                :func:`content_hash` byte-identical to its pre-knob value
                so no index already on disk is invalidated. Extraction
                never fails a document — a reply that yields no fields
                leaves it indexed and retrievable exactly as it is
                today, and logs.
            expand_query: keyword-only. Let :meth:`expand_query` and
                :meth:`retrieve_expanded` spend one LLM call rewriting a
                question into alternative phrasings before retrieval.
                Off by default: ``retrieve()`` makes zero LLM calls and
                that is a documented property of the library, so opting
                in is explicit. With it off, :meth:`expand_query` returns
                no variants without calling the model at all. Held on the
                instance as ``expansion_enabled``, since
                ``self.expand_query`` is the method.
            expansion_variants: keyword-only. How many alternative
                phrasings to ask for. Each becomes one more lexical
                ranking in the fusion, and RRF weights rankings equally,
                so raising it dilutes the original question's vote.
            _warn_settings: internal, keyword-only. False when the caller
                already ran :func:`check_prefix_cap` and
                :func:`check_index_settings` with these same values, so the
                pool and rerank-cap warnings are not emitted twice — it
                silences the advice, never the validation. Keyword-only
                because it trails six public arguments: as a positional it
                would catch a miscounted seventh argument and silence the
                warnings without the caller ever naming it.
        """
        self.llm = llm
        self.source: str = ""                    # every indexed document, joined
        self.documents: dict[str, str] = {}      # doc_id → text
        self.doc_summaries: dict[str, str] = {}  # doc_id → LLM summary
        # doc_id → identifying fields, when `extract_fields` is on. A
        # document whose extraction yielded nothing is **absent**, never
        # present-and-empty: downstream, absence is "no evidence, do not
        # filter this out", and the two must not be told apart by whether
        # a dict happens to be empty.
        self.doc_fields: dict[str, dict[str, str]] = {}
        self.chunks: list[Chunk] = []
        # Atomic assertions extracted from the chunks, when `extract_facts`
        # is on. Each carries the `doc_id` and `chunk` it came from, so a
        # retrieved fact is as traceable as a retrieved chunk. Parallel to
        # `chunks` rather than inside them: a fact is its own retrievable
        # unit, and burying it in a Chunk would make it unrankable.
        self.facts: list[dict[str, str]] = []
        self.fact_bm25: BM25Okapi | None = None
        self.fact_embeddings: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self.chunk_embeddings: np.ndarray | None = None
        self.embed_model = None   # SentenceTransformer, lazy loaded
        self.reranker = None      # CrossEncoder, lazy loaded
        # The last failure `_load_reranker` reported, so the retry it
        # deliberately keeps does not re-report the same thing per
        # document. Model state, not index state — `_reset` keeps loaded
        # models and so keeps this, for the same reason it keeps
        # `_model_fingerprint`.
        self._reranker_report: tuple | None = None
        self.use_chunking: bool = False
        self.use_reranker: bool = use_reranker
        self.max_context_tokens, self.reserved_output_tokens = (
            check_context_budget(
                max_context_tokens, reserved_output_tokens,
                warn=_warn_settings,
            )
        )
        self.strict: bool = strict
        self.max_concurrent_prefixes: int = check_prefix_cap(
            llm, max_concurrent_prefixes, warn=_warn_settings
        )
        self.top_k: int = check_top_k(top_k)
        settings = check_index_settings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_threshold=chunk_threshold,
            rrf_k=rrf_k,
            rerank_top=rerank_top,
            rerank_pool=rerank_pool,
            rerank_threshold=rerank_threshold,
            chars_per_token=chars_per_token,
            embed_model=embed_model,
            rerank_model=rerank_model,
            extract_fields=extract_fields,
            extract_facts=extract_facts,
            top_k=self.top_k,
            use_reranker=use_reranker,
            warn=_warn_settings,
        )
        self.chunk_size: int = settings["chunk_size"]
        self.chunk_overlap: int = settings["chunk_overlap"]
        self.chunk_threshold: int = settings["chunk_threshold"]
        self.rrf_k: int = settings["rrf_k"]
        self.rerank_top: int = settings["rerank_top"]
        self.rerank_pool: int | None = settings["rerank_pool"]
        self.rerank_threshold: float = settings["rerank_threshold"]
        self.chars_per_token: int = settings["chars_per_token"]
        # `embed_model` is taken: it holds the loaded SentenceTransformer,
        # which is what the rest of the class reaches for. The configured
        # *name* therefore lives under a distinct attribute rather than
        # being overwritten the first time the model loads.
        self.embed_model_name: str = settings["embed_model"]
        self.rerank_model: str = settings["rerank_model"]
        # Unlike `expand_query` there is no method of this name to shadow,
        # so the attribute keeps the argument's spelling.
        self.extract_fields: bool = settings["extract_fields"]
        self.extract_facts: bool = settings["extract_facts"]
        # Query-time, so deliberately *not* in `check_index_settings`
        # and so not in `content_hash`: changing the mix must not
        # invalidate a store that cost one LLM call per chunk.
        self.fact_mix: float = check_fact_mix(fact_mix)
        self.cite_sources: bool = _check_bool_setting(
            cite_sources, "cite_sources")
        expansion = check_expansion_settings(
            expand_query=expand_query,
            expansion_variants=expansion_variants,
            expansion_style=expansion_style,
        )
        # NOT `self.expand_query`: that name belongs to the method, and
        # binding the flag to it would shadow the coroutine the setting
        # exists to enable — `await lib.expand_query(q)` would raise
        # "bool is not callable". The constructor argument keeps the
        # documented name; the attribute is the one that has to move.
        self.expansion_enabled: bool = expansion["expand_query"]
        self.expansion_variants: int = expansion["expansion_variants"]
        # NOT `expansion_mode` — that name is taken by the property
        # reporting whether expansion runs at all (off/on/inert),
        # which is a different question and is public API.
        self.expansion_style: str = expansion["expansion_style"]
        # Built on first use, not here: asyncio primitives bind to a loop.
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._gate: asyncio.Semaphore | None = None
        self._gate_loop: asyncio.AbstractEventLoop | None = None
        self._replace_lock: asyncio.Lock | None = None
        self._adds_idle: asyncio.Event | None = None
        self._replace_loop: asyncio.AbstractEventLoop | None = None
        # In-flight `add_document` registrations, **keyed by the loop they
        # were taken on** — read through :meth:`_adds_registered_on`, which
        # is the one place this shape is known.
        #
        # It was a bare `int`, and that conflated two different things: how
        # many adds can still commit into this index is a durable fact
        # about the librarian, while `_adds_idle` is a loop-local
        # *mechanism* for telling a waiter the count reached zero. They
        # were rebound together in `_replace_guard`'s new-loop branch, so
        # refreshing the mechanism destroyed the fact — a parked add's
        # `finally` then gave back a registration that had already been
        # taken away, the count reached -1, and `_replacing`'s drain read
        # that as busy against an Event nobody would set again. Keyed by
        # loop, a decrement always finds the increment it belongs to, and a
        # handover has nothing of another loop's to zero.
        self._adds_in_flight: dict[asyncio.AbstractEventLoop, int] = {}
        # The task inside `_replacing()`, per loop, for the same reason and
        # read through :meth:`_replace_owner_on`. An asyncio.Lock records
        # no owner, and the synchronous guard needs one: `index()` loads
        # and saves from inside its own hold, so "is the lock held" is
        # the wrong question and "is somebody else holding it" is the
        # right one. See :meth:`_replacing_elsewhere`.
        self._replace_owner: dict[
            asyncio.AbstractEventLoop, asyncio.Task
        ] = {}
        # doc_ids some in-flight call has taken responsibility for chunking.
        # The pending sweep reads `documents`, which a concurrent add has
        # already written to, so without a claim two calls would chunk the
        # same document — twice the LLM calls, twice the chunks.
        self._chunking: set[str] = set()
        self.doc_summary: str = ""   # summary of the most recent document
        self._dense_ok: bool = True  # False once the embed model proves unusable
        # The fingerprint of the weights that produced `chunk_embeddings`,
        # as far as this librarian can vouch for it: index state, so it
        # travels with the vectors through save/load/_reset. None means
        # "not recorded" — an index from before fingerprints existed, or
        # one whose model could not be fingerprinted — and is never
        # upgraded to a guess, because stamping the current model's
        # identity onto vectors this process did not produce would
        # launder an unknown into a verification.
        self._vector_fingerprint: dict | None = None
        # The loaded model's own fingerprint, cached against the object it
        # was computed from so a re-assigned `embed_model` is re-probed
        # and nothing else pays a second encode.
        self._model_fingerprint: dict | None = None
        self._fingerprint_of: object | None = None

    # ── Introspection ────────────────────────────────────────────

    @property
    def retrieval_mode(self) -> str:
        """Which retrieval path a query will actually take.

        ``"hybrid"`` — BM25 + embeddings, RRF-fused. The configuration
        the published benchmark numbers were measured on (evidence:
        benchmarks/README.md, which records the date, the model and the
        three suites those numbers came from).
        ``"bm25-only"`` — keyword search alone; the embedding half is
        unavailable.
        ``"whole-document"`` — short input, no retrieval, everything
        goes to the model.
        ``"unindexed"`` — nothing indexed yet.

        Read off how the index is configured, so it describes a settled
        index. While one is being built it can still report ``hybrid``
        or ``bm25-only`` over a query that will in fact degrade to whole
        documents — see :meth:`add_document`.
        """
        if not self.documents and not self.source:
            return "unindexed"
        if not self.use_chunking or self.bm25 is None:
            return "whole-document"
        if self.chunk_embeddings is not None and self._dense_ok:
            return "hybrid"
        return "bm25-only"

    @property
    def expansion_mode(self) -> str:
        """Whether a question will be expanded before it is retrieved for.

        ``"off"`` — ``expand_query=False`` (the default); no expansion
        call is made and :meth:`retrieve_expanded` is a plain
        :meth:`retrieve`.
        ``"on"`` — expansion is enabled and the fused rankings will
        include one per variant.
        ``"inert"`` — enabled, but this index answers from whole
        documents (or has nothing indexed), and that path returns every
        document regardless of any ranking. The call would still be paid
        for and change nothing.

        A property of its own rather than a suffix on
        :attr:`retrieval_mode`. Expansion is orthogonal to the
        lexical/dense axis, and "hybrid+expanded" would break every
        ``retrieval_mode == "hybrid"`` comparison already written against
        the four documented strings — including the one in this
        project's own README.
        """
        if not self.expansion_enabled:
            return "off"
        if self.retrieval_mode in ("whole-document", "unindexed"):
            return "inert"
        return "on"

    @property
    def prefixed_chunks(self) -> list[str]:
        """Chunk texts with their context prefixes — what BM25 indexes."""
        return [chunk.prefixed for chunk in self.chunks]

    # ── LLM helpers ──────────────────────────────────────────────

    async def _llm_call(
        self, system: str, user: str, *, temperature: float = 0.1
    ) -> str:
        """Single LLM call via the injected client.

        *temperature* is an argument rather than the hardcoded 0.1 it
        used to be because one caller genuinely wants something else:
        expansion is asking for wording the model would not otherwise
        reach for (see :data:`EXPANSION_TEMPERATURE`). Every other call
        site here is extracting or copying text, where 0.1 is the point,
        so the default stays where it was.
        """
        return await self.llm.complete(
            system, user, think=False, temperature=temperature
        )

    async def _llm_call_json(self, system: str, user: str) -> dict | list:
        """LLM call that extracts JSON from response. Retries once."""
        for attempt in range(2):
            text = await self._llm_call(system, user)
            try:
                return self._extract_json(text)
            except ValueError:
                if attempt == 0:
                    system += "\n\nIMPORTANT: Respond with ONLY valid JSON."
                else:
                    raise
        raise ValueError("unreachable")

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Pull JSON from model output.

        Kept as a method because callers and tests reach it by this name;
        the body moved to :func:`_find_json` so :func:`parse_fields` can
        use it too. Behaviour is unchanged, ValueError included.
        """
        return _find_json(text)

    # ── Context budget ───────────────────────────────────────────

    def check_context(self, system: str, user: str) -> None:
        """Raise if this prompt plus the reserved reply exceeds the budget.

        Counts the *whole* call — system prompt, user turn, and the
        headroom kept for the reply — not just the retrieved material.
        """
        prompt_tokens = estimate_tokens(
            system, user, chars_per_token=self.chars_per_token
        )
        total = prompt_tokens + self.reserved_output_tokens
        if total <= self.max_context_tokens:
            return
        raise ValueError(
            f"This call is ~{total} tokens (~{prompt_tokens} prompt + "
            f"{self.reserved_output_tokens} reserved for the reply), which "
            f"exceeds max_context_tokens={self.max_context_tokens}. This will "
            f"overflow the model's context window.\n\n"
            f"Fixes:\n"
            f"  - Reduce ScrapeConfig.max_pages or max_depth (for WebLibrarian)\n"
            f"  - Lower top_k — per call (ask(..., top_k=5)) or for every "
            f"query (Librarian(llm, top_k=5))\n"
            f"  - Lower chunk_size (Librarian(llm, chunk_size=1000))\n"
            f"  - Lower reserved_output_tokens if you expect a short reply\n"
            f"  - Increase max_context_tokens if using a larger model"
        )

    # ── Indexing ─────────────────────────────────────────────────

    def _index_lock(self) -> asyncio.Lock:
        """The lock serializing index mutations, for the running loop.

        Narrow on purpose: it is held across the *mutations* only, never
        across an LLM call. Holding it for a whole document is what made
        a corpus index one document at a time; holding it for the reads
        and writes that have to see a consistent index — the doc_id
        counter, the chunk numbering, `source`, the embedding append —
        costs nothing and is what lets the calls overlap.

        Built lazily rather than in ``__init__`` because an
        ``asyncio.Lock`` binds to the loop that first awaits it, so one
        created at construction time breaks across ``asyncio.run()``
        boundaries — the same reason the prefix gate is loop-keyed.
        """
        loop = asyncio.get_running_loop()
        if self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _prefix_gate(self) -> AbstractAsyncContextManager:
        """The cap on index-time LLM calls, for the running loop.

        One gate per *librarian*, not one per fan-out: documents overlap
        now, so a semaphore built inside ``_generate_prefixes`` would
        hand every concurrent document its own full allowance and the
        allowances would add — ``max_concurrent_prefixes`` would bound
        the invocation, when what it describes is the deployment.

        Loop-keyed and lazy for the same reason as :meth:`_index_lock`:
        an ``asyncio.Semaphore`` binds to the loop that first awaits it,
        so one built in ``__init__`` breaks across ``asyncio.run()``
        boundaries.
        """
        limit = self.max_concurrent_prefixes
        if not limit or limit <= 0:
            return nullcontext()
        loop = asyncio.get_running_loop()
        if self._gate_loop is not loop:
            self._gate = asyncio.Semaphore(limit)
            self._gate_loop = loop
        return self._gate

    def _replace_guard(
        self, action: str,
    ) -> tuple[asyncio.Lock, asyncio.Event, asyncio.AbstractEventLoop]:
        """The lock and idle flag separating :meth:`index` from the adds.

        *action* names the caller for :meth:`_handover_would_strand`'s
        complaint — "add_document" or "index" — so the refusal names the
        call the reader made rather than the guard they have never heard
        of.

        Required rather than defaulted to ``"index"``, which is what it
        was. Both callers pass it, so the default was unreachable, and the
        only thing an unreachable default can do is let a third caller
        added later say "cannot index on this event loop" about something
        that is not an ``index()`` — mislabelling the call in the one
        message a reader has to act on, which is the single thing this
        parameter exists to prevent. A required argument makes that a
        choice rather than an omission.

        The running loop is handed back with them rather than resolved
        again by the caller, because it is the **key** the bookkeeping is
        kept under: a registration `_adding` takes has to land in the same
        loop's entry as the primitives it was handed, and returning the
        loop that decided them is how that is guaranteed rather than
        assumed.

        ``_index_lock`` is deliberately too narrow to do this job. It is
        held across each mutation and released across the LLM calls, so
        an ``index()`` holding only that could ``_reset()`` in the window
        between a document's claim and its commit — and the commit then
        lands that document's chunks and summary in the index that
        replaced it, present in neither ``documents`` nor ``source`` but
        returned by ``retrieve`` and contradicting ``content_hash``. With
        a swept-up document in the same call it is worse: the second
        ``_chunk_document`` reads ``self.documents[doc_id]`` after the
        reset wiped it, and ``add_document`` raises a bare ``KeyError``.

        So ``index()`` takes the librarian exclusively: it waits out the
        adds already in flight and holds later ones off until it is done,
        which is what the whole-document lock did before it was narrowed.
        Adds still do not wait on each *other* — that is the parallelism
        this exists to protect, not to undo.

        Loop-keyed and lazy for the same reason as :meth:`_index_lock`.
        """
        loop = asyncio.get_running_loop()
        if self._replace_loop is not loop:
            # Asked before anything below adopts the new loop, so it is
            # the *previous* one this reports on.
            previous_live, _ = self._observe_replace_guard()
            # And refused before anything below rebinds a thing, if that
            # previous loop is still holding work. Not because the
            # bookkeeping below would be destroyed — it is keyed by loop
            # now, and this branch touches no entry belonging to a loop
            # that can still run — but because an `index()` on this loop
            # can only ever wait out adds registered *here*
            # (:meth:`_replacing` says why it physically cannot wait for
            # more), so a document parked over there would commit into
            # whatever index this loop has left by then. See
            # :meth:`_handover_would_strand` for what it costs and
            # :class:`LoopHandoverError` for why this is a refusal and
            # not a warning. Raising here leaves `_replace_loop` and the
            # whole of the bookkeeping untouched, so the same call
            # succeeds once the other loop is closed or drained.
            stranded = self._handover_would_strand(previous_live, action)
            if stranded is not None:
                raise LoopHandoverError(stranded)
            # The lock and the flag are rebuilt per loop rather than kept
            # per loop, because unlike the counts they *are* the loop-local
            # mechanism: both bind to the loop that first awaits them.
            #
            # The residual in that is worth naming rather than leaving to
            # be rediscovered. Re-adopting a loop that still holds
            # registrations of its own would hand this loop a fresh Event
            # while the add parked over there will set the old one, and the
            # drain would then wait for something nobody can signal. The
            # refusal above is what keeps that out of reach — a live
            # previous loop holding a count is exactly what it declines
            # over — so narrowing the refusal means keying these two by
            # loop as well, not just noting that the counts already are.
            self._replace_lock = asyncio.Lock()
            self._adds_idle = asyncio.Event()
            # Set because *this* loop holds no registrations yet, which is
            # a fact about its own (absent) entry rather than something
            # made true by zeroing another loop's count. That zeroing is
            # what used to live here, together with the matching
            # `_replace_owner = None`, and between them they produced both
            # deferred defects: a resuming parked add gave back a
            # registration already taken away and drove the count to -1,
            # which this drain reads as busy against a flag nobody clears
            # again; and the same zeroing destroyed the very reading
            # `save`/`load` refuse on, admitting a `load()` that left a
            # live document's chunks orphaned in the index that replaced
            # it. Both are gone at the root: the count and the owner are
            # keyed by loop (see `__init__`), so there is no other loop's
            # state here to reset. What is left to do is forget the loops
            # that cannot run again, which is what those resets were
            # reaching for and overshot.
            self._adds_idle.set()
            self._forget_closed_loops()
            # And the third piece of that bookkeeping, on the same
            # condition as the line above it and now for the same reason:
            # drop what is dead, keep what can still run. It is not keyed
            # by loop and does not need to be — a claim is keyed by doc_id,
            # and the loop it was taken on is irrelevant to what it says,
            # which is "some call is already chunking this document". What
            # it shares with the counts is who releases it: a claim is
            # given back in `_add_document`'s `finally`, which a task
            # destroyed pending at loop teardown never runs, so a loop
            # *closing* under a parked add leaves its doc_id claimed
            # forever. Left in place that wedges a *document* rather than a
            # call: the pending sweep skips anything already claimed
            # (`_add_document`), so that document sits in `documents` and
            # `source` with no summary and no chunks, silently, until some
            # `_reset()` happens to run — and a caller who only ever adds
            # never triggers one. Reproduced: park an add, close the loop
            # without awaiting it, add again on a fresh loop, and the
            # parked document is never chunked (evidence:
            # `TestAClaimStrandedByAClosedLoop`).
            #
            # Only for a loop that is *dead*, and through the same liveness
            # reading as everything else here so the two cannot drift. What
            # clearing it for a *live* loop costs is measured and is not a
            # lost refusal: a claim is the one thing stopping two sweeps
            # chunking the same document twice, and an add parked on a loop
            # that is open rather than closed is suspended, not dead — the
            # next turn of that loop resumes it and commits
            # (`_observe_replace_guard`). Clearing its claim here lets this
            # loop's sweep chunk that document again and the parked add
            # then commit its own chunks on top. Reproduced with nothing
            # but `add_document`, single-threaded: one document ends up
            # with 96 chunks instead of 48, no exception raised (evidence:
            # `TestAClaimHeldByAnAbandonedOpenLoop`, whose two tests reach
            # it only with `without_the_handover_refusal` installed, and
            # `mutation_registry.py --only ignore-claim` for the refusal
            # half that now stands in front of it).
            #
            # The condition is a backstop rather than the thing doing the
            # work: a claim held by a *live* loop is one of the three
            # holdings the refusal above declines over, so reaching this
            # line with `previous_live` true means `_chunking` is already
            # empty and clearing it would be a no-op either way. Kept
            # because the two are separate statements of the same fact
            # and this one is local — a later change to what the refusal
            # covers must not silently re-enable the 96-chunk doubling —
            # and because it costs one already-computed bool.
            if not previous_live:
                self._chunking.clear()
            self._replace_loop = loop
        return self._replace_lock, self._adds_idle, loop

    def _forget_closed_loops(self) -> None:
        """Drop bookkeeping keyed by loops whose tasks can never run again.

        What :meth:`_replace_guard`'s three resets were reaching for and
        overshot. "Drop what is dead" is the whole of what a handover
        needs; "assume nothing else exists" is what those resets did
        instead, and it cost a count driven below zero and a destroyed
        refusal (see there). A closed loop's registration is stranded by
        definition — the `finally` that would have given it back died with
        the loop — so leaving it would make every later ``index()`` wait
        out an add that ended with its loop, and every later
        ``save``/``load`` refuse over it.

        **This is also what keeps the two dicts from growing.** One
        ``asyncio.run()`` per request over one librarian is a realistic
        shape, and an entry per loop that ever indexed would be a leak for
        the life of the process. Two mechanisms bound them together, each
        catching what the other cannot:

        * A balanced entry removes *itself* — :meth:`_adding` drops the key
          when its count returns to zero and :meth:`_replacing` pops the
          owner in its ``finally`` — so the tidy case leaves nothing behind
          at all, without waiting for a handover that may never come.
        * This, for the entries that are *not* balanced, which are exactly
          the ones a loop closing under a parked task strands. It runs from
          the new-loop branch and nowhere else, because crossing loops is
          the only way to accumulate an entry, and that crossing always
          goes through there.

        So the bookkeeping is O(loops still open), not O(loops ever used).

        **Not a `weakref.WeakKeyDictionary`**, though event loops are
        weak-referenceable and it would bound the dicts on its own. It
        would tie the disappearance of a refusal to garbage collection: the
        state ``save``/``load`` refuse over would clear at a moment nothing
        in the library chose, and a stranded count would be forgotten
        because nothing happened to reference its loop rather than because
        that loop had closed. ``is_closed()`` is a definite answer to the
        question actually being asked, and it is the one every other reader
        of this bookkeeping already uses.
        """
        # A snapshot because this **pops from the dict it is iterating**,
        # which raises "dictionary changed size during iteration" on the
        # next step. Nothing about threads and nothing about timing: the
        # copy is required single-threaded, with this the only thing
        # running. Measured — remove both `list()`s and five existing
        # tests fail on the first stranded entry they produce, led by
        # `TestALoopThatEndedUnderPendingWork`; one at a time it is four
        # on the count prune and one on the owner prune.
        #
        # The two observers snapshot for a *different* reason — they do not
        # mutate, and what they are avoiding is a resize by another
        # thread's loop mid-walk (:meth:`_adds_in_flight_here`). Same call,
        # two arguments; this one is the mandatory half.
        for loop in list(self._adds_in_flight):
            if not _loop_is_live(loop):
                self._adds_in_flight.pop(loop, None)
        for loop in list(self._replace_owner):
            if not _loop_is_live(loop):
                self._replace_owner.pop(loop, None)

    def _adds_registered_on(
        self, loop: asyncio.AbstractEventLoop | None = None,
    ) -> int:
        """In-flight ``add_document`` registrations taken on *loop*.

        The one place ``_adds_in_flight``'s shape is known, so that a later
        change to it is one edit. *loop* defaults to the one the guard is
        keyed to, because that is the loop the two async callers mean: it is
        the only loop an :meth:`_adding` can be registering on, and it is
        "the previous loop" a handover asks about
        (:meth:`_handover_would_strand`). The synchronous observers want
        the librarian-wide total instead, and walk every entry to get it
        (:meth:`_adds_in_flight_here`).

        Never negative, and that is structural rather than checked:
        :meth:`_adding` decrements the entry it incremented, so a
        decrement cannot arrive at a count that was raised somewhere else
        and taken away in between. Zero for a loop with no entry, which is
        how a loop that has finished its adds reads (the entry is dropped
        rather than left at zero — see :meth:`_forget_closed_loops`).
        """
        if loop is None:
            loop = self._replace_loop
        return self._adds_in_flight.get(loop, 0)

    def _replace_owner_on(
        self, loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Task | None:
        """The task inside :meth:`_replacing` on *loop*, if any.

        :meth:`_adds_registered_on`'s sibling, keyed and defaulted the same
        way and for the same reasons. Keyed by loop because the owner is
        compared against ``asyncio.current_task()``, which only means
        anything within one loop: an owner recorded by another loop is a
        stranger to this caller by construction, and used to be *erased* by
        the handover instead — dropping a parked ``index()``'s hold and
        letting a later ``save``/``load`` past a replace that had not
        finished.
        """
        if loop is None:
            loop = self._replace_loop
        return self._replace_owner.get(loop)

    def _handover_would_strand(
        self, previous_live: bool, action: str,
    ) -> str | None:
        """The complaint for taking the librarian onto a new loop, or None.

        Composed here rather than raised inline for
        :meth:`_busy_with_adds`'s reason — the sentence is the whole
        value of the refusal, and it wants room to say three things: what
        is holding the librarian, what taking it anyway would cost, and
        which of the two exits the caller has.

        The three holdings are the three pieces of bookkeeping the other
        loop can still be holding, and every one of them means a task over
        there can still run: a registered ``add_document``
        (``_adds_in_flight``), a document claimed for chunking
        (``_chunking``), an ``index()`` holding the exclusive hold
        (``_replace_owner``). They are reported together rather than
        first-one-wins, because a parked add usually holds two of them at
        once — its LLM phase holds the count and a claim — and naming only
        the count would have the reader looking for one thing when two are
        stranded.

        **What is being refused changed shape, and the refusal did not.**
        These three used to be *reset* by :meth:`_replace_guard`, and this
        was what stopped that reset landing on a loop that could still run.
        Two of them are keyed by loop now, so the reset is gone and with it
        the -1 and the destroyed reading. What survives is the reason
        keying them was enough: this loop's ``index()`` can wait out the
        adds registered *here* and no others (:meth:`_replacing`), so a
        document parked on that loop would commit into whatever index is
        here by then whatever the bookkeeping says. The refusal is what
        makes "wait only for your own loop" a rule rather than a hole, and
        that is a permanent job, not scaffolding for the fields below it.

        ``previous_live`` is passed in rather than read again so this
        cannot answer about a different observation than the one the
        caller acted on; :meth:`_observe_replace_guard` is advisory and
        re-reading it is a second sample, not a confirmation.
        """
        # A closed loop's bookkeeping is stranded by definition — nothing
        # it counted can run again — which is precisely the case
        # `_forget_closed_loops` exists for.
        if not previous_live:
            return None

        holdings = []
        # The previous loop's own entry and its own recorded owner, because
        # the previous loop is what `previous_live` describes and what the
        # sentence below names.
        #
        # What makes that enough is **unreachability, not `_chunking`.** For
        # some *third* loop to be holding a registration while the guard
        # sits on the previous one, a handover away from a live holder must
        # already have been admitted: single-threaded this refusal declines
        # exactly that, and the only other way in is the cross-thread
        # advisory window :meth:`_observe_replace_guard` names as unfixed.
        # So there is nothing for a wider walk to find here.
        #
        # `_chunking` is a partial net inside that window and not the
        # guarantee an earlier version of this comment claimed. Being keyed
        # by doc_id rather than by loop, it does catch an add parked in its
        # **LLM phase**, which is where a parked add spends nearly all of
        # its life — but that is a fact about where adds park rather than an
        # invariant, and it is not even every parking point the suite
        # reaches. `_adding` registers before `_add_document` claims
        # anything, so an add suspended on `_index_lock` between the two
        # holds the count with nothing claimed, and the
        # under-``chunk_threshold`` path takes no claim for the whole of its
        # life. `TestWhatEachHoldingIsWorth` reaches the first of those
        # through the public API, which is why the count check below is
        # load-bearing rather than redundant-today.
        #
        # Walking every live loop was considered and declined. It would
        # narrow that window without closing it — the read is still one
        # sample, so a registration taken a moment later is missed either
        # way — while answering about a different observation than the
        # ``previous_live`` the caller acted on, in a sentence that names
        # the previous loop. What closes the window is a lock this module
        # deliberately does not have.
        in_flight = self._adds_registered_on()
        if in_flight:
            holdings.append(
                f"{in_flight} add_document call(s) registered there"
            )
        if self._chunking:
            holdings.append(
                f"document(s) claimed for chunking there: "
                f"{', '.join(sorted(self._chunking))}"
            )
        if self._replace_owner_on() is not None:
            holdings.append("an index() holding it exclusively there")
        if not holdings:
            # Live but idle, which is an ordinary and safe handover: a
            # loop that has finished its indexing but has not been closed
            # yet holds nothing for the reset to strand. This is the case
            # the refusal must stay quiet for, and it is checked here
            # rather than at the call site so the two cannot drift.
            return None

        return (
            f"cannot {action} on this event loop: this librarian is still "
            f"held by indexing on the event loop it was last used on, which "
            f"is open and can still resume — {'; '.join(holdings)}. "
            f"Indexing here cannot wait for indexing there — an index() "
            f"waits out the adds registered on its own loop, which is all "
            f"an asyncio.Event can signal — so the parked call would commit "
            f"into whatever index is here by then: measured, chunks left "
            f"behind for a document the index no longer names, still "
            f"returned by retrieve() and still written to disk. Await the "
            f"work parked on that loop, or close it if it is finished "
            f"(`asyncio.run` always closes one; a hand-rolled loop wants a "
            f"`close()`), then retry — nothing here has changed, so the same "
            f"call succeeds once it is."
        )

    @asynccontextmanager
    async def _adding(self) -> AsyncIterator[None]:
        """Register one in-flight ``add_document``, waiting out a replace.

        Raises :class:`LoopHandoverError` before registering anything if
        the loop this librarian last indexed on is still open and still
        holding work — see :meth:`_replace_guard`.
        """
        lock, idle, loop = self._replace_guard("add_document")
        async with lock:
            # Registered against the loop the guard just resolved, not
            # against the librarian at large. `loop` is closed over, so the
            # `finally` gives the registration back to the same entry it
            # took it from however long it is parked and whatever else has
            # happened to the librarian in between — which is the whole of
            # why the count can no longer go negative.
            self._adds_in_flight[loop] = self._adds_registered_on(loop) + 1
            idle.clear()
        try:
            yield
        finally:
            # Deliberately await-free: a cancelled add still has to give
            # its registration back, and an await here could be cancelled
            # in turn, leaving the count high and every later index()
            # waiting on adds that finished long ago.
            remaining = self._adds_registered_on(loop) - 1
            if remaining:
                self._adds_in_flight[loop] = remaining
            else:
                # Dropped rather than left at zero. An entry per loop that
                # ever indexed is a leak in the `asyncio.run()`-per-request
                # shape, and a balanced loop removing its own entry is what
                # bounds the dict without waiting for a handover that may
                # never come — `_forget_closed_loops` covers the entries
                # nothing gives back, and only those.
                self._adds_in_flight.pop(loop, None)
                idle.set()

    @asynccontextmanager
    async def _replacing(self) -> AsyncIterator[None]:
        """Take the librarian exclusively, for :meth:`index` to replace it.

        Raises :class:`LoopHandoverError` before taking anything if the
        loop this librarian last indexed on is still open and still
        holding work — see :meth:`_replace_guard`.
        """
        lock, idle, loop = self._replace_guard("index")
        async with lock:
            # Recorded from the acquire, not from the drain: the moment
            # the lock is taken this call is committed to `_reset()`, and
            # that is what the synchronous half of the guard has to see.
            # Under this loop's key, because the only question ever asked
            # of it is `owner is not current_task()`, which compares two
            # tasks of one loop (:meth:`_replacing_elsewhere`).
            self._replace_owner[loop] = asyncio.current_task()
            try:
                # Holding the lock is what stops new adds registering, so
                # by the time the flag is set the count can only fall.
                #
                # **This waits for the adds registered on this loop and no
                # others**, and the reason is `idle` itself: an
                # asyncio.Event binds to the loop that first awaits it, so
                # an add finishing on another loop cannot wake this task —
                # not "does not", *cannot*.
                #
                # Waiting more widely is writable — poll
                # `_adds_in_flight_here()` behind an `await
                # asyncio.sleep(0)` and no Event is needed. It is *wrong*,
                # which is the distinction worth keeping straight one line
                # after insisting on "cannot". A registration on another
                # loop is not something this loop can outlast: if that loop
                # was abandoned open, its parked add never decrements and
                # the poll never ends, so a refusal turns into a permanent
                # hang; and if that loop is genuinely running, waiting it
                # out is waiting on cross-thread mutation this module does
                # not support (:meth:`_observe_replace_guard`). Adds on
                # another live loop are `LoopHandoverError`'s job, which is
                # why that refusal is permanent rather than a stopgap.
                while self._adds_registered_on(loop):
                    await idle.wait()
                yield
            finally:
                # Await-free for `_adding`'s reason: a cancelled replace
                # still has to give the librarian back, and an await here
                # could be cancelled in turn — leaving a finished index()
                # recorded as the owner and every later load() refusing
                # over it. Popped rather than set to None for
                # `_adds_in_flight`'s reason: an entry per loop that ever
                # ran an index() would never be collected.
                self._replace_owner.pop(loop, None)

    def _observe_replace_guard(
        self, loop: asyncio.AbstractEventLoop | None = None,
    ) -> tuple[bool, bool]:
        """``(live, ours)`` — can bookkeeping kept by *loop* still commit?

        *loop* defaults to the one the guard is keyed to, which is what
        every caller meant while there was only one to ask about. The
        bookkeeping is keyed by loop now, so the observers ask this of
        every entry they find (see :meth:`_adds_in_flight_here`) and the
        answer is per loop rather than per librarian — the same question,
        asked of each key, in one place so the readings cannot drift.

        The one question both synchronous observers have to answer, and
        the one they used to get wrong by asking a different one. They
        compared loop **identity**, which is two tests wearing one coat:
        it ignores a count stranded by a finished ``asyncio.run()``
        (right — that work cannot still be running, and refusing over it
        would wedge every later ``save``/``load`` forever) *and* it
        ignores a loop that is alive on another thread (wrong — that work
        **is** running, and the refusal is exactly what should fire).
        Reproduced through the public API: a ``load()`` from
        ``asyncio.to_thread`` mid-add was admitted and left 24 chunks of a
        document ``documents`` no longer held, ranked first by
        ``retrieve`` and written to a file ``_load_if_current`` calls
        current (evidence:
        ``TestWhatTheLivenessCheckIsWorth::test_the_identity_test_admits_the_cross_thread_load``,
        which restores the identity test and asserts that damage).
        Identity cannot separate the two cases; **liveness**
        can, and the loop object already reports it.

        * ``live`` is false only for a loop that was never recorded or one
          that is **closed**. That state is stranded by definition:
          no task of a closed loop can run again, so nothing it counted
          can still commit here.
        * ``ours`` says that loop is the one running on this
          thread — the only case in which ``asyncio.current_task()`` says
          anything about this caller. It raises ``RuntimeError`` outside
          a running loop and answers about the *wrong* loop's task inside
          a different one, so :meth:`_replacing_elsewhere` may only ask
          it when this is true.

        **The in-between state — open but not running — reads as live**,
        and that is a decision rather than a fallout. A loop between
        ``run_until_complete`` calls is not executing, but its parked
        ``add_document`` is suspended, not dead: the next turn of that
        loop resumes it and commits into whatever index is here by then.
        ``is_running()`` is therefore deliberately *not* the test — it
        would report False for exactly that task and admit the reset that
        strands it, which is the silent failure this guard exists to
        convert into a loud one. The cost of erring this way is a
        narrower and louder one: abandoning an open loop with an add
        parked on it makes every later ``save``/``load`` refuse until the
        loop is closed (``asyncio.run`` always closes; a hand-rolled loop
        wants a ``close()``), and the refusal says which call to await.

        **These reads are advisory and stay advisory.** A dict lookup and
        an object reference under the GIL: nothing is synchronised, nothing
        is ordered, and a count can change between this call and the
        caller's next statement. That is enough to *refuse* — the state
        it reports was true a moment ago, which is all a complaint needs
        — and nowhere near enough to proceed on, which is why no caller
        takes it as permission to mutate. It does not make concurrent
        mutation from threads safe or supported; it stops the guard being
        blind to it.

        **What keying the bookkeeping by loop did and did not do to that.**
        It removed the way this reading could be *destroyed*: the count a
        ``save``/``load`` refuses on was zeroed by the next handover, so a
        live add became invisible and the refusal was lost. Nothing resets
        another loop's entry now, and the observers read every entry rather
        than one, so a live add is visible wherever it was registered.

        It did **not** make the read-then-act window go away for a caller
        on another *thread*. :meth:`_replace_guard` acts on
        ``previous_live`` — a negative answer from an advisory read — so a
        guard on thread B that samples a moment before thread A's
        :meth:`_adding` increments still adopts the new loop. What that
        costs is now bounded: A's registration survives the handover and
        every later ``save``/``load`` sees it, so the outcome is a refusal
        rather than the stranded chunks it used to be. Closing it properly
        means a lock around the guard's reads and writes, and this module
        deliberately does not have one (see ``retrieve()``'s "atomic
        against the event loop, though not against threads", which is
        unchanged). Single-threaded the window cannot open at all — a
        parked add on loop A cannot be incrementing while loop B runs on
        the same thread.
        """
        if loop is None:
            loop = self._replace_loop
        if not _loop_is_live(loop):
            return False, False
        try:
            return True, asyncio.get_running_loop() is loop
        except RuntimeError:
            # No loop on this thread at all: the ordinary way an index is
            # loaded, and also the thread half of the reproduction above
            # (`TestWhatTheLivenessCheckIsWorth`). Which of the two it is
            # depends on that loop, not on this thread, and that is what
            # `live` already answered.
            return True, False

    def _adds_in_flight_here(self) -> int:
        """In-flight ``add_document`` calls that can still commit here, or 0.

        The synchronous half of the replace guard. ``save`` and ``load``
        cannot ``async with self._replacing()`` from a ``def``, but the
        registrations are plain data: they cannot *wait* for the adds and
        they can *see* them, which is all a refusal needs.

        Read through :meth:`_observe_replace_guard`, so "here" means the
        librarian rather than the thread: an add running on a loop that is
        still open counts whether that loop is this thread's, another
        thread's, or suspended between runs, because in every one of those
        cases its commit lands in *this* index. A count belonging to a loop
        that has since **closed** does not count — that add ended with
        the loop, and reading it raw would make every later ``load()``
        refuse forever over work that cannot happen.

        **Every live loop's registrations, not just the loop the guard
        holds**, which is the whole reason the count is keyed by loop
        rather than kept per guard. "How much indexing can still commit
        into this index" is a fact about the librarian; which loop it was
        started on is not part of the question. Asking only the guard's
        loop is what the single ``int`` forced and what made this reading
        destroyable: the handover moved the guard to a new loop and
        everything registered on the old one became invisible, so a
        ``load()`` was admitted and a parked document's 24 chunks were left
        orphaned in the index that replaced it. That is D2, and reading the
        entries rather than one entry is what makes it *impossible* here
        instead of merely refused earlier — measured with
        `_handover_would_strand` neutralised, which is the only way to tell
        those two apart (evidence:
        ``TestTheDeferredDefectsDoNotReproduce::test_d2_does_not_reproduce_with_the_refusal_removed``).

        The reading is a plain count of calls again, and
        :meth:`_busy_with_adds` names it as one: a decrement now finds the
        entry its increment went into, so nothing drives it below zero.
        """
        # A snapshot for `_forget_closed_loops`'s reason — another loop can
        # be registering while a thread reads this — and `live` per entry,
        # so a loop that closed under a parked add is forgiven exactly as
        # the guard's own closed loop always was.
        total = 0
        for loop, count in list(self._adds_in_flight.items()):
            live, _ = self._observe_replace_guard(loop)
            if live:
                total += count
        return total

    def _replacing_elsewhere(self) -> bool:
        """Is another task inside :meth:`_replacing`?

        The other half of the synchronous replace guard, and the half a
        counter cannot cover: ``_adds_in_flight`` is only raised by
        ``add_document``, so ``index()`` — which replaces the whole index
        and is the very thing ``_replace_guard`` was built for — is
        invisible to it. A ``save`` or ``load`` landing there does the
        same damage an add's does.

        **Why not ``self._replace_lock.locked()``.** ``index()`` calls
        ``_load_if_current`` and ``save(cache)`` from *inside* its own
        ``_replacing()``, so a bare lock test would refuse the one path
        that must never be refused. An ``asyncio.Lock`` carries no owner,
        so ``_replacing`` records the task and this asks whether the
        holder is somebody else. ``current_task()`` is ``None`` outside a
        task — a loop callback, say — which is not the owner either, and
        is refused like any other stranger.

        Read through :meth:`_observe_replace_guard` exactly like
        :meth:`_adds_in_flight_here`, for the same reasons in the same
        words: every loop's recorded owner is asked about, an owner
        recorded by a **closed** loop is ignored, and an owner on a live
        loop is refused whichever thread that loop belongs to. And, like
        the count, the owner is now recorded per loop, so a handover no
        longer *drops* one: an ``index()`` parked on a loop abandoned open
        keeps its hold, and a later ``save``/``load`` keeps refusing over
        the window
        ``TestLoadRefusesMidIndex::test_the_corruption_the_refusal_prevents``
        measures corrupting. That is D3, closed the same way D2 was — by
        the record surviving rather than by nothing being able to reach the
        reset.

        ``current_task()`` is asked *only* when the guard's loop is this
        thread's running one, which is the only case where its answer is
        about this caller. Measured on 3.11: with no running loop it
        raises ``RuntimeError("no running event loop")`` rather than
        returning ``None`` — it resolves the loop before the lookup
        (evidence:
        ``TestTheOwnerCheckFromAThreadWithNoLoop::test_a_thread_gets_the_refusal_and_not_asyncios_complaint``,
        which fails with asyncio's own message if the gate goes) — so
        an unconditional comparison here would turn a synchronous
        ``save``/``load`` from a plain thread into a confusing crash. And
        from a *different* running loop it would answer about that loop's
        task, which can never be an owner recorded by another loop, so
        the comparison would be true by construction: there is nothing to
        ask, and the refusal is already decided.
        """
        # Cheapest and commonest case first, by construction rather than by
        # a special case: with nothing replacing anything the dict is empty
        # and this walks nothing. Snapshotted for
        # `_forget_closed_loops`'s reason.
        for loop, owner in list(self._replace_owner.items()):
            live, ours = self._observe_replace_guard(loop)
            if not live:
                continue
            if ours and owner is asyncio.current_task():
                # `index()` inside its own hold, calling `_load_if_current`
                # or `save(cache)`. The one path that must never be
                # refused, and the reason an owner is recorded at all.
                continue
            return True
        return False

    def _busy_with_adds(self, action: str, damage: str) -> str | None:
        """The complaint for a whole-index *action* landing mid-add, or None.

        Returns the sentence rather than raising it because its two
        callers do different things with it: ``load`` has no safe way to
        proceed and raises, ``save`` has one and logs. Composing the
        message here is what keeps them saying the same thing about the
        same situation.

        The message carries the *damage* and not just the rule, because
        the caller is holding a librarian that still looks fine.

        It says "on this librarian" and no longer "on this event loop":
        since :meth:`_observe_replace_guard` the refusal also fires for
        an add on a live loop belonging to *another thread*, and naming
        the caller's own loop would send that reader looking in the one
        place the add is not.

        **The reading is a count of calls, so the sentence names one.** It
        had a second branch for a *negative* reading, which
        ``_replace_guard``'s ``_adds_in_flight = 0`` could produce by
        zeroing a count a parked add would later give back; it refused, and
        said so without quoting "-1 add_document call(s)" as work to await.
        The registrations are keyed by loop now, so a decrement lands in
        the entry its increment came from and no path reaches a negative
        reading — and a message for a state that cannot occur is a message
        nothing keeps honest, so it is gone rather than kept as a
        defence. What would bring it back is a change that lets one loop
        write another's entry; :meth:`_adds_registered_on` is where that
        would show up.
        """
        in_flight = self._adds_in_flight_here()
        if not in_flight:
            return None
        return (
            f"cannot {action} safely while {in_flight} add_document call(s) "
            f"are in flight on this librarian: {damage}. Await that indexing "
            f"before you {action} — `await asyncio.gather(*adds)` if it is on "
            f"this event loop; if it was started on another one you have to "
            f"await it there, or close that loop, because nothing on this "
            f"loop can finish it. index(..., cache=...) is unaffected by its "
            f"*own* loop's adds, which it drains before it loads — but it "
            f"reads every live loop's, so an add on another one refuses "
            f"there too."
        )

    def _busy_replacing(self, action: str, damage: str) -> str | None:
        """The complaint for a whole-index *action* landing mid-``index``.

        The sibling of :meth:`_busy_with_adds`, kept separate because the
        two are not the same situation and ``save`` does not answer them
        the same way — an add leaves a file the next add repairs, a
        replace leaves one ``_load_if_current`` calls current. See
        :meth:`save`.
        """
        if not self._replacing_elsewhere():
            return None
        return (
            f"cannot {action} safely while index() is replacing this "
            f"librarian from another task: {damage}. Await that index() "
            f"before you {action}; if it is running on another event loop "
            f"you have to await it there, or close that loop, because "
            f"nothing on this loop can finish it. index(..., cache=...) is "
            f"unaffected by its *own* hold, which it loads and saves from "
            f"inside — but a replace on another live loop refuses there too."
        )

    async def _gated_llm_call(self, system: str, user: str) -> str:
        """An index-time LLM call, held to ``max_concurrent_prefixes``.

        Every call issued while indexing goes through here — the
        per-document summary as well as the per-chunk prefixes. They are
        the same burst hitting the same backend, and a summary that
        skipped the gate meant one extra request per concurrent document
        on top of the cap.
        """
        async with self._prefix_gate():
            return await self._llm_call(system, user)

    async def index(
        self,
        text: str,
        doc_id: str = DEFAULT_DOC_ID,
        cache: str | Path | None = None,
    ) -> None:
        """Index a document, **replacing** anything indexed before.

        Use :meth:`add_document` to build a multi-document index instead.

        Takes the librarian exclusively for the whole replacement (see
        :meth:`_replace_guard`): it waits out the adds already in flight
        and holds later ones off, so it serializes against other
        indexing — but not against queries, which take no lock. The
        index lock itself is held only across the mutations, not across
        the LLM calls. The documents being replaced are gone from
        the moment this starts, so a question asked before it returns is
        answered from the incoming document alone and still unchunked —
        which, for a document big enough to have needed chunking, can
        overflow ``max_context_tokens`` and raise from :meth:`answer`
        rather than answer badly. See :meth:`add_document` on what
        concurrent use covers.

        Args:
            text: the document
            doc_id: provenance label carried on every chunk and fact
            cache: directory to persist the index in. A cache whose
                content hash matches *text* is loaded instead of
                re-indexing; otherwise the fresh index is written there.
        """
        # Exclusive for the whole run, adds included: `_reset()` landing
        # between another document's claim and its commit leaves chunks
        # and a summary behind for a doc_id this index no longer holds.
        # See :meth:`_replace_guard`.
        async with self._replacing():
            async with self._index_lock():
                if self.documents:
                    logger.warning(
                        "  LIBRARIAN index() is replacing %d already-indexed "
                        "document(s): %s — use add_document() to add instead",
                        len(self.documents), ", ".join(self.documents),
                    )
                self._reset()

                if cache is not None and self._load_if_current(cache, text):
                    self._relabel_single_document(doc_id)
                    return

            # The index lock is released across this: `_add_document`
            # takes it for each of its own mutations and it is not
            # reentrant. Exclusion against the adds is the replace
            # guard's job, not this lock's.
            await self._add_document(text, doc_id=doc_id)

            if cache is not None:
                async with self._index_lock():
                    self.save(cache)

    async def add_document(self, text: str, doc_id: str | None = None) -> None:
        """Add a document to the index, keeping what is already there.

        Safe to call concurrently on one librarian — indexing a corpus
        with ``asyncio.gather`` is the natural way to hand a library many
        documents, and the documents genuinely overlap: only the index
        mutations are serialized, not the LLM calls. The cap bounds every
        index-time call the librarian has in flight *across* documents,
        so it still describes the deployment rather than the invocation.

        Concurrent with :meth:`index` it is held off rather than lost:
        the two never overlap, and an add that arrives during a replace
        adds to the index that replaced the old one.

        That safety is indexing against indexing — the adds against each
        other, and every one of them against a replace. :meth:`save` and
        :meth:`load` are the two whole-index operations that cannot join
        that scheme, being synchronous: they observe it instead. Rather
        than corrupt the index ``load`` raises :class:`IndexBusyError`
        while this is in flight, and ``save`` warns that the file it
        wrote is partial (raising under ``strict``); against a concurrent
        :meth:`index` rather than an add, both refuse outright. It does
        not extend to querying during an index: :meth:`retrieve` is
        synchronous and takes no lock, so a question asked while this is
        in flight reads the index as it stands part-built. Retrieval
        never blocks and never raises, but it answers from what is
        there:

        * Ranked retrieval covers the documents swept in so far and not
          the ones still to come — a document contributes no chunks
          until every one of its prefixes is written.
        * A sweep of two or more pending documents extends ``chunks``
          before it rebuilds the rankings, and across that await neither
          ranking covers them (see :meth:`_bm25_ranking`), so retrieval
          falls back to whole documents. That fallback ignores ``top_k``
          and returns every document in the index whole — including the
          one currently being indexed.
        * Whole documents are readily larger than
          ``max_context_tokens``, so :meth:`answer` can then raise
          ValueError from :meth:`check_context` on a question a finished
          index would have answered. The remedies that error suggests do
          not apply here; awaiting the indexing is the fix.
        """
        async with self._adding():
            await self._add_document(text, doc_id=doc_id)

    async def _add_document(self, text: str, doc_id: str | None = None) -> None:
        """Body of :meth:`add_document`. Callers must NOT hold the index lock.

        Three phases, and the LLM work is the one in the middle that
        holds no lock: claim the doc_id and the documents this call will
        chunk, generate summaries and prefixes, then commit the built
        index. Every mutation happens inside one of the two locked
        phases; nothing awaits while the lock is held.
        """
        async with self._index_lock():
            doc_id = doc_id or f"doc{len(self.documents) + 1}"
            if doc_id in self.documents:
                raise ValueError(
                    f"doc_id {doc_id!r} is already indexed — pass a distinct "
                    f"doc_id"
                )

            self.documents[doc_id] = text
            self.source = "\n\n".join(self.documents.values())

            if len(self.source) <= self.chunk_threshold and not self._chunking:
                # Small enough to hand to the model whole — no index-time
                # LLM calls.
                #
                # The summaries go with the chunks. `source` can shrink —
                # the rollback below removes a document — so this branch is
                # reachable with summaries already recorded, and a summary
                # left behind without chunks is invisible to the pending
                # sweep, which finds work by looking for the *absence* of a
                # summary. It would never be re-chunked.
                #
                # Nothing is lost that `documents` cannot re-derive, but it
                # is not free: dropping the summaries means the next
                # over-threshold add re-chunks every surviving document, at
                # one summary call plus one prefix call per chunk for each.
                #
                # Skipped while another call is mid-chunk: dropping the
                # chunks under a document that is about to commit more of
                # them would leave the numbering with a hole. A corpus that
                # is under the threshold only because a concurrent rollback
                # shrank it is chunked instead, which costs calls but is
                # never wrong.
                self.use_chunking = False
                self.chunks = []
                self.doc_summaries = {}
                # Goes with the summaries for the same reason they go with
                # the chunks: this branch has no document loop to hang
                # extraction on, so a field record surviving here would
                # describe documents nothing re-derives it for.
                self.doc_fields = {}
                self.facts = []
                self.fact_bm25 = None
                self.fact_embeddings = None
                self.doc_summary = ""
                self.bm25 = None
                self.chunk_embeddings = None
                self._vector_fingerprint = None  # it described those vectors
                logger.info(
                    "  LIBRARIAN indexed %d chars (full context)", len(self.source)
                )
                return

            # Anything not chunked yet gets chunked now — including earlier
            # documents that were under the threshold on their own. Claiming
            # them is what keeps concurrent adds from doing it twice: they
            # read the same `documents`, so without the claim every call in
            # a gather would sweep up every un-summarized document.
            pending = [d for d in self.documents
                       if d not in self.doc_summaries
                       and d not in self._chunking]
            self._chunking.update(pending)

        try:
            for pending_id in pending:
                await self._chunk_document(pending_id)

            async with self._index_lock():
                self.use_chunking = True
                self._build_bm25()
                self._refresh_embeddings()
                self._build_fact_index()
                self._load_reranker()
        except BaseException:
            async with self._index_lock():
                self._roll_back(doc_id)
            raise
        finally:
            # After the rollback, not before: a document is only free for
            # another call to sweep up once this one has finished undoing
            # whatever it managed to write.
            self._chunking.difference_update(pending)

    def _roll_back(self, doc_id: str) -> None:
        """Undo a failed :meth:`_add_document`. Callers hold the index lock."""
        # A document that failed to index must not be left behind: it
        # would still count towards `source` and `content_hash()`, and
        # so persist into a saved index, while contributing no chunks.
        # Earlier documents that were pending stay in `documents` on
        # purpose — without a summary they are still pending, and the
        # next add_document() re-chunks them.
        #
        # With documents overlapping this is quieter than it used to be.
        # Serially, the call that swept up a pending document was the
        # only call running, so its failure was the batch's failure and
        # the caller saw it. Concurrently, the sweeper can fail while its
        # siblings all return successfully, leaving that swept-up
        # document in `documents` and `source` with no summary and no
        # chunks and nobody raising about it. It is still within the
        # contract above — the next add_document() picks it up, and until
        # then it is served by the whole-document path — but a corpus
        # indexed with `gather` can come back with a document unchunked
        # and no exception to say so.
        self.documents.pop(doc_id, None)
        self.doc_summaries.pop(doc_id, None)
        # Committed in the same locked phase as the summary, so it comes
        # back out here with it.
        self.doc_fields.pop(doc_id, None)
        # `_chunk_document` writes `doc_summaries[doc_id]` and
        # `doc_summary` together, so popping only the former leaves
        # `doc_summary` naming a document that is no longer indexed —
        # and `save` writes it verbatim into index.json, from where
        # `load` restores it. Fall back to the newest summary that
        # survived rather than blanking it.
        self.doc_summary = next(reversed(self.doc_summaries.values()), "")
        self.source = "\n\n".join(self.documents.values())
        # This document's own chunks go too, or the failure that lands
        # after it chunked (a strict embedding error, say) leaves chunks
        # pointing at a doc_id no longer in the index. They are no longer
        # a suffix — a document that started later can have committed its
        # chunks first — so the survivors are renumbered rather than left
        # with a hole where these sat.
        # Their embedding rows go with them, selected by the same mask:
        # `_refresh_embeddings` only encodes `chunks[len(chunk_embeddings):]`
        # and so pairs vectors with chunks by position, which means rows a
        # rollback leaves behind are silently inherited by whichever chunk
        # slides into their place — and once the counts coincide again the
        # length guard in `_embedding_ranking` no longer fires.
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        if len(keep) != len(self.chunks):
            if self.chunk_embeddings is not None:
                # Only the kept chunks that actually have a vector: the
                # embeddings are a prefix of `chunks`, and dropping the
                # rest keeps them one.
                rows = [i for i in keep if i < len(self.chunk_embeddings)]
                self.chunk_embeddings = (
                    self.chunk_embeddings[rows] if rows else None
                )
            # Built from `keep` and *before* the renumbering below, which
            # is the only moment the old-to-new mapping still exists: once
            # `chunk.index` has been reassigned the chunks say nothing
            # about where they used to be.
            moved = {old: new for new, old in enumerate(keep)}
            self.chunks = [self.chunks[i] for i in keep]
            for position, chunk in enumerate(self.chunks):
                chunk.index = position
            # The rolled-back document's facts go with its chunks. Left
            # behind they would be retrievable entries citing a `doc_id`
            # the index no longer holds — the orphan shape
            # `_replace_guard` exists to prevent, reached here by an
            # ordinary indexing failure rather than a loop handover. The
            # survivors are repointed for the same reason the chunks are
            # renumbered: a stale `chunk` would aim at whatever slid into
            # its place.
            if self.facts:
                kept_facts = []
                for fact in self.facts:
                    if fact.get("doc_id") == doc_id:
                        continue
                    if isinstance(fact.get("chunk"), int):
                        fact["chunk"] = moved.get(fact["chunk"], fact["chunk"])
                    kept_facts.append(fact)
                self.facts = kept_facts
                self._build_fact_index()
        # Unconditionally, not only when this document had chunks: the
        # pending sweep can have appended an *earlier* document's chunks
        # before the failing one contributed any, and a bm25 index built
        # over fewer entries than `chunks` makes `retrieve` index past
        # the end of `bm25_scores`.
        self._build_bm25()
        if not self.chunks:
            # `use_chunking` is set before the index is built, so a
            # failure in any of those steps can leave it True over an
            # empty index — a state nothing else produces. It contradicts
            # `retrieval_mode` ("unindexed") and passes the router's
            # "is there anything to query" gate, which then answers a
            # question from empty source material.
            self.use_chunking = False

    async def _chunk_document(self, doc_id: str) -> None:
        """Chunk one document, summarize it, and write its context prefixes.

        Holds no lock across its LLM calls — that is the whole point of
        the split. The chunks it builds are private to this call until
        the commit at the end, which is the only part that mutates the
        index and the only part that takes the lock.
        """
        text = self.documents[doc_id]
        spans = chunk_spans(text, self.chunk_size, self.chunk_overlap)
        new_chunks = [
            Chunk(text=body, doc_id=doc_id, offset=offset)
            for body, offset in spans
        ]
        logger.info(
            "  LIBRARIAN chunked %s: %d chars -> %d chunks",
            doc_id, len(text), len(new_chunks),
        )

        sample = text[:4000]
        if len(text) > 5000:
            sample += "\n...\n" + text[-1000:]
        summary = await self._gated_llm_call(DOC_SUMMARY_SYSTEM, sample)
        logger.info("  LIBRARIAN doc summary (%s): %s", doc_id, summary[:120])

        # The same `sample` the summary reads, not the whole document: a
        # document states what it is and who issued it in its header, and
        # a policy's identity is not hiding on page 40. The tail is in
        # there too for anything that signs off at the end.
        fields = await self._extract_doc_fields(doc_id, sample)

        # Publish nothing until every prefix is written. A summary recorded
        # before the prefix calls would survive a failure among them and the
        # document would be left in `documents` with a summary and zero
        # chunks — invisible to the pending sweep in `_add_document`, which
        # finds work by looking for documents *without* a summary, so it
        # would never be re-chunked. Summary and chunks go in together.
        await self._generate_prefixes(new_chunks, summary)

        # After the prefixes, and outside the lock like them: this is
        # one call per chunk and holding `_index_lock` across it would
        # serialise every concurrent document behind this one.
        chunk_facts = await self._extract_chunk_facts(new_chunks)

        async with self._index_lock():
            # `index` is assigned here rather than at construction, above:
            # a `start = len(self.chunks)` read before those awaits is a
            # promise about where the chunks will land, made before any
            # other document has had a chance to land theirs. Two
            # overlapping documents both read the same `start` and both
            # numbered from it. Read it where it is used instead.
            for position, chunk in enumerate(new_chunks, start=len(self.chunks)):
                chunk.index = position
            self.doc_summaries[doc_id] = summary
            self.doc_summary = summary
            # Only when there is something to record. An empty dict here
            # would read downstream as "extracted, found nothing", which
            # is a different claim from "not extracted" and the one thing
            # a filter must not confuse — see `doc_fields` in __init__.
            if fields:
                self.doc_fields[doc_id] = fields
            self.chunks.extend(new_chunks)
            # Stamped here rather than in the extractor: `chunk.index`
            # is assigned in this same critical section, so a fact
            # numbered before it would point at whatever position the
            # chunk had been *predicted* to take.
            for chunk, facts in zip(new_chunks, chunk_facts):
                for fact in facts:
                    fact["doc_id"] = chunk.doc_id
                    fact["chunk"] = chunk.index
                self.facts.extend(facts)

    async def _extract_chunk_facts(
        self, chunks: list[Chunk]
    ) -> list[list[dict[str, str]]]:
        """Facts for each chunk, in order. One gated LLM call per chunk.

        **Never fails a chunk.** A reply that will not parse, one that
        yields nothing, and a call that raises all leave the chunk indexed
        and retrievable exactly as it was; `strict` is what turns any of
        them into an error. That is `_extract_doc_fields`'s rule and it is
        the right way round here too — a fact store is an optimisation for
        finding a chunk's contents, and trading the chunk for it would be
        backwards.

        The call goes through `_gated_llm_call`, so fact extraction shares
        `max_concurrent_prefixes` with the prefix and summary calls rather
        than opening a second uncapped fan-out. That matters more here
        than anywhere else in this file: this is one call **per chunk**,
        the largest index-time cost any knob in this class adds.
        """
        if not self.extract_facts or not chunks:
            return [[] for _ in chunks]

        async def one(chunk: Chunk) -> list[dict[str, str]]:
            try:
                reply = await self._gated_llm_call(FACT_SYSTEM, chunk.text)
            except Exception as error:                       # noqa: BLE001
                if self.strict:
                    raise
                logger.warning(
                    "  LIBRARIAN fact extraction failed (%s): %s",
                    chunk.doc_id, error)
                return []
            facts = parse_facts(reply)
            if not facts:
                logger.info(
                    "  LIBRARIAN fact extraction yielded nothing (%s)",
                    chunk.doc_id)
            return facts

        return list(await asyncio.gather(*(one(c) for c in chunks)))

    async def _extract_doc_fields(
        self, doc_id: str, sample: str
    ) -> dict[str, dict]:
        """Pull *doc_id*'s fields out of *sample*, with proof. Never raises.

        ``{}`` when extraction is off, when the reply yields nothing that
        survives verification, or when the call itself fails.
        **Extraction never fails a document**: a field record is an
        optimisation for filtering, and the document is retrievable by
        ranking without one exactly as it was before this feature
        existed. Losing an index to a malformed JSON reply would trade the
        whole document for a filter.

        The loop is ask → :func:`verify_fields` → *if anything failed,
        hand the model its own failures and ask again*, bounded by
        :data:`DOC_FIELDS_REASK`. What goes back is the specific defect —
        "your quote does not appear in the document" — never "are you
        sure". A model asked to re-judge its own answer agrees with
        itself; a model shown a quote that is not in the document has
        something to fix.

        The re-ask is kept only if it is **better**, counted as fields
        that survived verification. A second reply that verifies worse
        than the first is discarded, because nothing guarantees a retry
        improves and silently taking the newer answer would make this
        loop capable of losing information.

        ``strict`` raises, as on every other degradable path here.

        Gated through :meth:`_gated_llm_call` like the summary and the
        prefixes, so the retry queues on the librarian's cap rather than
        opening a fresh allowance per document.
        """
        if not self.extract_fields:
            return {}

        best: dict[str, dict] = {}
        last_reply = ""
        system = DOC_FIELDS_SYSTEM
        for attempt in range(DOC_FIELDS_REASK + 1):
            try:
                reply = await self._gated_llm_call(system, sample)
            except Exception:
                # Not BaseException: a CancelledError here is the caller
                # abandoning this document, and swallowing it would keep
                # indexing something nobody is waiting for.
                # `_generate_prefixes` draws the same line for the same
                # reason.
                if self.strict:
                    raise
                logger.warning(
                    "  LIBRARIAN field extraction failed for %s — indexing "
                    "it without fields", doc_id, exc_info=True,
                )
                return best
            last_reply = reply
            kept, complaints = verify_fields(parse_fields(reply), sample)
            if len(kept) > len(best):
                best = kept
            if not complaints or attempt == DOC_FIELDS_REASK:
                if complaints:
                    logger.info(
                        "  LIBRARIAN %s: %d field(s) still unverified after "
                        "re-ask: %s", doc_id, len(complaints),
                        "; ".join(c[:90] for c in complaints),
                    )
                break
            logger.info(
                "  LIBRARIAN %s: re-asking with %d failed check(s)",
                doc_id, len(complaints),
            )
            system = (
                DOC_FIELDS_SYSTEM
                + "\n\nYour previous answer had these problems. Fix exactly "
                "these and return the whole object again:\n- "
                + "\n- ".join(complaints)
            )

        if not best:
            # The whole reply, at WARNING, for `expand_query`'s reason: a
            # per-key record would say a field was missing without ever
            # showing what the model actually wrote, and this is the only
            # place that can.
            logger.warning(
                "  LIBRARIAN no verified fields for %s from reply: %r",
                doc_id, last_reply[:300],
            )
            if self.strict:
                raise ValueError(
                    f"field extraction for {doc_id!r} yielded no verified "
                    f"fields — reply was {last_reply[:200]!r}"
                )
        else:
            logger.info(
                "  LIBRARIAN fields (%s): %s", doc_id,
                {k: v["value"] for k, v in best.items()},
            )
        return best

    async def _generate_prefixes(
        self, chunks: list[Chunk], summary: str
    ) -> None:
        """Write a contextual prefix for each chunk, in parallel, in place.

        Calls run concurrently but no more than ``max_concurrent_prefixes``
        are in flight at once — the librarian's gate, shared with every
        other document being indexed and with the summary calls, not a
        fresh allowance per fan-out. Without the cap this fans out one
        request per chunk: fine for the ~30 chunks of a 63K document,
        roughly a thousand simultaneous requests for a 2MB one, which is
        not a graceful failure on any backend. The gate holds only the
        request, so building the context strings still overlaps freely.
        """
        limit = self.max_concurrent_prefixes

        async def _prefix_one(idx: int, chunk: Chunk) -> None:
            prev_tail = ""
            next_head = ""
            if idx > 0:
                lines = chunks[idx - 1].text.strip().split("\n")
                prev_tail = " ".join(lines[-2:])
            if idx < len(chunks) - 1:
                lines = chunks[idx + 1].text.strip().split("\n")
                next_head = " ".join(lines[:2])

            context = f"DOCUMENT SUMMARY: {summary}\n\n"
            if prev_tail:
                context += f"PREVIOUS CHUNK (ending): {prev_tail}\n\n"
            context += f"THIS CHUNK:\n{chunk.text}\n\n"
            if next_head:
                context += f"NEXT CHUNK (beginning): {next_head}"

            prefix = await self._gated_llm_call(CONTEXT_PREFIX_SYSTEM, context)
            chunk.prefix = prefix.strip()

        # Cancel the siblings when one call fails. `gather` re-raises the
        # first exception but leaves the rest running: they keep issuing
        # calls for a document the caller has already abandoned, and they
        # keep taking slots on the librarian's gate to do it — slots the
        # documents still being indexed are queueing for. Now that the gate
        # is shared rather than per-fan-out, abandoned work does not merely
        # add a second allowance, it steals from the only one there is.
        # Cancelling and draining before re-raising keeps it from outliving
        # its document.
        tasks = [
            asyncio.ensure_future(_prefix_one(i, chunk))
            for i, chunk in enumerate(chunks)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # BaseException, not Exception: a cancellation of *this*
            # coroutine has to take the children down with it too.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        logger.info(
            "  LIBRARIAN generated %d contextual prefixes (max %s in flight)",
            len(chunks), limit if limit and limit > 0 else "unbounded",
        )

    def _build_fact_index(self) -> None:
        """(Re)build the lexical and dense indexes over the fact store.

        Separate from `_build_bm25` and `_refresh_embeddings` rather than
        folded into them, because facts and chunks are two populations
        ranked independently and then mixed: one BM25 over both would let
        a chunk's length statistics set the norm for a 154-character fact,
        which is the thing the fact store exists to avoid.
        """
        if not self.facts:
            self.fact_bm25 = None
            self.fact_embeddings = None
            return
        texts = [fact_text(f) for f in self.facts]
        self.fact_bm25 = BM25Okapi([tokenize(t) for t in texts])
        logger.info("  LIBRARIAN fact index built (%d facts)", len(self.facts))
        if not self._dense_ok:
            self.fact_embeddings = None
            return
        model = self._ensure_embed_model()
        if model is None:
            self.fact_embeddings = None
            return
        try:
            # `embed_prompts` returns a (document, query) *tuple*, and
            # the document half is what an indexed fact takes — the
            # same call `_refresh_embeddings` makes for a chunk.
            document_prompt = embed_prompts(self.embed_model_name)[0]
            # `show_progress_bar=False` and nothing else, matching
            # `_refresh_embeddings` exactly: the encoder contract this
            # library relies on is the one its doubles implement, and
            # asking for `normalize_embeddings` here made every stub
            # raise `TypeError` — caught, logged, and degraded to
            # BM25-only, so the dense half was dead while every test
            # still passed.
            self.fact_embeddings = np.asarray(model.encode(
                [f"{document_prompt}{t}" for t in texts],
                show_progress_bar=False))
        except Exception as error:                           # noqa: BLE001
            # The chunk half is untouched: a fact store that cannot be
            # embedded still ranks lexically, and `_dense_ok` describes
            # the *model*, which is demonstrably fine here.
            logger.warning("  LIBRARIAN could not embed the fact store: %s",
                           error)
            self.fact_embeddings = None

    def _fact_ranking(self, question: str, want: int) -> list[int]:
        """The best *want* facts for a question, as indexes into `facts`.

        Fuses the fact store's own lexical and dense rankings the way
        `retrieve` fuses the chunk store's, then returns the head. No
        `await` anywhere: the facts were extracted at index time, so
        retrieving over them is the same synchronous work retrieving over
        chunks is, and `retrieve()`'s contract is unchanged.
        """
        if not self.facts or want <= 0:
            return []
        rankings: list[list[int]] = []
        if self.fact_embeddings is not None:
            try:
                model = self._ensure_embed_model()
                if model is not None:
                    query_prompt = embed_prompts(self.embed_model_name)[1]
                    vector = np.asarray(model.encode(
                        [f"{query_prompt}{question}"],
                        show_progress_bar=False))[0]
                    # Cosine computed here rather than assumed from
                    # normalised vectors, for `_embedding_ranking`'s
                    # reason: nothing promises the encoder normalises,
                    # and a zero-norm row would divide by zero.
                    norms = (np.linalg.norm(self.fact_embeddings, axis=1)
                             * np.linalg.norm(vector))
                    norms = np.where(norms == 0, 1.0, norms)
                    scores = np.dot(self.fact_embeddings, vector) / norms
                    rankings.append(list(np.argsort(scores)[::-1]))
            except Exception as error:                       # noqa: BLE001
                logger.warning(
                    "  LIBRARIAN fact embedding query failed: %s", error)
        if self.fact_bm25 is not None:
            scores = self.fact_bm25.get_scores(tokenize(question))
            # Truncated to what BM25 actually scored, for the reason
            # `_bm25_ranking(scored=True)` is: a query it cannot score
            # returns every fact in index order, and RRF cannot tell that
            # filler from a ranking.
            ranked = [i for i in np.argsort(scores)[::-1] if scores[i] > 0]
            if ranked:
                rankings.append(ranked)
        if not rankings:
            return []
        order = (reciprocal_rank_fusion(rankings, k=self.rrf_k)
                 if len(rankings) > 1 else rankings[0])
        return [int(i) for i in order[:want]]

    def _build_bm25(self) -> None:
        """(Re)build the lexical index over every prefixed chunk."""
        if not self.chunks:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi([tokenize(c.prefixed) for c in self.chunks])
        logger.info("  LIBRARIAN BM25 index built (%d entries)", len(self.chunks))

    # ── Model loading ────────────────────────────────────────────

    def _ensure_embed_model(self):
        """Load the embedding model on demand. None if unavailable.

        An `ImportError` means the ``embeddings`` extra is not installed
        — the documented BM25-only path, logged quietly. Anything else
        (CUDA OOM, a half-downloaded model, a trust_remote_code refusal,
        a sentence-transformers version mismatch) is a real failure and
        is logged as an error, because silently running on half the
        retrieval stack for weeks is worse than a loud stop.

        Every path that hands back a model runs
        :meth:`_check_embed_fingerprint` first, so no caller can use one
        against vectors it did not produce.
        """
        if self.embed_model is not None or not self._dense_ok:
            # On the cached path too, not only after a fresh load: the
            # stored vectors can arrive *after* the model does — `load()`
            # on a librarian that has already indexed something — and
            # assigning `embed_model` directly is a supported way to
            # supply one. Past the first call this is one dot product
            # over one probe vector.
            self._check_embed_fingerprint()
            return self.embed_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.warning(
                "  LIBRARIAN embeddings extra not installed (%s) — "
                "using BM25-only retrieval", e,
            )
            self._dense_ok = False
            return None
        try:
            logger.info(
                "  LIBRARIAN loading embedding model: %s", self.embed_model_name
            )
            # No dtype and no `model_kwargs`: the weights load at whatever
            # the checkpoint says, which is what keeps
            # EMBED_FINGERPRINT_TOLERANCE comfortable. Adding a dtype knob
            # here is not free — the worst reduced-precision pairing measured,
            # cpu/bfloat16 vs mps/float16, is 8.155e-05, only 1.23x inside
            # that tolerance, so re-measure with
            # benchmarks/fingerprint_tolerance.py before one lands.
            self.embed_model = SentenceTransformer(
                self.embed_model_name,
                # Only for models that ship remote code. bge-m3 is a plain
                # XLM-RoBERTa and needs none; nomic does. Passing it
                # unconditionally asks every user to trust arbitrary code
                # from whatever repo `embed_model` names.
                trust_remote_code=self.embed_model_name.startswith("nomic-ai/"),
                # None for anything we did not pin, which is also
                # SentenceTransformer's own default — a user-supplied
                # model loads exactly as it did before.
                revision=pinned_revision(self.embed_model_name),
            )
        except Exception as e:
            self._dense_ok = False
            logger.error(
                "  LIBRARIAN embedding model %s is installed but failed to "
                "load (%s: %s) — dense retrieval is NOT active, retrieval "
                "quality will be below the documented benchmark",
                self.embed_model_name, type(e).__name__, e,
            )
            if self.strict:
                raise
            return None
        # Gated on the model that *has* remote code, not on the model that
        # happens to be the default. EMBED_CODE_REPO and
        # EMBED_CODE_REVISION describe nomic's `nomic-bert-2048`; bge-m3 is
        # a plain XLM-RoBERTa with no remote modeling file at all, so
        # comparing it against them checks nothing and would report drift
        # from a repo it never loads. This was `== EMBED_MODEL` and became
        # wrong the moment the default moved off nomic — the same
        # closed-assumption failure CLAUDE.md's Conventions section
        # describes, where a condition true when written silently stops
        # describing what it meant.
        if self.embed_model_name.startswith("nomic-ai/"):
            _check_code_revision(
                self.embed_model,
                EMBED_CODE_REVISION,
                model_name=self.embed_model_name,
                code_repo=EMBED_CODE_REPO,
            )
        self._check_embed_fingerprint()
        return self.embed_model

    def _current_fingerprint(self) -> dict | None:
        """This librarian's loaded model's fingerprint, probed once.

        ``None`` when no model is loaded — which is what keeps this off
        the cache-check path: nothing here loads a model to get an
        answer, so a librarian that never needed embeddings never pays
        the probe and `load()` can ask the question without deciding it.
        """
        model = self.embed_model
        if model is None:
            return None
        # Keyed on the object, not a bool: tests and callers assign
        # `embed_model` directly, and a flag would carry the first
        # model's fingerprint onto the second one's vectors. The
        # reference is strong rather than `id()` or a weakref — `id()` is
        # reused after a collection, which is precisely how a stale
        # fingerprint would be handed to a *different* model. The cost is
        # that replacing `embed_model` keeps the old object alive until
        # something probes again, and nothing schedules that probe: only
        # two call sites reach this line, and both are conditional —
        # `_check_embed_fingerprint` returns above it whenever there are
        # no vectors or no recorded fingerprint, and `_refresh_embeddings`
        # reaches it only when it re-embeds a whole array. On a BM25-only
        # or unrecorded index the replaced model can therefore stay
        # reachable for the life of the librarian. Accepted rather than
        # fixed: swapping a loaded model out mid-life is a test and
        # power-user move, one extra model is a bounded cost, and the
        # alternatives buy that memory back at the price of a wrong
        # answer.
        if self._fingerprint_of is not model:
            self._fingerprint_of = model
            self._model_fingerprint = embed_fingerprint(model)
        return self._model_fingerprint

    def _model_declined_probe(self) -> bool:
        """True when the loaded model has been probed and produced nothing.

        The whole point is that it **does not probe**. A missing
        fingerprint has two remedies that are opposites — re-index, or
        stop expecting the check to work at all — and the only thing that
        can tell them apart is whether this librarian's own model already
        failed to produce one. Asking by probing here would put an encode
        on `load()`, which is exactly the cost `_current_fingerprint`'s
        "None when no model is loaded" was written to avoid; so this
        reports what is already known and says "no" when nothing is.

        It does not distinguish *why* the probe came back empty (an
        encode that raised versus a degenerate vector), because the
        consequence for the index is identical: re-indexing will not
        record a fingerprint. The distinction is drawn where it is
        actionable — at the probe, where `embed_fingerprint` warns for
        the degenerate cases and stays silent for the raising one.
        """
        model = self.embed_model
        # `_fingerprint_of` starts None, so the identity test has to be
        # guarded: without it a librarian with no model at all would read
        # as "probed, and it declined".
        return (
            model is not None
            and self._fingerprint_of is model
            and self._model_fingerprint is None
        )

    def _check_embed_fingerprint(self) -> bool:
        """Discard stored vectors the loaded weights did not produce.

        Returns True when it discarded them, so `load()` can tell that
        this has already said what its own no-vectors warning would.

        **Why discard rather than warn and serve.** The two precedents
        pull apart, and the difference is what is stored.
        ``_check_code_revision`` only warns because nothing the remote
        code produced is on disk — the code that ran is unverified, not
        unusable. Here the stored artefact *is* the problem: a cosine
        between two embedding spaces is not a weak signal, it is a
        meaningless number that still sorts, so serving on wrong-space
        vectors is worse than serving without them — it looks like dense
        retrieval and ranks like noise. ``load()``'s chunk-count mismatch
        and its saved-``embed_model``-name mismatch both already discard
        for exactly that reason, and the name mismatch is this same
        failure caught by a cheaper signal; answering it two different
        ways would be the inconsistency.

        Unlike the name mismatch it leaves ``_dense_ok`` alone. That flag
        means "the model is unusable", and here it is not — only the
        vectors are. Leaving it set is what lets the remedy work in the
        process that was told about it: the next ``add_document`` sees no
        vectors and re-embeds under the loaded weights. Nothing on the
        *load* path re-embeds — that would silently turn a load into an
        arbitrarily long encode of the whole corpus.
        """
        if self.chunk_embeddings is None or self._vector_fingerprint is None:
            return False  # nothing stored, or nothing recorded to compare
        drift = fingerprint_drift(self._vector_fingerprint,
                                  self._current_fingerprint())
        if drift is None or drift <= EMBED_FINGERPRINT_TOLERANCE:
            return False
        self.chunk_embeddings = None
        self._vector_fingerprint = None
        message = (
            "the stored vectors were not produced by the embedding weights "
            "now loaded for %s — the fingerprint probe differs by %.3g, "
            "past the %g that separates float noise from different weights. "
            "Either upstream moved (this model's weights are fetched with no "
            "revision, so that needs no change here) or this process embeds "
            "on a different device or dtype. Cosine similarity across two "
            "embedding spaces is meaningless, so the vectors are discarded: "
            "retrieval is BM25-only until you re-index"
        )
        logger.warning("  LIBRARIAN " + message,
                       self.embed_model_name, drift,
                       EMBED_FINGERPRINT_TOLERANCE)
        if self.strict:
            raise StaleEmbeddingsError(
                (message % (self.embed_model_name, drift,
                            EMBED_FINGERPRINT_TOLERANCE))
                + " (strict=True)"
            )
        return True

    def _report_reranker_failure(self, level: int, message: str,
                                 *args: object) -> None:
        """Say a reranker load failed — once per distinct failure.

        The load is *retried* on every ``add_document`` (see
        :meth:`_load_reranker`), so without this a 500-document corpus on
        a core install writes 500 identical records. That is the same
        argument :func:`_report_unfingerprintable` makes one screen away:
        a report that repeats on every document is how a real one gets
        filtered out, and this branch was the inconsistency.

        **What "once" is keyed on: the report itself, per librarian.**
        The fingerprint report keys on the model object because it has
        one; here the ImportError path has no object at all — nothing
        loaded — so the only thing with a stable identity is the record
        that would be written. Keying on ``(level, template, args)``
        makes the rule exactly "the same sentence is not written twice in
        a row", which is the property being asked for, and it keeps a
        genuine change of circumstance reportable for free: the extra
        appearing and the load then failing for its own reason, a
        different exception type, a ``rerank_model`` that changed — each
        is a different record and each is said.

        Per librarian rather than per process, matching the fingerprint
        report's granularity: the repetition being paid for is one
        librarian indexing a corpus, and a process-wide key would silence
        a *second* librarian configured differently, whose failure is
        news. Cleared by a load that succeeds (in `_load_reranker`), so a
        failure after a recovery is reported again rather than being
        mistaken for the old one.

        The args are stringified into the key rather than held: the
        interesting one is an exception, which does not compare by value
        and whose ``__traceback__`` would pin every frame of the failed
        load for the life of the librarian.
        """
        # An exception enters the key by *type*, not by its text, because
        # the text is not stable across retries. The failures that reach
        # here from a real deployment come from the hub, and
        # huggingface_hub stamps a per-request token into every HTTP
        # error it raises — `"(Request ID: Root=1-...)"`, a fresh value
        # per attempt — which transformers then copies verbatim into the
        # OSError it re-raises ("There was a specific connection error
        # when trying to load ...: {e}"). Keyed on that text, a
        # rate-limited or briefly unreachable hub gives a 500-document
        # corpus 500 distinct keys and 500 records: exactly the
        # repetition this deduplication exists to remove, restored
        # silently. Normalising the text instead was rejected — the
        # request ID is only the volatile part we happen to have caught,
        # and a regex chasing the rest (temp paths, `PermissionError`'s
        # filename, timings) would be a permanent maintenance debt over a
        # message format nobody here owns.
        #
        # The cost is real and accepted: two failures of the same type
        # with different text now collapse into one report, and
        # transformers funnels a lot of distinct causes into `OSError`.
        # The second cause going unsaid is much the smaller loss — the
        # librarian has already said once that reranking is off, it
        # retries the load on every add regardless, and a flood is what
        # makes the whole log unreadable. The ERROR branch below passes
        # `type(e).__name__` as its own argument, so the distinction the
        # key keeps is also the one a reader sees; the ImportError branch
        # has one type by construction.
        report = (level, message, tuple(
            type(a).__name__ if isinstance(a, BaseException) else str(a)
            for a in args
        ))
        if report == self._reranker_report:
            return
        self._reranker_report = report
        logger.log(level, message, *args)

    def _load_reranker(self) -> None:
        """Load the cross-encoder reranker on demand, if enabled.

        Both failure paths leave ``self.reranker`` at ``None``, which is
        what makes the guard below fall through again on the next
        ``add_document``: the retry is deliberate, so a transient failure
        (a disk that filled, a hub that was down) recovers without
        rebuilding the librarian. Only the *reporting* is deduplicated —
        see :meth:`_report_reranker_failure`.
        """
        if not self.use_reranker or self.reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            self._report_reranker_failure(
                logging.WARNING,
                "  LIBRARIAN embeddings extra not installed (%s) — "
                "reranking disabled", e,
            )
            return
        try:
            logger.info("  LIBRARIAN loading reranker: %s", self.rerank_model)
            # No trust_remote_code: bge-reranker-v2-m3 is a plain
            # XLMRobertaForSequenceClassification with no auto_map, so
            # `revision` pins every file it loads — the embedding
            # model's cross-repo code problem has no counterpart here.
            self.reranker = CrossEncoder(
                self.rerank_model, revision=pinned_revision(self.rerank_model)
            )
        except Exception as e:
            self._report_reranker_failure(
                logging.ERROR,
                "  LIBRARIAN reranker %s is installed but failed to load "
                "(%s: %s) — reranking disabled",
                self.rerank_model, type(e).__name__, e,
            )
            # Unchanged by the deduplication: `strict` raises on every
            # failure, not only the one that got logged. The report is
            # about noise in a log nobody can filter; the raise is the
            # caller's own stop, and swallowing a later one because an
            # earlier one was already mentioned would be exactly the
            # silent degradation `strict` exists to prevent.
            if self.strict:
                raise
            self.reranker = None
        else:
            # A load that worked ends the previous report's life, so a
            # failure after a recovery is news again rather than being
            # compared against a record from before the model was ever
            # working.
            self._reranker_report = None

    def _refresh_embeddings(self) -> None:
        """Embed any chunk that does not have a vector yet."""
        # Bail before any encode work when dense retrieval is off.
        # _ensure_embed_model() cannot be relied on for this: it returns the
        # cached model whenever one is loaded, so a model that loaded fine and
        # was later disabled (e.g. a query-time embed failure setting
        # _dense_ok = False) still comes back non-None.  Without this guard
        # every add_document() pays the full encode cost to build vectors that
        # _embedding_ranking then refuses to use.
        if not self._dense_ok:
            self.chunk_embeddings = None
            # With the array, not after it: the fingerprint describes the
            # vectors being dropped here. This was the one place that
            # nulled `chunk_embeddings` and left the identity behind, and
            # the state is reachable through the public API alone — a
            # query-time embed failure sets `_dense_ok = False`, and the
            # next `add_document()` lands here still holding a
            # fingerprint recorded for vectors that no longer exist.
            self._vector_fingerprint = None
            return

        have = 0 if self.chunk_embeddings is None else len(self.chunk_embeddings)
        if have >= len(self.chunks):
            return

        model = self._ensure_embed_model()
        if model is None:
            self.chunk_embeddings = None
            self._vector_fingerprint = None
            return

        # Re-read after the load, never before it: `_ensure_embed_model`
        # is where the fingerprint check runs, and a drift verdict throws
        # the stored vectors away. Carrying the earlier count across it
        # would embed only `chunks[have:]` into an array that now starts
        # empty, so row *i* would hold chunk *have + i*'s vector: every
        # row off by `have`, and `len(chunks) - have` of them for
        # `len(chunks)` chunks. `_embedding_ranking`'s length guard
        # catches that count and drops the dense half, so the immediate
        # cost is retrieval silently degrading to BM25-only. The reason
        # to fix it here rather than lean on that guard is that the
        # misalignment outlives it: the next `add_document()` appends
        # onto the short array and can bring the count back into
        # agreement while every row is still shifted, at which point the
        # guard passes and wrong vectors are scored as right ones.
        have = 0 if self.chunk_embeddings is None else len(self.chunk_embeddings)
        pending = self.chunks[have:]
        try:
            # `c.prefixed`, not `c.text`: the same text `_build_bm25`
            # indexes, so the two halves of the fusion work from the same
            # material and the per-chunk LLM call that wrote the prefix
            # pays for both. No extra call — the prefix is already on the
            # chunk. `EMBED_SOURCE` is the constant that puts this choice
            # in `content_hash` and carries the numbers;
            # benchmarks/retrieval_arms.py is what produced them.
            vectors = np.asarray(model.encode(
                [f"{embed_prompts(self.embed_model_name)[0]}{c.prefixed}"
                 for c in pending],
                show_progress_bar=False,
            ))
        except Exception as e:
            self._dense_ok = False
            self.chunk_embeddings = None
            self._vector_fingerprint = None
            logger.error(
                "  LIBRARIAN failed to embed chunks (%s: %s) — dense "
                "retrieval is NOT active", type(e).__name__, e,
            )
            if self.strict:
                raise
            return

        if have == 0:
            # Every vector in the array came from this model, so this
            # librarian can vouch for the whole of it. Only then: with
            # `have > 0` the value already there covers the earlier rows,
            # and it is either a fingerprint that just matched (still
            # true of the whole array) or None because those rows arrived
            # with no recorded identity — which stays unknown, because
            # what this model produced says nothing about what produced
            # them.
            self._vector_fingerprint = self._current_fingerprint()
        self.chunk_embeddings = (
            vectors if self.chunk_embeddings is None
            else np.vstack([self.chunk_embeddings, vectors])
        )
        logger.info(
            "  LIBRARIAN embedded %d chunks (%d dims)",
            len(self.chunks), self.chunk_embeddings.shape[1],
        )

    # ── Persistence ──────────────────────────────────────────────

    def content_hash(self) -> str:
        """Hash of the indexed text plus the settings that shaped the index."""
        return self._content_hash(self.documents)

    def _content_hash(self, documents: dict[str, str]) -> str:
        """This librarian's :func:`content_hash` over arbitrary documents.

        Exists so the cache check can hash a *saved* set of documents
        under *this* librarian's settings, which is the comparison that
        decides whether the cache is reusable.
        """
        return content_hash(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunk_threshold=self.chunk_threshold,
            embed_model=self.embed_model_name,
            extract_fields=self.extract_fields,
        )

    def save(self, path: str | Path) -> None:
        """Write the index to *path* so it survives a process restart.

        Layout: ``index.json`` (documents, chunks, prefixes, summaries)
        and ``embeddings.npy``. The BM25 index is rebuilt from the
        prefixed chunks on load rather than pickled — cheap, and it
        avoids carrying a third-party object across versions.

        Called while another task is indexing this librarian it refuses
        or warns, and **which one depends on what that task is doing**.
        The two cases were measured separately; the answer is not the
        same for both (evidence: ``TestSaveMidAdd``,
        ``TestSaveRefusesMidIndex``).

        **An ``add_document`` in flight: a WARNING and a partial file** —
        :class:`IndexBusyError` instead under ``strict``. A document
        registers in ``documents``, and so in ``source`` and in
        ``content_hash``, one locked phase before its chunks are
        committed, so a save landing in between claims to be the index
        for a document it holds no chunks for. Measured on a
        two-document librarian: the file records both documents and four
        chunks, all of them the first document's, and a question about
        the second is answered from the first's material (evidence:
        ``TestSaveMidAdd::test_it_warns_and_still_writes_the_partial_file``).
        Worse for a
        single document, where the file records the document and *no*
        chunks: ``_load_if_current`` reads its content hash as current
        and serves it, so a document big enough to need chunking is
        served whole from that cache for as long as the cache lives.

        Warned rather than refused, unlike :meth:`load`, because the
        damage differs. ``load``'s race corrupts *live state*
        irrecoverably — chunks under a doc_id the index no longer holds,
        which nothing can re-derive away. This one writes a *file*, and
        a partial one is a state the library already tolerates and can
        recover from: the un-chunked document carries no summary either,
        so it stays pending and the next ``add_document`` chunks it, and
        ``TestSaveMidAdd::test_the_partial_file_recovers_on_the_next_add``
        pins that recovery rather than asserting it. Refusing would also
        take away the only reason to
        call this during a long index — a checkpoint — in favour of no
        file at all, which is not obviously better than a file that
        announces what it is missing. ``strict`` is what chooses between
        them, which is exactly the choice that flag exists to make.

        **An :meth:`index` in flight from another task: refused, with or
        without ``strict``.** The recovery argument above is the add
        case's and does not survive the move: ``index()`` has already
        reset, so the file records its one in-flight document and
        *nothing else* — no other document's chunks to make it visibly
        partial — and there is no next ``add_document`` coming to chunk
        it. Measured: ``documents:['fresh'] chunks:0 summaries:[]``,
        which ``_load_if_current`` accepts as **current** and serves
        whole with ``use_chunking`` false, for as long as that cache
        lives (evidence:
        ``TestSaveRefusesMidIndex::test_the_file_it_refuses_to_write_reloads_as_if_complete``,
        which writes the file the refusal declines and then loads it).
        That is a file which lies rather than a file that
        announces what it is missing, so there is no degraded outcome
        for ``strict`` to select — the same reason :meth:`load` does not
        consult it either. The checkpoint argument goes too: a
        replacement in progress is not a state worth checkpointing, and
        ``index(..., cache=...)`` already writes the finished one.

        Both at once — an add still draining under an ``index()`` that is
        waiting for it — is the replace case, and refusing wins. The
        add's repair depends on the document staying pending in *this*
        librarian until the next add reaches it, and the ``index()``
        holding the guard is about to discard it along with everything
        else.

        ``index(..., cache=...)`` saves from inside ``_replacing()``, as
        the *owner* of the hold and with this loop's adds already drained,
        so neither complaint fires — **single-threaded**, and that
        qualification is :meth:`load`'s: the drain guarantees zero on its
        own loop only, while :meth:`_busy_with_adds` reads every live
        loop's registrations. In the same cross-thread window
        (:meth:`_observe_replace_guard`) this ``save`` can warn from inside
        ``index(..., cache=...)``, and raise there under ``strict``. The
        direction is right even on the path this paragraph calls
        unaffected — there genuinely is an add on another loop that can
        still commit into this librarian, and the warning now scopes its
        "unaffected" to that path's own loop instead of claiming it whole —
        but the sentence it warns with describes the mid-add file, where the
        damage in that window is the in-memory one :meth:`load` documents.
        """
        # Checked first, and refused whatever ``strict`` says: this is the
        # case with no degraded outcome to offer. See the docstring.
        complaint = self._busy_replacing(
            "save",
            "the file would record the document index() is building with "
            "none of its chunks — an index _load_if_current calls current "
            "and serves whole, which no later add repairs",
        )
        if complaint:
            raise IndexBusyError(complaint)

        complaint = self._busy_with_adds(
            "save",
            "the index will be written with a document registered but not "
            "yet chunked — a partial index that loads without complaint",
        )
        if complaint:
            if self.strict:
                raise IndexBusyError(complaint)
            # WARNING and not INFO for this module's usual reason: the
            # library installs no handler, so an INFO sits under the
            # last-resort threshold and would report a partial index in a
            # way indistinguishable from not reporting it — and this file
            # outlives the process that wrote it.
            logger.warning("  LIBRARIAN %s", complaint)

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": INDEX_FORMAT_VERSION,
            "content_hash": self.content_hash(),
            "use_chunking": self.use_chunking,
            "embed_model": (
                self.embed_model_name
                if self.chunk_embeddings is not None else ""
            ),
            # What text those vectors are of, gated on the vectors exactly
            # as `embed_model` is and for the same reason: it describes
            # them, so with none there is nothing to describe.
            #
            # `content_hash` also folds `EMBED_SOURCE` in, and that is a
            # different guarantee — it decides whether a *cache* is
            # current, which only `_load_if_current` asks. A direct
            # `load()` consults no hash, so without this key a pre-change
            # index installed text-space vectors at `mode=hybrid` in
            # silence: the fingerprint passes because the weights really
            # are the same, which is the hole the fingerprint check is
            # structurally unable to see. Reproduced against an index
            # really built by the pre-change code — `git archive cbc62e5`
            # into a scratch tree, index there, load here — and it
            # compounds, because `_refresh_embeddings` only appends: the
            # next `add_document` left rows 0-1 in text space and row 2 in
            # prefixed space in one array, scored by one cosine, never
            # repaired. That sha is the citation because it is what makes
            # the reproduction available to somebody who is not its
            # author; re-running it needs the encoder, so no test can.
            #
            # Optional by construction like `embed_fingerprint`, so no
            # format bump: a file without it predates the change, which
            # `load()` reads as `"text"` rather than as "unknown".
            "embed_source": (
                EMBED_SOURCE if self.chunk_embeddings is not None else ""
            ),
            # Optional by construction — null when there are no vectors,
            # and null when the model behind them could not be probed —
            # so an older reader ignores it and this one reads its
            # absence as "not recorded". That is why the format version
            # is not bumped for it.
            #
            # Gated on the vectors the same way `embed_model` is, and
            # kept even though every path that nulls `chunk_embeddings`
            # now nulls the fingerprint with it: this is the boundary
            # where a wrong pairing stops being process state and
            # becomes a file. With no vectors `save()` *deletes*
            # `embeddings.npy` below, so an ungated write would leave
            # `index.json` claiming an identity for an array that is not
            # there — a claim no later `load()` can see to correct,
            # because it reads `embed_fingerprint` only after the
            # vectors file has loaded.
            "embed_fingerprint": (
                self._vector_fingerprint
                if self.chunk_embeddings is not None else None
            ),
            "documents": self.documents,
            "doc_summaries": self.doc_summaries,
            # Both optional by construction, so no INDEX_FORMAT_VERSION
            # bump — the same bargain `embed_fingerprint` and
            # `embed_source` struck, and for the same price: a bump costs
            # every index on disk a full re-index, one LLM call per chunk.
            # An older reader ignores these two keys; this one reads their
            # absence as "no fields recorded", which is what every index
            # written before this feature genuinely is.
            #
            # The version is gated on there being fields for the reason
            # `embed_model` is gated on there being vectors: it describes
            # them, so with none there is nothing to describe — and a
            # version written beside an empty record would make an index
            # that never extracted indistinguishable from one whose
            # extraction found nothing.
            "doc_fields": self.doc_fields,
            "doc_fields_version": (
                DOC_FIELDS_VERSION if self.doc_fields else ""
            ),
            # Optional by construction, exactly as `doc_fields` is, so
            # `INDEX_FORMAT_VERSION` is **not** bumped: an index written
            # before facts existed loads silently, and a bump would cost
            # every index on disk a full re-index for a knob most
            # librarians never turn on.
            "facts": self.facts,
            "facts_version": FACTS_VERSION if self.facts else "",
            "doc_summary": self.doc_summary,
            "chunks": [c.model_dump() for c in self.chunks],
        }
        (directory / "index.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

        vectors = directory / "embeddings.npy"
        if self.chunk_embeddings is not None:
            np.save(vectors, self.chunk_embeddings)
        elif vectors.exists():
            vectors.unlink()

        logger.info(
            "  LIBRARIAN saved index to %s (%d chunks, embeddings=%s)",
            directory, len(self.chunks), self.chunk_embeddings is not None,
        )

    def load(self, path: str | Path) -> None:
        """Load an index written by :meth:`save`.

        Raises FileNotFoundError if there is no index at *path*,
        ValueError if it was written by an incompatible version, and —
        under ``strict``, and only when a model is already loaded to
        compare against — :class:`StaleEmbeddingsError` if the stored
        vectors did not come from those weights. That last one leaves a
        fully loaded, usable index behind: the vectors are discarded and
        the reranker loaded *before* it raises, so a caller that catches
        it holds a working BM25-only librarian rather than an empty or a
        half-built one.

        Raises :class:`IndexBusyError` if another task is indexing this
        librarian — an ``add_document`` in flight, or an :meth:`index`
        holding the replace guard — whether that task belongs to this
        thread's loop or to a live loop on another thread
        (:meth:`_observe_replace_guard`; the cross-thread case is the
        ``asyncio.to_thread(lib.load, ...)`` one, measured to produce
        shape 1 below — evidence:
        ``TestWhatTheLivenessCheckIsWorth::test_the_identity_test_admits_the_cross_thread_load``).
        This is the same replacement
        :meth:`index` performs, and :meth:`_replace_guard` exists
        because a concurrent add commits into a freshly-reset index —
        but ``load`` is synchronous and cannot take that guard, so it
        refuses instead. It is not a theoretical window; both halves
        reproduce on the first try (evidence:
        ``TestTheCorruptionTheRefusalPrevents``, whose ``shape_1`` and
        ``shape_2`` tests remove the refusal and assert each):

        * The ``_reset`` here wipes ``documents`` while the add is
          between its claim and its commit, and the commit then lands
          that document's chunks and summary in the index that replaced
          it. The result is chunks whose ``doc_id`` is in neither
          ``documents`` nor ``source``, returned by ``retrieve`` and
          cited in answers, excluded from ``content_hash`` — and
          therefore written by ``save`` into a file that
          ``_load_if_current`` then calls current and serves. Nothing
          re-derives those chunks away, because the document they came
          from is no longer in the index at all.
        * With a second document in the same pending sweep, the next
          ``_chunk_document`` reads ``self.documents[doc_id]`` after the
          reset removed it and ``add_document`` raises a bare
          ``KeyError`` — the failure ``_replace_guard``'s docstring
          predicts for an unguarded reset, reached through ``load``.

        A concurrent :meth:`index` produces the first shape and not the
        second — it chunks one document, so there is no sweep to trip the
        ``KeyError`` — and it is otherwise identical, measured the same
        way: 24 chunks of the document ``index()`` was building, stranded
        in the loaded index and ranked first for a term only they
        contain (evidence:
        ``TestLoadRefusesMidIndex::test_the_corruption_the_refusal_prevents``).

        ``_load_if_current`` calls this from inside ``index()``, which
        holds ``_replacing()`` and has already drained the adds, and it is
        the *owner* of the replace hold rather than a stranger to it
        (:meth:`_replacing_elsewhere`) — so the replace half cannot fire
        there either, on the same single-threaded terms as the add half
        below and not on stronger ones. ``_replacing_elsewhere`` walks
        every live loop's owner, so a *foreign* loop's ``index()`` makes it
        true here too; what rules that out is the same invariant, that a
        handover away from a live holder is refused, and the same
        cross-thread window is where it stops holding.

        **The add half is refusal-free single-threaded, which is a narrower
        promise than it used to be.** The drain empties *this loop's* entry
        (``while self._adds_registered_on(loop)`` in :meth:`_replacing`)
        while :meth:`_adds_in_flight_here` sums every live loop's, so what
        makes the reading zero is no longer the drain by itself: it is that
        no other loop can be holding a registration, because the guard's
        loop follows the last registrant and
        :meth:`_handover_would_strand` refuses a handover away from a live
        loop that holds a count. In the cross-thread advisory window that
        reading leaves open (:meth:`_observe_replace_guard`), an
        ``index(..., cache=...)`` really can raise
        :class:`IndexBusyError` from here, and its remedy line now says so
        rather than promising that this path is unaffected — the clause is
        scoped to the adds that path drains, which are its own loop's.
        Stated rather than smoothed over, because the
        alternative on that route is not "works": before the registrations
        were keyed by loop, that same race read a count the handover had
        zeroed and *admitted* the load, which is the corruption above.
        """
        complaint = self._busy_with_adds(
            "load",
            "the reset would strand the in-flight document's chunks in the "
            "loaded index, under a doc_id that index does not hold",
        ) or self._busy_replacing(
            "load",
            "the reset would land between that call's own reset and its "
            "commit, stranding the document it is indexing in the loaded "
            "index, under a doc_id that index does not hold",
        )
        if complaint:
            raise IndexBusyError(complaint)

        directory = Path(path)
        meta_file = directory / "index.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"no librarium index at {directory}")

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        found = meta.get("format")
        if found != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"index at {directory} is format {found}, this librarium "
                f"reads format {INDEX_FORMAT_VERSION} — re-index to upgrade"
            )

        self._reset()
        self.documents = dict(meta.get("documents", {}))
        self.doc_summaries = dict(meta.get("doc_summaries", {}))
        self._restore_doc_fields(meta, directory)
        self._restore_facts(meta, directory)
        self.doc_summary = meta.get("doc_summary", "")
        self.source = "\n\n".join(self.documents.values())
        self.chunks = [Chunk(**c) for c in meta.get("chunks", [])]
        self.use_chunking = bool(meta.get("use_chunking", bool(self.chunks)))

        self._build_bm25()
        self._build_fact_index()

        saved_model = meta.get("embed_model", "")
        if saved_model and saved_model != self.embed_model_name:
            logger.warning(
                "  LIBRARIAN index was embedded with %s but this librarian "
                "uses %s — vectors are not comparable, re-index to use dense "
                "retrieval", saved_model, self.embed_model_name,
            )
            self._dense_ok = False

        # The same question as `embed_model` above, asked about the text
        # rather than the weights, and it has to be asked here because
        # nothing else on this path can. `content_hash` covers
        # `EMBED_SOURCE`, but only `_load_if_current` consults a hash; a
        # caller who reaches for the documented `load()` gets no such
        # check, and the fingerprint cannot help — it compares the model,
        # and the model is identical. Without this branch a pre-change
        # index loaded text-space vectors at `mode=hybrid` silently.
        #
        # A **missing** key is a mismatch, not an unknown: every file
        # written before the change lacks it and every one of them holds
        # `"text"` vectors. That asymmetry with `embed_model` — where ""
        # means "no vectors were saved" and is skipped — is why this
        # compares `meta.get(..., "text")` and not a truthiness test.
        # `""` written beside absent vectors is the one case that is
        # genuinely nothing to check, and there are no vectors to discard
        # there either.
        saved_source = meta.get("embed_source", "text")
        source_mismatch = bool(saved_source) and saved_source != EMBED_SOURCE

        vectors = directory / "embeddings.npy"
        discarded = False
        if vectors.exists() and self._dense_ok and source_mismatch:
            # Discarded rather than refused, and `_dense_ok` deliberately
            # left alone: these are honest vectors of the wrong text, so
            # the next `add_document` re-embeds into the current space —
            # the same recovery the fingerprint-drift and count-mismatch
            # discards below and beside this rely on.
            #
            # **Nothing raises, `strict` included, and that is a deliberate
            # divergence from the fingerprint-drift discard rather than an
            # oversight in copying it.** Somebody tidying the two into
            # symmetry would break no test today, so here is the argument.
            # `strict` chooses between degrading and raising, and it is
            # worth raising on drift because drift says something changed
            # *underneath* the caller — different weights behind the same
            # name, an environment problem they may need to stop on, and
            # one that re-embedding does not necessarily fix. This says
            # something changed inside *librarium*: the vectors are exactly
            # what the same model produced from less text, the cause is a
            # library upgrade, and one `add_document` repairs it completely.
            # There is nothing for a caller to do differently and nothing
            # for them to investigate. Raising would instead make every
            # index written before the upgrade fail to load for anyone who
            # set `strict` — turning a migration this branch deliberately
            # made recoverable into an outage, on a flag that is about
            # *their* data being consistent. Degrading here *is* the
            # migration path; `retrieval_mode` reports `bm25-only` while it
            # happens.
            #
            # If a future change makes the old vectors genuinely unusable
            # rather than merely narrower, that argument stops holding and
            # this is the branch to revisit.
            discarded = True
            logger.warning(
                "  LIBRARIAN index embedded %r text but this librarium "
                "embeds %r — the vectors are of a different space, "
                "discarding them; BM25-only until the next add_document "
                "re-embeds", saved_source, EMBED_SOURCE,
            )
        elif vectors.exists() and self._dense_ok:
            loaded = np.load(vectors)
            if len(loaded) == len(self.chunks):
                self.chunk_embeddings = loaded
                self._vector_fingerprint = meta.get("embed_fingerprint")
            else:
                discarded = True
                logger.warning(
                    "  LIBRARIAN saved embeddings cover %d chunks but the "
                    "index has %d — discarding, BM25-only until re-indexed",
                    len(loaded), len(self.chunks),
                )

        # Above the fingerprint check, not after it: under `strict` that
        # check raises, and `_load_reranker` has no other call site on
        # any path a loaded librarian reaches — `retrieve()` skips
        # reranking on a falsy `self.reranker` without a word. A caller
        # that catches `StaleEmbeddingsError`, which this method's
        # docstring invites, would otherwise keep `use_reranker=True`
        # with no reranker for the life of the librarian, so the branch
        # that exists to stop silent degradation would introduce some.
        # Loading it here also makes strict and non-strict leave the same
        # object behind: the only difference between them should be
        # whether this method returns or raises.
        self._load_reranker()

        # Only if a model is already loaded, which is what
        # `_current_fingerprint` returning None for "no model" buys:
        # nothing here loads one to answer the question, so `load()` stays
        # as cheap as it was and an offline process stays offline. When no
        # model is loaded yet the comparison happens at the first thing
        # that needs one — `_ensure_embed_model` runs it on every path
        # that hands a model back.
        if self._check_embed_fingerprint():
            discarded = True
        elif (
            self.chunk_embeddings is not None
            and self._vector_fingerprint is None
        ):
            # No fingerprint recorded — two different situations that used
            # to produce one message, and the message was the one that is
            # wrong here. "Re-index to record one" is right for an index
            # written before fingerprints existed and actively misleading
            # for a model that cannot produce one: following it changes
            # nothing, forever, and the user never learns the check is
            # off. `_model_declined_probe` is what separates them, and it
            # is only ever able to answer because `_reset` keeps the
            # model's fingerprint across a `load()`.
            if self._model_declined_probe():
                # INFO, not WARNING, and not a contradiction of
                # `_report_unfingerprintable`: that warning has already
                # fired for this model — being probed is the precondition
                # for reaching this branch — and it carries the fact.
                # This line only maps the fact onto the index just
                # loaded, for a reader who has INFO on.
                logger.info(
                    "  LIBRARIAN index at %s records no embedding "
                    "fingerprint and %s cannot produce one (see the warning "
                    "from its probe) — re-indexing will not record one "
                    "either, so these vectors will never be checked against "
                    "the weights", directory, self.embed_model_name,
                )
            else:
                # An index from before fingerprints existed — the ordinary
                # state on upgrade, and the reason this is INFO beside
                # `load()`'s own INFO rather than a warning. There is
                # nothing to act on: the vectors may be perfectly current,
                # nothing here can tell, and a WARNING on every load of
                # every pre-existing index is how the real one two
                # branches up gets filtered out. Re-indexing records one;
                # nothing stamps the current model's identity onto vectors
                # it did not produce, which would turn "unverified" into a
                # false "verified".
                #
                # This is also where a librarian with no model loaded
                # lands, and correctly: it cannot yet know which case it
                # is in, and if the model turns out to be
                # unfingerprintable the probe will say so when something
                # first loads it.
                logger.info(
                    "  LIBRARIAN index at %s records no embedding fingerprint "
                    "(saved before this librarium recorded one) — its vectors "
                    "cannot be checked against the current weights; re-index "
                    "to record one", directory,
                )

        # An index built without vectors otherwise loads as BM25-only in
        # total silence, and caches as current: `save` writes
        # ``embed_model: ""`` when there were none, the mismatch guard
        # above only fires on a *different* name, and `content_hash` folds
        # in the model this librarian is configured with — identical
        # whether or not the backend was importable when the index was
        # built. So `_load_if_current` calls that cache current and serves
        # it, and installing the extra to get dense retrieval appears to
        # work and does nothing. Nothing here rebuilds the vectors:
        # re-indexing is one LLM call per chunk, and turning a cache hit
        # into hundreds of metered calls because a package appeared is
        # worse than the bug. Saying so is the whole fix.
        #
        # `retrieval_mode` already reported this truthfully — the gap was
        # that nobody was asking it.
        #
        # Silent when the backend is absent: BM25-only is the intended,
        # documented mode for a core install, and a warning on every load
        # for every user who never wanted dense retrieval is how a real
        # warning gets filtered out. Silent too for an index with no
        # chunks — whole-document and empty indexes hold no vectors by
        # design, and "re-index" is no remedy for them — and after the
        # count mismatch above, which has just said the same thing better.
        if (
            self._dense_ok
            and self.chunk_embeddings is None
            and not discarded
            and self.use_chunking
            and self.chunks
            and embedding_backend_available()
        ):
            logger.warning(
                "  LIBRARIAN index at %s was built without embeddings, but "
                "the embeddings extra is installed — retrieval is BM25-only "
                "and will stay that way; re-index to use dense retrieval",
                directory,
            )

        logger.info(
            "  LIBRARIAN loaded index from %s (%d chunks, mode=%s)",
            directory, len(self.chunks), self.retrieval_mode,
        )

    def _load_if_current(self, path: str | Path, text: str) -> bool:
        """Load a cached index if it was built from exactly this text."""
        directory = Path(path)
        meta_file = directory / "index.json"
        if not meta_file.exists():
            return False
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("  LIBRARIAN cache at %s is unreadable (%s)", directory, e)
            return False

        if meta.get("format") != INDEX_FORMAT_VERSION:
            logger.info("  LIBRARIAN cache at %s is an older format — re-indexing", directory)
            return False

        cached_docs = meta.get("documents", {})
        if list(cached_docs.values()) != [text]:
            logger.info("  LIBRARIAN cache at %s is for different text — re-indexing", directory)
            return False
        if meta.get("content_hash") != self._content_hash(cached_docs):
            logger.info("  LIBRARIAN cache at %s is stale — re-indexing", directory)
            return False

        try:
            self.load(directory)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning("  LIBRARIAN cache at %s failed to load (%s)", directory, e)
            self._reset()
            return False
        logger.info("  LIBRARIAN reused cached index at %s", directory)
        return True

    def _relabel_single_document(self, doc_id: str) -> None:
        """Carry the caller's doc_id onto an index restored from cache.

        The cache is keyed on content, not on the label, so the same text
        under a new doc_id still hits — but the provenance on every chunk
        has to follow the caller, not the file it was saved under.
        """
        if len(self.documents) != 1:
            return
        ((cached_id, text),) = self.documents.items()
        if cached_id == doc_id:
            return
        self.documents = {doc_id: text}
        self.doc_summaries = {doc_id: self.doc_summaries.get(cached_id, "")}
        # Only if the cache carried fields at all — `.get(cached_id, {})`
        # unconditionally would turn "this index has no fields" into
        # "this document was extracted and yielded nothing", which is the
        # one distinction `doc_fields` exists to keep.
        if cached_id in self.doc_fields:
            self.doc_fields = {doc_id: self.doc_fields[cached_id]}
        for chunk in self.chunks:
            chunk.doc_id = doc_id

    def _restore_doc_fields(self, meta: dict, directory: Path) -> None:
        """Restore ``doc_fields`` from a loaded index, or discard them.

        Discarded — silently — when the file carries none, which is every
        index written before this feature and every index written with
        ``extract_fields`` off. That is not a mismatch worth a warning; it
        is the overwhelmingly common case.

        Discarded **with a warning** when the file's
        ``doc_fields_version`` is not this one's, because then the records
        exist and cannot be trusted: a changed prompt or field set means
        the values under those keys mean something else, and a filter
        applied to them would exclude documents on a comparison nobody
        checked. Nothing re-derives them here — that costs an LLM call per
        document — so the librarian carries on ranking-only until the next
        ``add_document`` re-extracts.

        This is the direct analogue of `load()`'s `embed_source` check and
        exists for the same reason: `content_hash` folds the version in,
        but only `_load_if_current` consults a hash, so a caller reaching
        for the documented `load()` gets no check at all from it.

        Not gated on ``self.extract_fields``: a librarian with extraction
        off can still *read* an index built with it on, and the records
        are as valid as the version says they are. Whether anything
        filters on them is that librarian's business.
        """
        saved = meta.get("doc_fields")
        if not saved:
            return
        saved_version = meta.get("doc_fields_version", "")
        if saved_version != DOC_FIELDS_VERSION:
            logger.warning(
                "  LIBRARIAN index at %s carries fields from %s but this "
                "librarium writes %s — discarding them; the next "
                "add_document re-extracts",
                directory, saved_version or "an unrecorded version",
                DOC_FIELDS_VERSION,
            )
            return
        # Values are re-filtered rather than trusted: the file is
        # editable, and `parse_fields`' placeholder and length bars are
        # what keep a filter from matching on "N/A". A record that empties
        # out is dropped, not stored empty — see `doc_fields` in __init__.
        #
        # Re-parsed but **not** re-verified: `verify_fields` needs the
        # document sample the extraction saw, and `load()` has the whole
        # document rather than that sample, so re-running the groundedness
        # check here would fail quotes that were legitimately taken from
        # material the sample included. The `flags` recorded at extraction
        # are carried across instead, which is what they are for.
        for doc_id, fields in saved.items():
            if not isinstance(fields, dict):
                continue
            kept = parse_fields(json.dumps(fields))
            for name, record in kept.items():
                stored = fields.get(name)
                if isinstance(stored, dict) and isinstance(
                    stored.get("flags"), list
                ):
                    record["flags"] = [
                        f for f in stored["flags"] if isinstance(f, str)
                    ]
                else:
                    # A record from a file that predates flags, or a
                    # hand-edited one. "Unknown provenance" is the honest
                    # reading and the one a filter should be cautious
                    # about, not an empty list meaning "clean".
                    record["flags"] = ["unverified"]
            if kept:
                self.doc_fields[doc_id] = kept

    def _restore_facts(self, meta: dict, directory: Path) -> None:
        """Load a saved fact store, or discard it and say so.

        `_restore_doc_fields`' argument applied to the second store, and
        for the same reason: `content_hash` decides whether a **cache** is
        current and only `_load_if_current` asks it, while a direct
        `load()` consults no hash at all — so without this check a store
        extracted under an older prompt would be installed silently and
        retrieved from as though it were this version's.

        Records are re-parsed rather than trusted, because the file is
        editable and `parse_facts`' bars are what keep a 4,000-character
        "value" out of a store whose whole advantage is short entries.
        """
        saved = meta.get("facts")
        if not saved or not isinstance(saved, list):
            return
        saved_version = meta.get("facts_version", "")
        if saved_version != FACTS_VERSION:
            logger.warning(
                "  LIBRARIAN index at %s carries facts from %s but this "
                "librarium writes %s — discarding them; the next "
                "add_document re-extracts",
                directory, saved_version or "an unrecorded version",
                FACTS_VERSION,
            )
            return
        restored: list[dict[str, str]] = []
        for item in saved:
            if not isinstance(item, dict):
                continue
            kept = parse_facts(json.dumps({"facts": [item]}))
            if not kept:
                continue
            record = kept[0]
            # Provenance travels with the record and is not re-derived:
            # `chunk` indexes this index's own chunk list, which load()
            # has just replaced wholesale.
            record["doc_id"] = str(item.get("doc_id", ""))
            if isinstance(item.get("chunk"), int):
                record["chunk"] = item["chunk"]
            restored.append(record)
        self.facts = restored

    def _reset(self) -> None:
        """Drop every trace of the current index (keeps loaded models)."""
        self.source = ""
        self.documents = {}
        self.doc_summaries = {}
        self.doc_fields = {}
        self.facts = []
        self.fact_bm25 = None
        self.fact_embeddings = None
        self.doc_summary = ""
        self.chunks = []
        self.bm25 = None
        self.chunk_embeddings = None
        # Index state, not model state: it describes vectors that are
        # going away. `_model_fingerprint` stays — it belongs to the
        # loaded model, which `_reset` deliberately keeps.
        self._vector_fingerprint = None
        self.use_chunking = False
        # The claims describe documents that no longer exist. A claim is
        # only ever held by a live indexing call — `_add_document` gives
        # it back in a `finally` that runs before `_adding` decrements,
        # and it is reached only from `add_document` (counted) or from
        # `index()` (holding the guard) — and both callers now exclude
        # one: `index()` drains the adds and holds the guard exclusively,
        # and `load()` refuses against an add in flight *and* against
        # another task's replace, on this thread's loop or on a live one
        # belonging to another (`_observe_replace_guard`).
        #
        # The remaining case — a claim stranded by the loop ending under
        # it, whose `finally` never runs — is handled where it is
        # detectable rather than here: `_replace_guard` clears the claims
        # when it meets a new loop and finds the old one closed.
        #
        # Which makes this `clear()` a backstop and not the only place a
        # replace drops them, whatever an earlier version of this comment
        # said. Measured on both routes into it: from `index()` the set
        # is empty by that method's own construction (the adds are
        # drained and each claim is given back before its count is) —
        # over the whole suite the one `index()` reset that met a
        # non-empty set met a *foreign* loop's claim, not its own
        # (unverified: that came from instrumenting `_reset` and running
        # the suite once; nothing in the tree records it, and it is the
        # kind of count a new test moves. What would discharge it: log the
        # owning loop beside each non-empty set this method meets, run the
        # suite, and read the tally back) — and
        # from `load()` it drops the dead-loop claims the next
        # `_replace_guard` would drop anyway, only sooner. Removing the
        # line changes no outcome on either route, just the intermediate
        # state `test_the_reset_clears_a_claim_the_dead_loop_left_behind`
        # reads. It is kept because it is cheap and because it belongs to
        # what this method means: the index is gone, so nothing is
        # part-way through being chunked into it.
        #
        # Unconditional, unlike `_replace_guard`'s, which clears only for
        # a closed loop. What makes that one conditional is a pending
        # sweep finding a still-live add's document in `documents` and
        # chunking it a second time; here `documents` has just been
        # emptied, so a claim dropped leaves no document behind to sweep.
        self._chunking.clear()

    # ── Retrieval ────────────────────────────────────────────────

    def _resolve_top_k(self, top_k: int | None) -> int:
        """How many chunks this call retrieves: override, else the default."""
        return self.top_k if top_k is None else check_top_k(top_k)

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        also: Sequence[str] = (),
        exclude: Collection[int] = (),
    ) -> list[Chunk]:
        """Return the chunks most relevant to *question*, best first.

        This is the reusable half of librarium: use it to build your own
        prompt, stream the answer yourself, or render citations from
        each chunk's ``doc_id`` and ``offset``. When the input was short
        enough to skip chunking, this returns one chunk per document.

        *top_k* defaults to the librarian's own, set at construction.

        *exclude* drops chunks by ``index`` before the cut, which is how
        a caller asks for the **next page**: retrieve, and if the answer
        was not there, retrieve again passing the indexes already seen.

        Worth having because the material usually *is* within reach.
        Measured on the 194-document corpus at a pool depth of 50: 70% of
        expected literals appear in ranks 1-10, another 12% in ranks
        11-20 and 12% more in 21-50, with only **6% absent from the top
        50 entirely**. Of the eight questions that find nothing at all on
        the first page, **seven are recovered by paging** — four on page
        two, three deeper — and they are the categories that hurt:
        `vocab_mismatch` four of five, `versioning` two of two (evidence:
        `benchmarks/README.md`).

        This is the cheap half of "look again": no LLM call and no
        ``await``, so the contract below is untouched. The other half is
        *also*, when the caller knows something about the answer.

        **It filters chunks, not facts.** A fact mixed in by ``fact_mix``
        carries ``index = -1`` because it is not a position in
        ``self.chunks``, so excluding by index cannot name one and a
        second page redraws the same facts. That is a real limit rather
        than an oversight: paging the fact store wants its own identity
        scheme, and nothing has asked for it yet.

        *also* supplies extra phrasings of the same question — from
        :meth:`expand_query`, or from your own client — whose rankings
        fuse with the question's own. It changes nothing about this
        method's contract: **no LLM call, no ``await``, no lock.** That
        is why generating the phrasings is a separate coroutine and not
        an argument that would make retrieval async. Empty, repeated, and
        question-identical phrasings are dropped before fusion
        (:func:`distinct_variants`).

        The decisions that live in the fusion:

        * **The question's own ranking is built first, and no phrasing
          is ever fused ahead of it** (when the question scored nothing
          its ranking is empty, and an empty ranking is dropped rather
          than fused — it carries no RRF mass either way). A drifting
          paraphrase must not displace a literal
          match, which is the case BM25 is best at and the one failure
          that would make expansion worse than not having it. RRF breaks
          ties by first appearance, so building the list in this order is
          what makes that concrete rather than aspirational.
        * **A lexical ranking votes only for the chunks it actually
          scored — whenever it is not the only ranking in the fusion.**
          That is: with extra phrasings, with a dense half, or with both.
          BM25 scores every chunk, so a query it cannot score at all
          still returns all *n* of them in index order, and RRF cannot
          tell that filler from a ranking (:meth:`_bm25_ranking`). Fused
          against anything real it wins on position, which is how a
          zero-scoring lexical half used to outvote a dense ranking that
          had put the answer first. Alone the filler is harmless — an
          arbitrary but non-empty answer — so the cut is not applied
          there, and **BM25-only results are exactly what they were**.
          The visible consequence: on a BM25-only index, a question whose
          phrasings between them match fewer than *top_k* chunks gets
          fewer chunks back, because the rest would have been chunks
          nothing asked for. On a hybrid index the dense half is
          full-length, so *top_k* still arrives — but **which** chunks
          arrive changed when the cut was extended to the hybrid path,
          deliberately and in the answer's favour.
        * **Variants get lexical rankings only; the dense ranking is the
          question's alone.** Expansion exists to give BM25 the
          vocabulary it has no other way to reach — an embedding already
          places "auto policy" near "car insurance", so a variant adds
          least exactly where it costs most. Dense would multiply
          query-time encoding by ``1 + len(also)`` (a model call per
          phrasing, on the slowest step of the query path) to re-rank
          neighbourhoods the question's own vector already covers. If a
          measurement later shows variants earning their encode, this is
          the line to change.
        * **Reranking scores against the original question**, not the
          variants. The cross-encoder judges relevance to what was
          actually asked; expansion is a recall device and stops at the
          candidate set.
        * **The fused ranking is cut to ``rerank_pool``, not to
          ``top_k``, before the cross-encoder sees it** — and only when
          one is loaded, since on every other path the extra candidates
          would be returned unranked. ``rerank_pool`` is the reranker's
          input width and ``rerank_top`` its output width; the default
          ``None`` ties the input to ``top_k``, which is where this cut
          has always been.

        Synchronous and lock-free, so it never waits on an index that is
        still being built — it reads whatever is there at the time, and
        where no ranking covers the chunks it finds it degrades to whole
        documents, for which ``top_k`` and *also* do not apply. See
        :meth:`add_document` for what that means for a query issued
        while a document is being indexed.
        """
        top_k = self._resolve_top_k(top_k)
        variants = distinct_variants(question, also)

        if not self.use_chunking or self.bm25 is None or not self.chunks:
            return self._whole_documents()

        n = len(self.chunks)

        # The dense ranking is computed *first* and fused *last*, and the
        # split is the point.
        #
        # First, because the lexical block below has to know whether it is
        # the only vote in the fusion, and the only honest way to know is
        # to have the answer already. A precondition check
        # (`chunk_embeddings is not None and _dense_ok`) reads the same in
        # the common case and is a forecast `_embedding_ranking` can
        # falsify four ways after the fact: a vector/chunk count mismatch,
        # a model that fails to load, an encode that raises, `_dense_ok`
        # flipping mid-call. Every one of those would leave a *sole*
        # lexical ranking truncated, which is the one thing BM25-only mode
        # may not do.
        #
        # Last, because RRF breaks ties by first appearance and the
        # question's own lexical ranking has to keep that slot — see the
        # docstring. Computing in one order and appending in another is
        # what buys both properties at once.
        cos_ranked = self._embedding_ranking(question, n)

        # Question first, so a tie between a chunk the question found and
        # one only a variant found resolves towards the question.
        rankings: list[list[int]] = []
        # `scored` once this ranking is not the only one in the fusion:
        # extra phrasings, a dense half, or both. Filler in index order
        # cannot be told from a ranking by RRF, and beats one — measured
        # against variants when expansion landed (`TestScoredTruncation`),
        # and measured again against the dense half
        # (`TestAZeroScoringLexicalHalfNoLongerOutvotesTheDenseOne`), where
        # a query whose vocabulary the corpus did not share scored every
        # chunk zero and handed the answer to whichever document happened
        # to be indexed first.
        # With neither, this is the same call it has always been, so
        # `retrieve(q)` on a BM25-only index returns what it returned
        # before by construction rather than by measurement.
        # Question and variants have to use the same rule: truncating the
        # variants alone leaves the question voting index order for chunks
        # it never scored, and a chunk a variant really found then loses
        # to the question's own filler vote.
        scored = bool(variants) or cos_ranked is not None
        bm25_ranked = self._bm25_ranking(question, n, scored=scored)
        if bm25_ranked is not None:
            lexical = [bm25_ranked]
            # Only when the question's own lexical ranking worked. The
            # guard inside `_bm25_ranking` is about the index being
            # behind the chunks, which is a property of the index and not
            # of the query: if it fired for the question it fires for
            # every variant, so there is no `is not None` to test here —
            # the only falsy result a variant can produce is the empty
            # ranking of a phrasing that matched nothing, and a phrasing
            # that matched nothing has no vote to cast.
            for variant in variants:
                variant_ranked = self._bm25_ranking(variant, n, scored=True)
                if variant_ranked:
                    lexical.append(variant_ranked)
            if not any(lexical) and cos_ranked is None:
                # Question and every phrasing matched nothing at all, and
                # the lexical side is the whole fusion. Take the
                # untruncated ranking back rather than hand the fusion
                # nothing: arbitrary is what `retrieve(question)` alone
                # would have returned here, and expansion is not allowed to
                # leave a question with less than it started with.
                #
                # Deliberately not taken back when a dense ranking exists.
                # The guard is there so the fusion is never left empty,
                # and with the dense half present it is not empty — it is
                # a full-length ranking of every chunk, so the caller
                # still gets `top_k`. Restoring index order here would put
                # back exactly the filler this truncation removes, on the
                # one query shape where BM25 knows nothing and the dense
                # half is the only vote with anything to say. Hybrid loses
                # nothing by dropping it; that is the whole trade.
                lexical = [self._bm25_ranking(question, n) or []]
            rankings.extend(ranking for ranking in lexical if ranking)

        # Embedding on raw chunks, for the question only — see the
        # docstring on why variants stay lexical. Computed above; appended
        # here so the question's lexical ranking keeps first place.
        if cos_ranked is not None:
            rankings.append(cos_ranked)

        if not rankings:
            # Neither half can rank these chunks. Same answer the top of
            # this method gives an index with no lexical half at all:
            # hand back whole documents rather than nothing.
            return self._whole_documents()
        # Fuse to the reranker's pool, not to the answer width. With the
        # cut at `top_k` the cross-encoder can only reorder a set it
        # cannot add to: a chunk RRF ranked eleventh is gone before the
        # one model actually built to judge relevance is asked about it,
        # and no reranking setting can recover it.
        #
        # Widened only when a reranker is really loaded and will really
        # cut back to `rerank_top`. With no cross-encoder nothing scores
        # the extra candidates, so they would be fused and then returned
        # unranked — a longer answer, not a better one, and `top_k` is
        # the whole of what the caller asked for there.
        #
        # The gate is the reranker and *not* the retrieval mode: a
        # librarian whose encoder failed to load but whose cross-encoder
        # did is BM25-only with something to rank the extra candidates,
        # so it widens like any other and should. Worth stating because
        # the wrong predicate is the intuitive one — every test in
        # `TestRerankPool` is that librarian, since `indexed()` clears
        # `_dense_ok`.
        #
        # `max`, not the pool alone: a pool under `top_k` would narrow
        # the result through a knob whose whole job is to widen the
        # input, so it is clamped and `check_index_settings` warns.
        pool = top_k
        if (self.rerank_pool is not None
                and self.use_reranker and self.reranker):
            pool = max(self.rerank_pool, top_k)
        ranked = (
            reciprocal_rank_fusion(rankings, k=self.rrf_k)
            if len(rankings) > 1
            else rankings[0]
        )
        # Applied to the whole fused ranking and **before** the pool is
        # cut, not after: cutting first and filtering second would hand the
        # cross-encoder the very chunks the caller is trying to move past,
        # and a second page would be a reranked view of the first.
        if exclude:
            skip = set(exclude)
            ranked = [i for i in ranked if i not in skip]
        top_indices = ranked[:pool]

        top_indices = self._rerank(question, top_indices)

        logger.info(
            "  LIBRARIAN retrieved %d chunks (%s, %d ranking(s) over %d extra "
            "phrasing(s)) for: %s",
            len(top_indices), self.retrieval_mode, len(rankings),
            len(variants), question[:80],
        )
        chunks = [self.chunks[i] for i in top_indices]
        return self._mix_in_facts(question, chunks, top_k)

    def _mix_in_facts(
        self, question: str, chunks: list[Chunk], top_k: int
    ) -> list[Chunk]:
        """Spend part of the slot budget on facts, when a store exists.

        **A mix, never a replacement.** Measured on the 194-document
        benchmark corpus, a fact store *alone* loses to the chunk store by
        10.6 headline points — atomising an answer throws away the
        connective tissue a chunk carries for free, so the two categories
        needing several documents at once collapse. The two units fail on
        opposite categories, which is why combining them beats either:
        against a chunk arm handed *more* characters, the mix is +2.4 to
        +3.5, with `versioning` — the category that fixture declares
        unanswerable — going 61.5% to 94.9% and identical across every
        cell of a noise 2x2 (evidence: `benchmarks/fact_retrieval.py`,
        `benchmarks/README.md`).

        Facts are returned as `Chunk`s so that every existing consumer —
        `ask`, `facts`, `check_context`, `chunk_header`, a caller's own
        prompt — keeps working unchanged. A fact carries the `doc_id` and
        the chunk offset it came from, so provenance and citation are as
        good as a chunk's.

        Inert when there is no store, when `fact_mix` is 0.0, or when the
        share rounds to no slots — and in each case the chunk list comes
        back exactly as it went in, which is what keeps a librarian that
        never asked for facts byte-identical to its pre-knob self.
        """
        if not self.facts or self.fact_mix <= 0.0:
            return chunks
        want = int(round(top_k * self.fact_mix))
        if want <= 0:
            return chunks
        picked = self._fact_ranking(question, want)
        if not picked:
            return chunks
        # The facts *replace* slots rather than extend the budget: a mix
        # that returned `top_k` chunks plus facts on top would be a wider
        # retrieval wearing a mix's name, and the run this rests on held
        # the character budget equal — the chunk arm it beats was handed
        # *more* characters, not fewer (evidence:
        # `benchmarks/fact_retrieval.py`, `benchmarks/README.md`).
        keep = max(0, top_k - len(picked))
        out: list[Chunk] = list(chunks[:keep])
        for i in picked:
            fact = self.facts[i]
            source = None
            position = fact.get("chunk")
            if isinstance(position, int) and 0 <= position < len(self.chunks):
                source = self.chunks[position]
            out.append(Chunk(
                text=fact_text(fact),
                doc_id=str(fact.get("doc_id", "")),
                # The offset of the chunk the fact was drawn from, so a
                # citation points into the document rather than at 0.
                offset=source.offset if source is not None else 0,
                # Facts are not positions in `self.chunks` and must not
                # claim to be: `index` is what a caller would use to ask
                # for neighbours, and a fact has none.
                index=-1,
                prefix="",
            ))
        logger.info(
            "  LIBRARIAN mixed %d fact(s) into %d chunk(s) for: %s",
            len(picked), len(out) - len(picked), question[:80],
        )
        return out

    # ── Query expansion ──────────────────────────────────────────

    async def expand_query(self, question: str) -> list[str]:
        """Rewrite *question* into alternative phrasings. One LLM call.

        The bridge across a vocabulary mismatch: a question asking about
        "car insurance" retrieves nothing from a document that says
        *auto policy*, because BM25 has no lexical overlap to score.
        Feed the result to ``retrieve(..., also=...)``.

        Returns ``[]`` and never raises, unless ``strict=True`` — a
        question must not fail because an optional recall aid did.
        Everything that goes wrong here degrades to single-query
        retrieval and says so at WARNING or ERROR, deliberately louder
        than the quiet ``_dense_ok`` flip: a silent expansion failure
        looks exactly like an expansion that did not help, which is the
        one thing that would make this feature impossible to evaluate.

        Returns ``[]`` without calling the model at all when
        ``expand_query=False``, which is the default. The LLM call is
        budgeted through :meth:`check_context` like every other call —
        it is small, but "small" is how a call ends up being the one
        that is never checked.

        The reply's temperature is :data:`EXPANSION_TEMPERATURE`, not
        ``_llm_call``'s 0.1; the reasoning is there.
        """
        if not self.expansion_enabled:
            return []
        if not question.strip():
            return []

        hyde = self.expansion_style == "hyde"
        system = (HYDE_SYSTEM if hyde
                  else EXPANSION_SYSTEM.format(n=self.expansion_variants))
        user = f"QUESTION: {question}"
        # An expansion memo would go here — question → variants is the
        # same query-keyed shape as the K-line cache in klines.py, and
        # this call is deterministic enough in practice to reuse. It is
        # deliberately not built: caching is its own decision, and this
        # feature ships unmeasured.
        try:
            self.check_context(system, user)
            reply = await self._llm_call(
                system, user, temperature=EXPANSION_TEMPERATURE
            )
        except Exception as e:
            # Exception, not BaseException: a cancellation is the caller
            # giving up on the whole question, not an expansion failure
            # to degrade past.
            logger.error(
                "  LIBRARIAN query expansion failed (%s: %s) — retrieving "
                "for the original question alone: %s",
                type(e).__name__, e, question[:80],
            )
            if self.strict:
                raise
            return []

        if hyde:
            # No parser. `parse_expansion`'s prose bar drops any line
            # carrying a comma, which is every line of a passage, so
            # running it here would discard the whole reply — see
            # HYDE_SYSTEM. `retrieve()` still applies `distinct_variants`,
            # which drops an empty passage and one identical to the
            # question, so a degenerate reply cannot reach the fusion.
            cleaned, _ = parse_thinking(reply)
            passage = " ".join(cleaned.split())
            variants = [passage] if passage else []
        else:
            variants = parse_expansion(
                reply, question, self.expansion_variants
            )
        if not variants:
            logger.warning(
                "  LIBRARIAN query expansion returned nothing usable — "
                "retrieving for the original question alone. Reply was: %r",
                reply[:200],
            )
            if self.strict:
                raise ValueError(
                    f"query expansion returned no usable phrasings for "
                    f"{question[:80]!r} (strict=True); reply was "
                    f"{reply[:200]!r}"
                )
            return []
        logger.info(
            "  LIBRARIAN expanded into %d phrasing(s): %s",
            len(variants), " | ".join(v[:60] for v in variants),
        )
        return variants

    async def retrieve_expanded(
        self, question: str, top_k: int | None = None
    ) -> list[Chunk]:
        """:meth:`expand_query` then :meth:`retrieve`, and nothing else.

        Deliberately a two-line composition. Everything interesting is in
        the halves: the call is in :meth:`expand_query`, the fusion is in
        :meth:`retrieve`, and a caller who wants to batch, cache or
        precompute the phrasings — or produce them from their own client
        — uses those two directly and gets the identical result. Logic
        added here would be logic that path silently skips.
        """
        return self.retrieve(
            question, top_k=top_k, also=await self.expand_query(question)
        )

    def _bm25_ranking(
        self, question: str, n: int, *, scored: bool = False
    ) -> list[int] | None:
        """Chunk indices ranked by BM25, or None if the index is stale.

        *scored* keeps only the chunks BM25 actually scored above zero. A
        zero-scored chunk was not ranked by this query, it was enumerated:
        with every score equal, ``sorted`` falls back to index order, so a
        query matching nothing returns all ``n`` chunks in index order and
        calls it a ranking. As the *only* ranking that is harmless — the
        caller gets an arbitrary but non-empty result. As one vote among
        several it is corrosive, because RRF cannot tell index-order
        filler from a real ranking: at ``rrf_k=60`` over 8 chunks a
        variant matching nothing gave chunk 0 ``1/61 + 1/60 = 0.0331``
        against the true hit's ``1/60 + 1/67 = 0.0316``, and the literal
        match lost to noise.

        The dense half loses the same arithmetic, and worse. Over three
        documents whose vocabulary a query missed entirely, BM25 scored
        every chunk zero — index order — while the embedding ranked the
        answer first; fused at equal weight the filler took it,
        ``0.032522`` against the answer's ``0.032266``. That is where
        dense retrieval is supposed to earn its keep, and the bias is
        systematic: it always favours whatever was indexed earliest.

        So the caller asks for it exactly when this ranking is **not the
        only one in the fusion** — extra phrasings, a dense ranking, or
        both — and not otherwise. Making it unconditional would be the
        tidier rule and is deliberately not taken: on a BM25-only index
        this is the only ranking there is, and truncating it would return
        fewer than ``top_k`` chunks — sometimes none — for any question
        BM25 cannot score, changing what every existing BM25-only caller
        of ``retrieve()`` gets back. Neither expansion nor the dense half
        is allowed to cost that.

        Above zero rather than non-zero: BM25Okapi floors the negative IDF
        of a term carried by most of the corpus, and a phrasing whose only
        purchase is a term that common discriminates nothing worth voting
        on.

        ``get_scores`` returns one score per entry of the corpus bm25 was
        built from, which is not always ``chunks``: the pending sweep in
        ``_add_document`` extends ``chunks`` a document at a time and
        rebuilds bm25 once, at the end, so a query landing on one of the
        awaits in between sees more chunks than there are scores. Nothing
        holds the index still for a query — see :meth:`add_document` on
        what concurrent use is and is not safe — and ``save``/``load``
        can carry that state to disk and back, because ``load`` rebuilds
        bm25 from whatever chunks it read.

        Ranking past the end of the score array raises IndexError. The
        same mismatch on the dense side already drops that half of
        retrieval rather than crashing; drop this one the same way.
        """
        scores = self.bm25.get_scores(tokenize(question))
        if len(scores) != n:
            logger.error(
                "  LIBRARIAN BM25 index covers %d chunks but the index has "
                "%d — skipping lexical retrieval until it is rebuilt",
                len(scores), n,
            )
            return None
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        if scored:
            ranked = [i for i in ranked if scores[i] > 0]
        return ranked

    def _embedding_ranking(self, question: str, n: int) -> list[int] | None:
        """Chunk indices ranked by cosine similarity, or None if unavailable."""
        if self.chunk_embeddings is None or not self._dense_ok:
            return None
        if len(self.chunk_embeddings) != n:
            logger.error(
                "  LIBRARIAN have %d embeddings for %d chunks — skipping "
                "dense retrieval until the index is rebuilt",
                len(self.chunk_embeddings), n,
            )
            return None
        model = self._ensure_embed_model()
        if model is None:
            return None
        # Re-checked after the load, not folded into the guard above:
        # `_ensure_embed_model` is where the fingerprint check runs, so
        # the vectors this was about to score against can have been
        # discarded between the two lines.
        if self.chunk_embeddings is None:
            return None
        # `search_query:` against the documents' `search_document:`, and the
        # asymmetry is nomic's design rather than an oversight to match up.
        # Giving the document side `c.prefixed` did not change this and is
        # not an argument to: what the document side was missing was
        # *content* the lexical half already had, and a question has no
        # parent document to be situated in. Nothing measured supports
        # touching it: benchmarks/retrieval_arms.py varied the document
        # input only, so there is no query-side arm to point at.
        try:
            q_emb = model.encode(
                [f"{embed_prompts(self.embed_model_name)[1]}{question}"],
                show_progress_bar=False
            )[0]
        except Exception as e:
            self._dense_ok = False
            logger.error(
                "  LIBRARIAN failed to embed the query (%s: %s) — falling "
                "back to BM25-only for this librarian",
                type(e).__name__, e,
            )
            if self.strict:
                raise
            return None

        norms = np.linalg.norm(self.chunk_embeddings, axis=1) * np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1.0, norms)
        cos_scores = np.dot(self.chunk_embeddings, q_emb) / norms
        return sorted(range(n), key=lambda i: cos_scores[i], reverse=True)

    def _rerank(self, question: str, top_indices: list[int]) -> list[int]:
        """Re-order candidates with the cross-encoder, if one is loaded."""
        if not (self.use_reranker and self.reranker and top_indices):
            return top_indices
        pairs = [[question, self.chunks[i].text] for i in top_indices]
        rerank_scores = self.reranker.predict(pairs)
        scored = sorted(
            zip(top_indices, rerank_scores), key=lambda x: x[1], reverse=True
        )
        reranked = [
            (i, s) for i, s in scored if s > self.rerank_threshold
        ][:self.rerank_top]
        if not reranked:
            reranked = scored[:self.rerank_top]
        return [i for i, _ in reranked]

    def _whole_documents(self) -> list[Chunk]:
        """Every indexed document as a single chunk each."""
        if self.documents:
            return [
                Chunk(text=text, doc_id=doc_id, index=i)
                for i, (doc_id, text) in enumerate(self.documents.items())
            ]
        if self.source:
            return [Chunk(text=self.source, doc_id=DEFAULT_DOC_ID)]
        return []

    def chunk_header(self, chunk: Chunk) -> str:
        """The provenance line prefixed to *chunk* in a prompt, or ``""``.

        **The model currently sees no provenance at all.** `retrieve()`
        returns chunks carrying `doc_id` and `offset`, and index-time
        extraction can add `doc_type` / `issuer` / `effective_date` — and
        `_retrieve` discarded every one of them, handing the model bare
        text. So a question about the March 2024 policy is answered from
        a chunk that says only "Section 5 — Part D", with nothing
        anywhere in the prompt saying which of three near-identical
        policy years it came from.

        That is the `near_duplicate` failure restated at the answering
        step rather than the retrieval one, and it is the cheap half:
        adding a header **cannot remove an answer**, which is what made
        both filter designs measure negative. The worst case is spent
        tokens.

        Fields are used when `extract_fields` produced them and skipped
        silently when it did not, so this works on an index that never
        ran extraction — `doc_id` alone is already more than the model
        had.
        """
        if not self.cite_sources:
            return ""
        parts = [chunk.doc_id] if chunk.doc_id else []
        record = self.doc_fields.get(chunk.doc_id) or {}
        for name in ("doc_type", "issuer", "effective_date"):
            value = (record.get(name) or {}).get("value")
            if value:
                parts.append(value)
        return f"[{' · '.join(parts)}]\n" if parts else ""

    def _retrieve(self, question: str, *, top_k: int | None = None) -> str:
        """Retrieved material as one string, ready to drop into a prompt."""
        return CHUNK_JOINER.join(
            f"{self.chunk_header(c)}{c.text}"
            for c in self.retrieve(question, top_k=top_k)
        )

    # ── Answer ───────────────────────────────────────────────────

    async def answer(
        self,
        question: str,
        context_facts: list[Fact] | None = None,
        refine: bool = False,
        max_rounds: int = 2,
        *,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Answer a question against the source material.

        When *refine* is True, the librarian checks whether the first-pass
        facts are sufficient to answer the question.  If not, it re-queries
        with a focused refinement question.  Capped at *max_rounds* total
        passes (default 2) to keep cost bounded.

        *top_k* overrides the librarian's default for this question,
        refinement rounds included — they read the same context budget.
        """
        result = await self._answer_single(question, context_facts, top_k=top_k)

        if not refine or not result.items:
            return result

        all_facts = list(result.items)
        seen = {f.content for f in all_facts}
        coverage = result.coverage

        for rnd in range(1, max_rounds):
            gap = await self._find_gap(question, all_facts)
            if gap is None:
                logger.info("  LIBRARIAN refine round %d: sufficient → stop", rnd)
                # The gap check confirmed every part of the question is
                # answered, which is a stronger signal than the first pass's
                # self-reported coverage.  Promoting it is not cosmetic:
                # LibrarianRouter falls through to the web librarian whenever
                # coverage != "complete", so leaving "partial" here sends a
                # fully-answered question out for a search + scrape + index.
                #
                # Only this exit promotes.  Leaving via no-new-facts or round
                # exhaustion means the gap was never closed, so those keep the
                # original coverage and stay honest.
                coverage = "complete"
                break

            # Re-query with the gap as the question — different search
            logger.info("  LIBRARIAN refine round %d: gap=%r → re-query", rnd, gap[:80])
            extra = await self._answer_single(gap, top_k=top_k)

            # Merge new facts, skip duplicates
            added = 0
            for fact in extra.items:
                if fact.content not in seen:
                    all_facts.append(fact)
                    seen.add(fact.content)
                    added += 1

            if added == 0:
                logger.info("  LIBRARIAN refine round %d: no new facts → stop", rnd)
                break

            logger.info("  LIBRARIAN refine round %d: added %d new facts", rnd, added)

        return LibrarianAnswer(items=all_facts, coverage=coverage)

    async def _find_gap(
        self, question: str, facts: list[Fact]
    ) -> str | None:
        """Return a search query for missing information, or None if sufficient."""
        facts_text = "\n".join(f"- {f.content}" for f in facts)
        prompt = (
            f"QUESTION: {question}\n\n"
            f"FACTS FOUND SO FAR:\n{facts_text}\n\n"
            f"Break the question into parts. For each part, check if the "
            f"facts above answer it. If ANY part is unanswered, respond "
            f"with:\nQUERY: <search keywords for the missing part>\n"
            f"If ALL parts are answered, respond with: SUFFICIENT"
        )
        response = await self._llm_call(GAP_SYSTEM, prompt)
        return parse_gap_query(response)

    async def _answer_single(
        self,
        question: str,
        context_facts: list[Fact] | None = None,
        *,
        top_k: int | None = None,
    ) -> LibrarianAnswer:
        """Single-pass answer against the source material."""
        retrieved = self.retrieve(question, top_k=top_k)
        source_context = CHUNK_JOINER.join(c.text for c in retrieved)
        origin = _provenance(retrieved)

        system = ANSWER_SYSTEM.format(source=source_context)
        user = f"QUESTION: {question}"
        if context_facts:
            known = "\n".join(f"- {f.content}" for f in context_facts)
            user += f"\n\nALREADY KNOWN:\n{known}"

        self.check_context(system, user)

        logger.info("  LIBRARIAN answering: %s", question[:120])

        try:
            data = await self._llm_call_json(system, user)
            raw_items = data.get("items", []) if isinstance(data, dict) else data
            items = []
            for item in raw_items:
                if isinstance(item, dict):
                    items.append(Fact(content=item.get("content", str(item)), source=origin))
                elif isinstance(item, str) and item.strip():
                    items.append(Fact(content=item.strip(), source=origin))
            coverage = data.get("coverage", "complete") if isinstance(data, dict) else "complete"
        except Exception as e:
            logger.warning("  LIBRARIAN JSON failed, text fallback: %s", e)
            text = await self._llm_call(
                "You are a librarian. Answer the question using ONLY "
                "information from the source material below. List each "
                f"fact on its own line, prefixed with a dash.\n\nSOURCE MATERIAL:\n{source_context}",
                user,
            )
            items = [
                Fact(content=line.lstrip("- ").strip(), source=origin)
                for line in text.split("\n")
                if line.strip() and (line.strip()[0] in "-•" or len(line.strip()) > 10)
            ]
            coverage = "uncertain"

        logger.info("  LIBRARIAN returned %d facts (coverage: %s)", len(items), coverage)
        return LibrarianAnswer(items=items, coverage=coverage)
