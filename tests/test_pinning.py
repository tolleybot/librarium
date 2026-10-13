"""The revision pins on the two default models, and the drift warning.

Part A of this change established — by running it, not by reading it —
that the default embedding model's *executed code* cannot be pinned
through sentence-transformers' public API: ``nomic-embed-text-v1.5``'s
``auto_map`` names a class in a **different** repo with no revision, and
sentence-transformers 5.7.0 pops ``model_kwargs["code_revision"]`` in
``base/model.py:1371`` to resolve the modules.json class ref and never
forwards it to the model loader. So no sham pin was shipped. What did
ship is three things, and this file covers all three:

* ``pinned_revision`` — a pin that applies to the two models we verified
  and to **nothing else**. The negative half is the important half: a
  nomic sha handed to a user's own ``embed_model`` resolves to nothing
  on their repo, or to a same-named branch of something unrelated. That
  is a fatal load, not a degraded one.
* The loader wiring — that the revision actually reaches
  ``SentenceTransformer`` and ``CrossEncoder``, and that the deliberate
  asymmetry between them holds: the embed model runs remote code and the
  reranker does not.
* ``_check_code_revision`` — the only genuinely new runtime behaviour.
  Detection where prevention was impossible: it warns when the sha that
  actually executed is not the one the ``embeddings`` extra was verified
  against.

**Nothing here loads a model or touches the network.** The doubles are
content-derived per CLAUDE.md: a fake model carries the sha it was built
for *inside its module path*, exactly where transformers puts the real
one, so a warning that reports the wrong sha fails an assertion rather
than passing on a constant. The module path shape was taken from the
real cached class rather than guessed — ``NomicBertModel.__module__`` is
``transformers_modules.nomic_hyphen_ai.nomic_hyphen_bert_hyphen_2048.<40
hex>.modeling_hf_nomic_bert``.
"""

from __future__ import annotations

import re
import sys
import types

import pytest

from librarium.source import (
    EMBED_CODE_REPO,
    EMBED_CODE_REVISION,
    EMBED_MODEL,
    EMBED_REVISION,
    RERANK_MODEL,
    RERANK_REVISION,
    SourceLibrarian,
    _check_code_revision,
    pinned_revision,
)
from tests.conftest import MockLLM

#: A model name nobody pinned. Realistic rather than nonsense: the point
#: of the guard is the user who legitimately swaps the default out.
OTHER_EMBED = "acme/embed-v9"
OTHER_RERANK = "acme/rerank-v9"


# ── Doubles ──────────────────────────────────────────────────────


def remote_code_model(sha: str) -> list:
    """A stand-in for a loaded SentenceTransformer running remote code.

    Identity-carrying on purpose: the *only* place ``sha`` appears is
    the module path of the class, which is where transformers records
    the commit a dynamically-loaded module came from. So a check that
    reports a hard-coded sha, or reads the wrong path component, cannot
    pass — it has to recover this exact one.
    """
    cls = type(
        "NomicBertModel",
        (),
        {
            "__module__": "transformers_modules.nomic_hyphen_ai"
            f".nomic_hyphen_bert_hyphen_2048.{sha}.modeling_hf_nomic_bert"
        },
    )
    transformer_module = types.SimpleNamespace(auto_model=cls())
    # SentenceTransformer is a Sequential; module 0 is the Transformer.
    return [transformer_module, object()]


def local_code_model() -> list:
    """A model whose class comes from installed code, not a hub snapshot."""
    cls = type(
        "XLMRobertaModel",
        (),
        {"__module__": "transformers.models.xlm_roberta.modeling_xlm_roberta"},
    )
    return [types.SimpleNamespace(auto_model=cls()), object()]


def stub_sentence_transformers(monkeypatch) -> dict:
    """Record every argument each loader was called with.

    The existing stub in ``test_settings.py`` records only the model
    name, which is what it was written for. This one keeps the kwargs,
    because the whole subject here is a kwarg.
    """
    calls: dict[str, dict] = {}
    module = types.ModuleType("sentence_transformers")

    def make(kind, result):
        def factory(name, **kwargs):
            calls[kind] = {"name": name, **kwargs}
            return result
        return factory

    module.SentenceTransformer = make("SentenceTransformer", object())
    module.CrossEncoder = make("CrossEncoder", object())
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


# ── pinned_revision ──────────────────────────────────────────────


class TestPinnedRevision:
    """The lookup that decides whether a pin exists at all."""

    def test_the_default_embedding_model_is_pinned(self):
        # Catches: the embed entry dropped from the table, or wired to
        # the reranker's sha — either way the default model would load
        # from a revision nobody verified the `embeddings` extra against.
        assert pinned_revision(EMBED_MODEL) == EMBED_REVISION

    def test_the_default_reranker_is_pinned(self):
        # Catches: the same for the reranker, whose pin is the *whole*
        # one — it has no auto_map, so `revision` covers its weights too.
        assert pinned_revision(RERANK_MODEL) == RERANK_REVISION

    def test_the_two_defaults_are_not_pinned_to_the_same_revision(self):
        # Catches a copy-paste of one sha into both entries, which the
        # two tests above would each still pass.
        assert EMBED_REVISION != RERANK_REVISION

    @pytest.mark.parametrize("name", [EMBED_MODEL, RERANK_MODEL])
    def test_a_pin_is_an_immutable_commit_not_a_branch(self, name):
        # The spec's central rule: a pin that appears to work and does
        # not is worse than none. "main" or "v1.5" would satisfy every
        # equality assertion in this class while pinning nothing at all,
        # so the *shape* is asserted separately from the value.
        assert re.fullmatch(r"[0-9a-f]{40}", pinned_revision(name))

    @pytest.mark.parametrize("name", [OTHER_EMBED, OTHER_RERANK])
    def test_a_user_supplied_model_gets_no_revision(self, name):
        # The guard the pin exists behind. Catches a lookup that falls
        # back to a default: nomic's sha against someone else's repo
        # resolves to nothing, or to a same-named branch of something
        # unrelated. Not degraded retrieval — a failed load.
        assert pinned_revision(name) is None

    def test_the_code_repo_is_not_itself_a_pinned_model(self):
        # `nomic-bert-2048` is where the executed code lives, not
        # something either loader is ever handed. Catches the code
        # revision being added to the model table by association.
        assert pinned_revision(EMBED_CODE_REPO) is None

    @pytest.mark.parametrize(
        "name",
        [
            "nomic-ai/nomic-embed-text-v1",  # a real, different model
            "nomic-ai/nomic-embed-text-v1.5-Q",
            "NOMIC-AI/NOMIC-EMBED-TEXT-V1.5",
            "hf.co/" + EMBED_MODEL,
            EMBED_MODEL + "/",
        ],
    )
    def test_a_near_miss_name_is_not_pinned(self, name):
        # Catches a substring or prefix match replacing exact lookup —
        # `startswith`/`in` would hand nomic's v1.5 sha to v1, to a
        # quantised fork, and to anything with the name embedded in it.
        assert pinned_revision(name) is None


# ── The loader wiring ────────────────────────────────────────────


class TestTheRevisionReachesTheLoader:
    """A pin nothing passes is a comment. Asserted against a stubbed
    loader, so no model is downloaded or constructed.
    """

    def test_the_default_embedding_model_loads_at_its_pinned_revision(
        self, monkeypatch
    ):
        # Catches `revision=` being dropped from the SentenceTransformer
        # call, or passed the wrong constant.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(MockLLM())._ensure_embed_model()
        assert calls["SentenceTransformer"]["name"] == EMBED_MODEL
        assert calls["SentenceTransformer"]["revision"] == EMBED_REVISION

    def test_a_user_supplied_embedding_model_loads_unpinned(self, monkeypatch):
        # Catches the pin leaking onto a model it was not measured
        # against — the failure mode that is fatal rather than merely
        # wrong. `None` is SentenceTransformer's own default, so such a
        # model loads exactly as it did before pinning existed.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(MockLLM(), embed_model=OTHER_EMBED)._ensure_embed_model()
        assert calls["SentenceTransformer"]["name"] == OTHER_EMBED
        assert calls["SentenceTransformer"]["revision"] is None

    def test_a_model_with_remote_code_still_gets_trust_remote_code(
        self, monkeypatch
    ):
        """nomic ships its model class in a foreign repo and cannot load
        without the flag. The default no longer does, so this pins the
        half that must not regress: the flag follows the *model*, not the
        fact of being the default."""
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(
            MockLLM(), embed_model="nomic-ai/nomic-embed-text-v1.5"
        )._ensure_embed_model()
        assert calls["SentenceTransformer"]["trust_remote_code"] is True

    def test_the_default_embedding_model_does_not_get_trust_remote_code(
        self, monkeypatch
    ):
        # Adding a kwarg is where a neighbouring one gets lost.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(MockLLM())._ensure_embed_model()
        # Conditional now, and deliberately: bge-m3 is a plain
        # XLM-RoBERTa with no remote code, and passing trust_remote_code
        # unconditionally asks a user to trust arbitrary code from
        # whatever repo `embed_model` names. nomic ships remote code and
        # still gets it.
        assert calls["SentenceTransformer"]["trust_remote_code"] is False

    def test_no_code_revision_is_passed(self, monkeypatch):
        # Part A measured that sentence-transformers 5.7.0 pops
        # `code_revision` out of model_kwargs to resolve the modules.json
        # class ref and never forwards it to the model loader. Passing it
        # would be a pin that appears to work and does not. Catches
        # someone reinstating it from the docstring without re-measuring.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(MockLLM())._ensure_embed_model()
        call = calls["SentenceTransformer"]
        assert "code_revision" not in call
        assert "code_revision" not in (call.get("model_kwargs") or {})

    def test_the_default_reranker_loads_at_its_pinned_revision(self, monkeypatch):
        # Catches `revision=` being dropped from the CrossEncoder call.
        # This one is the pin that actually holds end to end, so losing
        # it loses real protection rather than a partial one.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(MockLLM(), use_reranker=True)._load_reranker()
        assert calls["CrossEncoder"]["name"] == RERANK_MODEL
        assert calls["CrossEncoder"]["revision"] == RERANK_REVISION

    def test_a_user_supplied_reranker_loads_unpinned(self, monkeypatch):
        # The same fatal-mispin guard on the second call site. It is a
        # separate call, so it can regress on its own.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(
            MockLLM(), use_reranker=True, rerank_model=OTHER_RERANK
        )._load_reranker()
        assert calls["CrossEncoder"]["name"] == OTHER_RERANK
        assert calls["CrossEncoder"]["revision"] is None

    def test_the_reranker_is_not_loaded_with_trust_remote_code(self, monkeypatch):
        # The asymmetry is deliberate and load-bearing: bge-reranker-v2-m3
        # declares no auto_map, so it needs no remote code and `revision`
        # covers every file it loads. Catches the two loaders being
        # "harmonised" into executing third-party code for the reranker
        # too — which would silently widen this library's remote-code
        # surface to a repo nobody checked for it.
        calls = stub_sentence_transformers(monkeypatch)
        SourceLibrarian(MockLLM(), use_reranker=True)._load_reranker()
        assert "trust_remote_code" not in calls["CrossEncoder"]


# ── _check_code_revision ─────────────────────────────────────────


class TestTheCodeDriftWarning:
    """Detection standing in for a pin that cannot exist.

    All four branches, because the two silent ones are not decoration:
    a check that warns on a local model, or raises on a layout it does
    not recognise, would turn an advisory into a defect on paths that
    are working perfectly.
    """

    DRIFTED = "0123456789abcdef0123456789abcdef01234567"

    def test_it_warns_when_the_executed_code_is_not_the_verified_revision(
        self, caplog
    ):
        # The whole point of the function. Catches it going silent — the
        # exact silence that let a clean install degrade to BM25-only
        # when nomic's imports outgrew the `embeddings` extra.
        with caplog.at_level("WARNING", logger="librarium.source"):
            _check_code_revision(
                remote_code_model(self.DRIFTED),
                EMBED_CODE_REVISION,
                model_name=EMBED_MODEL,
                code_repo=EMBED_CODE_REPO,
            )
        assert self.DRIFTED in caplog.text

    def test_the_warning_names_the_verified_revision_and_the_code_repo(
        self, caplog
    ):
        # A warning that says only "upstream moved" leaves the reader
        # with no way to diff anything. Catches the message losing the
        # two identifiers that make it actionable.
        with caplog.at_level("WARNING", logger="librarium.source"):
            _check_code_revision(
                remote_code_model(self.DRIFTED),
                EMBED_CODE_REVISION,
                model_name=EMBED_MODEL,
                code_repo=EMBED_CODE_REPO,
            )
        assert EMBED_CODE_REVISION in caplog.text
        assert EMBED_CODE_REPO in caplog.text

    def test_the_warning_names_the_model_and_repo_it_was_given(self, caplog):
        # A sha is not an identity: the message is only actionable if it
        # names the repo the reader should go and diff. Hard-coded, the
        # names described nomic no matter whose `expected` was passed, so
        # a second caller's drift warning sent the reader to a repo that
        # never contained the sha being reported — worse than silence,
        # because it reads as a specific instruction. Catches the message
        # going back to the module constants: nomic's names must not
        # appear at all when someone else's were supplied.
        with caplog.at_level("WARNING", logger="librarium.source"):
            _check_code_revision(
                remote_code_model(self.DRIFTED),
                EMBED_CODE_REVISION,
                model_name=OTHER_EMBED,
                code_repo="acme/embed-code",
            )
        assert OTHER_EMBED in caplog.text
        assert "acme/embed-code" in caplog.text
        assert EMBED_MODEL not in caplog.text
        assert EMBED_CODE_REPO not in caplog.text

    def test_it_is_silent_when_the_executed_code_is_the_verified_revision(
        self, caplog
    ):
        # Catches an inverted comparison, and catches a warning that
        # fires unconditionally — either would train users to ignore it,
        # which costs the signal exactly when it matters.
        with caplog.at_level("WARNING", logger="librarium.source"):
            _check_code_revision(
                remote_code_model(EMBED_CODE_REVISION),
                EMBED_CODE_REVISION,
                model_name=EMBED_MODEL,
                code_repo=EMBED_CODE_REPO,
            )
        assert caplog.text == ""

    def test_it_is_silent_for_a_model_that_runs_no_remote_code(self, caplog):
        # No hub snapshot in the module path means either a local
        # directory or a model that stopped needing trust_remote_code.
        # Neither is drift. Catches a check that treats "no sha found"
        # as a mismatch and warns on every well-behaved model.
        with caplog.at_level("WARNING", logger="librarium.source"):
            _check_code_revision(
                local_code_model(),
                EMBED_CODE_REVISION,
                model_name=EMBED_MODEL,
                code_repo=EMBED_CODE_REPO,
            )
        assert caplog.text == ""

    @pytest.mark.parametrize(
        "model",
        [
            object(),                       # not subscriptable at all
            [],                             # empty Sequential
            [object()],                     # module 0 has no auto_model
            [types.SimpleNamespace(auto_model=None)],
        ],
        ids=["not-a-sequence", "empty", "no-auto-model", "auto-model-none"],
    )
    def test_an_unrecognised_layout_is_silent_and_does_not_raise(
        self, model, caplog
    ):
        # This runs *after* a successful load, so anything it raises
        # turns a working dense stack into a hard failure over an
        # advisory. Catches the try/except being narrowed or dropped
        # when sentence-transformers changes its module layout.
        with caplog.at_level("WARNING", logger="librarium.source"):
            _check_code_revision(
                model,
                EMBED_CODE_REVISION,
                model_name=EMBED_MODEL,
                code_repo=EMBED_CODE_REPO,
            )
        assert caplog.text == ""


class TestTheDriftWarningIsWiredIntoTheLoader:
    """The check has to run where a real load happens, and only for the
    model whose code revision was actually verified.
    """

    DRIFTED = "0123456789abcdef0123456789abcdef01234567"

    def _stub_returning(self, monkeypatch, model):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = lambda name, **kw: model
        module.CrossEncoder = lambda name, **kw: object()
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    NOMIC = "nomic-ai/nomic-embed-text-v1.5"

    def test_loading_a_remote_code_model_reports_drift(self, monkeypatch, caplog):
        # Catches the check being defined and never called — the state
        # the function was in before this test existed had zero callers'
        # worth of assertion behind it.
        #
        # Pointed at nomic explicitly rather than at the default. The
        # check is gated on the model having remote code, not on it being
        # the default, and the default is now bge-m3 — a plain
        # XLM-RoBERTa with no remote modeling file, so there is nothing
        # for a code revision to drift from. Using the default here would
        # assert the warning fires for a model the check deliberately
        # skips.
        self._stub_returning(monkeypatch, remote_code_model(self.DRIFTED))
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM(), embed_model=self.NOMIC)._ensure_embed_model()
        assert self.DRIFTED in caplog.text
        # The loader now names the model and repo explicitly rather than
        # letting the check fall back to constants, so the call site can
        # regress on its own: point `code_repo` at the model repo instead
        # of the code repo and the message sends the reader somewhere the
        # sha has never existed. The warning has to carry both.
        assert self.NOMIC in caplog.text
        assert EMBED_CODE_REPO in caplog.text

    def test_loading_a_user_supplied_model_reports_nothing(
        self, monkeypatch, caplog
    ):
        # Only the default model has a verified code revision to compare
        # against; another model's remote code is the user's to track.
        # Catches the `== EMBED_MODEL` guard being removed, which would
        # warn every user of a custom model that their model is "not the
        # verified nomic revision" — which it never was.
        self._stub_returning(monkeypatch, remote_code_model(self.DRIFTED))
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(
                MockLLM(), embed_model=OTHER_EMBED
            )._ensure_embed_model()
        assert caplog.text == ""

    def test_a_matching_load_of_the_default_model_is_quiet(
        self, monkeypatch, caplog
    ):
        # The happy path a user on the verified revision actually takes.
        self._stub_returning(monkeypatch, remote_code_model(EMBED_CODE_REVISION))
        with caplog.at_level("WARNING", logger="librarium.source"):
            SourceLibrarian(MockLLM())._ensure_embed_model()
        assert caplog.text == ""
