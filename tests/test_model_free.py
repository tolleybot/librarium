"""The suite is model-free, and stays model-free.

CLAUDE.md has said "no model, no network, no API key" since the project
started, and for five cycles it was false: a full run built **21 real
models** — 16 ``nomic-ai/nomic-embed-text-v1.5`` and 5
``BAAI/bge-reranker-v2-m3``. None of them was used for anything. The
reranker ones were constructed by `index()` and then overwritten by a
stub on the next line; the embedding ones came from librarians built
*inside* `LibrarianRouter.index()` and `WebLibrarian.answer()`, where
there is no handle to switch the dense half off before it runs.

Nothing local ever complains about that — the models are cached here and
the extra is installed — which is exactly why the doubles alone are not
enough. Without an enforcement the next test that indexes through a
lazily-built librarian silently puts the download back, and the only
person who finds out is someone running the suite on a clean core
install, from a network they may not have.

`tests.conftest._RealModelGuard` is the enforcement and this file is
what pins it. The guard's own value is the property in
:class:`TestTheRefusalCannotBeSwallowed`: both loaders catch
``Exception`` and degrade quietly, so a guard that raised anything
ordinary would be absorbed into the BM25-only path and the offending
test would go green — louder in the log, still silent in the result.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

import pytest

from librarium.source import SourceLibrarian, embedding_backend_available
from tests.conftest import (
    MODEL_GUARD,
    MockLLM,
    ModelFreeViolation,
    WordHashEmbedder,
    stub_models,
)


def wrapped_module() -> types.ModuleType:
    """A ``sentence_transformers`` that has been through the guard.

    Stands in for the real one having been imported: the guard wraps
    modules at import time, so this is the state a test that reached a
    real load would actually be in, without paying for torch.
    """
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = lambda name, **kwargs: object()
    module.CrossEncoder = lambda name, **kwargs: object()
    MODEL_GUARD.wrap(module)
    return module


class FakeLoader:
    """A loader that defines the two classes, as the real one does.

    Stands in for `sentence_transformers`' own loader so the guard's
    wrapping path can be driven without importing torch: the only thing
    about the real loader that matters here is that `exec_module` leaves
    the two model classes on the module.
    """

    def __init__(self):
        #: Modules this was asked to execute, so a test can say the guard
        #: still ran the original rather than replacing it.
        self.executed: list[types.ModuleType] = []

    def exec_module(self, module) -> None:
        self.executed.append(module)
        module.SentenceTransformer = lambda name, **kwargs: object()
        module.CrossEncoder = lambda name, **kwargs: object()


class FakeFinder:
    """Whatever `_delegate` would otherwise have found, made local.

    Answers only for the guard's target, like every real finder answers
    only for what it can supply.
    """

    def __init__(self, loader):
        self.loader = loader

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sentence_transformers":
            return None
        return importlib.machinery.ModuleSpec(fullname, self.loader)


@pytest.fixture
def ahead_of_the_real_finders():
    """Put a `FakeFinder` at the front of ``sys.meta_path``, and take it out.

    At index 0 rather than after the guard so the delegation cannot fall
    through to the real `sentence_transformers` on a machine that has it
    — the point of these tests is that the wrapping path runs with no
    torch anywhere.
    """
    installed = []

    def install(loader):
        finder = FakeFinder(loader)
        sys.meta_path.insert(0, finder)
        installed.append(finder)
        return finder

    yield install
    for finder in installed:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


@pytest.fixture
def deliberate_violation():
    """Trip the guard on purpose, and take the record back out again.

    Yields the length `MODEL_GUARD.attempts` had on the way in, so a test
    can assert on what it added. The teardown truncates back to exactly
    that, never clearing the whole list: a blunt clear here would swallow
    a *real* violation recorded by an earlier test, which is the one
    thing the session-scoped assertion in `conftest.py` exists to catch.
    """
    before = len(MODEL_GUARD.attempts)
    yield before
    del MODEL_GUARD.attempts[before:]


class TestNothingBuildsARealModel:
    def test_the_guard_is_installed_for_the_whole_session(self):
        # The autouse session fixture is the only thing putting it there,
        # so this catches that fixture being renamed out of autouse — the
        # quiet way the whole enforcement would stop existing while every
        # other test kept passing.
        assert MODEL_GUARD in sys.meta_path

    def test_nothing_has_built_one_so_far(self):
        # The count that Part A is judged against, stated in the suite
        # rather than only in a report. It is a running total, so a test
        # ordered after this one is covered by the same assertion
        # repeated in the session fixture's teardown, not by this line.
        assert MODEL_GUARD.attempts == []

    def test_a_real_construction_is_refused_and_named(self,
                                                      deliberate_violation):
        # Catches the guard being installed but toothless. The model name
        # is in the message because the first question about a violation
        # is which of the two models it was.
        module = wrapped_module()

        with pytest.raises(ModelFreeViolation, match="nomic-embed"):
            module.SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
        with pytest.raises(ModelFreeViolation, match="bge-reranker"):
            module.CrossEncoder("BAAI/bge-reranker-v2-m3")

    def test_the_refusal_is_recorded_as_well_as_raised(self,
                                                       deliberate_violation):
        # So a violation is still nameable at session end even if some
        # frame between here and pytest swallows it.
        module = wrapped_module()

        with pytest.raises(ModelFreeViolation):
            module.CrossEncoder("BAAI/bge-reranker-v2-m3")

        assert MODEL_GUARD.attempts[deliberate_violation:] == [
            ("CrossEncoder", "BAAI/bge-reranker-v2-m3")
        ]

    def test_it_intercepts_nothing_but_sentence_transformers(self):
        # A finder sitting at the front of sys.meta_path is in the path of
        # every import in the process. Catches the name test being
        # loosened to a prefix or dropped, which would route the whole
        # standard library through this class.
        assert MODEL_GUARD.find_spec("json") is None
        assert MODEL_GUARD.find_spec("librarium.source") is None


class TestTheGuardReachesTheModuleWhicheverWayItArrives:
    """The wiring everything else here rests on.

    `wrapped_module()` calls `MODEL_GUARD.wrap` by hand, so the tests
    above say what happens *once* the guard has hold of the module. These
    say it gets hold of it: the two routes are "already in `sys.modules`
    when the guard is installed" and "arrives later through the finder".
    Both were previously demonstrated only by mutating the tree.
    """

    def test_a_module_imported_before_the_guard_is_wrapped_on_install(
        self, monkeypatch, deliberate_violation
    ):
        # `sys.modules` is consulted before `sys.meta_path`, so a module
        # imported during startup — a pytest plugin, a `-p` module —
        # never reaches the finder at all. Catches `install`'s
        # already-imported branch being dropped, which would leave the
        # guard installed and silently blind on exactly the machines
        # where something else pulls the extra in first.
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = lambda name, **kwargs: object()
        module.CrossEncoder = lambda name, **kwargs: object()
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

        MODEL_GUARD.uninstall()
        try:
            MODEL_GUARD.install()

            with pytest.raises(ModelFreeViolation, match="nomic-embed"):
                module.SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
            with pytest.raises(ModelFreeViolation, match="bge-reranker"):
                module.CrossEncoder("BAAI/bge-reranker-v2-m3")
        finally:
            # Idempotent, so this restores the session's guard whether
            # the body got that far or not.
            MODEL_GUARD.install()

    def test_the_finder_wraps_the_module_its_loader_built(
        self, ahead_of_the_real_finders, deliberate_violation
    ):
        # The normal route. The guard has to leave the real loader's work
        # intact and swap the classes *after* it, not instead of it:
        # replacing `exec_module` outright would hand back a module with
        # nothing else on it, and refusing the import outright would make
        # the availability probe lie. Catches either, and catches the
        # wrap being dropped from the wrapper.
        loader = FakeLoader()
        ahead_of_the_real_finders(loader)

        spec = MODEL_GUARD.find_spec("sentence_transformers")
        assert spec is not None
        module = types.ModuleType("sentence_transformers")
        spec.loader.exec_module(module)

        # The original ran ...
        assert loader.executed == [module]
        # ... and what it defined was replaced on the way out.
        with pytest.raises(ModelFreeViolation, match="nomic-embed"):
            module.SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
        with pytest.raises(ModelFreeViolation, match="bge-reranker"):
            module.CrossEncoder("BAAI/bge-reranker-v2-m3")

    def test_a_spec_with_no_loader_is_declined(self,
                                               ahead_of_the_real_finders):
        # A namespace package's spec carries `loader is None`. Without
        # the guard's own check, `loader.exec_module` is an
        # `AttributeError` on None and every import of the name dies
        # inside the guard instead of falling through to the next finder.
        ahead_of_the_real_finders(None)

        assert MODEL_GUARD.find_spec("sentence_transformers") is None


class TestTheStubIsAWellFormedModule:
    """`stub_models` is what the guard's error message tells the next
    author to reach for, so it has to survive being asked the ordinary
    questions the import system asks about a module."""

    def test_the_availability_probe_answers_with_the_stub_installed(
        self, monkeypatch
    ):
        # `find_spec` *raises* on a module in `sys.modules` whose
        # `__spec__` is None, which is what a bare `types.ModuleType`
        # has, and `embedding_backend_available()` is an unguarded
        # `find_spec(...) is not None`. Catches the spec being dropped
        # from `stub_models`, which turns `load()`'s BM25-only branch —
        # and anything else asking the question — into
        # `ValueError: sentence_transformers.__spec__ is None`.
        stub_models(monkeypatch)

        assert embedding_backend_available() is True


class TestTheRefusalCannotBeSwallowed:
    """The property that makes the guard worth having.

    `_ensure_embed_model` catches ``Exception`` and sets ``_dense_ok =
    False``; `_load_reranker` catches ``Exception`` and sets ``reranker =
    None``. Both then return normally. A guard raising anything ordinary
    would be caught by exactly the handler it is trying to report on, and
    the test that triggered it would pass on the degraded path.
    """

    def test_the_embedding_loader_cannot_degrade_past_it(
        self, monkeypatch, deliberate_violation
    ):
        monkeypatch.setitem(sys.modules, "sentence_transformers",
                            wrapped_module())
        lib = SourceLibrarian(MockLLM())

        with pytest.raises(ModelFreeViolation):
            lib._ensure_embed_model()

        # It really did reach the `except Exception` handler's territory:
        # had the guard raised an Exception, this is the state the test
        # would have gone green in.
        assert lib.embed_model is None

    def test_the_reranker_loader_cannot_degrade_past_it(
        self, monkeypatch, deliberate_violation
    ):
        monkeypatch.setitem(sys.modules, "sentence_transformers",
                            wrapped_module())
        lib = SourceLibrarian(MockLLM(), use_reranker=True, top_k=3)

        with pytest.raises(ModelFreeViolation):
            lib._load_reranker()

        assert lib.reranker is None

    def test_strict_is_not_what_makes_it_escape(self, monkeypatch,
                                                deliberate_violation):
        # The default librarian, not a strict one: a violation has to be
        # loud in the configuration the whole suite actually uses.
        monkeypatch.setitem(sys.modules, "sentence_transformers",
                            wrapped_module())
        lib = SourceLibrarian(MockLLM(), use_reranker=True, top_k=3,
                              strict=False)

        with pytest.raises(ModelFreeViolation):
            lib._load_reranker()


class TestTheGuardChangesNothingElse:
    """It sits in front of every import in the process, so what it does
    *not* do is as much the subject as what it does."""

    def test_a_stub_in_sys_modules_reaches_the_loader_untouched(
        self, monkeypatch
    ):
        # sys.modules is consulted before sys.meta_path, which is the
        # whole reason four separate stubbing helpers in this suite keep
        # working. Catches the guard being reimplemented as an import
        # block, which would make every one of them unreachable.
        stub_models(monkeypatch)
        lib = SourceLibrarian(MockLLM(), embed_model="acme/embed-v9")

        model = lib._ensure_embed_model()

        assert isinstance(model, WordHashEmbedder)
        assert model.name == "acme/embed-v9"

    def test_the_availability_probe_answers_the_same_either_way(self):
        # `embedding_backend_available()` asks find_spec, which goes through
        # the guard. Blocking the import there would make it answer
        # "extra not installed" on a machine where it is, silently moving
        # several tests onto the degraded path they are not about. The
        # comparison is against the guard removed, so this holds on a
        # core install and on this one alike.
        with_guard = embedding_backend_available()

        MODEL_GUARD.uninstall()
        try:
            without_guard = (
                importlib.util.find_spec("sentence_transformers") is not None
            )
        finally:
            MODEL_GUARD.install()

        assert with_guard is without_guard

    def test_the_probe_does_not_import_the_module(self):
        # Finding a spec must not execute it: sentence_transformers pulls
        # torch, which is seconds of wall clock, and `test_dense_honesty`
        # asserts elsewhere that neither is imported on the BM25-only
        # path. Catches the guard being written as "import it, then wrap
        # it", which is the obvious implementation and the wrong one.
        already = "sentence_transformers" in sys.modules

        embedding_backend_available()

        assert ("sentence_transformers" in sys.modules) is already


class TestTheDoublesAreContentDerived:
    """CLAUDE.md's rule for test doubles, applied to the one this cycle
    added: a constant vector would let a test pass with the dense half
    ranking on nothing."""

    def test_different_texts_get_different_vectors(self):
        rows = WordHashEmbedder().encode(
            ["motor vehicle deductible", "dentist cleaning hygienist"]
        )
        assert not (rows[0] == rows[1]).all()

    def test_the_same_text_gets_the_same_vector(self):
        # Determinism across instances, which is what lets a saved index
        # be loaded back into a fresh librarian without the embedding
        # fingerprint reporting drift.
        first = WordHashEmbedder().encode(["motor vehicle deductible"])
        second = WordHashEmbedder().encode(["motor vehicle deductible"])
        assert (first == second).all()

    def test_a_missing_reranker_double_is_refused_not_ignored(self,
                                                              monkeypatch):
        # `stub_models` without a reranker means "this test does not
        # rerank". If it quietly handed back an object, a test that did
        # rerank would silently rank on nothing; if it raised an ordinary
        # error, `_load_reranker` would swallow it.
        stub_models(monkeypatch)
        lib = SourceLibrarian(MockLLM(), use_reranker=True, top_k=3)

        with pytest.raises(ModelFreeViolation, match="no reranker double"):
            lib._load_reranker()
