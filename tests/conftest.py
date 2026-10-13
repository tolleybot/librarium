"""Shared fixtures — MockLLM keeps every test model-free and fast.

The other half of "model-free" lives here too. `MockLLM` covers the
*language* model; nothing covered the two `sentence_transformers` models
until now, and the suite quietly built 21 of them per run (16
`nomic-ai/nomic-embed-text-v1.5`, 5 `BAAI/bge-reranker-v2-m3`) while
the project's notes said it built none. On this machine they are cached and the
extra is installed, so the cost was invisible; on a clean core install it
is multiple GB over the network from a suite documented as needing none.

Two things fix that and keep it fixed:

* :func:`stub_models` — the supply side. Both loaders do
  ``from sentence_transformers import X`` *inside* the function that
  needs it, so a module in ``sys.modules`` is the entire seam. Same shape
  as the existing `test_stale_vectors.stub_cross_encoder` and
  `test_pinning.stub_sentence_transformers`, generalised so the embedder
  side has one too.
* :data:`MODEL_GUARD` — the enforcement side. Without it this is a
  one-time cleanup: nothing stops the next test that indexes without a
  double from silently reintroducing the download, because on a
  developer machine it just works.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import sys
import types

import numpy as np
import pytest


class MockLLM:
    """LLMClient that returns canned responses in order (last repeats)."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ["Mock response."])
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, think: bool = False,
                       temperature: float = 0.1) -> str:
        self.calls.append((system, user))
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]


@pytest.fixture
def mock_llm():
    return MockLLM()


# ── The model-free guarantee ─────────────────────────────────────


class ModelFreeViolation(BaseException):
    """A real model was about to be built inside the test suite.

    **Deliberately a `BaseException`, not an `Exception`.** Both loaders
    wrap their construction in `except Exception` and degrade —
    `_ensure_embed_model` sets ``_dense_ok = False`` and returns None,
    `_load_reranker` sets ``reranker = None`` — so an ordinary error
    raised here would be swallowed and the offending test would go green
    on the BM25-only path. Passing quietly on a degraded path is exactly
    the silence this guard exists to end, so the guard has to be
    something those handlers cannot catch.
    """


class _RealModelGuard:
    """Fail loudly if anything constructs a real model.

    Implemented as a `sys.meta_path` finder that wraps
    `sentence_transformers` *as it is imported* rather than importing it
    up front, for three reasons:

    * ``sys.modules`` is consulted before ``sys.meta_path``, so every
      test that injects a stub module — there are four such helpers
      already — bypasses this untouched, including the ones that inject
      ``None`` to simulate the extra being absent.
    * `_embeddings_available()` asks `importlib.util.find_spec`, which
      goes through here. Blocking the *import* would make it answer
      "extra not installed" on a machine where it is, changing what
      several tests exercise. Finding the spec and wrapping only the
      loader keeps that answer honest and still costs nothing: a spec
      lookup does not execute the module.
    * The suite must run on a core install, where the module does not
      exist at all. Then no spec is found, this never fires, and the
      degraded paths behave exactly as they document.
    """

    TARGET = "sentence_transformers"
    WRAPPED = ("SentenceTransformer", "CrossEncoder")

    def __init__(self):
        #: Every real construction that was blocked, as
        #: ``(class name, model name)``. Populated before the raise so a
        #: violation is still nameable if some frame swallows it.
        self.attempts: list[tuple[str, str]] = []

    # -- installation --------------------------------------------

    def install(self) -> None:
        # Once, whatever happens: two copies in `sys.meta_path` make
        # `_delegate` hand the lookup straight back to this class and
        # recurse until the stack runs out.
        if self in sys.meta_path:
            return
        sys.meta_path.insert(0, self)
        # Already imported (a plugin, an earlier session-scoped import):
        # the finder will never see it, so wrap what is there.
        if self.TARGET in sys.modules:
            module = sys.modules[self.TARGET]
            if module is not None:
                self.wrap(module)

    def uninstall(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)

    # -- the finder ----------------------------------------------

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.TARGET:
            return None
        spec = self._delegate(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module, _original=original):
            _original(module)
            self.wrap(module)

        # Set on the loader instance the import system is about to use;
        # the finders below hand back a fresh loader per lookup, so this
        # does not leak into anyone else's import.
        loader.exec_module = exec_module
        return spec

    def _delegate(self, fullname, path, target):
        """Whatever the *rest* of ``sys.meta_path`` would have found.

        Asking `importlib.util.find_spec` instead would re-enter this
        method; skipping ourselves in the walk keeps the recursion
        impossible rather than merely guarded.
        """
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                return spec
        return None

    # -- the block itself ----------------------------------------

    def wrap(self, module) -> None:
        """Replace the two model classes with something that refuses."""
        for name in self.WRAPPED:
            if getattr(module, name, None) is None:
                continue
            module.__dict__[name] = self._refuser(name)

    def _refuser(self, class_name: str):
        def refuse(model_name="<unnamed>", *args, **kwargs):
            self.attempts.append((class_name, str(model_name)))
            raise ModelFreeViolation(
                f"{class_name}({model_name!r}) — a real model would have "
                "been downloaded and built here. The suite is model-free "
                "(see tests/conftest.py); supply a double with "
                "tests.conftest.stub_models(monkeypatch, ...) instead."
            )
        return refuse


#: The live guard, so a test can assert on it. Module-level rather than a
#: fixture value because the interesting assertion — nothing was blocked
#: — is about the whole session, not about one test's fixtures.
MODEL_GUARD = _RealModelGuard()


@pytest.fixture(scope="session", autouse=True)
def _model_free_suite():
    MODEL_GUARD.install()
    yield MODEL_GUARD
    MODEL_GUARD.uninstall()
    # Belt and braces on the raise above: a `ModelFreeViolation` that
    # some frame swallowed anyway still fails the run here, at session
    # scope, where no ordering between tests can hide it.
    assert MODEL_GUARD.attempts == [], (
        "real models were constructed during the run: %r" % (
            MODEL_GUARD.attempts,)
    )
    # The guard swaps two names on the *top-level* module, so a future
    # `from sentence_transformers.cross_encoder import CrossEncoder` in
    # `source.py` would build a real model without ever touching them.
    # Closed from the other side instead: no double in this suite causes
    # a genuine import, and importing any submodule imports the package
    # first, so the name being here at session end means something really
    # loaded it — and therefore torch.
    assert "sentence_transformers" not in sys.modules, (
        "sentence_transformers was genuinely imported during the run — "
        "the guard only wraps the top-level module's two classes, so a "
        "submodule import slips past it"
    )


# ── Doubles for the two models ───────────────────────────────────


class WordHashEmbedder:
    """A `SentenceTransformer` stand-in whose vectors carry their text.

    Content-derived, per the suite's convention: each word lands on an axis picked by
    a hash of the word, with a magnitude drawn from the same digest, so
    two different chunks embed differently and by different amounts. A
    constant or length-derived vector would make every dense ranking a
    near-tie that fusion then resolves out of BM25 alone — a test built
    on one could not tell a live dense half from a dead one, which is
    precisely the difference the tests using this are relying on.
    """

    DIMS = 16

    def __init__(self, name: str = "stub/embed", **kwargs):
        #: The name it was asked to load, so a test can say which model
        #: the librarian went looking for.
        self.name = name
        self.kwargs = kwargs
        #: Every batch handed to encode, for tests that want to say what
        #: was embedded rather than only how much.
        self.batches: list[list[str]] = []

    def encode(self, texts, show_progress_bar: bool = False):
        texts = list(texts)
        self.batches.append(texts)
        rows = []
        for text in texts:
            row = np.zeros(self.DIMS)
            for word in text.lower().split():
                digest = hashlib.sha256(word.encode()).digest()
                row[digest[0] % self.DIMS] += 1.0 + digest[1] / 256.0
            rows.append(row)
        return np.asarray(rows, dtype="float32")


def stub_models(monkeypatch, *, embedder=None, reranker=None):
    """Put a stub ``sentence_transformers`` where the loaders import from.

    *embedder* / *reranker* are what the corresponding class hands back;
    each defaults to something that refuses, so a test that reaches a
    loader it did not mean to reach hears about it instead of quietly
    taking a degraded path. Returns the module, whose ``asked`` list
    records ``(class name, model name)`` for tests that care.

    ``monkeypatch.setitem`` rather than a plain assignment so the stub
    comes back out: a later test in the same session that wants the real
    module — or wants it genuinely absent — must not find ours sitting
    in ``sys.modules``.
    """
    module = types.ModuleType("sentence_transformers")
    #: `importlib.util.find_spec` *raises* `ValueError` on a module that
    #: is in `sys.modules` with a `__spec__` of None — which is what a
    #: bare `types.ModuleType` has. `embedding_backend_available()` is a
    #: plain `find_spec(...) is not None` with no guard, so without this
    #: line any path reaching it while the stub is installed dies with
    #: "sentence_transformers.__spec__ is None" instead of answering —
    #: and that helper is the one the guard's own error message sends
    #: every future test author towards. A stub is installed precisely to
    #: say "the extra is here", so the honest spec is one that resolves.
    #: `loader=None`: nothing may ever execute this module.
    module.__spec__ = importlib.machinery.ModuleSpec(
        "sentence_transformers", loader=None,
    )
    module.asked = []

    def factory(class_name, supplied, default):
        def build(model_name="<unnamed>", **kwargs):
            module.asked.append((class_name, model_name))
            if supplied is not None:
                return supplied
            return default(model_name, **kwargs)
        return build

    def no_reranker(model_name, **kwargs):
        raise ModelFreeViolation(
            f"CrossEncoder({model_name!r}) — this test supplied no reranker "
            "double. Pass one to stub_models(reranker=...)."
        )

    module.SentenceTransformer = factory(
        "SentenceTransformer", embedder, WordHashEmbedder,
    )
    module.CrossEncoder = factory("CrossEncoder", reranker, no_reranker)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module
