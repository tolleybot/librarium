"""Makes ``tests`` a real package, so ``conftest.py`` is imported once.

Without this file pytest loads the conftest as the top-level module
``conftest`` while every test module that says ``from tests.conftest
import MockLLM`` loads it *again* as ``tests.conftest`` — two module
objects, two copies of everything defined at module level.

That was harmless while the conftest held only `MockLLM`, where nothing
compares identities. It stops being harmless the moment anything there
is a singleton or an exception class: the model-free guard installed by
the session fixture would be one object and the one a test asserts on
would be another, a second guard in ``sys.meta_path`` recurses through
the first's delegation, and `pytest.raises(ModelFreeViolation)` catches
nothing because the class raised is the other copy of the same name.
"""
