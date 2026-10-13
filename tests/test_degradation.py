"""What happens when the environment is missing pieces.

Every path here is a fallback that only fires on a machine unlike the
one the tests usually run on — no NLTK corpora, no network, a read-only
data directory. They are easy to break without noticing precisely
because the happy path never touches them.

Nothing in this file reaches the network: downloads are either
forbidden via LIBRARIUM_OFFLINE or stubbed out.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import nltk
import pytest

from librarium import KLineStore, klines
from librarium import chunking


def _raise_lookup(*args, **kwargs):
    raise LookupError("Resource averaged_perceptron_tagger_eng not found.")


@pytest.fixture
def store(tmp_path):
    return KLineStore(db_path=str(tmp_path / "klines.db"))


# ── NLTK data ────────────────────────────────────────────────────

AIRGAP = """
import os, sys
import nltk
nltk.data.path = [os.environ["EMPTY_NLTK"]]     # no corpora anywhere

import librarium                                 # used to raise LookupError
from librarium.chunking import chunk_text, tokenize

tokens = tokenize("The bakeries were running quickly")
assert tokens, "tokenize returned nothing"
assert "bakeri" in tokens, tokens               # stemming still works
assert "the" in tokens, tokens                  # stopwords cannot be removed

chunks = chunk_text("The bakery opened in 1987. " * 300, chunk_size=1000)
assert chunks and all(len(c) <= 1000 for c in chunks), "chunking broke"
print("AIRGAP-OK")
"""


# The other half of the airgap: the network is *available*, the download
# reports success, and the corpus still is not on the search path. That
# happens whenever the download directory and nltk.data.path are
# configured independently — a writable NLTK_DATA nltk.download does not
# write to, a read-only HOME, a mirror returning a truncated archive.
# _ensure_nltk() returns the truthiness of nltk.download(), so it cannot
# tell the difference; the loaders have to survive being told "yes" and
# then failing anyway.
#
# nltk.data.path is overwritten wholesale rather than steered with
# NLTK_DATA: the env var only *prepends* to the default list, so
# ~/nltk_data stays searchable and on any machine that has ever used
# NLTK the corpora are still found and nothing is being tested.
DOWNLOAD_OFF_PATH = """
import os, sys
import nltk
nltk.data.path[:] = [os.environ["EMPTY_NLTK"]]  # no corpora anywhere

attempted = []
nltk.download = lambda pkg, **kw: attempted.append(pkg) or True

from librarium.chunking import tokenize

tokens = tokenize("The bakery sells 120 loaves")
assert tokens, "tokenize returned nothing"
assert "bakeri" in tokens, tokens               # stemming still works
assert "the" in tokens, tokens                  # stopwords cannot be removed
assert "stopwords" in attempted, attempted      # the download was attempted
print("OFF-PATH-OK")
"""

# The same conditions reached through the front door, because a
# LookupError out of tokenize() surfaces from index() — the difference
# between an index that is slightly worse and no index at all.
INDEX_OFF_PATH = """
import asyncio, os
import nltk
nltk.data.path[:] = [os.environ["EMPTY_NLTK"]]
nltk.download = lambda pkg, **kw: True

from librarium import Librarian


class StubLLM:
    async def complete(self, system, user, *, think=False, temperature=0.1):
        return "A context prefix."


lib = Librarian(StubLLM())
lib.source._dense_ok = False        # BM25 only — no embedding model to fetch
asyncio.run(lib.index("The bakery opened in 1987 and sells bread. " * 200))

assert lib.retrieval_mode == "bm25-only", lib.retrieval_mode
assert lib.source.chunks, "nothing was indexed"
assert lib.retrieve("bakery"), "retrieval returned nothing"
print("INDEX-OK")
"""


class TestWithoutCorpora:
    @staticmethod
    def _airgapped(script: str, empty_dir, **env):
        """Run *script* in a clean interpreter with no corpora on disk.

        A subprocess because NLTK caches its loaders on first use — in
        process, whatever an earlier test resolved stays resolvable and
        the fallback is never reached. It also keeps the mutilated
        nltk.data.path out of the rest of the suite.
        """
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "EMPTY_NLTK": str(empty_dir), **env},
        )

    @pytest.fixture
    def empty_nltk(self, tmp_path):
        empty = tmp_path / "nltk-data"
        empty.mkdir()
        return empty

    def test_a_download_that_lands_off_the_path_falls_back(self, empty_nltk):
        """Download says yes, the corpus is still missing — regex, not a raise."""
        result = self._airgapped(
            DOWNLOAD_OFF_PATH, empty_nltk, LIBRARIUM_OFFLINE="",
        )
        assert result.returncode == 0, result.stderr
        assert "OFF-PATH-OK" in result.stdout

    def test_indexing_survives_a_download_that_lands_off_the_path(self, empty_nltk):
        result = self._airgapped(
            INDEX_OFF_PATH, empty_nltk, LIBRARIUM_OFFLINE="",
        )
        assert result.returncode == 0, result.stderr
        assert "INDEX-OK" in result.stdout

    def test_stopwords_give_up_after_one_failed_retry(self, monkeypatch):
        """The loader's own shape, without paying for a subprocess.

        A corpus that raises however often it is asked must leave an
        empty set behind, not a LookupError, even when the download
        claims to have fixed it.
        """
        class Missing:
            @staticmethod
            def words(lang):
                raise LookupError(f"Resource {lang!r} not found.")

        attempts = []
        monkeypatch.setattr(chunking, "_stopword_cache", None)
        monkeypatch.setattr(chunking, "nltk_stopwords", Missing)
        monkeypatch.setattr(
            chunking, "_ensure_nltk", lambda pkg: attempts.append(pkg) or True
        )
        assert chunking._stopwords() == set()
        assert attempts == ["stopwords"]  # chased once, not once per call

    def test_import_and_tokenize_on_an_airgapped_box(self, empty_nltk):
        """A clean interpreter, no corpora on disk, downloads forbidden."""
        result = self._airgapped(
            AIRGAP, empty_nltk,
            LIBRARIUM_OFFLINE="1", NLTK_DATA=str(empty_nltk),
        )
        assert result.returncode == 0, result.stderr
        assert "AIRGAP-OK" in result.stdout

    def test_offline_flag_blocks_the_download(self, monkeypatch):
        attempts = []
        monkeypatch.setenv("LIBRARIUM_OFFLINE", "1")
        monkeypatch.setattr(chunking, "_downloaded", set())
        monkeypatch.setattr(chunking, "_warned_offline", False)
        monkeypatch.setattr(
            chunking.nltk, "download", lambda *a, **k: attempts.append(a) or True
        )
        assert chunking._ensure_nltk("punkt_tab") is False
        assert attempts == []

    def test_unwritable_data_dir_does_not_escape(self, monkeypatch):
        # nltk.download() raises OSError on a read-only NLTK_DATA rather
        # than returning False, which used to crash at import time.
        monkeypatch.delenv("LIBRARIUM_OFFLINE", raising=False)
        monkeypatch.setattr(chunking, "_downloaded", set())

        def unwritable(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(chunking.nltk, "download", unwritable)
        assert chunking._ensure_nltk("stopwords") is False

    def test_a_successful_download_is_not_repeated(self, monkeypatch):
        calls = []
        monkeypatch.delenv("LIBRARIUM_OFFLINE", raising=False)
        monkeypatch.setattr(chunking, "_downloaded", set())
        monkeypatch.setattr(
            chunking.nltk, "download", lambda *a, **k: calls.append(a) or True
        )
        for _ in range(3):
            assert chunking._ensure_nltk("punkt_tab") is True
        assert len(calls) == 1


# ── K-line named entities ────────────────────────────────────────


class TestEntityExtraction:
    """The entity guard rejects a fuzzy K-line match whose entities
    disagree. Without the tagger corpora it extracts nothing, every
    query looks entity-free, and the guard silently passes everything.
    """

    def test_degrades_to_empty_and_says_so(self, monkeypatch, caplog):
        monkeypatch.setattr(klines, "_entity_support", None)
        monkeypatch.setattr(klines, "_fetch_entity_corpora", lambda: False)
        monkeypatch.setattr(nltk, "pos_tag", _raise_lookup)

        with caplog.at_level(logging.WARNING, logger="librarium.klines"):
            assert KLineStore._extract_entities("Deepa Rao at Lumen") == set()

        assert "entity" in caplog.text.lower()

    def test_missing_corpora_are_only_chased_once(self, monkeypatch):
        attempts = []
        monkeypatch.setattr(klines, "_entity_support", None)
        monkeypatch.setattr(
            klines, "_fetch_entity_corpora", lambda: attempts.append(1) or False
        )
        monkeypatch.setattr(nltk, "pos_tag", _raise_lookup)

        for _ in range(5):
            KLineStore._extract_entities("Deepa Rao")

        assert len(attempts) == 1  # not one download attempt per lookup

    def test_other_failures_do_not_disable_extraction(self, monkeypatch, caplog):
        def boom(*args, **kwargs):
            raise ValueError("something else went wrong")

        monkeypatch.setattr(klines, "_entity_support", None)
        monkeypatch.setattr(nltk, "pos_tag", boom)
        with caplog.at_level(logging.WARNING, logger="librarium.klines"):
            assert KLineStore._extract_entities("Deepa Rao") == set()
        assert klines._entity_support is not False  # not a missing-corpus verdict

    def test_extracts_entities_when_corpora_are_present(self, monkeypatch):
        # Never downloads: if the corpora are absent, there is nothing
        # to assert and the test skips.
        monkeypatch.setattr(klines, "_entity_support", None)
        monkeypatch.setattr(klines, "_fetch_entity_corpora", lambda: False)
        entities = KLineStore._extract_entities(
            "Deepa Rao at Lumen Technologies earned $4.9 million"
        )
        if not entities:
            pytest.skip("NLTK entity corpora not present on this machine")
        assert any("rao" in e for e in entities)
        assert "lumen technologies" in entities
        # The figure survives as a whole. A token scan would have yielded
        # a bare "$" here, because word_tokenize splits "$4.9" in two.
        assert "$4.9 million" in entities
        assert "$" not in entities


class TestAmountExtraction:
    """Figures are pulled from raw text, not from the token list."""

    @pytest.mark.parametrize("text,expected", [
        ("earned $4.9 million", {"$4.9 million"}),
        ("a $500 deductible", {"$500"}),
        ("revenue of $4,900,000", {"$4900000"}),
        ("$ 1200 in fees", {"$1200"}),
        ("up 12% year over year", {"12%"}),
        ("margin fell 3.5 %", {"3.5%"}),
        ("$4.9B in assets", {"$4.9b"}),
        ("the deductible is $500, plus tax", {"$500"}),
        ("both $500 and $1000 apply", {"$500", "$1000"}),
        ("no figures here at all", set()),
        # A lone currency symbol is not an entity.
        ("costs in $ terms", set()),
    ])
    def test_extract_amounts(self, text, expected):
        assert klines._extract_amounts(text) == expected

    def test_amounts_are_classified_for_exact_matching(self):
        assert klines._is_amount("$500")
        assert klines._is_amount("12%")
        assert not klines._is_amount("deepa rao")
        assert not klines._is_amount("lumen technologies")


class TestEntityGuard:
    """The guard's own logic, independent of whether corpora exist."""

    @staticmethod
    def _entities(monkeypatch, mapping):
        monkeypatch.setattr(
            KLineStore, "_extract_entities",
            staticmethod(lambda text: mapping.get(text, set())),
        )

    def test_mismatched_entities_reject_the_match(self, store, monkeypatch):
        self._entities(monkeypatch, {"a": {"lumen"}, "b": {"deepa"}})
        assert store._entities_compatible("a", "b") is False

    def test_fuzzy_overlap_accepts(self, store, monkeypatch):
        self._entities(monkeypatch, {"a": {"rao"}, "b": {"deepa rao"}})
        assert store._entities_compatible("a", "b") is True

    def test_one_sided_entities_reject(self, store, monkeypatch):
        self._entities(monkeypatch, {"a": {"lumen"}, "b": set()})
        assert store._entities_compatible("a", "b") is False

    def test_a_substring_amount_does_not_count_as_a_match(self, store, monkeypatch):
        # "$50" is a substring of "$500" — right for names, wrong for money.
        self._entities(monkeypatch, {"a": {"$50"}, "b": {"$500"}})
        assert store._entities_compatible("a", "b") is False

    def test_identical_amounts_match(self, store, monkeypatch):
        self._entities(monkeypatch, {"a": {"$500"}, "b": {"$500"}})
        assert store._entities_compatible("a", "b") is True

    def test_a_shared_name_still_matches_despite_differing_amounts(
        self, store, monkeypatch
    ):
        self._entities(monkeypatch,
                       {"a": {"deepa rao", "$500"}, "b": {"rao", "$1000"}})
        assert store._entities_compatible("a", "b") is True

    def test_no_entities_either_side_defers_to_cosine(self, store, monkeypatch):
        # This is also what a machine without the corpora sees for every
        # query — hence the warning the extractor now logs.
        self._entities(monkeypatch, {})
        assert store._entities_compatible("a", "b") is True
