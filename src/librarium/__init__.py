"""librarium — retrieval that feeds language models well.

Chunking with LLM-generated contextual prefixes, dual-index retrieval
(BM25 + embeddings, RRF-fused, optional cross-encoder rerank), verified
answer caching (K-lines), and optional web search — behind one facade:

    from librarium import Librarian, OpenAICompatClient

    llm = OpenAICompatClient("http://localhost:8000/v1", "Qwen/Qwen3-8B")
    lib = Librarian(llm)
    await lib.index(document_text)
    print(await lib.ask("What is the net cash position?"))
"""

from librarium.kline_librarian import KLineLibrarian
from librarium.klines import KLine, KLineStore, NegativeKLine
from librarium.librarian import Librarian
from librarium.llm import LLMClient, OpenAICompatClient, parse_thinking
from librarium.models import Chunk, Fact, LibrarianAnswer
from librarium.router import LibrarianRouter
from librarium.source import (
    IndexBusyError,
    LoopHandoverError,
    SourceLibrarian,
    StaleEmbeddingsError,
)

__version__ = "0.1.0"

__all__ = [
    "Librarian",
    "LLMClient",
    "OpenAICompatClient",
    "SourceLibrarian",
    "LibrarianRouter",
    # Exported because `strict=True` is the only way to raise it, and a
    # caller who asked for that has to be able to name it in an `except`.
    "StaleEmbeddingsError",
    # Exported for the same reason, one step further out: save()/load()
    # raise it on any librarian that indexes concurrently, no flag
    # required, so a caller cannot even opt out of needing the name.
    "IndexBusyError",
    # An IndexBusyError, so `except IndexBusyError` already catches it —
    # exported under its own name because the remedy is the one thing a
    # caller cannot guess from the base class: this one does not clear by
    # waiting, only by closing or draining the *other* loop.
    "LoopHandoverError",
    "KLineStore",
    "KLine",
    "NegativeKLine",
    "KLineLibrarian",
    "Chunk",
    "Fact",
    "LibrarianAnswer",
    "parse_thinking",
]


def __getattr__(name):
    # WebLibrarian pulls in httpx/bs4/trafilatura — import lazily so the
    # core package works without the [web] extra installed.
    if name == "WebLibrarian":
        from librarium.web import WebLibrarian
        return WebLibrarian
    raise AttributeError(f"module 'librarium' has no attribute {name!r}")
