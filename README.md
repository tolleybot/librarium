# librarium

Retrieval that feeds language models well.

`librarium` indexes documents and retrieves the passages that answer a
question, tuned to work with a small local model. Call `retrieve()` and
build your own prompt, or call `ask()` and get an answer in one LLM call.

## What it does

- **Contextual chunking.** Each chunk gets a short LLM-written prefix
  (entities, topics, synonyms) so search can find it.
- **Hybrid retrieval.** BM25 and embeddings index the same prefixed
  chunks and are merged by Reciprocal Rank Fusion, with optional
  cross-encoder reranking. Without `sentence-transformers` it falls back
  to BM25 only.
- **`retrieve()`** returns ranked chunks with provenance. It makes no LLM
  call and is synchronous.
- **`ask()`** sends the retrieved chunks to the model unparaphrased and
  returns the answer in one call.
- **`facts()`** extracts structured facts, with optional gap-finding
  refinement.
- **Persistence** with `save()` / `load()` or `index(..., cache=dir)`, so
  a restart does not pay the indexing cost again.
- **K-lines**, a SQLite cache of verified question/answer chains with
  similarity lookup.
- **Web librarian** (optional), which searches DuckDuckGo, scrapes the
  results and runs the same retrieval over them.

The internals are diagrammed in [`docs/architecture.md`](docs/architecture.md).

## Install

```bash
pip install "librarium[all] @ git+https://github.com/tolleybot/librarium"
# minimal: BM25-only retrieval, bring your own LLM client
pip install "librarium @ git+https://github.com/tolleybot/librarium"
```

| Extra | Adds |
|---|---|
| `embeddings` | dense retrieval and reranking (`sentence-transformers`, `einops`) |
| `web` | web search and scraping |
| `llm` | the bundled OpenAI-compatible client |
| `all` | all of the above |

Requires Python 3.11 or later.

## Quick start

```python
import asyncio
from librarium import Librarian, OpenAICompatClient

async def main():
    llm = OpenAICompatClient(
        base_url="http://localhost:8000/v1",   # vLLM, Ollama, OpenAI, ...
        model="Qwen/Qwen3-8B",
        api_key="sk-...",
    )
    lib = Librarian(llm)

    # cache= persists the index; rerunning over the same text skips
    # re-indexing, and an edited document re-indexes itself
    await lib.index(open("report.md").read(),
                    doc_id="report.md",
                    cache=".librarium/report")

    print(await lib.ask("What is the company's net cash position?"))

    answer = await lib.facts("When did it open and at what price?", refine=True)
    for fact in answer.items:
        print("-", fact.content)

asyncio.run(main())
```

Any backend works if it implements one method:

```python
class MyLLM:
    async def complete(self, system, user, *, think=False, temperature=0.1) -> str:
        ...
```

## Retrieval

```python
for chunk in lib.retrieve("What is the net cash position?", top_k=5):
    print(f"{chunk.doc_id}@{chunk.offset}: {chunk.text[:80]}")
```

**Paging.** If the first page missed, pass back what you have seen and
get the next results:

```python
page = lib.retrieve(question)
more = lib.retrieve(question, exclude={c.index for c in page})
```

**Query expansion** helps when the question and the document use
different words, such as *car insurance* against *auto policy*. It is off
by default and costs one LLM call per question. The default style,
`"hyde"`, writes a short passage that would answer the question and fuses
its ranking with the question's own.

```python
lib = Librarian(llm, expand_query=True)
chunks = await lib.retrieve_expanded("What is my car insurance deductible?")

# or in two steps, keeping retrieve() synchronous
variants = await lib.expand_query("What is my car insurance deductible?")
chunks = lib.retrieve("What is my car insurance deductible?", also=variants)
```

**Which path you got.** Retrieval silently degrades when the embedding
extra is missing or broken, so check:

```python
lib.retrieval_mode   # "hybrid" | "bm25-only" | "whole-document" | "unindexed"
lib.expansion_mode   # "off" | "on" | "inert"
```

Pass `strict=True` to raise instead of degrading.

## Several documents

```python
await lib.index(auto_policy, doc_id="auto-policy.md")       # replaces the index
await lib.add_document(home_policy, doc_id="home-policy.md")  # adds to it
```

Every chunk and fact carries the `doc_id` it came from. Documents can be
indexed concurrently:

```python
await asyncio.gather(*(
    lib.add_document(p.read_text(), doc_id=str(p))
    for p in Path("docs").glob("*.md")
))
```

Await indexing before you query, save or load. A query during indexing
sees a partial index. `save()` and `load()` refuse with `IndexBusyError`
while indexing is in flight rather than writing or loading a corrupt
index.

## Configuration

The defaults are sized for a small local model: `top_k=10` chunks of
2000 characters against a 6000-token context. Every setting is a
constructor argument on `Librarian`, `SourceLibrarian`, `LibrarianRouter`
and `WebLibrarian`.

```python
lib = Librarian(
    llm,
    top_k=10,                    # also settable per call
    max_context_tokens=6000,     # raise for a large-context model
    chunk_size=2000,             # chars per chunk
    chunk_overlap=200,
    chunk_threshold=3000,        # at or under this, documents are served whole
    use_reranker=False,
    rerank_top=5,                # with the reranker on, caps results below top_k
    max_concurrent_prefixes=8,   # index-time LLM calls in flight
    embed_model="BAAI/bge-m3",
    rerank_model="BAAI/bge-reranker-v2-m3",
)
```

Things worth knowing:

- **`top_k` per call** wins over the constructor value, on `ask()`,
  `facts()`, `retrieve()` and `answer()`.
- **With the reranker on, `rerank_top` caps the results.** Raise it
  alongside `top_k`, or you will get 5 chunks back.
- **Indexing costs one LLM call per chunk** plus one per document.
  `max_concurrent_prefixes` caps those across all documents. Use 1-2 for
  Ollama and more for vLLM, and raise `OpenAICompatClient`'s
  `max_connections` to match.
- **A hosted model needs a deliberate `max_context_tokens`.** Set it to
  what you want to pay for, not to what the model accepts.
- **Changing chunk geometry or `embed_model` invalidates saved indexes**
  automatically, so a re-tuned librarian re-indexes instead of loading
  mismatched chunks.

## Optional features

| Setting | Default | What it does |
|---|---|---|
| `cite_sources` | on | Heads each chunk in `ask()`'s prompt with its `doc_id`, type, issuer and date, so near-identical documents stay distinguishable. |
| `strict_grounding` | off | Tells the model to say so when the material does not contain the answer. It reduces correct answers, so turn it on only if a wrong answer costs much more than a missing one. |
| `extract_fields` | off | One extra LLM call per document to record `doc_type`, `issuer`, `effective_date`, `period` and `account`. |
| `extract_facts` | off | One extra LLM call per chunk to build a store of atomic facts that retrieval ranks beside chunks. `fact_mix` sets their share of the results. |
| `use_reranker` | off | Cross-encoder reranking of the fused candidates. `rerank_pool` widens what it sees. |

None of these invalidate an existing saved index when left off.

## Embedding models

The default models are pinned to fixed revisions:

| Model | Revision |
|---|---|
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

Any other `embed_model` or `rerank_model` loads unpinned.

A saved index records a fingerprint of the embedding weights that built
it. If the weights loaded later differ, the stored vectors are discarded
with a warning and retrieval falls back to BM25 until you re-index.
`strict=True` raises `StaleEmbeddingsError` instead. An index built
without the `embeddings` extra stays BM25-only after you install it,
and `load()` warns so you know to re-index.

## Offline use

NLTK data is fetched on first use. Set `LIBRARIUM_OFFLINE=1` to forbid
the download. Tokenization then falls back to regex splitting, and
nothing in the core retrieval path touches the network.

## K-line caching

```python
from librarium import KLineStore, LibrarianRouter

store = KLineStore(db_path="klines.db")
router = LibrarianRouter(llm, kline_store=store, source_librarian=lib.source)
answer = await router.answer("What is the runway?")
router.verify("What is the runway?", confidence=0.95)  # after your checks pass
```

Only verified answers are served from the cache. `router.mark_failed()`
records dead ends so they short-circuit next time. The router tries the
cache, then your documents, then the web librarian if it is installed.

## Development

```bash
pip install -e ".[dev,llm]"
pytest    # mock LLMs and stub encoders: no model, network or API key
```

## License

MIT
