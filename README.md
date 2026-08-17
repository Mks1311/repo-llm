# Repo LLM

Clone a Python GitHub repo, ask an LLM questions about it. Built to learn how RAG actually works end to end — chunking, embeddings, lexical search, hybrid fusion, reranking, and grounding an LLM in retrieved context — rather than calling a framework that hides all of it.

The motivating use case: getting oriented in an unfamiliar codebase fast, which is genuinely useful when deciding whether to contribute to an open source project.

## How it works

```
clone repo (GitPython)
        │
        ▼
walk .py / .md files
        │
        ▼
chunk two ways, kept separate for comparison
   ├── fixed-size word chunks
   └── AST-based chunks (grouped by function/class)
        │
        ├──────────────► embed (MiniLM) ──► ChromaDB
        │                                    (one collection per chunking method)
        └──────────────► BM25 index (per chunking method,
                                      tokens include the file's relative path)
        │
        ▼
retrieval, per method: vector search / BM25 / hybrid (RRF) → cross-encoder rerank
        │
        ▼
chat: retrieved chunks → Groq LLM, with numbered citations
      and client-side multi-turn memory
```

## Project structure

```
src/repo_llm/
  cli.py                 entry point: clone → chunk → embed/index → chat
  repo_cloner.py          clone a repo locally with GitPython
  file_discovery.py       walk a repo for .py/.md files
  chunking/
    fixed_char_chunker.py    fixed-size word chunking
    ast_chunker.py           AST-based chunking (grouped by function/class)
    markdown_chunker.py      splits markdown on ## headers
    chunk_writer.py          write/read chunks as .txt files (the chunk cache)
    pipeline.py              orchestrates chunking across a repo
  embedder.py               sentence-transformers embedding (MiniLM)
  vector_store.py           embeds + upserts chunks into Chroma, one collection per method
  bm25_index.py              builds a BM25 index per method, path-aware tokenization
  search/
    vector_search.py         semantic search, scoped by repo_name
    bm25_search.py            lexical search
    hybrid_search.py          combines both via Reciprocal Rank Fusion (RRF)
    reranker.py               cross-encoder reranking on top of the fused pool
  llm_client.py              Groq chat completion, context budgeting, citations, history
  chat.py                    interactive REPL that ties retrieval + LLM together
  test_vector.py / test_bm25.py / test_hybrid.py   standalone scripts to test each
                                                      retrieval strategy in isolation
```

## Setup

```bash
uv sync
```

Set a Groq API key (get one free at [console.groq.com](https://console.groq.com)) in a `.env` file at the project root:

```
GROQ_API_KEY=your-key-here
```

Run the full pipeline — clone, chunk, embed, index, then chat:

```bash
uv run python -m repo_llm.cli
```

To test a single retrieval strategy in isolation (useful for comparing char vs AST chunking, or vector vs BM25 vs hybrid):

```bash
uv run python -m repo_llm.test_vector
uv run python -m repo_llm.test_bm25
uv run python -m repo_llm.test_hybrid
```

## Design decisions

A few things that came out of actually building this, not just reading about RAG:

- **Char and AST chunks are stored in separate Chroma collections and separate BM25 indexes**, not merged — the whole point is being able to compare which chunking strategy retrieves better, not to pick one upfront.
- **Reciprocal Rank Fusion (RRF), not score blending**, combines vector + BM25 results. Cosine distance and BM25 score live on incompatible scales; RRF only cares about each result's *rank* in each list, so no normalization is needed.
- **File paths are tokenized into the BM25 index, but never into the embedding.** BM25 is a pure lexical matcher, so adding path tokens is a natural extension — it fixed filename queries like `api.py` immediately. Embeddings are a meaning signal; folding filenames in would skew semantic similarity toward incidental name overlap instead of actual content relevance. Metadata filtering, not search, is the right tool for exact identity lookups.
- **A cross-encoder reranks the fused candidate pool, not the whole corpus** — it reads the query and a chunk together in one pass (more accurate than independent scoring), but is too slow to run over every chunk. RRF cheaply narrows the field first.
- **Groq's chat API is stateless** — no server-side session like Gemini's chat objects or OpenAI's `previous_response_id`. Multi-turn memory is a client-side message list, resent every turn, capped to a character budget (and the retrieval context is capped too) to stay under the free-tier tokens-per-minute limit.
- **Citations are numbered and returned structurally**, not just requested from the model — each retrieved chunk is labeled `[1]`, `[2]`... in the prompt, and the mapping back to file/line is printed regardless of whether the model actually cited correctly, so there's always a verifiable trail.

## Evaluation

No formal eval harness yet — these are real runs against [psf/requests](https://github.com/psf/requests) during development, kept here as a starting point for a proper benchmark later.

| Query | Vector search | BM25 | Hybrid (RRF) + rerank | Notes |
|---|---|---|---|---|
| `session cookies` | Ranked `sessions.py`/`cookies.py` chunks highly on meaning | Ranked test files highest on raw term overlap | Best of both — `sessions.py` chunk rose to #1 with support from both rankers | Good example of hybrid actually improving over either alone |
| `api.py` | Ranked `api.py` chunks #1 (docstring happens to repeat the module name) | Found nothing until file paths were added to the BM25 tokens; correct after | Both agree, `api.py` docstring chunk ranks #1 | Filename lookup needed a deliberate fix (path-aware BM25 tokens), not automatic |
| `moddels.py` (typo) | Top result was unrelated (`packages.py`) | Scored 0.0 everywhere — no literal token match | Still wrong — fusing two wrong rankings doesn't produce a right one | Neither lexical nor embedding search handles typos; this is an identity-lookup problem, not a relevance problem — motivates giving the LLM a search tool it can query with a corrected term |

## Roadmap

- [x] Repo cloning + file discovery
- [x] Dual chunking: fixed-size and AST-based, stored separately
- [x] Embeddings + ChromaDB storage, one collection per chunking method
- [x] BM25 lexical index, path-aware tokenization
- [x] Hybrid search via Reciprocal Rank Fusion
- [x] Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L6-v2`)
- [x] LLM chat (Groq, Llama 3.3 70B) with multi-turn memory and inline citations
- [ ] **Next: function calling** — give the LLM an actual search tool instead of always force-feeding it fixed retrieval results, so it decides what and when to search. Should also fix typo queries like `moddels.py`: with the LLM formulating the search query itself, it can normalize an obvious typo before searching, instead of the pipeline blindly matching the raw string.

## Stack

Python, GitPython, ChromaDB, `sentence-transformers` (MiniLM embeddings + cross-encoder reranker), `rank-bm25`, Groq.
