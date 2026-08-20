# Repo LLM

Clone a Python GitHub repo, index it, and ask an LLM questions about the code.

I built this to learn how RAG works by writing each piece myself instead of calling a framework that hides it. The use case is getting oriented in an unfamiliar codebase quickly, which is useful when deciding whether to contribute to an open source project.

## Architecture

```
  repo URL
     │
     ▼
  clone (GitPython) ──► walk for .py / .md files
     │
     ▼
  CHUNK each Python file two ways, kept separate so they can be compared
     ├── fixed-size word chunks
     ├── AST chunks (grouped by function / class)
     └── markdown split on ## headers
     │
     ├──────────────► embed (MiniLM) ──► ChromaDB
     │                                   one collection per chunking method
     └──────────────► BM25 index
                      one per method, file paths included in the tokens
     │
     ▼
  RETRIEVE
     vector search ─┐
                    ├─► RRF fusion ─► cross-encoder rerank ─► top results
     BM25 search ───┘
     │
     ▼
  CHAT
     your question
         │
         ▼
     ┌─► LLM (Groq)  ──► no tool call? ──► answer
     │       │
     │       └──► tool call
     │              │
     │              ▼
     │        MCP client ──JSON-RPC over stdio──► MCP server (separate process)
     │                                              search_codebase
     │                                              read_file
     │                                              list_files
     │              │
     └──────────────┘  result goes back as a message, loop again
                       (capped rounds; last round runs with no tools
                        attached so the model has to answer)
```

## Evaluation

10 hand-written queries against [psf/requests](https://github.com/psf/requests), each with a known correct source file (e.g. "HTTP basic authentication" → `auth.py`). Run on AST chunks, top 5 results. A hit means the correct file appeared in the top 5.

| Method | Hit@5 | MRR |
|---|---|---|
| Vector only | 90.0% | 0.725 |
| BM25 only | 90.0% | 0.695 |
| **Hybrid (RRF) + rerank** | **100.0%** | **0.792** |

Hybrid beat vector-only by 11.1% on Hit@5 and 9.2% on MRR, and BM25-only by 11.1% and 13.9%.

Per-query rank of the first correct file:

| Query | Vector | BM25 | Hybrid + rerank |
|---|---|---|---|
| cookies in a session | 1 | 1 | 3 |
| sending a GET request | 4 | 5 | 3 |
| HTTP basic auth | 2 | 1 | 1 |
| session redirects | 1 | 1 | 1 |
| connection pooling / adapters | 1 | 1 | 1 |
| custom exceptions | miss | miss | 4 |
| the Response object | 1 | 4 | 1 |
| hooks | 1 | 1 | 1 |
| preparing a request body | 1 | 2 | 1 |
| proxy configuration | 2 | 1 | 1 |

The clearest win is "custom exceptions", which both single methods missed completely and only turned up once they were fused. Hybrid is not better on every query though. It pushed "cookies in a session" from rank 1 down to rank 3. It wins on average, it doesn't win every time.

This is a small eval on one repo, so treat it as a sanity check rather than a benchmark.

## Design decisions

Char and AST chunks go into separate Chroma collections and separate BM25 indexes, because the point was to compare chunking strategies rather than pick one up front. Vector and BM25 results are merged with Reciprocal Rank Fusion instead of blending scores, since cosine distance and BM25 scores are on completely different scales and RRF only looks at rank. File paths are tokenized into the BM25 index but deliberately kept out of the embeddings, because BM25 is already a literal matcher so paths fit naturally, while putting filenames into embeddings would drag semantic similarity toward name overlap instead of what the code actually does.

## What I'd improve with more time

**A real eval harness.** The numbers above came from a throwaway script. It should live in the repo, cover more repos than one, and measure answer quality rather than just retrieval rank.

**Cheaper agent loops.** Every round resends the whole conversation, so a question that needs four tool calls costs a lot of tokens. Old tool results should be summarized or dropped once they've been used.

**Better handling of unknown tool arguments.** When the model passes an argument a tool doesn't declare, MCP silently ignores it and returns normal-looking results, which sent the model into retry loops. Rejecting unknown arguments with a clear message would be much easier to debug.

**Typo handling that doesn't depend on the LLM.** A misspelled filename currently recovers through the model calling `list_files` and spotting the real name. Fuzzy matching filenames directly would be faster and more reliable.

## Project structure

```
src/repo_llm/
  cli.py                  entry point: clone → chunk → index → chat
  repo_cloner.py          clone a repo with GitPython
  file_discovery.py       walk a repo for .py / .md files
  chunking/
    fixed_char_chunker.py fixed-size word chunking
    ast_chunker.py        AST chunking, grouped by function / class
    markdown_chunker.py   splits markdown on ## headers
    chunk_writer.py       writes and reads chunks as .txt files
    pipeline.py           runs chunking across a repo
  embedder.py             MiniLM embeddings
  vector_store.py         embeds and upserts into Chroma
  bm25_index.py           BM25 index per chunking method
  search/
    vector_search.py      semantic search, scoped by repo_name
    bm25_search.py        lexical search
    hybrid_search.py      RRF fusion of both
    reranker.py           cross-encoder reranking
  llm_client.py           Groq calls, context budget, citations, history
  chat.py                 the chat loops
  mcp/
    server.py             the three tools, runs as its own process
    client.py             spawns the server and opens a session
    agent.py              the tool-calling loop
  test/                   scripts to try each retrieval method on its own
```

## Setup

```bash
uv sync
```

Put a Groq API key in a `.env` file at the project root ([get one free](https://console.groq.com)):

```
GROQ_API_KEY=your-key-here
```

Run it. It asks for a repo URL, indexes it, then starts the chat:

```bash
uv run python -m repo_llm.cli
```

To try a single retrieval method on its own:

```bash
uv run python -m repo_llm.test.test_hybrid
```

## Models

| Job | Model |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Chat | `openai/gpt-oss-120b` on Groq (override with `GROQ_MODEL`) |

## Stack

Python, GitPython, ChromaDB, sentence-transformers, rank-bm25, MCP, Groq.
