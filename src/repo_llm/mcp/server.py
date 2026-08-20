"""
MCP server exposing the codebase search as a tool the LLM can call.

Runs as its own process, speaking MCP over stdio. The repo to search is
passed in through the REPO_LLM_REPO environment variable when the client
spawns this process, so the tool itself only needs a `query` argument.
"""

import os
from pathlib import Path

from mcp.server import MCPServer

from repo_llm.file_discovery import discover_source_files
from repo_llm.repo_cloner import LOCAL_REPO_DIR
from repo_llm.search.hybrid_search import hybrid_search

# Chunking methods to search: code and docs.
SEARCH_METHODS = ("ast", "markdown")
RESULTS_PER_METHOD = 3
FILTERED_RESULTS_PER_METHOD = 15

# Tool results are fed back into the conversation, so cap them to stay
# under the LLM's tokens-per-minute limit.
MAX_RESULT_CHARS = 4000
MAX_READ_CHARS = 8000

# Cross-encoder score below which results are treated as "nothing matched".
# Measured on this corpus: real queries score above -1, typos below -6.
WEAK_MATCH_SCORE = -4.0

mcp = MCPServer("repo-llm")


@mcp.tool()
def search_codebase(query: str, file_path: str = "") -> str:
    """Search the indexed code repository for snippets relevant to a query.

    Use this whenever you need to see actual code or documentation to answer
    a question about the repository. Pass a short, descriptive search query
    (e.g. "how sessions handle cookies"), not a filename alone.

    file_path is optional: pass it to restrict results to a single file, using
    a path as returned by list_files or by an earlier search.
    """
    repo_name = os.environ.get("REPO_LLM_REPO")
    if not repo_name:
        return "Error: REPO_LLM_REPO is not set, so there is no indexed repo to search."

    # Widen the net when filtering, or the file we want may not survive it.
    n_results = FILTERED_RESULTS_PER_METHOD if file_path else RESULTS_PER_METHOD

    chunks = []
    for method in SEARCH_METHODS:
        try:
            chunks.extend(
                hybrid_search(query, method, repo_name, n_results=n_results)
            )
        except Exception as e:
            return f"Error searching '{method}' chunks: {e}"

    if file_path:
        needle = file_path.replace("\\", "/").lower()
        chunks = [
            chunk for chunk in chunks
            if needle in chunk["metadata"]["file_path"].replace("\\", "/").lower()
        ]
        if not chunks:
            return (
                f"No matches for '{query}' inside {file_path}. "
                "Use read_file to read that file directly, or search without "
                "file_path to search the whole repo."
            )
        return _format_results(chunks)

    if not chunks:
        return f"No results found for query: {query}"

    # Search always returns its top N, even when nothing really matches, so
    # so this tells the LLM when to stop calling search_codebase and try list_files instead.
    best_score = max((chunk.get("rerank_score", 0.0) for chunk in chunks), default=0.0)
    if best_score < WEAK_MATCH_SCORE:
        return (
            f"No good match for '{query}' — nothing in this repo is close to it. "
            "The name may be misspelled; call list_files to see the real "
            "filenames instead of searching again.\n\n"
            "Closest (probably irrelevant) results:\n\n"
            + _format_results(chunks[:2])
        )

    return _format_results(chunks)


@mcp.tool()
def list_files() -> str:
    """List every source file in the repository, as paths relative to its root.

    Use this when a filename in the question might be misspelled, or when
    search returns nothing useful and you need to see what actually exists.
    """
    repo_name = os.environ.get("REPO_LLM_REPO")
    if not repo_name:
        return "Error: REPO_LLM_REPO is not set."

    root = LOCAL_REPO_DIR / repo_name
    paths = sorted(
        str(Path(path).relative_to(root)) for path in discover_source_files(root)
    )
    return "\n".join(paths)[:MAX_RESULT_CHARS]


@mcp.tool()
def read_file(file_path: str, start_line: int = 1, end_line: int = 150) -> str:
    """Read a range of lines from one file in the repository.

    Use this after search_codebase or list_files when you know which file you
    need. file_path may be absolute (as search returns it) or relative to the
    repo root (as list_files returns it).
    """
    repo_name = os.environ.get("REPO_LLM_REPO")
    if not repo_name:
        return "Error: REPO_LLM_REPO is not set."

    root = (LOCAL_REPO_DIR / repo_name).resolve()

    # search_codebase reports absolute paths, list_files relative ones —
    # accept either rather than failing on one of them.
    path = Path(file_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()

    # Only ever read inside the cloned repo.
    if not path.is_relative_to(root):
        return f"Refused: {file_path} is outside the repo."

    if not path.is_file():
        return f"No such file: {file_path}"

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[start_line - 1:end_line]
    last_line = min(end_line, len(lines))

    body = "\n".join(selected)
    note = ""
    if len(body) > MAX_READ_CHARS:
        body = body[:MAX_READ_CHARS]
        # Say so explicitly, or the model assumes it got the whole file and
        # keeps re-reading wider ranges looking for the rest.
        note = "\n\n[output truncated — request a narrower line range for more]"

    return f"{path} (lines {start_line}-{last_line} of {len(lines)})\n{body}{note}"


def _format_results(chunks, max_chars=MAX_RESULT_CHARS):
    """Render chunks as numbered, cited snippets the LLM can read and quote."""
    parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        header = (
            f"[{i}] {metadata['file_path']} "
            f"(lines {metadata['start_line']}-{metadata['end_line']})"
        )
        part = f"{header}\n{chunk['text']}"

        if total_chars + len(part) > max_chars:
            break

        parts.append(part)
        total_chars += len(part)

    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
