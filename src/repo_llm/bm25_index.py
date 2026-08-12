import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from repo_llm.chunking.chunk_writer import read_chunks
from repo_llm.repo_cloner import LOCAL_REPO_DIR

BASE_DIR = Path(__file__).resolve().parents[2]
BM25_INDEX_DIR = BASE_DIR / "bm25_index"


def tokenize(text):
    return text.lower().split()


def _tokenize_path(file_path, repo_root):
    """Turn a file's path (relative to the repo root) into search tokens."""
    relative_path = Path(file_path).relative_to(repo_root)
    return tokenize(str(relative_path).replace("\\", " ").replace("/", " "))


def save_chunks_to_bm25_index(chunks_path):
    """
    Build a BM25 index per chunking method over the same chunks stored in
    the vector store, so lexical and semantic retrieval can be compared.
    """
    chunks_path = Path(chunks_path)
    repo_name = chunks_path.name
    repo_root = LOCAL_REPO_DIR / repo_name

    sources = {
        "char": chunks_path / "python" / "char_chunks",
        "ast": chunks_path / "python" / "ast_chunks",
        "markdown": chunks_path / "markdown",
    }

    destination = BM25_INDEX_DIR / repo_name
    destination.mkdir(parents=True, exist_ok=True)

    for method, directory in sources.items():
        chunks = list(read_chunks(directory))

        if not chunks:
            print(f"  [SKIP] No {method} chunks found in {directory}")
            continue

        ids = [f"{repo_name}_{method}_{chunk['chunk_id']}" for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "repo_name": repo_name,
                "file_path": chunk["file_path"],
                "chunking_method": chunk["chunking_method"],
                "type": chunk["type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "file_name": Path(chunk["file_path"]).name,
            }
            for chunk in chunks
        ]

        tokenized_corpus = [
            _tokenize_path(chunk["file_path"], repo_root) + tokenize(chunk["text"])
            for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_corpus)

        index_file = destination / f"{method}.pkl"
        with open(index_file, "wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "ids": ids,
                    "texts": texts,
                    "metadatas": metadatas,
                },
                f,
            )

        print(f"  [OK] Indexed {len(chunks)} {method} chunks in '{index_file}'")

    print("BM25 indexing complete.")
