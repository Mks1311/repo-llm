from pathlib import Path

import chromadb

from repo_llm.chunking.chunk_writer import read_chunks
from repo_llm.embedder import embed_texts

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# One collection per chunking method, so they can be compared side by side.
COLLECTION_NAMES = {
    "char": "python_char_collection",
    "ast": "python_ast_collection",
    "markdown": "markdown_collection",
}


def save_chunks_to_vector_store(chunks_path):
    """
    Embed every chunk previously written under `chunks_path` and store it
    in the Chroma collection matching its chunking method.
    """
    chunks_path = Path(chunks_path)
    repo_name = chunks_path.name

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    sources = {
        "char": chunks_path / "python" / "char_chunks",
        "ast": chunks_path / "python" / "ast_chunks",
        "markdown": chunks_path / "markdown",
    }

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
            }
            for chunk in chunks
        ]

        embeddings = embed_texts(texts)

        collection = client.get_or_create_collection(name=COLLECTION_NAMES[method])
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  [OK] Stored {len(chunks)} {method} chunks in '{COLLECTION_NAMES[method]}'")

    print("Embedding and storage complete.")
