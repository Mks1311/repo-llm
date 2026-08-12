from pathlib import Path

import chromadb

from repo_llm.embedder import embed_texts

BASE_DIR = Path(__file__).resolve().parents[3]
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAMES = {
    "char": "python_char_collection",
    "ast": "python_ast_collection",
    "markdown": "markdown_collection",
}


def vector_search(query, method, repo_name, n_results=10):
    """
    Semantic search: embed the query and find the nearest chunks in Chroma.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAMES[method])

    query_embedding = embed_texts([query])[0]

    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where={"repo_name": repo_name},
    )

    return [
        {"id": id_, "text": text, "metadata": metadata, "score": distance}
        for id_, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]
