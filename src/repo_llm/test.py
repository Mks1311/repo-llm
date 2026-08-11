import chromadb
from pathlib import Path

from repo_llm.embedder import embed_texts

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAMES = {
    "char": "python_char_collection",
    "ast": "python_ast_collection",
    "markdown": "markdown_collection",
}


def test():
    query = input("Enter your query: ")
    method = input("Enter the chunking method (char, ast, markdown): ")
    embeddings = embed_texts(query)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAMES[method])
    result = collection.query(
        query_embeddings=[embeddings],
        n_results=10
    )
    print(f"Results for query '{query}' using method '{method}':")
    for id, document, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
        print(id, document, metadata)

if __name__ == "__main__":
    test()