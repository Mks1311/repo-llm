import pickle

from repo_llm.bm25_index import BM25_INDEX_DIR, tokenize


def bm25_search(query, method, repo_name, n_results=10):
    """
    Lexical search: score every chunk in the repo's BM25 index against the query.
    """
    index_file = BM25_INDEX_DIR / repo_name / f"{method}.pkl"

    with open(index_file, "rb") as f:
        data = pickle.load(f)

    scores = data["bm25"].get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:n_results]

    return [
        {
            "id": data["ids"][i],
            "text": data["texts"][i],
            "metadata": data["metadatas"][i],
            "score": float(scores[i]),
        }
        for i in ranked_indices
    ]
