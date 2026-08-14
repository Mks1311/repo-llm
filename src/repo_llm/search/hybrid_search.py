from repo_llm.search.bm25_search import bm25_search
from repo_llm.search.reranker import rerank
from repo_llm.search.vector_search import vector_search

RRF_K = 60


def hybrid_search(
    query,
    method,
    repo_name,
    n_results=10,
    candidate_pool_size=20,
    k=RRF_K,
    use_reranker=True,
):
    """
    Combine vector and BM25 rankings using Reciprocal Rank Fusion (RRF),
    then optionally re-order the fused candidates with a cross-encoder.

    Each ranker contributes 1 / (k + rank) per document it returns, so a
    chunk ranked highly by both semantic and lexical search rises to the
    top, and a chunk found by only one ranker still gets a chance.
    """
    vector_results = vector_search(query, method, repo_name, n_results=candidate_pool_size)
    bm25_results = bm25_search(query, method, repo_name, n_results=candidate_pool_size)

    fused = {}

    for rank, result in enumerate(vector_results, start=1):
        entry = fused.setdefault(result["id"], {
            "text": result["text"],
            "metadata": result["metadata"],
            "rrf_score": 0.0,
            "vector_rank": None,
            "bm25_rank": None,
        })
        entry["rrf_score"] += 1 / (k + rank)
        entry["vector_rank"] = rank

    for rank, result in enumerate(bm25_results, start=1):
        entry = fused.setdefault(result["id"], {
            "text": result["text"],
            "metadata": result["metadata"],
            "rrf_score": 0.0,
            "vector_rank": None,
            "bm25_rank": None,
        })
        entry["rrf_score"] += 1 / (k + rank)
        entry["bm25_rank"] = rank

    ranked_ids = sorted(fused, key=lambda id_: fused[id_]["rrf_score"], reverse=True)

    fused_results = [
        {
            "id": id_,
            "text": fused[id_]["text"],
            "metadata": fused[id_]["metadata"],
            "score": fused[id_]["rrf_score"],
            "vector_rank": fused[id_]["vector_rank"],
            "bm25_rank": fused[id_]["bm25_rank"],
        }
        for id_ in ranked_ids
    ]

    if not use_reranker:
        return fused_results[:n_results]

    return rerank(query, fused_results)[:n_results]
