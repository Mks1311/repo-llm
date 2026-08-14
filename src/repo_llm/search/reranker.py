from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL_NAME)
    return _model


def rerank(query, results):
    """
    Re-score `results` (as returned by vector_search/bm25_search/hybrid_search)
    against `query` with a cross-encoder, and return them sorted by that score.

    Unlike vector/BM25 search, which score the query and a chunk independently
    and compare the two representations, a cross-encoder reads the query and
    chunk together in one pass, so it can judge relevance more precisely at
    the cost of being too slow to run over a whole corpus.
    """
    model = _get_model()
    pairs = [(query, result["text"]) for result in results]
    scores = model.predict(pairs)

    reranked = [
        {**result, "rerank_score": float(score)}
        for result, score in zip(results, scores)
    ]
    reranked.sort(key=lambda result: result["rerank_score"], reverse=True)

    return reranked
