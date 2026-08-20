from repo_llm.search.hybrid_search import hybrid_search


def test():
    query = input("Enter your query: ")
    method = input("Enter the chunking method (char, ast, markdown): ")
    # repo_name = input("Enter the repo name (e.g. requests): ")
    repo_name = "requests"

    results = hybrid_search(query, method, repo_name)

    print(f"\nHybrid (RRF + reranker) search results for '{query}' (method={method}, repo={repo_name}):\n")
    for rank, result in enumerate(results, start=1):
        rerank_score = result.get("rerank_score")
        rerank_part = f"rerank_score={rerank_score:.4f} " if rerank_score is not None else ""
        print(
            f"{rank}. [{result['id']}] {rerank_part}rrf_score={result['score']:.4f} "
            f"(vector_rank={result['vector_rank']}, bm25_rank={result['bm25_rank']})"
        )
        print(f"   {result['metadata']['file_path']}")
        print(f"   {result['text'][:200]!r}\n")


if __name__ == "__main__":
    test()
