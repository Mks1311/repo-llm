from repo_llm.search.vector_search import vector_search


def test():
    query = input("Enter your query: ")
    method = input("Enter the chunking method (char, ast, markdown): ")
    # repo_name = input("Enter the repo name (e.g. requests): ")
    repo_name = "requests"

    results = vector_search(query, method, repo_name)

    print(f"\nVector search results for '{query}' (method={method}, repo={repo_name}):\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result['id']}] distance={result['score']:.4f}")
        print(f"   {result['metadata']['file_path']}")
        print(f"   {result['text'][:200]!r}\n")


if __name__ == "__main__":
    test()