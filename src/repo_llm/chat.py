from repo_llm.llm_client import generate_answer
from repo_llm.search.hybrid_search import hybrid_search

CHUNKING_METHODS = ("ast", "markdown")


def retrieve_context(question, repo_name, n_results_per_method=3):
    """
    Run hybrid search against every chunking method and pool the results,
    so the LLM gets context drawn from code and docs alike.
    """
    context_chunks = []

    for method in CHUNKING_METHODS:
        try:
            context_chunks.extend(
                hybrid_search(question, method, repo_name, n_results=n_results_per_method)
            )
        except Exception as e:
            print(f"  [WARN] Retrieval failed for method '{method}': {e}")

    return context_chunks


def start_chat(repo_name):
    print(f"\nAsk questions about '{repo_name}' (type 'exit' or 'quit' to stop).\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        context_chunks = retrieve_context(question, repo_name)

        if not context_chunks:
            print("AI: I couldn't find any relevant context for that question.\n")
            continue

        answer = generate_answer(question, context_chunks)
        print(f"AI: {answer}\n")
