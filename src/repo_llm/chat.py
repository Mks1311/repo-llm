from repo_llm.llm_client import generate_answer
from repo_llm.mcp.agent import answer_with_tools
from repo_llm.mcp.client import connect
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

    history = []

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

        answer, citations = generate_answer(question, context_chunks, history)
        print(f"AI: {answer}\n")

        if citations:
            print("Sources:")
            for citation in citations:
                print(
                    f"  [{citation['id']}] {citation['file_path']} "
                    f"(lines {citation['start_line']}-{citation['end_line']})"
                )
            print()


async def start_mcp_chat(repo_name):
    """
    Same chat, but the LLM decides when to search instead of every question
    being preceded by a fixed retrieval step. Search is exposed as an MCP
    tool, so the session stays open for the whole conversation.
    """
    print(f"\nAsk questions about '{repo_name}' (type 'exit' or 'quit' to stop).")
    print("The AI will search the codebase on its own when it needs to.\n")

    history = []

    async with connect(repo_name) as session:
        tools = (await session.list_tools()).tools
        print(f"Connected to MCP server. Tools: {[tool.name for tool in tools]}\n")

        while True:
            question = input("You: ").strip()

            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not question:
                continue

            # Keep the session alive if one question fails (bad model name,
            # rate limit, ...) instead of tearing down the whole chat.
            try:
                answer = await answer_with_tools(question, history, session)
                print(f"AI: {answer}\n")
            except Exception as e:
                print(f"AI: [error] {type(e).__name__}: {e}\n")
