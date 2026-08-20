"""
Connects MCP tools to the LLM.

The key idea: the LLM never runs anything itself. It only *asks* for a tool
by name with arguments; our code executes that tool via MCP and feeds the
result back as a new message. Repeat until the model answers without asking
for a tool.

    LLM  --"call search_codebase(query=...)"-->  us  --MCP-->  server
    LLM  <--"here is what it returned"--------  us  <--MCP--  server
"""

import json

from groq import BadRequestError

from repo_llm.llm_client import GROQ_MODEL, _get_client, _trim_history

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a code repository. "
    "You cannot see the code directly — use search_codebase to find relevant "
    "code, and read_file to read more of a file you found. If a filename "
    "might be misspelled, or a search returns nothing useful, call list_files "
    "once to see what actually exists and work from those real names — do not "
    "repeat a search that already failed. Results include their file path; "
    "cite the file you used in your answer. Once you have enough to answer, "
    "answer — don't keep calling tools. If the results don't contain the "
    "answer, say you don't know."
)

# Stops a misbehaving model from looping on tool calls forever.
MAX_TOOL_ROUNDS = 8


def to_groq_tool(mcp_tool):
    """Translate an MCP tool definition into Groq's function-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.input_schema,
        },
    }


async def answer_with_tools(question, history, session):
    """
    Answer `question`, letting the LLM call MCP tools as it sees fit.

    Appends this turn's question and final answer to `history` so follow-up
    questions keep their context.
    """
    client = _get_client()

    groq_tools = [to_groq_tool(tool) for tool in (await session.list_tools()).tools]

    user_message = {"role": "user", "content": question}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_trim_history(history),
        user_message,
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=groq_tools,
            )
        except BadRequestError as e:
            # The model invented arguments that don't fit the tool's schema,
            # and Groq rejected its own generation. Tell it what went wrong
            # so it can retry with the real parameters.
            if "tool_use_failed" not in str(e):
                raise
            print("  [tool] invalid tool call, asking the model to retry")
            messages.append({
                "role": "user",
                "content": (
                    "That tool call was rejected: the arguments did not match "
                    "the tool's schema. Call the tool again using only the "
                    "parameters it actually declares."
                ),
            })
            continue

        message = completion.choices[0].message

        # No tool requested: this is the final answer.
        if not message.tool_calls:
            answer = message.content
            history.append(user_message)
            history.append({"role": "assistant", "content": answer})
            return answer

        # The model asked for tools — record the request, then run them.
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"  [tool] {name}({arguments})")

            try:
                result = await session.call_tool(name, arguments=arguments)
                output = "\n".join(
                    block.text for block in result.content if block.type == "text"
                )
            except Exception as e:
                output = f"Tool call failed: {e}"

            # Tool output goes back in as its own message, tied to the call id.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })

    answer = "I couldn't finish answering — too many tool calls in a row."
    history.append(user_message)
    history.append({"role": "assistant", "content": answer})
    return answer
