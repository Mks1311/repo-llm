import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a codebase. "
    "Use the provided code context to answer, and the earlier turns of this "
    "conversation for follow-up questions. If the answer isn't in the "
    "context, say you don't know. Each context snippet is labeled with a "
    "bracketed number like [1]; cite the snippets you used inline in your "
    "answer using those numbers, e.g. \"Sessions persist cookies [2].\""
)

# Keeps a single request comfortably under Groq's free-tier tokens-per-minute limit.
MAX_CONTEXT_CHARS = 6000
MAX_HISTORY_CHARS = 6000

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a key from https://console.groq.com and set it before chatting."
            )
        _client = Groq(api_key=api_key)
    return _client


def generate_answer(question, context_chunks, history):
    """
    Ask the LLM to answer `question`, using freshly retrieved `context_chunks`
    plus the running conversation `history` so follow-up questions have
    context from earlier turns.

    `history` is a list of {"role", "content"} dicts. This function reads it
    to build the request and appends this turn's question/answer to it, so
    the caller's list grows across turns and the next call sees it.

    Returns (answer, citations), where citations maps each bracketed number
    used in the context back to the source chunk's file path and lines.
    """
    client = _get_client()

    context, citations = _build_context(context_chunks)
    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_trim_history(history),
        user_message,
    ]

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
    )

    answer = completion.choices[0].message.content

    history.append(user_message)
    history.append({"role": "assistant", "content": answer})

    return answer, citations


def _build_context(context_chunks, max_chars=MAX_CONTEXT_CHARS):
    """
    Number and concatenate chunks into a context block, capped to a character
    budget so a single request doesn't exceed Groq's tokens-per-minute limit.

    Returns (context_text, citations), where citations is the list of
    {"id", "file_path", "start_line", "end_line"} for each numbered chunk
    that made it into the context.
    """
    parts = []
    citations = []
    total_chars = 0

    for i, chunk in enumerate(context_chunks, start=1):
        metadata = chunk["metadata"]
        location = f"{metadata['file_path']} (lines {metadata['start_line']}-{metadata['end_line']})"
        part = f"[{i}] {location}\n{chunk['text']}"

        if total_chars + len(part) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 0:
                parts.append(part[:remaining])
                citations.append({
                    "id": i,
                    "file_path": metadata["file_path"],
                    "start_line": metadata["start_line"],
                    "end_line": metadata["end_line"],
                })
            break

        parts.append(part)
        citations.append({
            "id": i,
            "file_path": metadata["file_path"],
            "start_line": metadata["start_line"],
            "end_line": metadata["end_line"],
        })
        total_chars += len(part)

    return "\n\n".join(parts), citations


def _trim_history(history, max_chars=MAX_HISTORY_CHARS):
    """
    Keep only the most recent messages that fit within a character budget,
    so a long conversation doesn't grow past the tokens-per-minute limit.
    """
    trimmed = []
    total_chars = 0

    for message in reversed(history):
        length = len(message["content"])

        if total_chars + length > max_chars:
            break

        trimmed.append(message)
        total_chars += length

    trimmed.reverse()
    return trimmed
