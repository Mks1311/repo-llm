import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

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


def generate_answer(question, context_chunks):
    """
    Ask the LLM to answer `question` using only the given retrieved chunks
    as context.
    """
    client = _get_client()

    context = "\n\n".join(
        f"# {chunk['metadata']['file_path']}\n{chunk['text']}"
        for chunk in context_chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant answering questions about a codebase. "
                "Use only the provided code context to answer. If the context doesn't "
                "contain the answer, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
    )

    return completion.choices[0].message.content
