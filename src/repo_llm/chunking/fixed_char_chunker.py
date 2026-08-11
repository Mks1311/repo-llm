def chunk_by_fixed_size(file_path, chunk_size=500, overlap=50):
    """
    Split Python files into fixed-size chunks.

    Currently chunk_size and overlap are based on words,
    not actual tokenizer tokens.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    words_with_lines = []

    for line_number, line in enumerate(lines, start=1):
        words = line.split()

        for word in words:
            words_with_lines.append((word, line_number))

    if not words_with_lines:
        return []

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words_with_lines), step):

        chunk_words = words_with_lines[
            start:start + chunk_size
        ]

        if not chunk_words:
            break

        text = " ".join(
            word for word, _ in chunk_words
        )

        start_line = chunk_words[0][1]
        end_line = chunk_words[-1][1]

        chunks.append({
            "text": text,
            "file_path": str(file_path),
            "start_line": start_line,
            "end_line": end_line,
            "type": "python",
        })

        if start + chunk_size >= len(words_with_lines):
            break

    return chunks
