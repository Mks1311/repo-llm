def chunk_markdown(file_path):
    """
    Split Markdown files on ## headers.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return []

    chunks = []

    current_chunk = []
    start_line = 1

    for line_number, line in enumerate(lines, start=1):

        if line.startswith("## ") and current_chunk:

            chunks.append({
                "text": "".join(current_chunk),
                "file_path": str(file_path),
                "start_line": start_line,
                "end_line": line_number - 1,
                "type": "markdown",
            })

            current_chunk = []
            start_line = line_number

        current_chunk.append(line)

    if current_chunk:

        chunks.append({
            "text": "".join(current_chunk),
            "file_path": str(file_path),
            "start_line": start_line,
            "end_line": len(lines),
            "type": "markdown",
        })

    return chunks
