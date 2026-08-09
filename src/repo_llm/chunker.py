from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_REPO_DIR = BASE_DIR / "repo_chunks"


def chunker(file_paths, repo_url):
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    destination = LOCAL_REPO_DIR / repo_name

    if destination.exists():
        print(f"Repository already chunked at {destination}. Skipping chunking.")
        return []

    destination.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    chunk_number = 1

    for file_path in file_paths:

        extension = Path(file_path).suffix.lower()

        if extension == ".py":
            chunks = chunk_python_file(file_path)

        elif extension == ".md":
            chunks = chunk_markdown_file(file_path)

        else:
            continue

        for chunk in chunks:
            chunk["chunk_id"] = chunk_number

            save_chunk(chunk, destination, chunk_number)

            all_chunks.append(chunk)

            chunk_number += 1

    return all_chunks


def chunk_python_file(file_path, chunk_size=500, overlap=50):
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

        chunk_words = words_with_lines[start:start + chunk_size]

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
            "type": "python"
        })

        if start + chunk_size >= len(words_with_lines):
            break

    return chunks


def chunk_markdown_file(file_path):
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
                "type": "markdown"
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
            "type": "markdown"
        })

    return chunks


def save_chunk(chunk, destination, chunk_number):
    """
    Save a chunk and its metadata as a .txt file.
    """

    chunk_file = destination / f"chunk_{chunk_number:04d}.txt"

    with open(chunk_file, "w", encoding="utf-8") as f:

        f.write(f"FILE: {chunk['file_path']}\n")
        f.write(f"TYPE: {chunk['type']}\n")
        f.write(f"START_LINE: {chunk['start_line']}\n")
        f.write(f"END_LINE: {chunk['end_line']}\n")
        f.write(f"CHUNK_ID: {chunk_number}\n")

        f.write("\n")
        f.write("-" * 80)
        f.write("\n\n")

        f.write(chunk["text"])