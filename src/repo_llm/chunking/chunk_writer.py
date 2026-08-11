from pathlib import Path

SEPARATOR = "-" * 80


def write_chunk(chunk, destination, chunk_number):
    """
    Save a chunk and its metadata as a .txt file.
    """

    chunk_file = destination / f"chunk_{chunk_number:04d}.txt"

    with open(chunk_file, "w", encoding="utf-8") as f:

        f.write(f"FILE: {chunk['file_path']}\n")
        f.write(f"TYPE: {chunk['type']}\n")
        f.write(f"CHUNKING_METHOD: {chunk.get('chunking_method', 'unknown')}\n")
        f.write(f"START_LINE: {chunk['start_line']}\n")
        f.write(f"END_LINE: {chunk['end_line']}\n")
        f.write(f"CHUNK_ID: {chunk_number}\n")

        f.write("\n")
        f.write(SEPARATOR)
        f.write("\n\n")

        f.write(chunk["text"])

    return chunk_file


def read_chunks(directory):
    """
    Read back every chunk .txt file previously written by `write_chunk`,
    in ascending chunk order.
    """

    directory = Path(directory)

    if not directory.exists():
        return

    for chunk_file in sorted(directory.glob("chunk_*.txt")):

        with open(chunk_file, "r", encoding="utf-8") as f:
            content = f.read()

        header, _, text = content.partition(f"{SEPARATOR}\n\n")
        fields = dict(
            line.split(": ", 1) for line in header.strip().splitlines()
        )

        yield {
            "chunk_id": int(fields["CHUNK_ID"]),
            "file_path": fields["FILE"],
            "type": fields["TYPE"],
            "chunking_method": fields["CHUNKING_METHOD"],
            "start_line": int(fields["START_LINE"]),
            "end_line": int(fields["END_LINE"]),
            "text": text,
        }
