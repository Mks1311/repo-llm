from pathlib import Path

from repo_llm.chunking.ast_chunker import chunk_by_ast
from repo_llm.chunking.chunk_writer import write_chunk
from repo_llm.chunking.fixed_char_chunker import chunk_by_fixed_size
from repo_llm.chunking.markdown_chunker import chunk_markdown

BASE_DIR = Path(__file__).resolve().parents[3]
CHUNKS_DIR = BASE_DIR / "repo_chunks"


def run_chunking_pipeline(file_paths, repo_url):
    """
    Chunk all supported files in a repository.
    """
    print(f"Chunking {len(file_paths)} files from {repo_url}...")

    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    destination = CHUNKS_DIR / repo_name

    python_char_destination = destination / "python" / "char_chunks"
    python_ast_destination = destination / "python" / "ast_chunks"
    markdown_destination = destination / "markdown"

    python_char_destination.mkdir(parents=True, exist_ok=True)
    python_ast_destination.mkdir(parents=True, exist_ok=True)
    markdown_destination.mkdir(parents=True, exist_ok=True)

    failed_files = []

    char_chunk_number = 1
    ast_chunk_number = 1
    markdown_chunk_number = 1

    for file_path in file_paths:

        extension = Path(file_path).suffix.lower()

        try:
            # -------------------------
            # Python
            # -------------------------
            if extension == ".py":

                # Character/word-based chunking
                try:
                    char_chunks = chunk_by_fixed_size(file_path)
                except Exception as e:
                    print(f"  [WARN] char chunking failed for {file_path}: {e}")
                    char_chunks = []

                for chunk in char_chunks:
                    chunk["chunking_method"] = "char"
                    write_chunk(chunk, python_char_destination, char_chunk_number)
                    char_chunk_number += 1

                # AST-based chunking
                try:
                    ast_chunks = chunk_by_ast(file_path)
                except Exception as e:
                    print(f"  [WARN] AST chunking failed for {file_path}: {e}")
                    ast_chunks = []

                for chunk in ast_chunks:
                    chunk["chunking_method"] = "ast"
                    write_chunk(chunk, python_ast_destination, ast_chunk_number)
                    ast_chunk_number += 1

            # -------------------------
            # Markdown
            # -------------------------
            elif extension == ".md":
                chunks = chunk_markdown(file_path)

                for chunk in chunks:
                    chunk["chunking_method"] = "markdown"
                    write_chunk(chunk, markdown_destination, markdown_chunk_number)
                    markdown_chunk_number += 1

            else:
                continue

        except Exception as e:
            print(f"  [ERROR] Skipping {file_path} due to unexpected error: {e}")
            failed_files.append((file_path, str(e)))
            continue

    if failed_files:
        print(f"\n{len(failed_files)} file(s) failed and were skipped:")
        for fp, err in failed_files:
            print(f"  - {fp}: {err}")

    print(f"\nChunking complete. \nChar chunks created for py file: {char_chunk_number}\nAST chunks created for py file: {ast_chunk_number}\nMarkdown chunks created: {markdown_chunk_number}")

    return destination
