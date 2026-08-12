from repo_llm.bm25_index import save_chunks_to_bm25_index
from repo_llm.chunking.pipeline import run_chunking_pipeline
from repo_llm.file_discovery import discover_source_files
from repo_llm.repo_cloner import clone_repository
from repo_llm.vector_store import save_chunks_to_vector_store


def main():
    start_cli()


def start_cli():
    print("Welcome to Repo LLM, Paste a repo url and ask AI about it! (Python repo only please)")
    # repo_url = input("Enter the repository URL: ")
    repo_url = "https://github.com/psf/requests"
    cloned_repo_destination = clone_repository(repo_url)
    file_paths = discover_source_files(cloned_repo_destination)
    chunks_path = run_chunking_pipeline(file_paths, repo_url)
    save_chunks_to_vector_store(chunks_path)
    save_chunks_to_bm25_index(chunks_path)


if __name__ == "__main__":
    main()
