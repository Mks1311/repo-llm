from repo_llm.chunker import chunker
from repo_llm.git_actions import clone_repo
from repo_llm.file import list_files

def main():
    start_cli()

def start_cli():
    print("Welcome to Repo LLM, Paste a repo url and ask AI about it! (Python repo only please)")
    repo_url = input("Enter the repository URL: ")
    cloned_repo_destination = clone_repo(repo_url)
    file_paths =  list_files(cloned_repo_destination,)
    chunker(file_paths,repo_url)

if __name__ == "__main__":
    main()