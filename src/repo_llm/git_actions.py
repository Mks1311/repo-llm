from pathlib import Path
from git import Repo


BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_REPO_DIR = BASE_DIR / "local_repo"


def clone_repo(repo_url):
    repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    destination = LOCAL_REPO_DIR / repo_name

    if destination.exists():
        print(f"Repository already exists at {destination}. Skipping clone.")
        return destination

    LOCAL_REPO_DIR.mkdir(parents=True, exist_ok=True)

    Repo.clone_from(repo_url, destination)

    print(f"Repository '{repo_name}' cloned successfully.")
    return destination