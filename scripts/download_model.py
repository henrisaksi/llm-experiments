"""Utility to download compatible Gemma 4 model weights into the ./models directory."""
import argparse
import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def download_model(repo_id: str, target_name: str | None = None):
    folder_name = target_name or repo_id.split("/")[-1]
    dest_dir = os.path.join(MODELS_DIR, folder_name)
    print(f"Downloading '{repo_id}' into '{dest_dir}'...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest_dir,
    )
    print(f"Successfully downloaded to '{dest_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download compatible Gemma 4 weight checkpoints.")
    parser.add_argument("repo_id", help="Hugging Face repo id, e.g. google/gemma-4-E2B-it")
    parser.add_argument("--name", help="Custom folder name under ./models", default=None)
    args = parser.parse_args()

    download_model(args.repo_id, args.name)
