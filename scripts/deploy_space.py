"""Publish OmniAI Cloud to a Hugging Face Space.

    hf auth login                                       # once, with a WRITE token
    python scripts/deploy_space.py NanicyberAI/omniai-cloud

The Space builds the repository's Dockerfile: the paper profile, with every
model fetched from the Hub during the build. Only what the image needs is
uploaded, plus the Space card from deploy/huggingface/README.md in place of the
GitHub README.

The token comes from ``hf auth login`` (or the HF_TOKEN environment variable)
and is never read, printed or stored by this script. Never commit a token.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everything the Docker build reads, and nothing else.
INCLUDE = ["Dockerfile", ".dockerignore", "requirements", "backend"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "downloaded_models", "*.pt", "*.safetensors")


def _stage(directory: Path) -> None:
    for name in INCLUDE:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, directory / name, ignore=SKIP)
        else:
            shutil.copy2(source, directory / name)
    shutil.copy2(ROOT / "deploy" / "huggingface" / "README.md", directory / "README.md")


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "working tree"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", help="owner/name, for example NanicyberAI/omniai-cloud")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception:
        print("Not logged in to Hugging Face. Run `hf auth login` with a write token first.", file=sys.stderr)
        return 1

    print(f"Logged in as {who}. Publishing to spaces/{args.space} ...")
    try:
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        with tempfile.TemporaryDirectory() as staging:
            _stage(Path(staging))
            api.upload_folder(
                folder_path=staging,
                repo_id=args.space,
                repo_type="space",
                commit_message=f"Deploy OmniAI Cloud ({_commit()})",
                # Mirror: files removed from the repository leave the Space too.
                delete_patterns=["backend/**", "requirements/**"],
            )
    except HfHubHTTPError as error:
        print(f"Hugging Face refused the upload: {error}", file=sys.stderr)
        return 1

    owner, name = args.space.split("/", 1)
    print(f"Uploaded. The Space is building: https://huggingface.co/spaces/{args.space}")
    print(f"When it is running, the app is at https://{owner.lower()}-{name.lower()}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
