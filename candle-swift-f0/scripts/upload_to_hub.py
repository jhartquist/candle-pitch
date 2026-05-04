#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.20"]
# ///
"""
Upload SwiftF0 safetensors to the Hugging Face Hub.

Pushes the model file plus the model card and upstream license to a single
model repository (default jhartquist/swift-f0). Requires an HF write token;
run `hf auth login` first.

Usage:
    ./upload_to_hub.py <weights.safetensors> [--repo-id <id>] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "jhartquist/swift-f0"
WEIGHTS_NAME_IN_REPO = "swift-f0.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("weights", type=Path, help="local path to swift-f0.safetensors")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dry-run", action="store_true", help="print plan without uploading")
    args = parser.parse_args()

    here = Path(__file__).parent
    plan: list[tuple[Path, str]] = [
        (here / "model_card.md", "README.md"),
        (here / "UPSTREAM_LICENSE", "LICENSE"),
        (args.weights, WEIGHTS_NAME_IN_REPO),
    ]

    missing = [str(p) for p, _ in plan if not p.exists()]
    if missing:
        print("missing files:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    print(f"target: https://huggingface.co/{args.repo_id}")
    for path, name in plan:
        print(f"  {path}  ->  {name}")

    if args.dry_run:
        print("(dry run, nothing uploaded)")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    for path, name in plan:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"uploaded {name}")


if __name__ == "__main__":
    main()
