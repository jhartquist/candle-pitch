#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.20"]
# ///
"""
Upload CREPE safetensors to the Hugging Face Hub.

Pushes the five capacity files plus the model card and upstream license to a
single model repository (default jhartquist/crepe). Requires an HF write
token; run `hf auth login` first.

Usage:
    ./upload_to_hub.py <weights_dir> [--repo-id <id>] [--dry-run]

<weights_dir> must contain {tiny,small,medium,large,full}.safetensors as
produced by export_safetensors.py.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


CAPACITIES = ("tiny", "small", "medium", "large", "full")
DEFAULT_REPO_ID = "jhartquist/crepe"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("weights_dir", type=Path, help="directory holding the five capacity files")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dry-run", action="store_true", help="print plan without uploading")
    args = parser.parse_args()

    here = Path(__file__).parent
    plan: list[tuple[Path, str]] = [
        (here / "model_card.md", "README.md"),
        (here / "UPSTREAM_LICENSE", "LICENSE"),
    ]
    for capacity in CAPACITIES:
        plan.append((args.weights_dir / f"{capacity}.safetensors", f"{capacity}.safetensors"))

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
