"""Push a trained adapter to HuggingFace Hub.

Usage:
    python scripts/export_model.py \\
        --local-dir ./out_tpu \\
        --repo-id your-org/gemma-2-2b-tpu-lora \\
        --token $HF_TOKEN
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if not args.token:
        raise SystemExit("Set --token or HF_TOKEN.")

    local = Path(args.local_dir)
    if not local.is_dir():
        raise SystemExit(f"{local} is not a directory.")

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"[vforge] Uploading {local} → {args.repo_id}...")
    api.upload_folder(
        folder_path=str(local), repo_id=args.repo_id, repo_type="model"
    )
    print("[vforge] Done.")


if __name__ == "__main__":
    main()
