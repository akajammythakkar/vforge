"""CLI: generate an instruction-tuning dataset with an open LLM.

Mirrors the backend `dataset_generator` service so you can run the pipeline
end-to-end without spinning up the FastAPI server.

Usage:
    python scripts/generate_dataset.py \\
        --domain code \\
        --description "Pandas one-liner code completions" \\
        --rows 500 \\
        --provider ollama \\
        --model llama3.1:8b \\
        --out data/code.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make the backend services importable when run from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from services.dataset_generator import generate_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", default="general")
    p.add_argument("--description", default=None)
    p.add_argument("--rows", type=int, default=100)
    p.add_argument("--provider", default="ollama")
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--out", required=True, help="Output JSONL path.")
    return p.parse_args()


async def run() -> None:
    args = parse_args()
    description = args.description or f"Diverse instruction/response pairs for {args.domain}."

    rows = await generate_rows(
        description=description,
        domain=args.domain,
        num_rows=args.rows,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[vforge] Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    asyncio.run(run())
