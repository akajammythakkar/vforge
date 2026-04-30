"""LoRA fine-tuning on Cloud TPU with JAX + Keras 3.0.

Target: TPU v5e-4 (Sprint hardware), but works on any TPU/GPU/CPU.

Why JAX/Keras here:
  - Direct, first-class TPU support via the JAX backend.
  - `keras_hub` ships LoRA-enabled GemmaCausalLM out of the box.
  - SPMD sharding via `keras.distribution` covers v5e-4 in two lines.

Usage:
    python scripts/finetune_tpu.py \\
        --model google/gemma-4-E2B-it \\
        --data data/code.jsonl \\
        --rank 8 --epochs 1 --batch-size 4 \\
        --lr 1e-4 --max-seq-len 1024 \\
        --output ./out_tpu
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Set the Keras backend BEFORE importing keras.
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras  # noqa: E402
import keras_hub  # noqa: E402
import numpy as np  # noqa: E402


# Map common HF-style identifiers to the matching Keras Hub presets.
HF_TO_PRESET = {
    "google/gemma-4-E2B-it": "gemma4_instruct_2b",
    "google/gemma-4-E4B-it": "gemma4_instruct_4b",
    "google/gemma-4-26B-A4B-it": "gemma4_instruct_26b",
    "google/gemma-4-31B-it": "gemma4_instruct_31b",
    # Legacy Gemma 2 (kept for backward compat)
    "google/gemma-2-2b": "gemma2_2b_en",
    "google/gemma-2-9b": "gemma2_9b_en",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or Keras Hub preset name.")
    p.add_argument("--data", required=True, help="Path to JSONL or Alpaca JSON dataset.")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=str, default="1e-4")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--output", type=str, default="./out_tpu")
    p.add_argument("--limit", type=int, default=None, help="Truncate dataset for smoke tests.")
    p.add_argument(
        "--seed", type=int, default=42, help="Reproducibility seed."
    )
    return p.parse_args()


def load_dataset(path: str, max_seq_len: int, limit: int | None) -> list[str]:
    """Read instruction-tuning rows and format them as a single text-completion stream.

    The Gemma chat template is intentionally simple here — just instruction +
    response — so the same data path works for the GPU script too.
    """
    p = Path(path)
    rows: list[dict] = []
    if p.suffix == ".json":
        rows = json.loads(p.read_text())
    else:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]

    texts: list[str] = []
    for r in rows:
        instruction = (r.get("instruction") or "").strip()
        ctx = (r.get("input") or "").strip()
        output = (r.get("output") or "").strip()
        prompt = f"<start_of_turn>user\n{instruction}"
        if ctx:
            prompt += f"\n\n{ctx}"
        prompt += f"<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        if len(prompt) > max_seq_len * 4:  # rough char→token bound
            prompt = prompt[: max_seq_len * 4]
        texts.append(prompt)
    return texts


def setup_tpu_distribution() -> None:
    """Set up SPMD data-parallel distribution across all TPU devices."""
    devices = keras.distribution.list_devices()
    if not devices:
        print("[vforge] No accelerators detected — running on CPU.")
        return
    if not any(d.lower().startswith("tpu") for d in devices):
        print(f"[vforge] {len(devices)} non-TPU devices: {devices}")
        return
    data_parallel = keras.distribution.DataParallel(devices=devices)
    keras.distribution.set_distribution(data_parallel)
    print(f"[vforge] TPU distribution set across {len(devices)} devices.")


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    setup_tpu_distribution()

    preset = HF_TO_PRESET.get(args.model, args.model)
    print(f"[vforge] Loading {preset} (Keras Hub)...")
    t0 = time.time()
    causal_lm = keras_hub.models.CausalLM.from_preset(preset)
    print(f"[vforge] Loaded in {time.time() - t0:.1f}s")

    print(f"[vforge] Enabling LoRA rank={args.rank}")
    causal_lm.backbone.enable_lora(rank=args.rank)

    causal_lm.preprocessor.sequence_length = args.max_seq_len

    causal_lm.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=float(args.lr), weight_decay=0.01
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    texts = load_dataset(args.data, args.max_seq_len, args.limit)
    print(f"[vforge] {len(texts)} training rows.")

    metrics = {"library": "keras_hub", "backend": "jax", "hardware": "tpu"}
    metrics["dataset_rows"] = len(texts)
    metrics["base_model"] = args.model
    metrics["rank"] = args.rank

    t0 = time.time()
    history = causal_lm.fit(
        x=texts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )
    elapsed = time.time() - t0
    metrics["train_time_sec"] = round(elapsed, 2)

    final_loss = history.history.get("loss", [None])[-1]
    metrics["final_loss"] = float(final_loss) if final_loss is not None else None

    # Tokens/sec estimate: rows × seq_len / wall-clock × epochs
    approx_tokens = len(texts) * args.max_seq_len * args.epochs
    if elapsed > 0:
        metrics["approx_train_tokens_per_sec"] = round(approx_tokens / elapsed, 2)

    print(f"[vforge] Training done in {elapsed:.1f}s. Saving to {out}/")
    causal_lm.save_weights(str(out / "lora.weights.h5"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "preset": preset,
                "rank": args.rank,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_seq_len": args.max_seq_len,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
