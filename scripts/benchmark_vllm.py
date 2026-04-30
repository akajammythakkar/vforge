"""vLLM inference benchmark — works on TPU and GPU backends.

Measures:
  - Throughput (tokens / sec)
  - Latency p50 / p95 / p99 (per-request)
  - Peak memory (GPU only — TPU memory is not exposed the same way)
  - $ / 1M tokens (using a configurable hourly rate)

Usage:
    python scripts/benchmark_vllm.py \\
        --model ./out_tpu \\
        --hardware tpu \\
        --prompts 256 --max-tokens 256 \\
        --hourly-cost-usd 1.20

Cost defaults (April 2026 list prices, on-demand, illustrative only):
    TPU v5e-4   ~ $1.20 / hour
    NVIDIA A100 ~ $3.67 / hour
    NVIDIA L4   ~ $0.71 / hour
    NVIDIA T4   ~ $0.35 / hour
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# vLLM is the only heavy import — guard it so this file can be parsed without it.
try:
    from vllm import LLM, SamplingParams
except ImportError as e:  # pragma: no cover
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _VLLM_IMPORT_ERROR = e

DEFAULT_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain the difference between processes and threads in 3 sentences.",
    "Write a SQL query to find the top 5 customers by total order value.",
    "Summarize the plot of Hamlet in two paragraphs.",
    "Translate this sentence to French: 'The TPU is faster than I expected.'",
    "Write a haiku about silicon chips.",
    "What is the time complexity of merge sort? Why?",
    "Write a regex that matches valid IPv4 addresses.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path.")
    p.add_argument(
        "--hardware",
        choices=["tpu", "gpu", "cpu"],
        required=True,
        help="What the benchmark is logically running on.",
    )
    p.add_argument(
        "--device",
        default=None,
        help='vLLM device override (e.g. "tpu", "cuda", "cpu"). Defaults to --hardware.',
    )
    p.add_argument("--prompts", type=int, default=128, help="Total prompts to run.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hourly-cost-usd", type=float, default=1.20)
    p.add_argument(
        "--output",
        default=None,
        help="Path to write JSON metrics (default: benchmarks/results/<hardware>_<ts>.json).",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Run name. Defaults to <hardware>-<model-basename>.",
    )
    return p.parse_args()


def measure_peak_gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    except Exception:
        pass
    return None


def main() -> None:
    args = parse_args()
    if LLM is None:  # pragma: no cover
        raise SystemExit(
            f"vllm import failed: {_VLLM_IMPORT_ERROR}. "
            "Install with `pip install vllm` (and the right TPU/GPU build)."
        )

    # Build the prompt list (cycle through DEFAULT_PROMPTS to reach --prompts).
    prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(args.prompts)]

    device = args.device or {"tpu": "tpu", "gpu": "cuda", "cpu": "cpu"}[args.hardware]
    print(f"[vforge] Loading {args.model} on {device}...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        device=device,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"[vforge] Loaded in {load_time:.1f}s")

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Warmup
    print("[vforge] Warmup (8 prompts)...")
    llm.generate(prompts[:8], sp)

    # Per-request latency (sequential subset, 16 prompts)
    print("[vforge] Per-request latency (16 prompts)...")
    latencies_ms: list[float] = []
    for p in prompts[:16]:
        s = time.time()
        llm.generate([p], sp)
        latencies_ms.append((time.time() - s) * 1000)

    # Bulk throughput (all prompts, batched)
    print(f"[vforge] Throughput run ({len(prompts)} prompts)...")
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    total_time = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    cost_per_1m = (
        (total_time / 3600.0) * args.hourly_cost_usd / (total_tokens / 1_000_000)
        if total_tokens > 0
        else 0.0
    )

    metrics = {
        "model": args.model,
        "hardware": args.hardware,
        "device": device,
        "load_time_sec": round(load_time, 2),
        "total_prompts": len(prompts),
        "total_tokens": total_tokens,
        "total_time_sec": round(total_time, 3),
        "throughput_tokens_per_sec": round(throughput, 2),
        "latency_p50_ms": round(statistics.median(latencies_ms), 2),
        "latency_p95_ms": round(
            statistics.quantiles(latencies_ms, n=20)[-1] if len(latencies_ms) >= 20 else max(latencies_ms),
            2,
        ),
        "latency_p99_ms": round(max(latencies_ms), 2),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "hourly_cost_usd": args.hourly_cost_usd,
        "cost_per_1m_tokens_usd": round(cost_per_1m, 4),
    }
    peak = measure_peak_gpu_memory_gb()
    if peak is not None:
        metrics["peak_memory_gb"] = peak

    name = args.name or f"{args.hardware}-{Path(args.model).name}"
    out_path = Path(args.output) if args.output else Path(
        f"benchmarks/results/{name}_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"name": name, "metrics": metrics, "config": vars(args)}, indent=2))
    print(f"[vforge] Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
