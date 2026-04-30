"""Aggregate raw vLLM benchmark JSON files and produce charts.

Reads `benchmarks/results/*.json` (written by `scripts/benchmark_vllm.py`),
groups by hardware, and renders bar charts for the blog.

Usage:
    python benchmarks/analysis.py --output benchmarks/charts/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="benchmarks/results", help="Dir of JSON files.")
    p.add_argument("--output", default="benchmarks/charts", help="Where to write charts.")
    p.add_argument("--no-charts", action="store_true", help="Skip matplotlib output.")
    return p.parse_args()


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    files = sorted(results_dir.glob("*.json"))
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            metrics = data.get("metrics") or data
            metrics["__file__"] = f.name
            out.append(metrics)
        except Exception as e:
            print(f"[skip] {f}: {e}")
    return out


def aggregate(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_hw: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_hw[(r.get("hardware") or "unknown").lower()].append(r)
    summary = {}
    for hw, items in by_hw.items():
        items.sort(key=lambda x: x.get("throughput_tokens_per_sec", 0), reverse=True)
        best = items[0]
        summary[hw] = {
            "n_runs": len(items),
            "best": {
                "model": best.get("model"),
                "throughput_tokens_per_sec": best.get("throughput_tokens_per_sec"),
                "latency_p50_ms": best.get("latency_p50_ms"),
                "latency_p95_ms": best.get("latency_p95_ms"),
                "peak_memory_gb": best.get("peak_memory_gb"),
                "cost_per_1m_tokens_usd": best.get("cost_per_1m_tokens_usd"),
            },
        }
    return summary


def render_charts(results: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[vforge] matplotlib not installed — skipping charts.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    by_hw: dict[str, list[float]] = defaultdict(list)
    for r in results:
        hw = (r.get("hardware") or "unknown").lower()
        if r.get("throughput_tokens_per_sec"):
            by_hw[hw].append(float(r["throughput_tokens_per_sec"]))

    if by_hw:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = list(by_hw.keys())
        values = [max(by_hw[h]) for h in labels]
        bars = ax.bar(labels, values, color=["#FF6F00", "#34A853", "#4285F4", "#999"])
        ax.set_title("vLLM throughput — best run per hardware")
        ax.set_ylabel("tokens / sec")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(out_dir / "throughput.png", dpi=150)
        print(f"[vforge] Wrote {out_dir / 'throughput.png'}")


def main() -> None:
    args = parse_args()
    results = load_results(Path(args.results))
    if not results:
        print(f"No results in {args.results}/. Run scripts/benchmark_vllm.py first.")
        return
    summary = aggregate(results)
    print(json.dumps(summary, indent=2))
    if not args.no_charts:
        render_charts(results, Path(args.output))


if __name__ == "__main__":
    main()
