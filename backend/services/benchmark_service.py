"""Benchmark aggregation & comparison helpers.

Actual benchmark execution lives in `scripts/benchmark_vllm.py` —
this module records / queries results stored in Postgres.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from models import BenchmarkResult


def record_benchmark(
    db: Session,
    *,
    name: str,
    hardware: str,
    benchmark_type: str,
    model: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    notes: str | None = None,
    training_run_id: uuid.UUID | None = None,
) -> BenchmarkResult:
    bm = BenchmarkResult(
        name=name,
        hardware=hardware,
        benchmark_type=benchmark_type,
        model=model,
        config=config,
        metrics=metrics,
        notes=notes,
        training_run_id=training_run_id,
    )
    db.add(bm)
    db.commit()
    db.refresh(bm)
    return bm


def compare(db: Session, model: str | None = None) -> dict[str, Any]:
    """Return TPU vs GPU comparison metrics for the latest matching benchmarks."""
    q = db.query(BenchmarkResult)
    if model:
        q = q.filter(BenchmarkResult.model == model)
    rows = q.order_by(BenchmarkResult.created_at.desc()).limit(50).all()

    by_hw: dict[str, list[BenchmarkResult]] = {}
    for r in rows:
        by_hw.setdefault(r.hardware, []).append(r)

    summary = {}
    for hw, items in by_hw.items():
        latest = items[0]
        summary[hw] = {
            "name": latest.name,
            "model": latest.model,
            "metrics": latest.metrics,
            "config": latest.config,
            "created_at": latest.created_at.isoformat(),
        }
    return {"comparison": summary, "count": len(rows)}
