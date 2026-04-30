import uuid
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database import get_db
from models import BenchmarkResult
from schemas import BenchmarkCreate, BenchmarkOut
from services.benchmark_service import compare, record_benchmark

router = APIRouter()


@router.post("/runs", response_model=BenchmarkOut, status_code=201)
def create_benchmark(payload: BenchmarkCreate, db: Session = Depends(get_db)):
    bm = record_benchmark(
        db,
        name=payload.name,
        hardware=payload.hardware,
        benchmark_type=payload.benchmark_type,
        model=payload.model,
        config=payload.config,
        metrics=payload.metrics,
        notes=payload.notes,
        training_run_id=payload.training_run_id,
    )
    return bm


@router.get("/runs", response_model=list[BenchmarkOut])
def list_benchmarks(
    benchmark_type: str | None = None,
    hardware: str | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(BenchmarkResult)
    if benchmark_type:
        q = q.filter(BenchmarkResult.benchmark_type == benchmark_type)
    if hardware:
        q = q.filter(BenchmarkResult.hardware == hardware)
    return q.order_by(BenchmarkResult.created_at.desc()).limit(200).all()


@router.get("/compare")
def compare_benchmarks(
    model: str | None = Query(None), db: Session = Depends(get_db)
):
    return compare(db, model=model)
