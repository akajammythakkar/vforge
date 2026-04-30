import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import get_db
from models import TrainingRun
from schemas import TrainingRunCreate, TrainingRunOut
from services.training_service import render_training_notebook

router = APIRouter()


DEFAULT_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "epochs": 1,
    "batch_size": 4,
    "lr": "1e-4",
    "max_seq_len": 1024,
    "warmup_ratio": 0.03,
    "seed": 42,
}


@router.post("/runs", response_model=TrainingRunOut, status_code=201)
def create_run(payload: TrainingRunCreate, db: Session = Depends(get_db)):
    config = {**DEFAULT_CONFIG, **(payload.config or {})}
    run = TrainingRun(
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        name=payload.name,
        base_model=payload.base_model,
        hardware=payload.hardware,
        method=payload.method,
        config=config,
        metrics={},
        status="created",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    dataset_uri = f"datasets/{payload.dataset_id}.json" if payload.dataset_id else "data.json"
    nb_path = render_training_notebook(
        run_id=run.id,
        base_model=payload.base_model,
        hardware=payload.hardware,
        method=payload.method,
        dataset_uri=dataset_uri,
        config=config,
    )
    run.notebook_path = str(nb_path)
    db.commit()
    db.refresh(run)
    return run


@router.get("/runs", response_model=list[TrainingRunOut])
def list_runs(
    project_id: uuid.UUID | None = None, db: Session = Depends(get_db)
):
    q = db.query(TrainingRun)
    if project_id:
        q = q.filter(TrainingRun.project_id == project_id)
    return q.order_by(TrainingRun.created_at.desc()).all()


@router.get("/runs/{run_id}", response_model=TrainingRunOut)
def get_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    r = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not r:
        raise HTTPException(404, "Training run not found")
    return r


@router.get("/runs/{run_id}/notebook")
def download_notebook(run_id: uuid.UUID, db: Session = Depends(get_db)):
    r = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not r or not r.notebook_path:
        raise HTTPException(404, "Notebook not found")
    return FileResponse(
        r.notebook_path,
        media_type="application/x-ipynb+json",
        filename=f"vforge_{r.hardware}_{r.id}.ipynb",
    )


@router.patch("/runs/{run_id}/metrics", response_model=TrainingRunOut)
def update_metrics(
    run_id: uuid.UUID, metrics: dict, db: Session = Depends(get_db)
):
    r = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not r:
        raise HTTPException(404, "Training run not found")
    r.metrics = {**(r.metrics or {}), **metrics}
    if metrics.get("status"):
        r.status = metrics["status"]
    db.commit()
    db.refresh(r)
    return r
