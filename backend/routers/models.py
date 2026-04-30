from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config import get_settings
from database import get_db
from models import TrainingRun
from schemas import ModelExportRequest
from services.export_service import push_model_to_hf

router = APIRouter()


@router.post("/export")
def export_model(payload: ModelExportRequest, db: Session = Depends(get_db)):
    s = get_settings()
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.id == payload.training_run_id)
        .first()
    )
    if not run:
        raise HTTPException(404, "Training run not found")
    if not run.artifact_uri:
        raise HTTPException(400, "Training run has no artifact_uri set yet.")

    if payload.target == "huggingface":
        if not s.hf_token:
            raise HTTPException(400, "HF_TOKEN not configured")
        if not payload.repo_id:
            raise HTTPException(400, "repo_id required for huggingface target")
        return push_model_to_hf(
            run, repo_id=payload.repo_id, token=s.hf_token, local_dir=run.artifact_uri
        )

    if payload.target == "local":
        return {"path": run.artifact_uri, "type": "local"}

    raise HTTPException(501, f"Target '{payload.target}' not yet implemented in MVP")
