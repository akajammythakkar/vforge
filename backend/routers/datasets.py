import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from database import get_db
from models import Dataset, DatasetRow
from schemas import DatasetCreate, DatasetOut, DatasetRowIn, DatasetRowOut
from services.export_service import export_dataset

router = APIRouter()


@router.get("", response_model=list[DatasetOut])
def list_datasets(
    project_id: uuid.UUID | None = None, db: Session = Depends(get_db)
):
    q = db.query(Dataset)
    if project_id:
        q = q.filter(Dataset.project_id == project_id)
    return q.order_by(Dataset.created_at.desc()).all()


@router.post("", response_model=DatasetOut, status_code=201)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    ds = Dataset(**payload.model_dump())
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    db.delete(ds)
    db.commit()


@router.get("/{dataset_id}/rows", response_model=list[DatasetRowOut])
def list_rows(
    dataset_id: uuid.UUID,
    skip: int = 0,
    limit: int = Query(500, le=2000),
    db: Session = Depends(get_db),
):
    return (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.position.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.post("/{dataset_id}/rows", response_model=DatasetRowOut, status_code=201)
def add_row(dataset_id: uuid.UUID, payload: DatasetRowIn, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    last = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.position.desc())
        .first()
    )
    pos = (last.position + 1) if last else 0
    row = DatasetRow(dataset_id=dataset_id, position=pos, **payload.model_dump())
    db.add(row)
    ds.row_count += 1
    db.commit()
    db.refresh(row)
    return row


@router.put("/rows/{row_id}", response_model=DatasetRowOut)
def update_row(row_id: uuid.UUID, payload: DatasetRowIn, db: Session = Depends(get_db)):
    row = db.query(DatasetRow).filter(DatasetRow.id == row_id).first()
    if not row:
        raise HTTPException(404, "Row not found")
    for k, v in payload.model_dump().items():
        setattr(row, k, v)
    db.commit()
    db.refresh(row)
    return row


@router.delete("/rows/{row_id}", status_code=204)
def delete_row(row_id: uuid.UUID, db: Session = Depends(get_db)):
    row = db.query(DatasetRow).filter(DatasetRow.id == row_id).first()
    if not row:
        raise HTTPException(404, "Row not found")
    ds = db.query(Dataset).filter(Dataset.id == row.dataset_id).first()
    if ds and ds.row_count > 0:
        ds.row_count -= 1
    db.delete(row)
    db.commit()


@router.post("/{dataset_id}/export")
def export(
    dataset_id: uuid.UUID,
    format: str = Query("alpaca", pattern="^(alpaca|sharegpt|jsonl)$"),
    db: Session = Depends(get_db),
):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(404, "Dataset not found")
    payload = export_dataset(db, dataset_id, format)
    media = "application/json" if format == "alpaca" else "application/x-ndjson"
    filename = f"{ds.name.replace(' ', '_')}_{format}.{'json' if format == 'alpaca' else 'jsonl'}"
    return Response(
        content=payload,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
