import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Project
from schemas import ProjectCreate, ProjectOut

router = APIRouter()


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.created_at.desc()).all()


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    p = Project(**payload.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: uuid.UUID, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: uuid.UUID, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    db.delete(p)
    db.commit()
