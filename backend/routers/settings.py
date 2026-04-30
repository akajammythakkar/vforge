from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Setting
from schemas import SettingIn, SettingOut

router = APIRouter()


@router.get("", response_model=list[SettingOut])
def list_settings(category: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Setting)
    if category:
        q = q.filter(Setting.category == category)
    return q.order_by(Setting.key.asc()).all()


@router.put("", response_model=SettingOut)
def upsert_setting(payload: SettingIn, db: Session = Depends(get_db)):
    s = db.query(Setting).filter(Setting.key == payload.key).first()
    if s:
        s.value = payload.value
        s.category = payload.category
        s.description = payload.description
    else:
        s = Setting(**payload.model_dump())
        db.add(s)
    db.commit()
    db.refresh(s)
    return s


@router.delete("/{key}", status_code=204)
def delete_setting(key: str, db: Session = Depends(get_db)):
    s = db.query(Setting).filter(Setting.key == key).first()
    if not s:
        raise HTTPException(404, "Setting not found")
    db.delete(s)
    db.commit()
