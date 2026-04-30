import uuid
from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base
from .base import TimestampMixin


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(100), nullable=True)

    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="project", cascade="all, delete-orphan")
    training_runs = relationship("TrainingRun", back_populates="project", cascade="all, delete-orphan")
