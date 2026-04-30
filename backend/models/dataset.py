import uuid
from sqlalchemy import ForeignKey, String, Text, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base
from .base import TimestampMixin


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    format: Mapped[str] = mapped_column(String(50), default="alpaca")  # alpaca | sharegpt | jsonl
    source: Mapped[str] = mapped_column(String(50), default="generated")  # generated | uploaded | hybrid
    row_count: Mapped[int] = mapped_column(Integer, default=0)

    project = relationship("Project", back_populates="datasets")
    rows = relationship("DatasetRow", back_populates="dataset", cascade="all, delete-orphan")


class DatasetRow(Base, TimestampMixin):
    __tablename__ = "dataset_rows"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    instruction: Mapped[str] = mapped_column(Text, nullable=False)
    input: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0)

    dataset = relationship("Dataset", back_populates="rows")
