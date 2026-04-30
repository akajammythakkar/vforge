import uuid
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base
from .base import TimestampMixin


class TrainingRun(Base, TimestampMixin):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    base_model: Mapped[str] = mapped_column(String(200), nullable=False)
    hardware: Mapped[str] = mapped_column(String(50), nullable=False)  # tpu | gpu
    method: Mapped[str] = mapped_column(String(50), default="lora")  # lora | qlora | full
    status: Mapped[str] = mapped_column(String(50), default="created")  # created | running | done | failed
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    notebook_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)

    project = relationship("Project", back_populates="training_runs")
