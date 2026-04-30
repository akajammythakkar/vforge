import uuid
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from database import Base
from .base import TimestampMixin


class BenchmarkResult(Base, TimestampMixin):
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hardware: Mapped[str] = mapped_column(String(50), nullable=False)  # tpu_v5e_4 | rtx_4070 | a100 | etc.
    benchmark_type: Mapped[str] = mapped_column(String(50), nullable=False)  # training | inference
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
