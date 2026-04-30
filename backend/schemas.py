"""Pydantic schemas for the API surface."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Project ────────────────────────────────────────────────
class ProjectBase(BaseModel):
    name: str
    description: str | None = None
    domain: str | None = None


class ProjectCreate(ProjectBase):
    pass


class ProjectOut(ProjectBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


# ── Chat ────────────────────────────────────────────────────
class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    project_id: uuid.UUID | None = None
    messages: list[ChatMessageIn]
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 1024


class GenerateDatasetRequest(BaseModel):
    project_id: uuid.UUID
    description: str = Field(..., description="What the dataset should teach the model.")
    domain: str = Field("general", description="e.g., code, support, medical, legal")
    num_rows: int = Field(50, ge=1, le=5000)
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    seed_examples: list[dict[str, str]] = Field(default_factory=list)
    dataset_name: str | None = None


# ── Dataset ────────────────────────────────────────────────
class DatasetRowIn(BaseModel):
    instruction: str
    input: str | None = None
    output: str
    system_prompt: str | None = None
    quality_score: float | None = None


class DatasetRowOut(DatasetRowIn):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    position: int
    created_at: datetime


class DatasetCreate(BaseModel):
    project_id: uuid.UUID
    name: str
    description: str | None = None
    format: Literal["alpaca", "sharegpt", "jsonl"] = "alpaca"
    source: Literal["generated", "uploaded", "hybrid"] = "generated"


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    description: str | None
    format: str
    source: str
    row_count: int
    created_at: datetime
    updated_at: datetime


# ── Training ───────────────────────────────────────────────
class TrainingRunCreate(BaseModel):
    project_id: uuid.UUID
    dataset_id: uuid.UUID | None = None
    name: str
    base_model: str
    hardware: Literal["tpu", "gpu"]
    method: Literal["lora", "qlora", "full"] = "lora"
    config: dict[str, Any] = Field(default_factory=dict)


class TrainingRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    project_id: uuid.UUID
    dataset_id: uuid.UUID | None
    name: str
    base_model: str
    hardware: str
    method: str
    status: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    notebook_path: str | None
    artifact_uri: str | None
    created_at: datetime
    updated_at: datetime


# ── Benchmarks ─────────────────────────────────────────────
class BenchmarkCreate(BaseModel):
    training_run_id: uuid.UUID | None = None
    name: str
    hardware: str
    benchmark_type: Literal["training", "inference"]
    model: str
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class BenchmarkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    training_run_id: uuid.UUID | None
    name: str
    hardware: str
    benchmark_type: str
    model: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    notes: str | None
    created_at: datetime


# ── Model export ───────────────────────────────────────────
class ModelExportRequest(BaseModel):
    training_run_id: uuid.UUID
    target: Literal["huggingface", "gcs", "drive", "local"]
    repo_id: str | None = None
    bucket_path: str | None = None
    public: bool = False


# ── Settings ───────────────────────────────────────────────
class SettingIn(BaseModel):
    key: str
    value: str
    category: str = "general"
    description: str | None = None


class SettingOut(SettingIn):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    updated_at: datetime
