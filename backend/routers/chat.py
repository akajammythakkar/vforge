import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from database import get_db
from models import ChatMessage, Dataset, DatasetRow
from schemas import ChatRequest, GenerateDatasetRequest, DatasetOut
from services import dataset_generator
from services.llm_provider import LLMError

router = APIRouter()


@router.post("/stream")
async def stream_chat(req: ChatRequest, db: Session = Depends(get_db)):
    """Server-Sent Events streaming chat endpoint."""

    async def event_gen():
        try:
            async for delta in dataset_generator.discover(
                [m.model_dump() for m in req.messages],
                provider=req.provider,
                model=req.model,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                yield {"event": "delta", "data": delta}
            yield {"event": "done", "data": ""}
        except LLMError as e:
            yield {"event": "error", "data": str(e)}

    if req.project_id:
        last_user = next(
            (m for m in reversed(req.messages) if m.role == "user"), None
        )
        if last_user:
            db.add(
                ChatMessage(
                    project_id=req.project_id,
                    role="user",
                    content=last_user.content,
                )
            )
            db.commit()

    return EventSourceResponse(event_gen())


@router.post("/generate-dataset", response_model=DatasetOut, status_code=201)
async def generate_dataset(
    req: GenerateDatasetRequest, db: Session = Depends(get_db)
):
    try:
        rows = await dataset_generator.generate_rows(
            description=req.description,
            domain=req.domain,
            num_rows=req.num_rows,
            provider=req.provider,
            model=req.model,
            seed_examples=req.seed_examples,
        )
    except LLMError as e:
        raise HTTPException(502, f"LLM error: {e}") from e

    if not rows:
        raise HTTPException(422, "Generator produced 0 valid rows. Try a different model.")

    ds = Dataset(
        project_id=req.project_id,
        name=req.dataset_name or f"{req.domain} — {req.num_rows} rows",
        description=req.description,
        format="alpaca",
        source="generated",
        row_count=len(rows),
    )
    db.add(ds)
    db.flush()

    for i, r in enumerate(rows):
        db.add(
            DatasetRow(
                dataset_id=ds.id,
                instruction=r["instruction"],
                input=r.get("input"),
                output=r["output"],
                system_prompt=r.get("system_prompt"),
                quality_score=r.get("quality_score"),
                position=i,
            )
        )
    db.commit()
    db.refresh(ds)
    return ds


@router.get("/history/{project_id}")
def chat_history(project_id: uuid.UUID, db: Session = Depends(get_db)):
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.project_id == project_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [
        {"id": str(m.id), "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in msgs
    ]
