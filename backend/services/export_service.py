"""Dataset & model export utilities."""
from __future__ import annotations

import io
import json
from typing import Iterable

from sqlalchemy.orm import Session

from models import Dataset, DatasetRow, TrainingRun


def _row_to_alpaca(r: DatasetRow) -> dict:
    return {
        "instruction": r.instruction,
        "input": r.input or "",
        "output": r.output,
    }


def _row_to_sharegpt(r: DatasetRow) -> dict:
    convs = []
    if r.system_prompt:
        convs.append({"from": "system", "value": r.system_prompt})
    user_text = r.instruction + (("\n\n" + r.input) if r.input else "")
    convs.append({"from": "human", "value": user_text})
    convs.append({"from": "gpt", "value": r.output})
    return {"conversations": convs}


def _row_to_jsonl(r: DatasetRow) -> dict:
    return {
        "messages": [
            *(
                [{"role": "system", "content": r.system_prompt}]
                if r.system_prompt
                else []
            ),
            {
                "role": "user",
                "content": r.instruction
                + (("\n\n" + r.input) if r.input else ""),
            },
            {"role": "assistant", "content": r.output},
        ]
    }


def export_dataset(db: Session, dataset_id, fmt: str) -> bytes:
    rows: Iterable[DatasetRow] = (
        db.query(DatasetRow)
        .filter(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.position.asc())
        .all()
    )
    fmt = fmt.lower()
    buf = io.StringIO()
    if fmt == "alpaca":
        json.dump([_row_to_alpaca(r) for r in rows], buf, ensure_ascii=False, indent=2)
    elif fmt == "sharegpt":
        for r in rows:
            buf.write(json.dumps(_row_to_sharegpt(r), ensure_ascii=False) + "\n")
    elif fmt == "jsonl":
        for r in rows:
            buf.write(json.dumps(_row_to_jsonl(r), ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return buf.getvalue().encode("utf-8")


def push_dataset_to_hf(
    db: Session, dataset_id, repo_id: str, token: str, private: bool = True
) -> dict:
    """Upload the dataset (Alpaca JSON) to HF Hub as a dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    payload = export_dataset(db, dataset_id, "alpaca")
    api.upload_file(
        path_or_fileobj=io.BytesIO(payload),
        path_in_repo="alpaca.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    return {"repo_id": repo_id, "type": "dataset", "path": "alpaca.json"}


def push_model_to_hf(run: TrainingRun, repo_id: str, token: str, local_dir: str) -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    return {"repo_id": repo_id, "type": "model"}
