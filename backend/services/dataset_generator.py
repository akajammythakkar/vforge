"""Conversational dataset generation.

Two paths:
  1. `discover()` — chat-style intent discovery (used by the chat router).
  2. `generate_rows()` — bulk QA-pair synthesis given a description + domain.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .llm_provider import ChatMsg, get_provider


SYSTEM_DISCOVERY = """You are vForge, a helpful assistant that helps users design fine-tuning datasets for LLMs.

Your job in this chat:
  1. Ask clarifying questions about WHAT the user wants the model to do.
  2. Suggest an instruction/response format and a few example rows.
  3. When the user is ready, call out: "Ready to generate dataset?"

Be concise (≤ 4 sentences per turn). Always ground suggestions in concrete examples."""


SYSTEM_GENERATOR = """You are an expert dataset author for LLM fine-tuning.

Generate diverse, high-quality instruction/response pairs in strict JSONL.
Each line must be a JSON object with EXACTLY these keys:
  - "instruction": a clear task description
  - "input": optional context (use "" if none)
  - "output": the model's expected response

Rules:
  - Vary phrasing, length, and difficulty. Avoid templated repetition.
  - For code: include realistic, runnable code with explanations.
  - For support / medical / legal: include domain-appropriate edge cases.
  - Output ONLY raw JSONL. No prose, no code fences, no numbering.
  - Generate the requested number of rows; no more, no less.
"""


async def discover(messages: list[dict[str, str]], provider: str, model: str, **kwargs):
    """Yield streaming text deltas from the discovery chat."""
    msgs = [ChatMsg(role="system", content=SYSTEM_DISCOVERY)] + [
        ChatMsg(role=m["role"], content=m["content"]) for m in messages
    ]
    p = get_provider(provider)
    async for delta in p.stream(msgs, model, **kwargs):
        yield delta


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse JSONL output, tolerating common LLM quirks."""
    text = text.strip()
    text = re.sub(r"^```(?:json|jsonl)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "instruction" in obj and "output" in obj:
            rows.append(
                {
                    "instruction": str(obj.get("instruction", "")).strip(),
                    "input": str(obj.get("input", "") or ""),
                    "output": str(obj.get("output", "")).strip(),
                    "system_prompt": obj.get("system_prompt"),
                    "quality_score": obj.get("quality_score"),
                }
            )
    return rows


async def generate_rows(
    *,
    description: str,
    domain: str,
    num_rows: int,
    provider: str,
    model: str,
    seed_examples: list[dict[str, str]] | None = None,
    batch_size: int = 25,
) -> list[dict[str, Any]]:
    """Generate `num_rows` rows by issuing multiple LLM calls of `batch_size`."""
    p = get_provider(provider)
    seed_examples = seed_examples or []
    seed_block = ""
    if seed_examples:
        seed_block = "\n\nFew-shot examples:\n" + "\n".join(
            json.dumps(ex, ensure_ascii=False) for ex in seed_examples[:5]
        )

    rows: list[dict[str, Any]] = []
    remaining = num_rows
    batch_idx = 0
    while remaining > 0 and batch_idx < 50:  # hard cap
        n = min(batch_size, remaining)
        user_prompt = (
            f"Domain: {domain}\n"
            f"Goal: {description}\n"
            f"Generate exactly {n} JSONL rows now."
            f"{seed_block}"
        )
        msgs = [
            ChatMsg(role="system", content=SYSTEM_GENERATOR),
            ChatMsg(role="user", content=user_prompt),
        ]
        text = await p.complete(msgs, model, temperature=0.8, max_tokens=4096)
        new_rows = _parse_jsonl(text)
        if not new_rows:
            batch_idx += 1
            continue
        rows.extend(new_rows)
        remaining = num_rows - len(rows)
        batch_idx += 1

    return rows[:num_rows]
