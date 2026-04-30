"""LoRA fine-tuning on a single CUDA GPU with PyTorch + PEFT.

Companion to scripts/finetune_tpu.py — same dataset, same model, same hyperparams.
Use this for the TPU-vs-GPU baseline.

Usage:
    python scripts/finetune_gpu.py \\
        --model google/gemma-4-E2B-it \\
        --data data/code.jsonl \\
        --rank 8 --epochs 1 --batch-size 4 \\
        --lr 1e-4 --max-seq-len 1024 \\
        --output ./out_gpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=str, default="1e-4")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--output", type=str, default="./out_gpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-accum", type=int, default=4)
    return p.parse_args()


def load_dataset_jsonl(path: str, limit: int | None) -> list[dict]:
    p = Path(path)
    if p.suffix == ".json":
        rows = json.loads(p.read_text())
    else:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    return rows


def format_row(r: dict) -> str:
    instruction = (r.get("instruction") or "").strip()
    ctx = (r.get("input") or "").strip()
    output = (r.get("output") or "").strip()
    prompt = f"<start_of_turn>user\n{instruction}"
    if ctx:
        prompt += f"\n\n{ctx}"
    prompt += f"<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
    return prompt


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[vforge] CUDA: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    else:
        print("[vforge] No CUDA — running on CPU (training will be slow).")

    print(f"[vforge] Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[vforge] Loading model {args.model}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    print(f"[vforge] Loaded in {time.time() - t0:.1f}s")

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = (
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.parameters()),
    )
    print(f"[vforge] LoRA r={args.rank}: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    rows = load_dataset_jsonl(args.data, args.limit)
    texts = [format_row(r) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    def tok(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"[vforge] {len(ds)} training rows.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        num_train_epochs=args.epochs,
        learning_rate=float(args.lr),
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        warmup_ratio=0.03,
        seed=args.seed,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"[vforge] Training done in {elapsed:.1f}s")

    metrics = {
        "library": "transformers+peft",
        "hardware": "gpu" if torch.cuda.is_available() else "cpu",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "base_model": args.model,
        "rank": args.rank,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataset_rows": len(ds),
        "train_time_sec": round(elapsed, 2),
        "final_loss": float(train_result.training_loss),
        "trainable_params": trainable,
        "total_params": total,
    }
    if torch.cuda.is_available():
        metrics["peak_memory_gb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3
        )

    approx_tokens = len(ds) * args.max_seq_len * args.epochs
    if elapsed > 0:
        metrics["approx_train_tokens_per_sec"] = round(approx_tokens / elapsed, 2)

    print(f"[vforge] Saving adapter to {out}/")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
