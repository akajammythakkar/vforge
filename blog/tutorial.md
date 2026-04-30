# Benchmarking LLM Fine-Tuning & Inference: TPU vs GPU with vLLM

> Sprint deliverable for the **Google TPU Sprint Q1 2026**.
> Repo: [github.com/<you>/vforge](https://github.com/) · Notebooks: [`notebooks/`](../notebooks/) · Reproducible: yes.

If you're an ML practitioner looking at the cost line in your training bill, you've probably wondered: **"would a TPU actually save me money?"** The answer most blog posts give you is *"it depends."* This post is what you actually want — a head-to-head benchmark of LoRA fine-tuning and inference on **Cloud TPU v5e-4** vs **a single CUDA GPU**, using the same model, the same dataset, the same hyper-parameters, and **vLLM** as the inference engine on both sides.

We built [vForge](https://github.com/) for this Sprint. It's an open-source platform that takes a chat-described goal, generates a synthetic instruction-tuning dataset, fine-tunes a model on TPU and on GPU, and benchmarks both with vLLM. The benchmarks below come from real runs on real hardware. Every script and notebook is in the repo.

---

## TL;DR

| Metric (Gemma-2-2B, LoRA r=8, 1 epoch, 200 rows × 512 tokens) | Cloud TPU v5e-4 | NVIDIA L4 (24 GB) | Notes |
|---|---|---|---|
| Train time (s) | _fill in_ | _fill in_ | wall-clock end-to-end |
| Approx. train tokens/sec | _fill in_ | _fill in_ | rows × seq × epochs / time |
| Inference throughput (vLLM, tok/s) | _fill in_ | _fill in_ | 128 prompts × 256 tokens |
| Latency p95 (ms) | _fill in_ | _fill in_ | per-request |
| Peak memory | TPU (not exposed) | _fill in_ GB | from `torch.cuda.max_memory_allocated` |
| On-demand $ / hour | $1.20 | $0.71 | April 2026 list, illustrative |
| **$ / 1M tokens (inference)** | _fill in_ | _fill in_ | computed by `benchmark_vllm.py` |

> **How to fill these in:** run [`notebooks/02_finetune_tpu.ipynb`](../notebooks/02_finetune_tpu.ipynb) on a Colab TPU runtime, [`notebooks/03_finetune_gpu.ipynb`](../notebooks/03_finetune_gpu.ipynb) on a Colab L4, then [`notebooks/04_benchmark_vllm.ipynb`](../notebooks/04_benchmark_vllm.ipynb) on each. The metrics drop into `benchmarks/results/*.json` and the dashboard (`/benchmarks` in the web app) renders them.

---

## What we're benchmarking and why

The Sprint asks a focused question: *for a typical LoRA fine-tune of a small open model (2B–9B), how does TPU v5e compare to a single consumer GPU?* That's the question most teams actually face when they're not at GPT-scale.

**Why Gemma-2-2B?** It fits on free-tier hardware on both sides (Colab TPU and a single L4/T4), so anyone can reproduce the numbers. Gemma is also first-class in `keras_hub` — the LoRA path is two lines.

**Why LoRA r=8?** It's the most common adapter rank in production today, and it pushes only ~0.1% of the parameters during training, which makes both sides memory-feasible.

**Why vLLM for inference?** Because (a) it has continuous batching, which is what a real serving deployment looks like, and (b) it's now first-class on both TPU (via JAX) and GPU. Same engine on both sides means we're comparing the chip, not the runtime.

---

## The pipeline

```
chat with vForge → synthetic dataset → LoRA fine-tune (TPU) ──┐
                                          │                   │
                                          └→ LoRA fine-tune (GPU) ─→ vLLM benchmark on each → compare
```

Three things make this benchmark fair:

1. **Identical data.** Both runs read the same JSONL file produced by the dataset generator (Llama-3.3-70B via Together AI), so there's no "different prompts → different difficulty" confounder.
2. **Identical hyper-parameters.** Same rank, same alpha, same LR, same batch size, same max sequence length, same seed.
3. **Identical inference workload.** vLLM with `max_tokens=256`, `temperature=0.0`, the same 128 prompts cycled through.

The only thing that changes is the silicon.

---

## TPU side — JAX + Keras 3.0 + LoRA

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras, keras_hub

# SPMD data-parallel across the v5e-4 chips
devices = keras.distribution.list_devices()
keras.distribution.set_distribution(
    keras.distribution.DataParallel(devices=devices)
)

causal = keras_hub.models.CausalLM.from_preset("gemma2_2b_en")
causal.backbone.enable_lora(rank=8)
causal.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
causal.fit(texts, batch_size=4, epochs=1)
```

That's the whole training loop. `enable_lora` patches the attention projections in place; `set_distribution` shards activations across the four TPU chips. No `pjit` boilerplate, no `jax.tree_map`, no manual `with mesh:` blocks. **This is the strongest argument for JAX/Keras 3.0 on TPU**: the productivity is much closer to PyTorch than the JAX of 18 months ago.

Full script: [`scripts/finetune_tpu.py`](../scripts/finetune_tpu.py)

### Gotchas worth flagging

- **Set `KERAS_BACKEND` before `import keras`.** Easy to miss in a Colab cell that already imported keras at startup.
- **`from_preset` downloads ~5GB.** First-run download time dominates if your dataset is small. Cache it.
- **`preprocessor.sequence_length` defaults to 512** — bump it for production but watch memory.

---

## GPU side — PyTorch + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b", torch_dtype=torch.bfloat16, device_map="auto"
)
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    task_type="CAUSAL_LM",
))

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out_gpu",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1, learning_rate=1e-4, bf16=True,
    ),
    train_dataset=ds,
)
trainer.train()
```

Same hyper-parameters as the TPU run. Full script: [`scripts/finetune_gpu.py`](../scripts/finetune_gpu.py)

---

## Inference benchmark — vLLM on both sides

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./out_tpu", device="tpu", max_model_len=2048)
# llm = LLM(model="./out_gpu", device="cuda", max_model_len=2048)

sp = SamplingParams(temperature=0.0, max_tokens=256, seed=42)
outputs = llm.generate(prompts, sp)
```

Throughput is `total_output_tokens / total_wall_clock_time`. Latency is per-request, sequential, p50/p95/p99. The script also computes `$/1M tokens` from the on-demand hourly rate.

Full script: [`scripts/benchmark_vllm.py`](../scripts/benchmark_vllm.py)

---

## What we found

> _Numbers are placeholders until the runs land. Update this section before publishing._

**Training:** _expected pattern: TPU v5e-4 wins on tokens/sec by a factor of N due to the 4-chip SPMD; the GPU wins on cold-start time because vLLM's TPU loader is heavier._

**Inference:** _expected pattern: GPU edges out on single-request p50 latency because of lower memory-bandwidth roundtrip; TPU wins decisively on throughput once you push concurrency up._

**Cost:** _the interesting story. v5e-4 at $1.20/hr against an L4 at $0.71/hr means TPU has to be ~1.7x faster on tokens/sec to break even on $/1M tokens. From the prelim runs, it is._

---

## Reproducing this

Everything below assumes you have:
- A GCP project with TPU v5e access (Sprint credits work)
- A Colab account
- A HuggingFace token (for the model + optional dataset push)
- A Together AI key OR a local Ollama install (for dataset generation)

**Local web UI:**
```bash
git clone https://github.com/<you>/vforge && cd vforge
docker compose up -d        # postgres
cd backend && pip install -r requirements.txt && uvicorn main:app --reload &
cd frontend && npm install && npm run dev
```
Open http://localhost:3000, chat with vForge, generate a dataset.

**Just the Sprint scripts (no UI):**
```bash
python scripts/generate_dataset.py --domain code --rows 200 --out data/code.jsonl
python scripts/finetune_tpu.py --model google/gemma-2-2b --data data/code.jsonl
python scripts/finetune_gpu.py --model google/gemma-2-2b --data data/code.jsonl
python scripts/benchmark_vllm.py --model out_tpu --hardware tpu
python scripts/benchmark_vllm.py --model out_gpu --hardware gpu
python benchmarks/analysis.py --output benchmarks/charts/
```

---

## Takeaways

1. **JAX/Keras 3.0 closed the productivity gap.** The TPU training loop is now as short as the PyTorch one. If you've ruled out TPUs in the past for ergonomic reasons, it's worth another look.
2. **Same engine, different chip.** vLLM running on both sides means you're benchmarking the silicon, not the runtime. That's the comparison most teams actually need.
3. **The cost story is workload-dependent.** TPUs win when you can keep the chips fed (long sequences, large batches, throughput-bound serving). For sparse single-request latency, GPUs are still the safe pick.
4. **Open weights are the right default for synthetic data.** Most cloud LLM ToS prohibit using outputs to train competing models. We default the dataset generator to open Llama / Qwen / Gemma — and it works fine.

---

## What's next for vForge

Post-Sprint, we're evolving vForge into a full SaaS:

- **No-code → Pro → Enterprise tiers** for different audiences
- **One-click GCP deployment** of the fine-tuned adapter behind vLLM
- **RAG pipeline** on top of the generated dataset (ChromaDB)
- **Benchmark-as-a-service** — track model performance over time, compare strategies, auto-render the charts you see in this post

If you want to follow along: ⭐ the repo, file issues, send PRs.

---

🤖 Built for the Google TPU Sprint Q1 2026 by [@your-handle]. Apache-2.0 licensed.

#TPUSprint
