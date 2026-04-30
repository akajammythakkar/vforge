# vForge

> Open-source platform for LLM fine-tuning benchmarking — TPU vs GPU — with conversational dataset generation.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![TPU Sprint Q1 2026](https://img.shields.io/badge/Google_TPU_Sprint-Q1_2026-4285F4.svg)](#)
[![Made with JAX/Keras](https://img.shields.io/badge/JAX_%2B_Keras_3.0-FF6F00.svg)](#)

vForge takes you from **"I have an idea"** to **"my fine-tuned model is benchmarked on TPU and GPU"** in three steps:

1. **Chat** with a conversational dataset builder. It discovers your intent, generates synthetic instruction/response pairs, and lets you edit them in a structured table.
2. **Fine-tune** on Cloud TPU (JAX + Keras 3.0 + LoRA) and on GPU as a baseline. One generated Colab notebook per hardware target.
3. **Benchmark** with vLLM — throughput, latency, memory, and cost — and compare TPU vs GPU side-by-side.

This repo is a Google **TPU Sprint Q1 2026** deliverable. The benchmarks, scripts, and notebooks are reproducible end-to-end.

---

## Quickstart

### Prerequisites

- Node 20+, Python 3.11+, Postgres 16 (or `docker compose up`)
- Optional: Ollama, Together AI key, HuggingFace token, GCP project with TPU access

### Local dev

```bash
git clone https://github.com/<you>/vforge.git
cd vforge
cp .env.example .env

# Database (one of):
docker compose up -d                       # local Postgres
# or set DATABASE_URL to a hosted Postgres (Neon/Supabase) in .env

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

### Run the Sprint scripts directly (no UI)

```bash
# Generate a code-completion dataset locally with Ollama
python scripts/generate_dataset.py --domain code --rows 500 --out data/code.jsonl

# Fine-tune Gemma-2-2B with LoRA on Cloud TPU v5e-4
python scripts/finetune_tpu.py --model google/gemma-2-2b --data data/code.jsonl

# Same, on a single GPU
python scripts/finetune_gpu.py --model google/gemma-2-2b --data data/code.jsonl

# Benchmark inference with vLLM
python scripts/benchmark_vllm.py --model checkpoints/gemma-2-2b-code-tpu --hardware tpu
python scripts/benchmark_vllm.py --model checkpoints/gemma-2-2b-code-gpu --hardware gpu
```

Results land in `benchmarks/results/*.json`. Compare with:

```bash
python benchmarks/analysis.py --output benchmarks/charts/
```

---

## Sprint Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Blog tutorial — "TPU vs GPU for LLM Fine-Tuning" | [`blog/tutorial.md`](blog/tutorial.md) |
| 2 | Dataset generation notebook | [`notebooks/01_dataset_generation.ipynb`](notebooks/01_dataset_generation.ipynb) |
| 3 | TPU fine-tuning notebook | [`notebooks/02_finetune_tpu.ipynb`](notebooks/02_finetune_tpu.ipynb) |
| 4 | GPU fine-tuning notebook | [`notebooks/03_finetune_gpu.ipynb`](notebooks/03_finetune_gpu.ipynb) |
| 5 | vLLM benchmarking notebook | [`notebooks/04_benchmark_vllm.ipynb`](notebooks/04_benchmark_vllm.ipynb) |
| 6 | Full pipeline notebook | [`notebooks/05_full_pipeline.ipynb`](notebooks/05_full_pipeline.ipynb) |

---

## Architecture

```
┌──────────────┐         ┌──────────────────┐         ┌────────────────┐
│   Next.js    │ ──API──►│   FastAPI        │ ──────► │  Postgres 16   │
│   (chat UI,  │         │   (services,     │         └────────────────┘
│    editor,   │         │    routers)      │
│   dashboard) │         └─────┬────────────┘
└──────────────┘               │
                               ├─► LLM providers (Ollama / Together / OpenAI-compat)
                               ├─► HF Hub / GCS / Drive (model export)
                               └─► generates Colab notebooks via Jinja2

Standalone scripts (Sprint deliverable, no UI required):
  scripts/finetune_tpu.py  ──►  Cloud TPU v5e-4   (JAX + Keras 3.0 + LoRA)
  scripts/finetune_gpu.py  ──►  CUDA GPU          (PyTorch + PEFT/LoRA)
  scripts/benchmark_vllm.py ─►  TPU or GPU        (vLLM throughput/latency)
```

---

## Why TPU vs GPU?

The Sprint asks a simple question: for a typical LoRA fine-tune of a small open model (2B–9B params), how do TPU v5e and a single consumer GPU compare on:

- Tokens/sec during training
- Wall-clock time to convergence
- Peak memory
- Inference throughput (vLLM)
- $ per 1M tokens

vForge runs the same model + dataset on both, captures the metrics in JSONB, and produces the comparison charts you see in the blog.

---

## Roadmap (post-Sprint)

- Web-UI tier system (no-code → pro → enterprise)
- One-click GCP deployment of fine-tuned model
- RAG pipeline (ChromaDB) on top of generated datasets
- Team features: shared datasets, shared benchmarks
- Benchmark-as-a-service: track model performance over time

---

## Contributing

Issues and PRs welcome. See [`CLAUDE.md`](CLAUDE.md) for repo conventions.

## License

[Apache 2.0](LICENSE)

---

🤖 Built for the Google TPU Sprint Q1 2026.
