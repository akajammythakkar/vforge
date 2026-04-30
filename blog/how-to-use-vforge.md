# How to Use vForge: From Idea to Fine-Tuned Model in Three Steps

vForge is an open-source platform that turns a plain-English description of what you want your model to do into a fine-tuned, benchmarked LLM — without writing data pipelines or training boilerplate from scratch.

This walkthrough covers the full lifecycle:

1. [Set up the repo locally](#1-set-up-locally)
2. [Chat to generate a dataset](#2-generate-a-dataset-via-chat)
3. [Edit and export your dataset](#3-edit-and-export-your-dataset)
4. [Launch a fine-tuning run](#4-fine-tune-with-a-colab-notebook)
5. [Run the vLLM benchmark and compare TPU vs GPU](#5-benchmark-with-vllm)
6. [Use the standalone scripts (no UI)](#6-standalone-scripts-no-ui-required)

---

## 1. Set Up Locally

### Prerequisites

| Requirement | Version |
|---|---|
| Node.js | 20 or later |
| Python | 3.11 or later |
| PostgreSQL | 16 (or Docker) |
| Git | any |

An LLM provider is needed to generate the dataset. The default and safest option is **Ollama** (runs locally, no ToS restrictions on using outputs for training). Together AI, any OpenAI-compatible endpoint, or a local Ollama instance all work.

### Clone and configure

```bash
git clone https://github.com/akajammythakkar/vforge.git
cd vforge
cp .env.example .env
```

Open `.env` and fill in at minimum:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vforge

# Pick ONE provider (Ollama is zero-cost and default)
OLLAMA_BASE_URL=http://localhost:11434
# TOGETHER_API_KEY=your-key
# OPENAI_API_KEY=your-key
```

Optional but useful for the full pipeline:

```env
HF_TOKEN=your-huggingface-token
GCP_PROJECT_ID=your-gcp-project
```

### Start Postgres

```bash
docker compose up -d          # starts postgres:16 on port 5432
```

No Docker? Point `DATABASE_URL` at a hosted Postgres (Neon and Supabase both have free tiers).

### Start the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API is now at `http://localhost:8000`. Open `http://localhost:8000/docs` to see the auto-generated Swagger UI.

### Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** and you'll see the vForge sidebar with Chat, Datasets, Training, and Benchmarks.

---

## 2. Generate a Dataset via Chat

The chat interface is the fastest way to build a training dataset. You describe what you want the model to learn, vForge asks clarifying questions, then synthesizes rows for you.

### Pick your provider and model

At the top of the Chat page there are two dropdowns:

- **Provider** — Ollama (local), Together AI, OpenAI-compatible, or custom endpoint
- **Model** — type any model name your provider serves (e.g. `llama3.3:70b`, `mistralai/Mixtral-8x7B-Instruct-v0.1`)

Providers in the "OpenAI / Anthropic / Google" group show a yellow ToS badge reminding you that their outputs may not be used to train competing models. Open-model providers (Ollama, Together with open weights) are safe by default.

### Describe your goal

Type something like:

> "I want to fine-tune a model to answer customer support questions for a SaaS product. It should be polite, concise, and always suggest escalation if it's unsure."

vForge will ask clarifying questions — typical domain, tone, edge cases — and then confirm before generating.

### Generate rows

Once the conversation converges on the task, click **Generate Dataset**. A panel slides in:

- **Name** — what to call the dataset
- **Rows** — how many instruction/output pairs to generate (default 200; 500–1000 for real fine-tuning)
- **Format** — Alpaca or ShareGPT

Hit Generate. A progress bar streams rows as they arrive. The job runs as a background task on the backend; you can navigate away and come back.

---

## 3. Edit and Export Your Dataset

Navigate to **Datasets** in the sidebar and open your new dataset. You'll see a table with every generated row.

### Row structure

| Field | Description |
|---|---|
| `instruction` | The task/question given to the model |
| `input` | Optional extra context (can be empty) |
| `output` | The ideal response to train toward |
| `system_prompt` | Optional per-row system message |

### Edit a row

Click the pencil icon on any row to open the edit dialog. All four fields are editable. Bad rows are common — the edit dialog is the fastest way to fix them one by one.

### Add a row manually

Click **Add row** (top right of the Datasets page). This is useful for high-value examples you want to write yourself.

### Delete a row

Click the trash icon. Deletion is immediate and cannot be undone from the UI (but it's just a database row).

### Export

Choose a format from the dropdown next to the Export button:

| Format | File type | Use with |
|---|---|---|
| **Alpaca** | `.json` | `finetune_gpu.py`, most HF trainers |
| **ShareGPT** | `.jsonl` | FastChat, Axolotl, ShareGPT-format trainers |
| **JSONL** | `.jsonl` | Any custom pipeline |

Click **Export** to download the file locally.

---

## 4. Fine-Tune with a Colab Notebook

vForge doesn't run training inside the web app — training is GPU/TPU-heavy and belongs on appropriate hardware. Instead, it generates a ready-to-run Colab notebook that you click through.

### Create a training run

Click the **Fine-tune** button on the dataset page (or navigate to **Training → New run**).

Fill in:

| Field | Recommended value |
|---|---|
| Base model | `google/gemma-4-E2B-it` (fits on free Colab) |
| Hardware | TPU v5e-4 (for Sprint) or GPU |
| Method | LoRA |
| LoRA rank | 8 |
| Epochs | 1–3 |
| Batch size | 4 |

Click **Create run**. vForge immediately downloads a `.ipynb` file.

### Run on Colab — TPU path

1. Go to **colab.research.google.com**, open the downloaded notebook
2. Runtime → Change runtime type → **TPU v5e** (requires GCP project with TPU quota)
3. Runtime → Run all

The notebook:
- Installs `keras`, `keras-hub`, `jax[tpu]`
- Sets `KERAS_BACKEND=jax` before any import
- Loads `gemma4_instruct_2b` from keras-hub preset
- Calls `backbone.enable_lora(rank=8)`
- Shards across all 4 TPU chips via `keras.distribution.DataParallel`
- Saves `lora.weights.h5` and `metrics.json` to your Drive

### Run on Colab — GPU path

Same steps, but select **L4** or **A100** as the runtime. The notebook uses PyTorch + PEFT instead of JAX:

- `get_peft_model(model, LoraConfig(r=8, ...))`
- `Trainer` from `transformers`
- Records `torch.cuda.max_memory_allocated()` for peak memory comparison

### After training

The notebook writes a `metrics.json` like:

```json
{
  "hardware": "tpu_v5e_4",
  "train_time_seconds": 312,
  "tokens_per_second": 4820,
  "final_loss": 0.87,
  "peak_memory_gb": null
}
```

In vForge, go to **Training → your run → Update metrics** and paste this JSON. The run status will change to `completed`.

---

## 5. Benchmark with vLLM

Once you have a fine-tuned checkpoint (TPU or GPU), the benchmark script measures real serving performance.

### Run the benchmark

```bash
# TPU checkpoint
python scripts/benchmark_vllm.py \
  --model ./out_tpu \
  --hardware tpu \
  --prompts data/bench_prompts.txt   # optional; uses built-in prompts if omitted

# GPU checkpoint
python scripts/benchmark_vllm.py \
  --model ./out_gpu \
  --hardware gpu
```

The script runs 128 prompts × 256 output tokens through vLLM's continuous-batching engine and records:

| Metric | What it measures |
|---|---|
| `tokens_per_second` | Throughput — higher is better |
| `latency_p50_ms` | Median per-request latency |
| `latency_p95_ms` | Tail latency |
| `latency_p99_ms` | Worst-case latency |
| `peak_memory_gb` | GPU VRAM peak (TPU doesn't expose this) |
| `cost_per_1m_tokens_usd` | Computed from on-demand hourly rate |

Results land in `benchmarks/results/<hardware>_<timestamp>.json`.

### View the comparison dashboard

Start the backend + frontend, navigate to **Benchmarks** in the sidebar. The dashboard renders a side-by-side bar chart (Recharts) for throughput and latency, plus a cost comparison table. It reads whichever JSON files are in `benchmarks/results/`.

### Generate charts for the blog

```bash
python benchmarks/analysis.py --output benchmarks/charts/
```

This produces `throughput.png`, `latency.png`, and `cost.png` — the images referenced in `blog/tutorial.md`.

---

## 6. Standalone Scripts (No UI Required)

If you just want the Sprint scripts without running the web app:

```bash
# 1. Generate a dataset (requires Ollama or a Together key in env)
python scripts/generate_dataset.py \
  --domain "Python code completion" \
  --rows 200 \
  --out data/code.jsonl

# 2. Fine-tune on TPU (run this in a Colab TPU runtime)
python scripts/finetune_tpu.py \
  --model google/gemma-4-E2B-it \
  --data data/code.jsonl \
  --rank 8 \
  --epochs 1

# 3. Fine-tune on GPU (run this in a Colab GPU or local CUDA machine)
python scripts/finetune_gpu.py \
  --model google/gemma-4-E2B-it \
  --data data/code.jsonl \
  --rank 8 \
  --epochs 1

# 4. Benchmark both checkpoints
python scripts/benchmark_vllm.py --model out_tpu --hardware tpu
python scripts/benchmark_vllm.py --model out_gpu --hardware gpu

# 5. Plot comparison
python benchmarks/analysis.py --output benchmarks/charts/
```

Each script exits with a `metrics.json` in its output directory. None of them need a running database or frontend.

---

## Colab Notebooks

Five notebooks live in `notebooks/`. Open them directly in Colab:

| Notebook | What it does |
|---|---|
| `01_dataset_generation.ipynb` | Generate a JSONL dataset via Together AI or Ollama |
| `02_finetune_tpu.ipynb` | LoRA fine-tune on Cloud TPU v5e-4 (JAX + Keras 3.0) |
| `03_finetune_gpu.ipynb` | Same run on an L4/A100 GPU (PyTorch + PEFT) |
| `04_benchmark_vllm.ipynb` | vLLM throughput + latency + cost on either hardware |
| `05_full_pipeline.ipynb` | All four steps in one notebook, end-to-end |

`05_full_pipeline.ipynb` is the fastest way to reproduce the full benchmark from scratch on a fresh Colab session.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | yes | PostgreSQL connection string |
| `OLLAMA_BASE_URL` | for Ollama | default `http://localhost:11434` |
| `TOGETHER_API_KEY` | for Together AI | |
| `OPENAI_API_KEY` | for OpenAI-compat | |
| `ANTHROPIC_API_KEY` | for Claude (ToS warning) | |
| `HF_TOKEN` | for model push | HuggingFace write token |
| `GCP_PROJECT_ID` | for TPU | your GCP project |
| `TPU_NAME` | for TPU | e.g. `local` or your TPU node name |
| `HOURLY_COST_USD` | optional | overrides $/hr for cost calculation |

---

## Troubleshooting

**Backend fails to start — "could not connect to server"**
Your Postgres container isn't running. Run `docker compose up -d` first, or check that `DATABASE_URL` points at a reachable instance.

**Chat generates no rows / hangs**
Check that Ollama is running (`ollama serve`) and your selected model is pulled (`ollama pull llama3.2`). For Together AI, verify your API key in `.env`.

**`KERAS_BACKEND` error in Colab**
You must set the env var before importing keras. Add this as the first code cell:
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
```

**vLLM complains about device**
Pass `--device tpu` or `--device cuda` explicitly. vLLM doesn't always auto-detect TPU.

**`enable_lora` is not a method on the backbone**
You need `keras-hub >= 0.3` and a preset model. Run `pip install -q keras-hub --upgrade` in the notebook.

---

## What's Next

Once you have benchmark results, fill in the placeholders in `blog/tutorial.md` and the comparison is complete. The repo is set up so that every chart in the blog traces back to a specific JSON file in `benchmarks/results/` — no screenshots, no made-up numbers.

For questions or issues: https://github.com/akajammythakkar/vforge/issues

---

*Built for the Google TPU Sprint Q1 2026. Apache-2.0 licensed.*
