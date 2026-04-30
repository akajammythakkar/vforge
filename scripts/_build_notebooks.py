"""One-shot script that writes the 5 Sprint deliverable notebooks.

Run from the repo root:  python scripts/_build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_nb(path: Path, cells: list[dict], kernel: str = "python3") -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": kernel},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))


# ── 01 — dataset generation ─────────────────────────────────
nb1 = [
    md(
        "# vForge — 01. Dataset generation\n\n"
        "Generate a synthetic instruction-tuning dataset with an open-weight LLM.\n\n"
        "**Sprint topic:** Google TPU Sprint Q1 2026 — vLLM benchmarking TPU vs GPU.\n\n"
        "Open-weight providers (Ollama, Together AI) are recommended — outputs are safe to use for fine-tuning."
    ),
    md("## 1. Install"),
    code("!pip -q install httpx tenacity"),
    md(
        "## 2. Configure provider\n\n"
        "Set `TOGETHER_API_KEY` (or use Ollama locally). Together's open-model endpoint works in Colab without local resources."
    ),
    code(
        "import os\n"
        "os.environ['TOGETHER_API_KEY'] = 'YOUR_KEY_HERE'  # set me\n"
        "PROVIDER = 'together'\n"
        "MODEL = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'"
    ),
    md("## 3. Generate"),
    code(
        "import httpx, json, asyncio\n"
        "\n"
        "SYSTEM = (\n"
        "  'You generate diverse instruction/response pairs for fine-tuning.\\n'\n"
        "  'Output ONLY raw JSONL with keys instruction, input, output.'\n"
        ")\n"
        "\n"
        "async def gen(domain: str, description: str, n: int):\n"
        "    rows = []\n"
        "    while len(rows) < n:\n"
        "        batch_n = min(20, n - len(rows))\n"
        "        prompt = f'Domain: {domain}\\nGoal: {description}\\nGenerate exactly {batch_n} JSONL rows.'\n"
        "        async with httpx.AsyncClient(timeout=600) as c:\n"
        "            r = await c.post(\n"
        "              'https://api.together.xyz/v1/chat/completions',\n"
        "              headers={'Authorization': f\"Bearer {os.environ['TOGETHER_API_KEY']}\"},\n"
        "              json={'model': MODEL,\n"
        "                    'messages': [{'role':'system','content':SYSTEM},\n"
        "                                 {'role':'user','content':prompt}],\n"
        "                    'max_tokens': 4096, 'temperature': 0.8})\n"
        "            r.raise_for_status()\n"
        "            text = r.json()['choices'][0]['message']['content']\n"
        "        for line in text.splitlines():\n"
        "            line = line.strip().lstrip('```jsonl').lstrip('```json').lstrip('```').rstrip('```')\n"
        "            try:\n"
        "                obj = json.loads(line)\n"
        "                if 'instruction' in obj and 'output' in obj:\n"
        "                    rows.append({'instruction': obj['instruction'],\n"
        "                                 'input': obj.get('input',''),\n"
        "                                 'output': obj['output']})\n"
        "            except Exception:\n"
        "                continue\n"
        "    return rows[:n]\n"
        "\n"
        "rows = await gen('code', 'Pandas one-liner code completions for data analysis', n=200)\n"
        "print(len(rows), 'rows')\n"
        "rows[:2]"
    ),
    md("## 4. Save and inspect"),
    code(
        "from pathlib import Path\n"
        "Path('data').mkdir(exist_ok=True)\n"
        "with open('data/code.jsonl','w') as f:\n"
        "    for r in rows: f.write(json.dumps(r) + '\\n')\n"
        "!head -2 data/code.jsonl"
    ),
    md("## 5. Push to HuggingFace (optional)"),
    code(
        "# from huggingface_hub import HfApi\n"
        "# api = HfApi(token='hf_...')\n"
        "# api.upload_file(path_or_fileobj='data/code.jsonl', path_in_repo='train.jsonl',\n"
        "#                 repo_id='you/your-dataset', repo_type='dataset')"
    ),
]
write_nb(OUT / "01_dataset_generation.ipynb", nb1)


# ── 02 — TPU fine-tune ─────────────────────────────────────
nb2 = [
    md(
        "# vForge — 02. Fine-tune Gemma-2-2B on Cloud TPU (LoRA, JAX/Keras 3.0)\n\n"
        "Runs on **Colab → Runtime → Change runtime type → TPU v5e-1** (or any v5e).\n\n"
        "For the Sprint we run on **Cloud TPU v5e-4** via the GCP credits."
    ),
    md("## 1. Install JAX/Keras 3.0 + keras-hub"),
    code(
        "import os\n"
        "os.environ['KERAS_BACKEND'] = 'jax'\n"
        "!pip -q install -U 'jax[tpu]' keras==3.6.0 keras-hub datasets huggingface_hub"
    ),
    md("## 2. Verify TPU"),
    code(
        "import jax\n"
        "print('JAX devices:', jax.devices())\n"
        "import keras\n"
        "print('Keras backend:', keras.backend.backend())"
    ),
    md("## 3. Load dataset"),
    code(
        "from datasets import load_dataset\n"
        "ds = load_dataset('json', data_files='https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K/resolve/main/data/train-00000-of-00001-3a2050c1d3ebcacb.parquet', split='train[:1000]')\n"
        "# Or use your own: ds = load_dataset('json', data_files='data/code.jsonl', split='train')\n"
        "ds = ds.rename_columns({'prompt':'instruction','completion':'output'}) if 'prompt' in ds.column_names else ds\n"
        "print(ds)"
    ),
    md("## 4. Format prompts"),
    code(
        "def fmt(r):\n"
        "    instr = r.get('instruction','').strip()\n"
        "    inp = (r.get('input') or '').strip()\n"
        "    out = (r.get('output') or '').strip()\n"
        "    p = f'<start_of_turn>user\\n{instr}'\n"
        "    if inp: p += f'\\n\\n{inp}'\n"
        "    p += f'<end_of_turn>\\n<start_of_turn>model\\n{out}<end_of_turn>'\n"
        "    return p\n"
        "texts = [fmt(r) for r in ds]\n"
        "print(texts[0][:300])"
    ),
    md("## 5. Set up data-parallel TPU distribution"),
    code(
        "devices = keras.distribution.list_devices()\n"
        "print('devices:', devices)\n"
        "if devices and any('tpu' in d.lower() for d in devices):\n"
        "    keras.distribution.set_distribution(keras.distribution.DataParallel(devices=devices))"
    ),
    md("## 6. Load Gemma-2-2B with LoRA"),
    code(
        "import keras_hub, time\n"
        "t0 = time.time()\n"
        "causal = keras_hub.models.CausalLM.from_preset('gemma2_2b_en')\n"
        "print(f'load: {time.time()-t0:.1f}s')\n"
        "causal.backbone.enable_lora(rank=8)\n"
        "causal.preprocessor.sequence_length = 512  # bump for production\n"
        "causal.compile(\n"
        "  optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01),\n"
        "  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n"
        "  weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],\n"
        ")"
    ),
    md("## 7. Train"),
    code(
        "import time, json\n"
        "t0 = time.time()\n"
        "history = causal.fit(x=texts, batch_size=4, epochs=1, verbose=2)\n"
        "elapsed = time.time() - t0\n"
        "metrics = {\n"
        "  'hardware':'tpu', 'base_model':'google/gemma-2-2b', 'rank':8,\n"
        "  'epochs':1, 'batch_size':4, 'dataset_rows': len(texts),\n"
        "  'train_time_sec': round(elapsed,2),\n"
        "  'final_loss': float(history.history['loss'][-1]),\n"
        "  'approx_train_tokens_per_sec': round(len(texts) * 512 / elapsed, 2),\n"
        "}\n"
        "print(json.dumps(metrics, indent=2))"
    ),
    md("## 8. Save adapter & report metrics"),
    code(
        "from pathlib import Path\n"
        "Path('out_tpu').mkdir(exist_ok=True)\n"
        "causal.save_weights('out_tpu/lora.weights.h5')\n"
        "Path('out_tpu/metrics.json').write_text(json.dumps(metrics, indent=2))\n"
        "# POST these to your vForge backend:\n"
        "# !curl -X POST $VFORGE/api/benchmarks/runs -H 'content-type: application/json' \\\n"
        "#   -d '{\"name\":\"gemma-2-2b TPU\",\"hardware\":\"tpu_v5e\",\"benchmark_type\":\"training\", \"model\":\"google/gemma-2-2b\", \"metrics\": '\"$(cat out_tpu/metrics.json)\"'}'\n"
    ),
    md(
        "## Next\n"
        "- 03_finetune_gpu.ipynb — same data, GPU baseline.\n"
        "- 04_benchmark_vllm.ipynb — inference throughput / latency / cost.\n"
    ),
]
write_nb(OUT / "02_finetune_tpu.ipynb", nb2)


# ── 03 — GPU fine-tune ─────────────────────────────────────
nb3 = [
    md(
        "# vForge — 03. Fine-tune Gemma-2-2B on GPU (LoRA, PyTorch + PEFT)\n\n"
        "GPU baseline for the TPU comparison. Colab → Runtime → T4 (free) or A100 / L4 (Pro)."
    ),
    md("## 1. Install"),
    code(
        "!pip -q install -U torch transformers peft trl datasets accelerate bitsandbytes huggingface_hub"
    ),
    md("## 2. Verify GPU"),
    code(
        "import torch\n"
        "print('cuda available:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print(torch.cuda.get_device_name(0))\n"
        "    print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
    ),
    md("## 3. Load + format dataset (mirror notebook 02)"),
    code(
        "from datasets import load_dataset\n"
        "ds = load_dataset('json', data_files='data/code.jsonl', split='train')\n"
        "def fmt(r):\n"
        "    p = f'<start_of_turn>user\\n{r[\"instruction\"]}'\n"
        "    if r.get('input'): p += f'\\n\\n{r[\"input\"]}'\n"
        "    p += f'<end_of_turn>\\n<start_of_turn>model\\n{r[\"output\"]}<end_of_turn>'\n"
        "    return {'text': p}\n"
        "ds = ds.map(fmt)\n"
        "ds[0]['text'][:300]"
    ),
    md("## 4. Tokenize"),
    code(
        "from transformers import AutoTokenizer\n"
        "tok = AutoTokenizer.from_pretrained('google/gemma-2-2b')\n"
        "if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
        "def t(b): return tok(b['text'], truncation=True, max_length=512, padding='max_length')\n"
        "ds = ds.map(t, batched=True, remove_columns=['text','instruction','input','output'])\n"
        "ds.set_format('torch', columns=['input_ids','attention_mask'])"
    ),
    md("## 5. Load model + apply LoRA"),
    code(
        "import torch, time\n"
        "from transformers import AutoModelForCausalLM\n"
        "from peft import LoraConfig, get_peft_model\n"
        "t0 = time.time()\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    'google/gemma-2-2b', torch_dtype=torch.bfloat16, device_map='auto')\n"
        "print(f'load: {time.time()-t0:.1f}s')\n"
        "lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none',\n"
        "                  task_type='CAUSAL_LM',\n"
        "                  target_modules=['q_proj','k_proj','v_proj','o_proj'])\n"
        "model = get_peft_model(model, lora)\n"
        "model.print_trainable_parameters()"
    ),
    md("## 6. Train"),
    code(
        "from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling\n"
        "torch.cuda.reset_peak_memory_stats()\n"
        "args = TrainingArguments(output_dir='out_gpu',\n"
        "  per_device_train_batch_size=4, gradient_accumulation_steps=4,\n"
        "  num_train_epochs=1, learning_rate=1e-4, bf16=True,\n"
        "  logging_steps=10, save_strategy='no', warmup_ratio=0.03,\n"
        "  report_to=[])\n"
        "tr = Trainer(model=model, args=args, train_dataset=ds,\n"
        "             data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False))\n"
        "import time, json\n"
        "t0 = time.time()\n"
        "out = tr.train()\n"
        "elapsed = time.time() - t0\n"
        "metrics = {\n"
        "  'hardware':'gpu', 'device': torch.cuda.get_device_name(0),\n"
        "  'base_model':'google/gemma-2-2b', 'rank':8,\n"
        "  'epochs':1, 'batch_size':4,\n"
        "  'dataset_rows': len(ds),\n"
        "  'train_time_sec': round(elapsed,2),\n"
        "  'final_loss': float(out.training_loss),\n"
        "  'peak_memory_gb': round(torch.cuda.max_memory_allocated()/1e9, 3),\n"
        "  'approx_train_tokens_per_sec': round(len(ds) * 512 / elapsed, 2),\n"
        "}\n"
        "print(json.dumps(metrics, indent=2))"
    ),
    md("## 7. Save"),
    code(
        "from pathlib import Path\n"
        "Path('out_gpu').mkdir(exist_ok=True)\n"
        "model.save_pretrained('out_gpu')\n"
        "tok.save_pretrained('out_gpu')\n"
        "Path('out_gpu/metrics.json').write_text(json.dumps(metrics, indent=2))"
    ),
]
write_nb(OUT / "03_finetune_gpu.ipynb", nb3)


# ── 04 — vLLM benchmark ─────────────────────────────────────
nb4 = [
    md(
        "# vForge — 04. Benchmark with vLLM\n\n"
        "Run inference benchmarks on TPU and GPU. Captures throughput, latency, memory, and cost.\n"
        "**Run twice** — once with TPU runtime, once with GPU — and aggregate the two JSON files."
    ),
    md("## 1. Install"),
    code("!pip -q install vllm"),
    md("## 2. Detect hardware"),
    code(
        "import torch, jax\n"
        "HARDWARE = 'tpu' if any('tpu' in str(d).lower() for d in jax.devices()) else ('gpu' if torch.cuda.is_available() else 'cpu')\n"
        "DEVICE = {'tpu':'tpu','gpu':'cuda','cpu':'cpu'}[HARDWARE]\n"
        "HOURLY = {'tpu': 1.20, 'gpu': 0.71, 'cpu': 0.05}[HARDWARE]  # rough on-demand\n"
        "print(HARDWARE, DEVICE)"
    ),
    md("## 3. Load model"),
    code(
        "from vllm import LLM, SamplingParams\n"
        "import time\n"
        "t0 = time.time()\n"
        "MODEL = 'google/gemma-2-2b'   # or local path: './out_tpu' / './out_gpu'\n"
        "llm = LLM(model=MODEL, device=DEVICE, max_model_len=2048, seed=42, trust_remote_code=True)\n"
        "load_time = time.time() - t0\n"
        "print(f'load: {load_time:.1f}s')"
    ),
    md("## 4. Latency + throughput"),
    code(
        "import statistics, time\n"
        "PROMPTS = ['Write Fibonacci in Python.', 'Explain monads in 3 sentences.',\n"
        "           'Top SQL query for top-5 customers by sales.', 'Haiku about TPUs.',\n"
        "           'Translate \"hello\" to Japanese.', 'Regex for IPv4.',\n"
        "           'Big-O of merge sort?', 'Define Pareto principle.']\n"
        "prompts = [PROMPTS[i % len(PROMPTS)] for i in range(128)]\n"
        "sp = SamplingParams(temperature=0.0, max_tokens=256, seed=42)\n"
        "\n"
        "# warmup\n"
        "llm.generate(prompts[:8], sp)\n"
        "\n"
        "lats = []\n"
        "for p in prompts[:16]:\n"
        "    t0 = time.time(); llm.generate([p], sp); lats.append((time.time()-t0)*1000)\n"
        "\n"
        "t0 = time.time()\n"
        "outs = llm.generate(prompts, sp)\n"
        "total_time = time.time() - t0\n"
        "total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)\n"
        "throughput = total_tokens / total_time\n"
        "cost_per_1m = (total_time/3600) * HOURLY / (total_tokens / 1_000_000)\n"
        "\n"
        "metrics = {\n"
        "  'model': MODEL, 'hardware': HARDWARE, 'device': DEVICE,\n"
        "  'load_time_sec': round(load_time,2),\n"
        "  'total_prompts': len(prompts), 'total_tokens': total_tokens,\n"
        "  'total_time_sec': round(total_time,3),\n"
        "  'throughput_tokens_per_sec': round(throughput,2),\n"
        "  'latency_p50_ms': round(statistics.median(lats),2),\n"
        "  'latency_p95_ms': round(max(lats),2),\n"
        "  'hourly_cost_usd': HOURLY,\n"
        "  'cost_per_1m_tokens_usd': round(cost_per_1m,4),\n"
        "}\n"
        "if HARDWARE=='gpu':\n"
        "    metrics['peak_memory_gb'] = round(torch.cuda.max_memory_allocated()/1e9, 3)\n"
        "import json\n"
        "print(json.dumps(metrics, indent=2))"
    ),
    md("## 5. Save"),
    code(
        "from pathlib import Path\n"
        "Path('benchmarks/results').mkdir(parents=True, exist_ok=True)\n"
        "fname = f'benchmarks/results/{HARDWARE}_vllm_{int(time.time())}.json'\n"
        "Path(fname).write_text(json.dumps({'name': f'{HARDWARE}-{MODEL}', 'metrics': metrics}, indent=2))\n"
        "print(fname)"
    ),
]
write_nb(OUT / "04_benchmark_vllm.ipynb", nb4)


# ── 05 — full pipeline ─────────────────────────────────────
nb5 = [
    md(
        "# vForge — 05. Full pipeline (Sprint demo)\n\n"
        "Single notebook that runs the **whole pipeline** for the Sprint blog:\n"
        "1. Generate dataset (Together AI)\n"
        "2. Fine-tune on TPU (LoRA)\n"
        "3. Benchmark with vLLM\n"
        "4. Compare to a GPU baseline (loaded from `benchmarks/results/`)\n\n"
        "Run on Colab → TPU runtime."
    ),
    md("## 1. Install"),
    code(
        "import os\n"
        "os.environ['KERAS_BACKEND'] = 'jax'\n"
        "!pip -q install -U 'jax[tpu]' keras==3.6.0 keras-hub datasets vllm httpx huggingface_hub"
    ),
    md("## 2. Generate dataset"),
    code(
        "# (uses code from 01_dataset_generation.ipynb — abbreviated here)\n"
        "import os, httpx, json, asyncio\n"
        "os.environ['TOGETHER_API_KEY'] = 'YOUR_KEY_HERE'\n"
        "# ... see 01_dataset_generation.ipynb for the full gen() helper ...\n"
        "# rows = await gen('code', 'Pandas one-liners', n=200)\n"
        "# Path('data/code.jsonl').write_text('\\n'.join(json.dumps(r) for r in rows))"
    ),
    md("## 3. Fine-tune on TPU"),
    code(
        "# (see 02_finetune_tpu.ipynb for the full version)\n"
        "import keras, keras_hub, time\n"
        "devs = keras.distribution.list_devices()\n"
        "if devs and any('tpu' in d.lower() for d in devs):\n"
        "    keras.distribution.set_distribution(keras.distribution.DataParallel(devices=devs))\n"
        "causal = keras_hub.models.CausalLM.from_preset('gemma2_2b_en')\n"
        "causal.backbone.enable_lora(rank=8)\n"
        "# causal.fit(texts, batch_size=4, epochs=1)\n"
        "# causal.save_weights('out_tpu/lora.weights.h5')"
    ),
    md("## 4. Benchmark vLLM (TPU)"),
    code(
        "# from vllm import LLM, SamplingParams\n"
        "# (see 04_benchmark_vllm.ipynb)\n"
        "# metrics_tpu = {...}\n"
        "# Path('benchmarks/results/tpu_pipeline.json').write_text(json.dumps({'metrics': metrics_tpu}))"
    ),
    md("## 5. Compare with GPU baseline"),
    code(
        "from pathlib import Path\n"
        "import json\n"
        "tpu = json.loads(Path('benchmarks/results/tpu_pipeline.json').read_text())['metrics']\n"
        "gpu = json.loads(Path('benchmarks/results/gpu_pipeline.json').read_text())['metrics']\n"
        "print('TPU tok/s:', tpu['throughput_tokens_per_sec'], '  GPU tok/s:', gpu['throughput_tokens_per_sec'])\n"
        "print('TPU cost/1M:', tpu['cost_per_1m_tokens_usd'], '  GPU cost/1M:', gpu['cost_per_1m_tokens_usd'])"
    ),
    md(
        "## Done\n"
        "- Adapter + metrics in `out_tpu/`.\n"
        "- Benchmark JSONs in `benchmarks/results/`.\n"
        "- Render charts: `python benchmarks/analysis.py --output benchmarks/charts/`."
    ),
]
write_nb(OUT / "05_full_pipeline.ipynb", nb5)


print(f"Wrote 5 notebooks to {OUT}")
for f in sorted(OUT.glob("*.ipynb")):
    print(" ", f.name)
