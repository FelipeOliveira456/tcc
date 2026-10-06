# Small Language Models for Agentic Workflow Generation

Experiment code for the paper *Small Language Models na Geração de Workflows Agênticos: Uma Avaliação Empírica no WorFBench* (Felipe Geraldo de Oliveira and Denilson Alves Pereira, DCC/UFLA).

Repository: [github.com/FelipeOliveira456/tcc](https://github.com/FelipeOliveira456/tcc)

Six SLMs of up to 4B parameters (Qwen 3.5 0.8B/2B/4B, Granite 4.1 3B, Gemma 3 1B, Ministral 3 3B) are evaluated on [WorFBench](https://github.com/zjunlp/WorFBench) under I0, RAG, SFT, and SFT+RAG. The non-SLM baseline is **GLM 5.2** (OpenRouter), in I0 and RAG only.

Lean pipeline. The **`--model`** argument is required wherever an SLM is involved (one model is downloaded at a time).

## Scripts

| Script | What it does |
|--------|----------------|
| `setup_project.py` | **One-time setup**: HF data + RAG index + WorFBench + llama.cpp |
| `run_model.py --model ID` | **Model pipeline**: download → Ollama → I0/RAG/SFT/SFT+RAG (infer + eval) |
| `download_data.py` | Train + test (`data/test/<task>/`, **without** a nested `gold_traj/`) |
| `download_model.py --model ID` | One HF model (`models/<id>/`) |
| `ollama_import.py --model ID` | Writes a Modelfile + the `ollama create` command |
| `build_vector_db.py` | **Deterministic** RAG vector store (`data/rag_index/`) |
| `finetune.py --model ID` | BF16 LoRA via Unsloth (`--dry-run` only writes the YAML) |
| `infer.py --model ID [--rag] [--finetuned]` | Local inference (Ollama) |
| `infer_openrouter.py [--rag]` | GLM 5.2 baseline inference (API) |
| `worfeval.py --setup` / `--model ID` | Clone WorFBench + WorFEval metrics |

Running a script **already performs** the action. Use `--dry-run` only to inspect paths and commands.

## Experiment flow

**Shortcut (recommended):**

```bash
python scripts/setup_project.py
python scripts/run_model.py --model qwen35-0.8b --limit 5   # pilot
python scripts/run_model.py --model qwen35-4b              # full run
```

`run_model.py` runs: download → **SFT** → Ollama (base + SFT) → **parallel inference** (base: I0→RAG ‖ SFT: SFT→SFT+RAG) → WorFEval for every setting.

Useful flags: `--skip-download`, `--skip-finetune` (SFT already trained), `--skip-sft` (base I0/RAG only), `--skip-ollama`.

**Manual steps:**

```text
1. download_data.py
2. download_model.py --model qwen35-4b
3. build_vector_db.py
4. setup_llama_cpp.py          # GGUF (Granite/Nemotron); included in setup_project
5. finetune.py --model qwen35-4b --export-merged
6. ollama_import.py --model qwen35-4b --run
7. ollama_import.py --model qwen35-4b --finetuned --run
8. infer.py --model qwen35-4b                       # I0
9. infer.py --model qwen35-4b --rag                 # RAG
10. infer.py --model qwen35-4b --finetuned           # SFT
11. infer.py --model qwen35-4b --finetuned --rag
12. worfeval.py --setup
13. worfeval.py --model qwen35-4b --all-scenarios
```

(On the shortcut, steps 8–11 run in parallel: base track ‖ SFT track.)

## Settings (`infer.py`)

| Flags | Setting |
|-------|---------|
| (none) | I0 — base, no RAG |
| `--rag` | RAG — base + 2 retrieved examples |
| `--finetuned` | SFT — checkpoint, no RAG |
| `--finetuned --rag` | SFT + RAG |

No fixed WorFBench few-shot. The test prompt is **system + user** (plus a RAG block if `--rag`).

## Inference (Ollama)

`infer.py` calls local **Ollama** (`POST /api/chat`). It does not use LangChain or `transformers` for generation.

1. Install and start [Ollama](https://ollama.com) (a recent build, for Qwen3.5 / Gemma 3).
2. Import the model under the **same name** as `--model` (see [HF → Ollama](#hf--ollama-importing-models) below).
3. With `--finetuned`, the default name is `{model}-sft` (e.g. `qwen35-0.8b-sft`). Change it in `config/default.yaml` → `inference.ollama` or in `config/backends.yaml`.

```bash
python scripts/infer.py --model qwen35-0.8b --limit 2   # pilot
python scripts/infer.py --model qwen35-0.8b --dry-run  # paths only
```

Config: `inference.ollama.base_url`, `timeout_s`, `temperature`, `think` (default `false` for every model), `num_predict`.

## HF → Ollama (importing models)

Recommended flow: download the weights into the repo, write a `Modelfile`, and register it in Ollama with `ollama create`. The Ollama name must match `--model` (base) or `{model}-sft` (after fine-tuning).

### 1. Download from Hugging Face

```bash
# Accept licenses on the HF website when required (Gemma, etc.)
huggingface-cli login          # optional; or export HF_TOKEN=...
python scripts/download_model.py --model qwen35-0.8b
# → models/qwen35-0.8b/  (safetensors + tokenizer)
```

### 2. Base model in Ollama

**Option A — project script** (writes `models/ollama/Modelfile.<id>`):

```bash
python scripts/ollama_import.py --model qwen35-0.8b
# prints: ollama create qwen35-0.8b -f models/ollama/Modelfile.qwen35-0.8b

python scripts/ollama_import.py --model qwen35-0.8b --run
# or run the printed command by hand
```

**Granite / Nemotron (GGUF):** setup clones `external/llama.cpp`. On import, if the
architecture requires GGUF (`GraniteForCausalLM`, and similar), `ollama_import.py` only runs
`convert_hf_to_gguf.py` and creates the model from `models/ollama/gguf/`.

```bash
python scripts/setup_llama_cpp.py   # or setup_project.py (step 4/4)
python scripts/ollama_import.py --model granite-3b --run
# GGUF is f16 (no quantization); outtype is inference.ollama.gguf.outtype
```

Config: `inference.ollama.gguf` in `config/default.yaml` (`force_architectures`, `outtype: f16`,
`llama_cpp_dir`). Per model: `ollama_via_gguf: true`.

**Option B — manual** (equivalent):

```bash
cd models/qwen35-0.8b
cat > Modelfile <<'EOF'
FROM .
PARAMETER temperature 0
EOF
ollama create qwen35-0.8b -f Modelfile
```

**Optional quantization** (safetensors create only; the GGUF path does not quantize):

```bash
python scripts/ollama_import.py --model qwen35-4b --quantize q4_K_M --run
# same as: ollama create qwen35-4b --quantize q4_K_M -f ...
```

**Check:**

```bash
ollama list
ollama run qwen35-0.8b "Hello"
```

### 3. SFT model in Ollama (after `finetune.py`)

SFT training runs outside Ollama (Unsloth, bf16 LoRA). To infer with `--finetuned`, export the weights and create `{model}-sft`.

```bash
pip install -r requirements.txt -r requirements-sft.txt
python scripts/finetune.py --model qwen35-0.8b --export-merged
# checkpoints/qwen35-0.8b/merged/  ← merged weights

python scripts/ollama_import.py --model qwen35-0.8b --finetuned --run
# → ollama create qwen35-0.8b-sft -f models/ollama/Modelfile.qwen35-0.8b-sft
```

**Qwen3.5 SFT:** use `finetune.py --export-merged`; import builds `checkpoints/<id>/ollama_sft/` (merge + base vision configs). Ollama does **not** support a PEFT `ADAPTER` on Qwen3.5.

```bash
python scripts/finetune.py --model qwen35-0.8b --export-merged
python scripts/ollama_import.py --model qwen35-0.8b --finetuned --run
```

**LoRA adapter** (only families Ollama supports: Llama/Mistral/Gemma — not Qwen3.5):

```bash
# the base must already exist: ollama list | grep qwen35-0.8b
python scripts/ollama_import.py --model qwen35-0.8b \
  --adapter checkpoints/qwen35-0.8b --run
# Modelfile: FROM models/qwen35-0.8b + ADAPTER checkpoints/...
```

Prefer **merge + create** for reproducibility; a direct adapter requires a base identical to the one used in training.

### 4. Notes by family

| Family | Safetensors import | Chat template (HF tokenizer) |
|--------|--------------------|------------------------------|
| Qwen3.5 | Ollama ≥0.17 with qwen35 support; otherwise GGUF via llama.cpp | Qwen3.5 / `enable_thinking=False` |
| Granite 4.1 | **GGUF via llama.cpp** (direct safetensors fails: `GraniteForCausalLM`) | Granite |
| Gemma 3 | safetensors; Google license on HF | Gemma |
| Ministral 3 | **BF16** (`…-Instruct-2512-BF16`); FP8 breaks `ollama create` | Ministral |
| Nemotron 3 Nano 4B | **GGUF via llama.cpp** (hybrid) or `ollama pull nemotron-3-nano:4b` | Nemotron |

Large models: if `ollama create` runs out of memory, use `GOMAXPROCS=1 ollama create ...`.

### 5. Custom names

In `config/local.yaml` or `config/backends.yaml`:

```yaml
inference:
  ollama:
    models:
      qwen35-0.8b:
        base: my-qwen-08
        sft: my-qwen-08-sft
```

## Fine-tuning (SFT / Unsloth)

Extra dependencies (two steps; see [Setup](#setup)):

```bash
pip install -r requirements.txt
pip install -r requirements-sft.txt
```

Backend: **Unsloth** (bf16 LoRA). On Qwen3.5, Unsloth reports about **3 GB** of VRAM for the 0.8B
(versus OOM on the previous stack without those kernels). Loss is computed only on the **last** assistant turn.

```bash
# Inspect the dataset + manifest (no GPU)
python scripts/finetune.py --model qwen35-0.8b --dry-run

# Real training (1 epoch)
python scripts/finetune.py --model qwen35-0.8b
python scripts/finetune.py --model qwen35-0.8b --export-merged   # 16-bit merge → Ollama
```

The pipeline writes:

- `data/sft/worfbench_sharegpt.json` — 7 turns; if an example exceeds `max_example_tokens` (2048),
  older few-shot demos are dropped; gold/RAG are **not** changed
- `data/sft/worfbench_sharegpt.filter.json` — full/truncated/dropped counts
- `outputs/manifests/finetune_<model>_<stamp>.json`
- `checkpoints/<model>/` — LoRA adapters; `merged/` after `--export-merged`

Defaults: `framework: unsloth`, `num_train_epochs: 1`, `cutoff_len: 2048`,
`load_in_16bit: true`, `load_in_4bit: false`, batch 1 × accum 8.

**LabRI:**

```bash
pip uninstall -y causal-conv1d   # if an incompatible wheel was installed
pip install -r requirements-sft.txt
git pull
python scripts/finetune.py --model qwen35-0.8b --export-merged
```

Loss is on the **last** assistant message (gold workflow) only; demos are context.

## Timestamps (inference and fine-tuning)

Each run stamps files with `YYYYMMDD_HHMMSS` (UTC):

- **`infer.py`**: `outputs/predictions/<model>/<task>/graph_eval_{setting}_{stamp}.json` + `outputs/predictions/<model>/run_{stamp}.json` (`started_at`, `finished_at`, tasks, paths).
- **`finetune.py`**: `outputs/manifests/finetune_{model}_{stamp}.yaml` + `.json` (`stamp`, `started_at`, `finished_at`).
- **`worfeval.py`**: automatically uses the **most recent** stamped prediction.

Later runs therefore do not overwrite earlier ones.

## Deterministic RAG

- Fixed embedding in `config/default.yaml` (`rag.embedding_model`, `rag.seed`)
- Per training example: **last user** turn (the actual question) + **last assistant** turn (gold workflow)
- Similarity uses only the question; the workflow goes into the prompt, **not** into the vector
- The training system turn and few-shot demos are **not** indexed
- Examples are ordered by a stable `id` before embedding
- `meta.json` stores the training hash + `chunk_fields` — the index is rebuilt if they change

After changing the RAG logic, rebuild the index:

```bash
python scripts/build_vector_db.py --force
```

## WorFEval (how it works)

1. **`infer.py`** writes, per task, a JSON list of `{query, workflow}`.
2. **`worfeval.py`** calls `external/WorFBench/node_eval.py --task eval_workflow` on the most recent stamped prediction.
3. It compares the prediction with `data/test/<task>/graph_eval.json` using **sentence-transformers** and matching.
4. By default it writes **two** results (as in the paper):
   - `*_node.json` → chain / **f1chain** (node order)
   - `*_graph.json` → graph / **f1graph** (nodes + edges)
5. Output goes to `outputs/eval_results/`. Chain only: `--eval-type node` or `run_model.py --eval-node-only`.

Evaluation does not use Ollama or LangChain.

## Models (`--model`)

Alias **without a dot** (safe in bash). The official HF name may contain a dot (e.g. **Qwen3.5** → `Qwen/Qwen3.5-4B`).

Models **used in the paper**:

| `--model` | Source | Settings |
|-----------|--------|----------|
| `qwen35-0.8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | I0, RAG, SFT, SFT+RAG |
| `qwen35-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | I0, RAG, SFT, SFT+RAG |
| `qwen35-4b` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | I0, RAG, SFT, SFT+RAG |
| `granite-3b` | [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b) | I0, RAG, SFT, SFT+RAG |
| `gemma3-1b` | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | I0, RAG, SFT, SFT+RAG |
| `ministral-3-3b` | [mistralai/Ministral-3-3B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) | I0, RAG, SFT, SFT+RAG |
| `glm-5.2-openrouter` | [z-ai/glm-5.2](https://huggingface.co/zai-org/GLM-5.2) via OpenRouter | **I0 and RAG** |

Example: `python scripts/download_model.py --model qwen35-4b`

Aliases still in `config/default.yaml`, **not evaluated in the paper**: `gemma3-4b`, `nemotron-nano-4b`, `qwen35-27b`.

## GLM 5.2 baseline (OpenRouter)

It does not go through Ollama. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

```bash
python scripts/infer_openrouter.py            # I0
python scripts/infer_openrouter.py --rag      # RAG (k=2, same index as the SLMs)
python scripts/worfeval.py --model glm-5.2-openrouter --all-scenarios
```

Temperature 0, reasoning off. An API failure on one example stores an empty workflow (`F1 = 0`). A local layer normalizes `1.Texto` → `1. Texto` before the WorFEval parser, without changing the upstream repository.

## Setup

Requires **Python 3.11 or 3.12** (do not use 3.14 — `pip install` for SFT fails with `resolution-too-deep`).

```bash
cd tcc
python3.12 -m venv .venv && source .venv/bin/activate
python --version   # must be 3.11.x or 3.12.x

pip install -r requirements.txt
# tests (optional; pytest for test_sft, unittest covers the rest)
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -q
# or: python -m pytest tests/ -q
# fine-tuning (GPU): a second step, so the resolver does not blow up
pip install -r requirements-sft.txt
# Ollama: install separately from https://ollama.com
```

If `pip install -r requirements-sft.txt` fails on LabRI, try:

```bash
pip install "unsloth" "unsloth_zoo" "bitsandbytes>=0.45.0"
```

More detail (Ollama, SFT): [`docs/stack.md`](docs/stack.md).

## Data

Train and test: [`zjunlp/WorFBench_train`](https://huggingface.co/datasets/zjunlp/WorFBench_train) and [`zjunlp/WorFBench_test`](https://huggingface.co/datasets/zjunlp/WorFBench_test). Official evaluator: [zjunlp/WorFBench](https://github.com/zjunlp/WorFBench).

## License

Code in this repository: free to use to reproduce the experiment. Weights and data follow their original licenses (Hugging Face / WorFBench). Do not commit `.env` or API keys.
