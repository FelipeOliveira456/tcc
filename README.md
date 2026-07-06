# TCC — WorFBench (6 scripts)

Pipeline enxuto. Parâmetro **`--model`** em tudo que envolve SLM (baixa um modelo por vez).

## Scripts

| Script | O que faz |
|--------|-----------|
| `download_data.py` | Treino + teste (`data/test/<task>/`, **sem** `gold_traj/` aninhado) |
| `download_model.py --model ID` | Um modelo HF (`models/<id>/`) |
| `build_vector_db.py` | BD vetorial RAG **determinístico** (`data/rag_index/`) |
| `finetune.py --model ID` | QLoRA (treina; `--dry-run` só gera YAML) |
| `infer.py --model ID [--rag] [--finetuned]` | Gera predições para o teste |
| `worfeval.py --setup` / `--model ID` | Clone WorFBench + métricas WorFEval |

Rodar o script **já executa** a ação. Use `--dry-run` só para inspecionar caminhos/comandos sem fazer nada.

## Fluxo do experimento

```text
1. download_data.py
2. download_model.py --model qwen35-4b
3. build_vector_db.py
4. infer.py --model qwen35-4b                       # I0
5. infer.py --model qwen35-4b --rag                 # RAG
6. finetune.py --model qwen35-4b
7. infer.py --model qwen35-4b --finetuned           # SFT
8. infer.py --model qwen35-4b --finetuned --rag
9. worfeval.py --setup
10. worfeval.py --model qwen35-4b --all-scenarios
```

## Cenários (`infer.py`)

| Flags | Cenário |
|-------|---------|
| (nenhuma) | I0 — base, sem RAG |
| `--rag` | RAG — base + 2 exemplos recuperados |
| `--finetuned` | SFT — checkpoint, sem RAG |
| `--finetuned --rag` | SFT + RAG |

Sem few-shot fixo do WorFBench. Prompt de teste = **system + user** (+ bloco RAG se `--rag`).

## Inferência (Ollama)

`infer.py` chama o **Ollama** local (`POST /api/chat`). Não usa LangChain nem `transformers` na geração.

1. Instale e inicie o [Ollama](https://ollama.com).
2. Importe o modelo com o **mesmo nome** do `--model` (ex.: `ollama pull qwen35-4b` ou `ollama create` a partir dos pesos HF).
3. Com `--finetuned`, o nome padrão é `{model}-sft` (ex.: `qwen35-4b-sft`). Ajuste em `config/default.yaml` → `inference.ollama` ou `config/backends.yaml`.

```bash
python scripts/infer.py --model qwen35-4b --limit 2   # piloto
python scripts/infer.py --model qwen35-4b --dry-run  # só caminhos
```

Config: `inference.ollama.base_url`, `timeout_s`, `temperature`.

## Marcadores de tempo (inferência e fine-tune)

Cada execução carimba os arquivos com `YYYYMMDD_HHMMSS` (UTC):

- **`infer.py`**: `outputs/predictions/<model>/<task>/graph_eval_{cenário}_{stamp}.json` + `outputs/predictions/<model>/run_{stamp}.json` (`started_at`, `finished_at`, tarefas, caminhos).
- **`finetune.py`**: `outputs/manifests/finetune_{model}_{stamp}.yaml` + `.json` (`stamp`, `started_at`, `finished_at`).
- **`worfeval.py`**: usa automaticamente a predição **mais recente** com stamp.

Assim múltiplas rodadas não se sobrescrevem.

## RAG determinístico

- Embedding fixo em `config/default.yaml` (`rag.embedding_model`, `rag.seed`)
- Exemplos ordenados por `id` estável antes de embedar
- `meta.json` com hash do treino — não reconstrói se o treino não mudou

## WorFEval (como funciona)

1. **`infer.py`** grava, por tarefa, um JSON com lista `{query, workflow}`.
2. **`worfeval.py`** chama `external/WorFBench/node_eval.py --task eval_workflow` sobre a predição mais recente (com stamp).
3. Compara predição vs `data/test/<task>/graph_eval.json` com **sentence-transformers** + matching de grafo.
4. Saída: precision, recall, F1 em `outputs/eval_results/`.

Não usa Ollama/LangChain na avaliação.

## Modelos (`--model`)

Apelido **sem ponto** (seguro no bash). Nome oficial no HF pode ter ponto (ex.: **Qwen3.5** → `Qwen/Qwen3.5-4B`).

| `--model` | Hugging Face | Família |
|-----------|--------------|---------|
| `granite-3b` | [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b) | Granite 4.1 ~3B |
| `qwen35-0.8b` | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Qwen3.5 0,8B |
| `qwen35-2b` | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Qwen3.5 2B |
| `qwen35-4b` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Qwen3.5 4B |
| `gemma3-1b` | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | Gemma 3 1B (licença Google no HF) |
| `gemma3-4b` | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Gemma 3 4B |
| `nemotron-nano-4b` | [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | Nemotron-3-Nano 4B |
| `ministral-3-3b` | [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Ministral 3 3B |

Exemplo: `python scripts/download_model.py --model qwen35-4b`

### Baseline fora do recorte SLM

| `--model` | Hugging Face | Cenários |
|-----------|--------------|----------|
| `qwen35-27b` | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Apenas **I0** |

## Setup

```bash
cd tcc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fine-tune (opcional): pip install -r requirements-sft.txt
```

Mais detalhes (Ollama, LangChain, QLoRA): [`docs/stack.md`](docs/stack.md).
