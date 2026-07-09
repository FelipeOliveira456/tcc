# TCC — WorFBench

Pipeline enxuto. Parâmetro **`--model`** em tudo que envolve SLM (baixa um modelo por vez).

## Scripts

| Script | O que faz |
|--------|-----------|
| `setup_project.py` | **Setup único**: dados HF + índice RAG + WorFBench |
| `run_model.py --model ID` | **Pipeline do modelo**: download → Ollama → I0/RAG/SFT/SFT+RAG (infer + eval) |
| `download_data.py` | Treino + teste (`data/test/<task>/`, **sem** `gold_traj/` aninhado) |
| `download_model.py --model ID` | Um modelo HF (`models/<id>/`) |
| `ollama_import.py --model ID` | Gera Modelfile + comando `ollama create` |
| `build_vector_db.py` | BD vetorial RAG **determinístico** (`data/rag_index/`) |
| `finetune.py --model ID` | QLoRA (treina; `--dry-run` só gera YAML) |
| `infer.py --model ID [--rag] [--finetuned]` | Gera predições para o teste |
| `worfeval.py --setup` / `--model ID` | Clone WorFBench + métricas WorFEval |

Rodar o script **já executa** a ação. Use `--dry-run` só para inspecionar caminhos/comandos sem fazer nada.

## Fluxo do experimento

**Atalho (recomendado):**

```bash
python scripts/setup_project.py
python scripts/run_model.py --model qwen35-0.8b --limit 5   # piloto
python scripts/run_model.py --model qwen35-4b              # completo
```

`run_model.py` faz: download → **SFT** → Ollama (base + SFT) → **infer paralelo** (base: I0→RAG ‖ SFT: SFT→SFT+RAG) → WorFEval de todos os cenários.

**Passo a passo manual:**

```text
1. download_data.py
2. download_model.py --model qwen35-4b
3. build_vector_db.py
4. finetune.py --model qwen35-4b --export-merged
5. ollama_import.py --model qwen35-4b --run
6. ollama_import.py --model qwen35-4b --finetuned --run
7. infer.py --model qwen35-4b                       # I0
8. infer.py --model qwen35-4b --rag                 # RAG
9. infer.py --model qwen35-4b --finetuned           # SFT
10. infer.py --model qwen35-4b --finetuned --rag
11. worfeval.py --setup
12. worfeval.py --model qwen35-4b --all-scenarios
```

(No atalho, os passos 7–10 rodam em paralelo: track base ‖ track SFT.)

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

1. Instale e inicie o [Ollama](https://ollama.com) (versão recente para Qwen3.5 / Gemma 3).
2. Importe o modelo com o **mesmo nome** do `--model` (veja seção [HF → Ollama](#hf--ollama-importar-modelos) abaixo).
3. Com `--finetuned`, o nome padrão é `{model}-sft` (ex.: `qwen35-0.8b-sft`). Ajuste em `config/default.yaml` → `inference.ollama` ou `config/backends.yaml`.

```bash
python scripts/infer.py --model qwen35-0.8b --limit 2   # piloto
python scripts/infer.py --model qwen35-0.8b --dry-run  # só caminhos
```

Config: `inference.ollama.base_url`, `timeout_s`, `temperature`, `think` (padrão `false` para todos os modelos), `num_predict`.

## HF → Ollama (importar modelos)

Fluxo recomendado: baixar pesos no repo, gerar `Modelfile`, registrar no Ollama com `ollama create`. O nome no Ollama deve coincidir com o `--model` (base) ou `{model}-sft` (após fine-tune).

### 1. Baixar do Hugging Face

```bash
# Aceite licenças no site HF quando necessário (Gemma, etc.)
huggingface-cli login          # opcional; ou export HF_TOKEN=...
python scripts/download_model.py --model qwen35-0.8b
# → models/qwen35-0.8b/  (safetensors + tokenizer)
```

### 2. Modelo base no Ollama

**Opção A — script do projeto** (gera `models/ollama/Modelfile.<id>`):

```bash
python scripts/ollama_import.py --model qwen35-0.8b
# imprime: ollama create qwen35-0.8b -f models/ollama/Modelfile.qwen35-0.8b

python scripts/ollama_import.py --model qwen35-0.8b --run
# ou execute o comando impresso manualmente
```

**Opção B — manual** (equivalente):

```bash
cd models/qwen35-0.8b
cat > Modelfile <<'EOF'
FROM .
PARAMETER temperature 0
EOF
ollama create qwen35-0.8b -f Modelfile
```

**Quantização na importação** (menos VRAM; modelos FP16/BF16 do HF):

```bash
python scripts/ollama_import.py --model qwen35-4b --quantize q4_K_M --run
# equivale a: ollama create qwen35-4b --quantize q4_K_M -f ...
```

**Verificar:**

```bash
ollama list
ollama run qwen35-0.8b "Olá"
```

### 3. Modelo SFT no Ollama (após `finetune.py`)

O treino QLoRA roda fora do Ollama (LLaMA-Factory). Para inferir com `--finetuned`, exporte pesos e crie `{model}-sft`.

```bash
pip install -r requirements.txt -r requirements-sft.txt
python scripts/finetune.py --model qwen35-0.8b --export-merged
# checkpoints/qwen35-0.8b/merged/  ← pesos fundidos

python scripts/ollama_import.py --model qwen35-0.8b --finetuned --run
# → ollama create qwen35-0.8b-sft -f models/ollama/Modelfile.qwen35-0.8b-sft
```

**Adapter LoRA sem merge** (Ollama aplica `ADAPTER` sobre o modelo base já criado):

```bash
# base já deve existir: ollama list | grep qwen35-0.8b
python scripts/ollama_import.py --model qwen35-0.8b \
  --adapter checkpoints/qwen35-0.8b --run
# Modelfile: FROM models/qwen35-0.8b + ADAPTER checkpoints/...
```

Preferir **merge + create** para reprodutibilidade; adapter direto exige base idêntica ao treino.

### 4. Notas por família

| Família | Import safetensors | Template SFT (LLaMA-Factory) |
|---------|-------------------|------------------------------|
| Qwen3.5 | Ollama ≥0.17 com suporte qwen35; senão GGUF via llama.cpp | `qwen3_5_nothink` |
| Granite 4.1 | safetensors ou quantize na create | `granite4` |
| Gemma 3 | safetensors; licença Google no HF | `gemma3` |
| Ministral 3 | safetensors (arquitetura Mistral) | `ministral3` |

Modelos grandes: se `ollama create` estourar memória, use `GOMAXPROCS=1 ollama create ...`.

### 5. Mapear nomes customizados

Em `config/local.yaml` ou `config/backends.yaml`:

```yaml
inference:
  ollama:
    models:
      qwen35-0.8b:
        base: meu-qwen-08
        sft: meu-qwen-08-sft
```

## Fine-tune (SFT / QLoRA)

Dependências extras (dois passos, ver [Setup](#setup)):

```bash
pip install -r requirements.txt
pip install -r requirements-sft.txt
```

Inclui **LLaMA-Factory** 0.9.5 (puxa torch, transformers, peft, etc.) e **bitsandbytes** para QLoRA.

Todos os SLMs do recorte usam **LLaMA-Factory** com template por modelo em `config/default.yaml`
(`qwen3_5_nothink`, `granite4`, `gemma3`, `ministral3`).

```bash
# Inspecionar YAML + dataset (sem GPU)
python scripts/finetune.py --model qwen35-0.8b --dry-run

# Treino real
python scripts/finetune.py --model qwen35-0.8b
python scripts/finetune.py --model qwen35-0.8b --export-merged   # merge para Ollama
```

O pipeline gera:

- `data/sft/worfbench_sharegpt.json` — 7 turnos (LLaMA-Factory + `mask_history: true`)
- `data/sft/dataset_info.json` — registro exigido pelo LLaMA-Factory
- `outputs/manifests/finetune_<model>_<stamp>.yaml` — config completa (`stage`, `lora_target`, `cutoff_len`, etc.)
- `checkpoints/<model>/` — adapters; `merged/` após `--export-merged`

Loss só na **7ª mensagem** (workflow ouro); as 6 primeiras são contexto — alinhado ao WorFBench.

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
| `ministral-3-3b` | [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Ministral 3 3B |

Exemplo: `python scripts/download_model.py --model qwen35-4b`

### Baseline fora do recorte SLM

| `--model` | Hugging Face | Cenários |
|-----------|--------------|----------|
| `qwen35-27b` | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Apenas **I0** |

## Setup

Requer **Python 3.11 ou 3.12** (não use 3.14 — `pip install` do SFT falha com `resolution-too-deep`).

```bash
cd tcc
python3.12 -m venv .venv && source .venv/bin/activate
python --version   # deve ser 3.11.x ou 3.12.x

pip install -r requirements.txt
# testes (opcional; pytest para test_qlora, unittest cobre o resto)
pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py' -q
# ou: python -m pytest tests/ -q
# fine-tune (GPU): em um segundo passo, para o resolver não explodir
pip install -r requirements-sft.txt
# Ollama: instale separadamente em https://ollama.com
```

Se o `pip install -r requirements-sft.txt` ainda falhar, instale só o LLaMA-Factory (ele puxa o resto):

```bash
pip install "llamafactory==0.9.5" "bitsandbytes>=0.45.0"
```

Mais detalhes (Ollama, LangChain, QLoRA): [`docs/stack.md`](docs/stack.md).
