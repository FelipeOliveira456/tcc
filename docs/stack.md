# TCC stack — training, inference, RAG, evaluation

## Roles in the pipeline

| Artifact | Role |
|----------|------|
| `scripts/infer.py` / `scripts/run_model.py` | I0 / RAG / SFT / SFT+RAG — writes predictions (Ollama) |
| `scripts/infer_openrouter.py` | I0 / RAG for the GLM 5.2 baseline |
| `scripts/finetune.py` | Unsloth SFT |
| `src/tcc/finetune/unsloth_sft.py` | bf16 LoRA, loss only on the last assistant turn, 16-bit merge |
| `src/tcc/finetune/sft.py` | Orchestrates dataset preparation + Unsloth |
| WorFBench `LLM/localLLM.py` | OpenAI-style HTTP client → local API (in the paper: LLaMA-Factory) |

### RAG (index)

| Artifact | Library |
|----------|---------|
| `scripts/build_vector_db.py` | **`sentence-transformers`** + numpy |
| `src/tcc/rag/` | Pickle + cosine similarity (no LangChain) |

---

## Ollama + LangChain — where each one is used

```text
                    ┌─────────────────────────────────────┐
                    │  This TCC pipeline (4 settings)      │
                    └─────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌───────────┐      ┌──────────────┐     ┌─────────────────┐
   │  SFT      │      │  Inference   │     │  Evaluation     │
   │  (train)  │      │  (write WF)  │     │  (WorFEval)     │
   └───────────┘      └──────────────┘     └─────────────────┘
         │                    │                    │
   ❌ Ollama            ✅ Ollama (opt.)      ❌ Ollama
   ❌ LangChain         ✅ LangChain (opt.)   ❌ LangChain
   ✅ Unsloth            ✅ Ollama API          ✅ node_eval.py
                         / ChatOllama            + sentence-
                                                 transformers
```

### Ollama

| Use | Suitable? |
|-----|-----------|
| Local **inference** (I0, SFT, RAG after exporting the model) | Yes |
| **SFT** on WorFBench (~18k × 7 messages) | **No** — Ollama does not replace gradient training |
| **WorFEval (metrics)** | No — it needs the prediction JSONs + `eval_workflow` |

### LangChain

| Use | Suitable? |
|-----|-----------|
| **Orchestrate RAG** (retriever + prompt assembly + `ChatOllama`) | Yes, optional |
| **Replace WorFEval** | No |
| **Train SFT** | No — use Unsloth |

---

## Recommended stack

| Stage | Recommendation | Reason |
|-------|----------------|--------|
| **SFT training** | **Unsloth** | 24 GB GPU, bf16 LoRA, 1 epoch, loss on the last assistant turn |
| **RAG** | `sentence-transformers` **or** LangChain + the same embedder | |
| **SLM inference** | **Ollama** after export | |
| **Evaluation (test)** | **Always** `node_eval.py eval_workflow` | Official metrics |

```text
Unsloth               →  SFT (bf16 LoRA, 1 epoch, loss on message 7)
LangChain + Ollama    →  inference + RAG
node_eval (WorFBench) →  evaluation
sentence-transformers →  RAG index + WorFEval metrics
```

---

## Training format

- **First 6 messages** = context (system + 2×(user, assistant) + final user).
- **7th message** = target with loss (gold workflow).
- Implementation: `unsloth_sft.build_masked_example`.

---

## Config

- `config/default.yaml` — `sft.framework: unsloth`, `num_train_epochs: 1`
- `src/tcc/finetune/` — dataset + Unsloth train/merge
