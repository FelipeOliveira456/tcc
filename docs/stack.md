# Stack do TCC — treino, inferência, RAG, avaliação

## Papéis no pipeline

| Artefato | Papel |
|----------|--------|
| `scripts/05_run_fewshot_rag.py` / `run_model.py` | I0 / RAG — gera predições |
| `scripts/finetune.py` | SFT Unsloth |
| `src/tcc/finetune/unsloth_sft.py` | bf16 LoRA, loss só no último assistant, merge 16-bit |
| `src/tcc/finetune/sft.py` | Orquestra prepare dataset + Unsloth |
| WorFBench `LLM/localLLM.py` | Cliente HTTP estilo OpenAI → API local (no paper: LLaMA-Factory) |

### RAG (índice)

| Artefato | Biblioteca |
|----------|------------|
| `scripts/04_build_rag_index.py` | **`sentence-transformers`** + numpy |
| `src/tcc/rag/` | Pickle + similaridade coseno (sem LangChain) |

---

## Ollama + LangChain — onde cada um serve

```text
                    ┌─────────────────────────────────────┐
                    │  Seu pipeline TCC (4 cenários)       │
                    └─────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌───────────┐      ┌──────────────┐     ┌─────────────────┐
   │  SFT      │      │  Inferência  │     │  Avaliação      │
   │  (treino) │      │  (gerar WF)  │     │  (WorFEval)     │
   └───────────┘      └──────────────┘     └─────────────────┘
         │                    │                    │
   ❌ Ollama            ✅ Ollama (opc.)      ❌ Ollama
   ❌ LangChain         ✅ LangChain (opc.)   ❌ LangChain
   ✅ Unsloth            ✅ Ollama API          ✅ node_eval.py
                         / ChatOllama            + sentence-
                                                 transformers
```

### Ollama

| Uso | Adequado? |
|-----|-----------|
| **Inferência** local (I0, SFT, RAG após exportar modelo) | Sim |
| **SFT** no WorFBench (~18k × 7 msgs) | **Não** — Ollama não substitui treino com gradiente |
| **WorFEval (métricas)** | Não — precisa dos JSONs de predição + `eval_workflow` |

### LangChain

| Uso | Adequado? |
|-----|-----------|
| **Orquestrar RAG** (retriever + montar prompt + `ChatOllama`) | Sim, opcional |
| **Substituir WorFEval** | Não |
| **Treinar SFT** | Não — use Unsloth |

---

## Stack recomendada

| Fase | Recomendação | Motivo |
|------|--------------|--------|
| **Treino SFT** | **Unsloth** | GPU 24 GB, bf16 LoRA, 1 época, loss no último assistant |
| **RAG** | `sentence-transformers` **ou** LangChain + mesmo embedder | |
| **Inferência SLM** | **Ollama** após export | |
| **Avaliação (teste)** | **Sempre** `node_eval.py eval_workflow` | Métricas oficiais |

```text
Unsloth               →  SFT (bf16 LoRA, 1 época, loss na 7ª msg)
LangChain + Ollama    →  inferência + RAG
node_eval (WorFBench) →  avaliação
sentence-transformers →  índice RAG + métricas WorFEval
```

---

## Formato de treino

- **6 primeiras mensagens** = contexto (system + 2×(user, assistant) + user final).
- **7ª mensagem** = alvo com loss (workflow ouro).
- Implementação: `unsloth_sft.build_masked_example`.

---

## Config

- `config/default.yaml` — `sft.framework: unsloth`, `num_train_epochs: 1`
- `src/tcc/finetune/` — dataset + Unsloth train/merge
