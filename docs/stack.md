# Stack experimental — TCC WorFBench

Documento de discussão (não executa treino nem teste). Alinha scripts atuais, bibliotecas e uso de Ollama / LangChain.

## O que já existe no repositório

### “Teste” = avaliação WorFEval (não pytest)

| Artefato | Papel |
|----------|--------|
| `scripts/08_run_evaluation.py` | Entrada CLI do passo de **teste/avaliação** |
| `src/tcc/pipelines/evaluation.py` | Chama `node_eval.py --task eval_workflow` do repo clonado |
| `external/WorFBench/node_eval.py` | Protocolo oficial (após passo 2) |
| `external/WorFBench/evaluator/` | Matching subgrafo/subsequência |
| **`sentence-transformers`** (`all-mpnet-base-v2`) | Similaridade semântica entre nós (métricas do paper) |

Ou seja: o “script de teste” **não** usa Ollama nem LangChain hoje. Usa o **WorFBench/WorFEval** + **SentenceTransformers** para pontuar predições já salvas em JSON.

### Geração (antes da avaliação)

| Artefato | Papel |
|----------|--------|
| `scripts/05_run_fewshot_rag.py` | I0 / RAG — gera `pred_traj/...json` |
| `scripts/07_run_sft_rag.py` | SFT+RAG |
| `src/tcc/pipelines/inference.py` | Hoje: subprocess → `node_eval.py --task gen_workflow` |
| WorFBench `LLM/localLLM.py` | Cliente HTTP estilo OpenAI → API local (no paper: **LLaMA-Factory**) |

### Treino (SFT / QLoRA)

| Artefato | Papel |
|----------|--------|
| `scripts/06_run_sft.py` | Entrada CLI |
| `src/tcc/pipelines/sft.py` | Converte JSON → ShareGPT; stub `llamafactory-cli train` |
| Biblioteca de treino **ainda não integrada** de fato | Planejado: **QLoRA** (base 4-bit, adapters bf16) |

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
   │  QLoRA    │      │  Inferência  │     │  Avaliação      │
   │  (treino) │      │  (gerar WF)  │     │  (WorFEval)     │
   └───────────┘      └──────────────┘     └─────────────────┘
         │                    │                    │
   ❌ Ollama            ✅ Ollama (opc.)      ❌ Ollama
   ❌ LangChain         ✅ LangChain (opc.)   ❌ LangChain
   ✅ Unsloth ou         ✅ Ollama API ou       ✅ node_eval.py
      LLaMA-Factory          LangChain              + sentence-
      ou TRL+PEFT            ChatOllama             transformers
```

### Ollama

| Uso | Adequado? |
|-----|-----------|
| **Inferência** local (I0, SFT, RAG após exportar modelo) | Sim |
| **QLoRA / SFT** no WorFBench (~18k × 7 msgs) | **Não** — Ollama não substitui treino com gradiente em adapters |
| **WorFEval (métricas)** | Não — precisa dos JSONs de predição + `eval_workflow` |

Export típico pós-treino: merge LoRA → GGUF ou `ollama create` a partir de safetensors (passo manual).

### LangChain

| Uso | Adequado? |
|-----|-----------|
| **Orquestrar RAG** (retriever + montar prompt + `ChatOllama`) | Sim, opcional |
| **Substituir WorFEval** | Não |
| **Treinar QLoRA** | Não — use TRL/PEFT, Unsloth ou LLaMA-Factory |

LangChain é camada de **composição** (chains, retrievers, chat models), não motor de fine-tuning.

---

## Stack recomendada para o TCC (eficácia × reprodutibilidade)

| Fase | Recomendação | Motivo |
|------|--------------|--------|
| **Treino QLoRA** | **Unsloth** ou **LLaMA-Factory** | GPU 24 GB, `mask_history`, 4-bit; alinhado ao paper WorFBench |
| **RAG** | `sentence-transformers` (já no repo) **ou** LangChain + mesmo embedder | LangChain só se quiser LCEL/documentar agente; ganho marginal |
| **Inferência SLM** | **vLLM** / LLaMA-Factory API **ou** **Ollama** após export | Ollama ok para protótipo; paper usa API OpenAI-compat |
| **Inferência 27B I0** | Ollama com **FP8/GGUF** ou modelo quantizado no Hub | bf16 não cabe em 24 GB |
| **Avaliação (teste)** | **Sempre** `node_eval.py eval_workflow` | Métricas oficiais; incomparável com “avaliar na mão” |

### Por que não “só Ollama + LangChain”?

1. **Treino**: QLoRA exige backward nos adapters — stack PyTorch (Unsloth/TRL/LLaMA-Factory).
2. **WorFEval**: algoritmo de grafo + embeddings fixos — código do benchmark.
3. **LangChain** adiciona dependências e abstração sem acelerar treino; útil se você quer RAG legível no texto do TCC.

### Híbrido razoável (documentar na monografia)

```text
Unsloth/LLaMA-Factory  →  QLoRA (SFT puro, loss na 7ª msg)
LangChain + Ollama     →  inferência + RAG (cenários I0, RAG, SFT, SFT+RAG)
node_eval (WorFBench)  →  geração batch opcional OU só avaliação
sentence-transformers  →  índice RAG + métricas WorFEval
```

---

## Cenários × backend (seu desenho final)

| Cenário | Inferência (sugestão) | Treino |
|---------|----------------------|--------|
| **I0** | Ollama modelo base | — |
| **RAG** | LangChain retriever + `ChatOllama` (2 shots) | — |
| **SFT** | Ollama com Modelfile/adapter exportado | QLoRA Unsloth/LF |
| **SFT+RAG** | LangChain + Ollama modelo ajustado | (mesmo checkpoint) |

Sem `--few_shot` fixo do WorFBench em nenhum cenário.

---

## Formato de treino (confirmado)

- **6 primeiras mensagens** = contexto (system + 2×(user, assistant) + user final).
- **7ª mensagem** = alvo com loss (workflow ouro).
- Implementação: `mask_history: true` (LLaMA-Factory) ou equivalente em Unsloth/TRL.

---

## Próximos arquivos no repo (planejado)

- `config/backends.yaml` — escolher `inference: ollama | llamafactory`, `train: unsloth | llamafactory`
- `src/tcc/backends/` — adaptadores finos (Ollama generate, LangChain RAG) sem substituir WorFEval
- Manter `08_run_evaluation.py` sempre no **WorFEval**
