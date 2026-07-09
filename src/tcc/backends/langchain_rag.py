"""Stub — RAG com LangChain (não executa).

Alternativa ao retriever nativo em tcc.rag.retriever.
Mesmos 2 exemplos (top_k=2), sem few-shot fixo do WorFBench.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LangChainRagConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 2
    ollama_model: str = "qwen35-4b"
    ollama_base_url: str = "http://127.0.0.1:11434"


def build_rag_chain_stub(cfg: LangChainRagConfig) -> Any:
    """
    Implementação futura (exemplo de peças):
      - HuggingFaceEmbeddings(model_name=cfg.embedding_model)
      - VectorStore from worfbench_train.json (FAISS/Chroma)
      - ChatOllama(base_url=..., model=...)
      - prompt: system + retrieved examples + user
    """
    raise NotImplementedError(
        "langchain-huggingface + langchain-ollama; ver docs/stack.md"
    )


def format_prompt_with_rag(
    system: str,
    user: str,
    retrieved_blocks: list[str],
) -> list[dict[str, str]]:
    """Monta messages para Ollama — 2 shots dinâmicos + pergunta."""
    rag_text = "\n\n".join(retrieved_blocks)
    return [
        {"role": "system", "content": f"{system}\n\nRetrieved examples:\n{rag_text}"},
        {"role": "user", "content": user},
    ]
