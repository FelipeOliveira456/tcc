"""Indexação do conjunto de treino para RAG (somente partição train)."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RagExample:
    example_id: str
    query_text: str
    workflow_text: str
    task_type: str | None
    embedding: np.ndarray | None = None


def load_train_examples(train_json: Path) -> list[dict[str, Any]]:
    with train_json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Treino WorFBench deve ser lista de exemplos.")
    return data


def _conversation(item: dict[str, Any]) -> list[dict[str, Any]]:
    return item.get("conversations") or item.get("messages") or []


def last_user_message(item: dict[str, Any]) -> str:
    """Último turno user/human — pergunta real (após demos few-shot)."""
    msg = ""
    for turn in _conversation(item):
        if turn.get("role") in ("user", "human"):
            msg = turn.get("content", "")
    return msg


def last_assistant_message(item: dict[str, Any]) -> str:
    """Último turno assistant/gpt — workflow ouro da tarefa real."""
    msg = ""
    for turn in _conversation(item):
        if turn.get("role") in ("assistant", "gpt"):
            msg = turn.get("content", "")
    return msg


def example_to_text(item: dict[str, Any], fields: list[str]) -> str:
    """Extrai campos do último par user/assistant (não dos demos iniciais)."""
    mapping = {
        "query": last_user_message(item),
        "workflow": last_assistant_message(item),
    }
    parts = [mapping[f] for f in fields if f in mapping and mapping[f]]
    return "\n".join(parts).strip()


def build_rag_index(
    train_json: Path,
    index_dir: Path,
    embedding_model_name: str,
    chunk_fields: list[str],
    *,
    seed: int = 42,
) -> Path:
    """
    Persiste índice (embeddings + metadados) em index_dir/rag_index.pkl.

    Embedding usa só a pergunta (último user). Workflow (último assistant)
    fica guardado para o prompt, mas não entra na similaridade.
    """
    from sentence_transformers import SentenceTransformer

    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(embedding_model_name)
    examples: list[RagExample] = []
    # Similaridade = só query; se chunk_fields omitir query, cai no último user.
    embed_fields = [f for f in chunk_fields if f == "query"] or ["query"]

    for i, item in enumerate(load_train_examples(train_json)):
        query = last_user_message(item)
        workflow = last_assistant_message(item)
        embed_text = example_to_text(item, embed_fields)
        if not embed_text or not workflow:
            continue
        ex_id = str(item.get("id", i))
        task = item.get("task_type") or item.get("source")
        examples.append(
            RagExample(
                example_id=ex_id,
                query_text=query,
                workflow_text=workflow,
                task_type=str(task) if task else None,
            )
        )

    texts = [e.query_text for e in examples]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for ex, emb in zip(examples, embeddings, strict=True):
        ex.embedding = emb

    out = index_dir / "rag_index.pkl"
    with out.open("wb") as f:
        pickle.dump({"examples": examples, "model": embedding_model_name, "seed": seed}, f)
    return out
