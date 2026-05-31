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


def example_to_text(item: dict[str, Any], fields: list[str]) -> str:
    parts: list[str] = []
    conv = item.get("conversations") or item.get("messages") or []
    user_msg = next(
        (c.get("content", "") for c in conv if c.get("role") in ("user", "human")),
        "",
    )
    asst_msg = next(
        (c.get("content", "") for c in conv if c.get("role") in ("assistant", "gpt")),
        "",
    )
    mapping = {"query": user_msg, "workflow": asst_msg}
    for field in fields:
        if field in mapping and mapping[field]:
            parts.append(mapping[field])
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

    Na execução real, carrega SentenceTransformer e codifica o corpus.
    """
    from sentence_transformers import SentenceTransformer

    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(embedding_model_name)
    examples: list[RagExample] = []
    texts: list[str] = []

    for i, item in enumerate(load_train_examples(train_json)):
        text = example_to_text(item, chunk_fields)
        if not text:
            continue
        ex_id = str(item.get("id", i))
        task = item.get("task_type") or item.get("source")
        examples.append(
            RagExample(
                example_id=ex_id,
                query_text=text,
                workflow_text=example_to_text(item, ["workflow"]),
                task_type=str(task) if task else None,
            )
        )
        texts.append(text)

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for ex, emb in zip(examples, embeddings, strict=True):
        ex.embedding = emb

    out = index_dir / "rag_index.pkl"
    with out.open("wb") as f:
        pickle.dump({"examples": examples, "model": embedding_model_name, "seed": seed}, f)
    return out
