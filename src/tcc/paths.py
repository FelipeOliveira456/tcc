"""Caminhos padronizados de dados, predições e checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcc.config import resolve_path


def train_json(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "data_dir") / "train" / "worfbench_train.json"


def test_gold(cfg: dict[str, Any], task: str) -> Path:
    return resolve_path(cfg, "data_dir") / "test" / task / "graph_eval.json"


def model_dir(cfg: dict[str, Any], model_id: str) -> Path:
    return resolve_path(cfg, "models_dir") / model_id


def checkpoint_dir(cfg: dict[str, Any], model_id: str) -> Path:
    return resolve_path(cfg, "checkpoints_dir") / model_id


def vector_db_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "rag_index_dir")


def prediction_path(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    rag: bool,
    task: str | None = None,
) -> Path:
    """Um JSON por cenário (todas as tarefas dentro) ou por tarefa se task setado."""
    if finetuned and rag:
        name = "sft_rag"
    elif finetuned:
        name = "sft"
    elif rag:
        name = "rag"
    else:
        name = "i0"
    base = resolve_path(cfg, "outputs_dir") / "predictions" / model_id
    if task:
        return base / task / f"graph_eval_{name}.json"
    return base / f"graph_eval_{name}.json"
