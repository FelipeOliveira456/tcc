"""Caminhos padronizados de dados, predições e checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcc.config import resolve_path


def scenario_name(*, finetuned: bool, rag: bool) -> str:
    if finetuned and rag:
        return "sft_rag"
    if finetuned:
        return "sft"
    if rag:
        return "rag"
    return "i0"


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
    stamp: str | None = None,
) -> Path:
    """JSON de predições por cenário/tarefa. Com stamp: graph_eval_{cenário}_{stamp}.json."""
    name = scenario_name(finetuned=finetuned, rag=rag)
    suffix = f"_{stamp}" if stamp else ""
    base = resolve_path(cfg, "outputs_dir") / "predictions" / model_id
    if task:
        return base / task / f"graph_eval_{name}{suffix}.json"
    return base / f"graph_eval_{name}{suffix}.json"


def latest_prediction_path(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    rag: bool,
    task: str,
) -> Path:
    """Predição mais recente com marcador de tempo; senão o caminho sem stamp."""
    name = scenario_name(finetuned=finetuned, rag=rag)
    base = resolve_path(cfg, "outputs_dir") / "predictions" / model_id / task
    stamped = sorted(
        base.glob(f"graph_eval_{name}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if stamped:
        return stamped[0]
    return prediction_path(cfg, model_id, finetuned=finetuned, rag=rag, task=task)


def inference_run_meta_path(cfg: dict[str, Any], model_id: str, stamp: str) -> Path:
    return resolve_path(cfg, "outputs_dir") / "predictions" / model_id / f"run_{stamp}.json"


def finetune_manifest_paths(cfg: dict[str, Any], model_id: str, stamp: str) -> tuple[Path, Path]:
    """YAML e JSON de manifesto com marcador de tempo."""
    base = resolve_path(cfg, "outputs_dir") / "manifests"
    stem = f"finetune_{model_id}_{stamp}"
    return base / f"{stem}.yaml", base / f"{stem}.json"
