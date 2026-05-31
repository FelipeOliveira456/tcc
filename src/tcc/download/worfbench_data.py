"""Download WorFBench (HF) e layout plano em data/test/<task>/."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download

from tcc.config import resolve_path


def _hf_token(cfg: dict[str, Any]) -> str | None:
    env_name = cfg.get("huggingface", {}).get("token_env", "HF_TOKEN")
    return os.environ.get(env_name) or None


def _flatten_test_gold_traj(data_dir: Path) -> None:
    """Move data/test/gold_traj/<task>/ → data/test/<task>/ e remove pasta aninhada."""
    nested = data_dir / "test" / "gold_traj"
    test_root = data_dir / "test"
    if not nested.is_dir():
        return
    for task_dir in sorted(nested.iterdir()):
        if not task_dir.is_dir():
            continue
        dest = test_root / task_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(task_dir), str(dest))
    shutil.rmtree(nested)
    # Limpa artefatos HF cache sob test/ se existirem
    cache = test_root / ".cache"
    if cache.exists():
        shutil.rmtree(cache)


def download_worfbench(cfg: dict[str, Any]) -> dict[str, Path]:
    data_dir = resolve_path(cfg, "data_dir")
    data_dir.mkdir(parents=True, exist_ok=True)
    hf = cfg.get("huggingface", {})
    token = _hf_token(cfg)

    train_path = Path(
        hf_hub_download(
            repo_id=hf["train_repo"],
            filename=hf.get("train_filename", "worfbench_train.json"),
            repo_type="dataset",
            local_dir=data_dir / "train",
            token=token,
        )
    )

    snapshot_download(
        repo_id=hf["test_repo"],
        repo_type="dataset",
        allow_patterns=["gold_traj/**"],
        local_dir=data_dir / "test",
        token=token,
    )
    _flatten_test_gold_traj(data_dir)

    return {"train": train_path, "test": data_dir / "test"}


def validate_train_dataset(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Esperada lista JSON em {path}")
    return len(data)


def list_test_tasks(cfg: dict[str, Any]) -> list[str]:
    test_root = resolve_path(cfg, "data_dir") / "test"
    tasks = []
    for p in sorted(test_root.iterdir()):
        if p.is_dir() and (p / "graph_eval.json").exists():
            tasks.append(p.name)
    return tasks


def validate_test_layout(cfg: dict[str, Any]) -> dict[str, int]:
    counts = {}
    for task in list_test_tasks(cfg):
        with (resolve_path(cfg, "data_dir") / "test" / task / "graph_eval.json").open(
            encoding="utf-8"
        ) as f:
            data = json.load(f)
        counts[task] = len(data) if isinstance(data, list) else 0
    return counts
