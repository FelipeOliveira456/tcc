"""Pipeline de ajuste fino supervisionado (Unsloth)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcc.finetune.sft import run_finetune


def run_sft(
    cfg: dict[str, Any],
    model_id: str,
    train_json: Path | None = None,
    *,
    dry_run: bool = True,
    export_merged: bool = False,
) -> Path:
    """Delega para ``tcc.finetune.sft.run_finetune`` (dataset a partir do train JSON)."""
    del train_json  # prepare_sharegpt_dataset lê paths.train via config
    return run_finetune(cfg, model_id, dry_run=dry_run, export_merged=export_merged)
