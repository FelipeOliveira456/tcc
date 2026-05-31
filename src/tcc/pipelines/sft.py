"""Pipeline de ajuste fino supervisionado (LLaMA-Factory, alinhado ao WorFBench)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tcc.config import resolve_path


def train_json_to_sharegpt(train_json: Path, out_path: Path) -> Path:
    """Converte WorFBench train para formato conversa do LLaMA-Factory."""
    with train_json.open(encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        conv = item.get("conversations") or item.get("messages") or []
        messages = []
        for turn in conv:
            role = turn.get("role", "user")
            if role in ("human",):
                role = "user"
            if role in ("gpt",):
                role = "assistant"
            messages.append({"role": role, "content": turn.get("content", "")})
        records.append({"messages": messages})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return out_path


def build_sft_command(cfg: dict[str, Any], model_id: str, train_file: Path) -> list[str]:
    sft = cfg.get("sft", {})
    out_dir = resolve_path(cfg, "checkpoints_dir") / model_id
    model_path = resolve_path(cfg, "models_dir") / model_id

    return [
        "llamafactory-cli",
        "train",
        "--stage",
        "sft",
        "--model_name_or_path",
        str(model_path),
        "--dataset",
        str(train_file),
        "--template",
        sft.get("template", "llama3"),
        "--output_dir",
        str(out_dir),
        "--num_train_epochs",
        str(sft.get("num_train_epochs", 3)),
        "--learning_rate",
        str(sft.get("learning_rate", 2e-5)),
        "--per_device_train_batch_size",
        str(sft.get("per_device_train_batch_size", 2)),
        "--seed",
        str(sft.get("seed", 42)),
    ]


def run_sft(
    cfg: dict[str, Any],
    model_id: str,
    train_json: Path,
    *,
    dry_run: bool = True,
) -> Path:
    data_dir = resolve_path(cfg, "data_dir")
    sharegpt = data_dir / "sft" / f"{model_id}_train_sharegpt.json"
    train_json_to_sharegpt(train_json, sharegpt)
    cmd = build_sft_command(cfg, model_id, sharegpt)

    manifest = resolve_path(cfg, "outputs_dir") / "manifests" / f"sft_{model_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        json.dump({"command": cmd, "dry_run": dry_run}, f, indent=2)

    if dry_run:
        return resolve_path(cfg, "checkpoints_dir") / model_id

    subprocess.run(cmd, check=True)
    return resolve_path(cfg, "checkpoints_dir") / model_id
