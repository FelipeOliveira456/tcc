"""Orquestração SFT via Unsloth (bf16 LoRA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.finetune.dataset import prepare_sharegpt_dataset
from tcc.models_registry import get_model_spec
from tcc.paths import checkpoint_dir, finetune_manifest_paths, model_dir
from tcc.run_stamp import run_stamp, utc_now_iso


def sft_dataset_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "data_dir") / "sft"


def sft_hparams(cfg: dict[str, Any], model_id: str, stamp: str) -> dict[str, Any]:
    sft = cfg.get("sft", {})
    return {
        "backend": "unsloth",
        "model_id": model_id,
        "stamp": stamp,
        "num_train_epochs": int(sft.get("num_train_epochs", 1)),
        "cutoff_len": int(sft.get("cutoff_len", 2048)),
        "max_example_tokens": int(
            sft.get("max_example_tokens", sft.get("cutoff_len", 2048))
        ),
        "per_device_train_batch_size": int(sft.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(sft.get("gradient_accumulation_steps", 8)),
        "learning_rate": sft.get("learning_rate", 2e-5),
        "lora_rank": int(sft.get("lora_rank", 16)),
        "lora_alpha": int(sft.get("lora_alpha", 32)),
        "load_in_4bit": bool(sft.get("load_in_4bit", False)),
        "load_in_16bit": bool(sft.get("load_in_16bit", True)),
        "optim": sft.get("optim", "adamw_8bit"),
        "mask": "last_assistant_only",
        "model_path": str(model_dir(cfg, model_id)),
    }


def run_finetune(
    cfg: dict[str, Any],
    model_id: str,
    *,
    dry_run: bool = False,
    export_merged: bool = False,
) -> Path:
    stamp = run_stamp()
    started_at = utc_now_iso()

    sharegpt = prepare_sharegpt_dataset(cfg, model_id=model_id)

    _, manifest_path = finetune_manifest_paths(cfg, model_id, stamp)
    manifest: dict[str, Any] = {
        "stamp": stamp,
        "started_at": started_at,
        "model_id": model_id,
        "hf_id": get_model_spec(cfg, model_id)["hf_id"],
        "sft_backend": "unsloth",
        "dataset": str(sharegpt),
        "checkpoint_dir": str(checkpoint_dir(cfg, model_id)),
        "dry_run": dry_run,
        "export_merged": export_merged,
        "hparams": sft_hparams(cfg, model_id, stamp),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if dry_run:
        manifest["finished_at"] = utc_now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return checkpoint_dir(cfg, model_id)

    from tcc.finetune.unsloth_sft import run_unsloth_train

    run_unsloth_train(cfg, model_id, dataset_path=sharegpt, export_merged=export_merged)
    if export_merged:
        manifest["merged_dir"] = str(checkpoint_dir(cfg, model_id) / "merged")

    manifest["finished_at"] = utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return checkpoint_dir(cfg, model_id)
