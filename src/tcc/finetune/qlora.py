"""QLoRA via LLaMA-Factory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.finetune.dataset import SHAREGPT_DATASET_NAME, prepare_sharegpt_dataset, write_dataset_info
from tcc.models_registry import get_model_spec, get_sft_template, get_trust_remote_code
from tcc.paths import checkpoint_dir, finetune_manifest_paths, model_dir
from tcc.run_stamp import run_stamp, utc_now_iso


def sft_dataset_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "data_dir") / "sft"


def build_llamafactory_yaml(cfg: dict[str, Any], model_id: str, stamp: str) -> Path:
    """YAML completo para llamafactory-cli train (QLoRA 4-bit, mask_history)."""
    sft = cfg.get("sft", {})
    yaml_path, _ = finetune_manifest_paths(cfg, model_id, stamp)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    template = get_sft_template(cfg, model_id)
    trust = get_trust_remote_code(cfg, model_id)
    dataset_dir = sft_dataset_dir(cfg)
    ckpt = checkpoint_dir(cfg, model_id)
    grad_accum = int(sft.get("gradient_accumulation_steps", 4))
    cutoff = int(sft.get("cutoff_len", 8192))
    yaml_text = f"""# Gerado pelo TCC — {stamp}
### model
model_name_or_path: {model_dir(cfg, model_id)}
trust_remote_code: {str(trust).lower()}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {int(sft.get('lora_rank', 16))}
lora_alpha: {int(sft.get('lora_alpha', 32))}
quantization_bit: {int(sft.get('quantization_bit', 4))}
quantization_method: bnb

### dataset
dataset_dir: {dataset_dir}
dataset: {SHAREGPT_DATASET_NAME}
template: {template}
cutoff_len: {cutoff}
mask_history: true
overwrite_cache: true
preprocessing_num_workers: {int(sft.get('preprocessing_num_workers', 4))}

### output
output_dir: {ckpt}
logging_steps: {int(sft.get('logging_steps', 10))}
save_steps: {int(sft.get('save_steps', 500))}
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: {int(sft.get('per_device_train_batch_size', 2))}
gradient_accumulation_steps: {grad_accum}
learning_rate: {sft.get('learning_rate', 2e-5)}
num_train_epochs: {int(sft.get('num_train_epochs', 3))}
lr_scheduler_type: cosine
warmup_ratio: {sft.get('warmup_ratio', 0.1)}
bf16: true
ddp_timeout: 180000000
seed: {int(sft.get('seed', 42))}
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def build_export_yaml(cfg: dict[str, Any], model_id: str, stamp: str) -> Path:
    """YAML para merge LoRA → pesos completos (importação no Ollama)."""
    export_dir = checkpoint_dir(cfg, model_id) / "merged"
    yaml_path, _ = finetune_manifest_paths(cfg, model_id, stamp)
    export_path = yaml_path.parent / f"export_{model_id}_{stamp}.yaml"
    template = get_sft_template(cfg, model_id)
    trust = get_trust_remote_code(cfg, model_id)
    yaml_text = f"""# Export merge LoRA — {stamp}
### model
model_name_or_path: {model_dir(cfg, model_id)}
adapter_name_or_path: {checkpoint_dir(cfg, model_id)}
template: {template}
trust_remote_code: {str(trust).lower()}
finetuning_type: lora

### export
export_dir: {export_dir}
export_size: 2
export_device: cpu
export_legacy_format: false
"""
    export_path.write_text(yaml_text, encoding="utf-8")
    return export_path


def run_llamafactory_train(yaml_path: Path) -> None:
    subprocess.run(["llamafactory-cli", "train", str(yaml_path)], check=True)


def run_finetune(
    cfg: dict[str, Any],
    model_id: str,
    *,
    dry_run: bool = False,
    export_merged: bool = False,
) -> Path:
    stamp = run_stamp()
    started_at = utc_now_iso()

    sharegpt = prepare_sharegpt_dataset(cfg)
    write_dataset_info(cfg)
    yaml_path = build_llamafactory_yaml(cfg, model_id, stamp)
    _, manifest_path = finetune_manifest_paths(cfg, model_id, stamp)
    manifest = {
        "stamp": stamp,
        "started_at": started_at,
        "model_id": model_id,
        "hf_id": get_model_spec(cfg, model_id)["hf_id"],
        "sft_backend": "llamafactory",
        "sft_template": get_sft_template(cfg, model_id),
        "dataset": str(sharegpt),
        "dataset_info": str(sft_dataset_dir(cfg) / "dataset_info.json"),
        "yaml": str(yaml_path),
        "checkpoint_dir": str(checkpoint_dir(cfg, model_id)),
        "dry_run": dry_run,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if dry_run:
        manifest["finished_at"] = utc_now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return checkpoint_dir(cfg, model_id)

    run_llamafactory_train(yaml_path)
    if export_merged:
        export_yaml = build_export_yaml(cfg, model_id, stamp)
        subprocess.run(["llamafactory-cli", "export", str(export_yaml)], check=True)
        manifest["export_yaml"] = str(export_yaml)
        manifest["merged_dir"] = str(checkpoint_dir(cfg, model_id) / "merged")

    manifest["finished_at"] = utc_now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return checkpoint_dir(cfg, model_id)
