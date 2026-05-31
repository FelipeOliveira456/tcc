"""QLoRA — gera config/manifest (Unsloth ou LLaMA-Factory); execução opcional."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.models_registry import get_model_spec
from tcc.paths import checkpoint_dir, model_dir, train_json


def prepare_sharegpt_dataset(cfg: dict[str, Any], model_id: str) -> Path:
    """Converte treino WorFBench para JSON de conversas (7 turnos preservados)."""
    out = resolve_path(cfg, "data_dir") / "sft" / f"{model_id}_sharegpt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with train_json(cfg).open(encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data:
        conv = item.get("messages") or item.get("conversations") or []
        messages = []
        for turn in conv:
            role = turn.get("role", "user")
            if role in ("human",):
                role = "user"
            if role in ("gpt",):
                role = "assistant"
            messages.append({"role": role, "content": turn.get("content", "")})
        records.append({"messages": messages})
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return out


def build_llamafactory_yaml(cfg: dict[str, Any], model_id: str, dataset_path: Path) -> Path:
    """YAML de referência para QLoRA (loss na última resposta = mask_history)."""
    sft = cfg.get("sft", {})
    out = resolve_path(cfg, "outputs_dir") / "manifests" / f"finetune_{model_id}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = f"""# Gerado pelo TCC — ajuste paths e rode: llamafactory-cli train {out.name}
model_name_or_path: {model_dir(cfg, model_id)}
dataset: {dataset_path}
template: {sft.get('template', 'llama3')}
finetuning_type: lora
quantization_bit: 4
lora_rank: 16
lora_alpha: 32
output_dir: {checkpoint_dir(cfg, model_id)}
num_train_epochs: {sft.get('num_train_epochs', 3)}
learning_rate: {sft.get('learning_rate', 2e-5)}
per_device_train_batch_size: {sft.get('per_device_train_batch_size', 2)}
bf16: true
train_on_prompt: false
mask_history: true
# Loss só na última mensagem assistant (7ª), contexto = 6 primeiras.
"""
    out.write_text(yaml_text, encoding="utf-8")
    return out


def run_finetune(cfg: dict[str, Any], model_id: str, *, dry_run: bool = False) -> Path:
    dataset = prepare_sharegpt_dataset(cfg, model_id)
    yaml_path = build_llamafactory_yaml(cfg, model_id, dataset)
    manifest = {
        "model_id": model_id,
        "hf_id": get_model_spec(cfg, model_id)["hf_id"],
        "dataset": str(dataset),
        "yaml": str(yaml_path),
        "checkpoint_dir": str(checkpoint_dir(cfg, model_id)),
        "note": "Integrar Unsloth ou llamafactory-cli train; não usa Ollama.",
    }
    manifest_path = yaml_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if dry_run:
        return checkpoint_dir(cfg, model_id)

    import subprocess

    subprocess.run(["llamafactory-cli", "train", str(yaml_path)], check=True)
    return checkpoint_dir(cfg, model_id)
