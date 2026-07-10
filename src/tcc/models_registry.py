"""Registro de modelos do TCC (id → hf_id, template SFT)."""

from __future__ import annotations

from typing import Any

# Templates de chat (metadado; Unsloth usa apply_chat_template do tokenizer HF).
_DEFAULT_SFT_TEMPLATE = "llama3"

# Fallback quando o modelo não define sft_template no YAML.
_TEMPLATE_BY_ID: dict[str, str] = {
    "granite-3b": "granite4",
    "qwen35-0.8b": "qwen3_5_nothink",
    "qwen35-2b": "qwen3_5_nothink",
    "qwen35-4b": "qwen3_5_nothink",
    "qwen35-27b": "qwen3_5_nothink",
    "gemma3-1b": "gemma3",
    "gemma3-4b": "gemma3",
    "ministral-3-3b": "ministral3",
    "nemotron-nano-4b": "nemotron",
}


def all_model_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    specs = list(cfg.get("models", {}).get("slm", []))
    specs.extend(cfg.get("models", {}).get("baseline_non_slm", []))
    return specs


def get_model_spec(cfg: dict[str, Any], model_id: str) -> dict[str, Any]:
    for spec in all_model_specs(cfg):
        if spec["id"] == model_id:
            return spec
    known = [s["id"] for s in all_model_specs(cfg)]
    raise KeyError(f"Modelo '{model_id}' não está em config. Opções: {known}")


def get_sft_template(cfg: dict[str, Any], model_id: str) -> str:
    spec = get_model_spec(cfg, model_id)
    return spec.get("sft_template") or _TEMPLATE_BY_ID.get(model_id, _DEFAULT_SFT_TEMPLATE)


def get_trust_remote_code(cfg: dict[str, Any], model_id: str) -> bool:
    spec = get_model_spec(cfg, model_id)
    return bool(spec.get("trust_remote_code", False))
