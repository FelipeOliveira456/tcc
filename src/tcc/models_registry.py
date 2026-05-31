"""Registro de modelos do TCC (id → hf_id)."""

from __future__ import annotations

from typing import Any


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
