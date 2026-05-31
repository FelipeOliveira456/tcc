"""Download de pesos OSS via Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from tcc.config import resolve_path


def _hf_token(cfg: dict[str, Any]) -> str | None:
    env_name = cfg.get("huggingface", {}).get("token_env", "HF_TOKEN")
    return os.environ.get(env_name) or None


def download_models(cfg: dict[str, Any], *, model_ids: list[str] | None = None) -> dict[str, Path]:
    """Baixa SLMs (e baseline) listados em config.models via snapshot_download."""
    models_dir = resolve_path(cfg, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token(cfg)

    specs: list[dict[str, Any]] = list(cfg.get("models", {}).get("slm", []))
    specs.extend(cfg.get("models", {}).get("baseline_non_slm", []))

    result: dict[str, Path] = {}
    for spec in specs:
        mid = spec["id"]
        if model_ids and mid not in model_ids:
            continue

        dest = models_dir / mid
        if dest.exists() and any(dest.iterdir()):
            result[mid] = dest
            continue

        hf_id = spec.get("hf_id")
        if not hf_id:
            raise ValueError(f"Modelo {mid}: defina hf_id em config.")

        path = snapshot_download(
            repo_id=hf_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )
        result[mid] = Path(path)
    return result
