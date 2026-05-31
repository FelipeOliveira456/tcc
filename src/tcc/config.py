"""Carregamento de configuração YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
LOCAL_CONFIG = Path(__file__).resolve().parents[2] / "config" / "local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrada: {path}")

    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if LOCAL_CONFIG.exists() and path.resolve() != LOCAL_CONFIG.resolve():
        with LOCAL_CONFIG.open(encoding="utf-8") as f:
            local = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local)

    # Paths relativos ao diretório do projeto (raiz do repo tcc).
    project_root = Path(cfg.get("paths", {}).get("project_root", "."))
    if not project_root.is_absolute():
        project_root = (path.parent.parent / project_root).resolve()
    cfg["_project_root"] = project_root
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve path relativo em cfg['paths'][key]."""
    rel = cfg.get("paths", {}).get(key, key)
    p = Path(rel)
    if not p.is_absolute():
        p = cfg["_project_root"] / p
    return p.resolve()


def env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")
