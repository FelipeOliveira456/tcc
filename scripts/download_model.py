#!/usr/bin/env python3
"""Baixa UM modelo do Hugging Face (--model obrigatório)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.download.models import download_models
from tcc.models_registry import get_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--model",
        required=True,
        help="ID em config (ex.: qwen35-4b, gemma3-4b, granite-3b)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    get_model_spec(cfg, args.model)
    paths = download_models(cfg, model_ids=[args.model])
    print(f"OK: {args.model} -> {paths[args.model]}")


if __name__ == "__main__":
    main()
