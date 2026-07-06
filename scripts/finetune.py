#!/usr/bin/env python3
"""Fine-tuning QLoRA de UM modelo (--model)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.finetune.qlora import run_finetune
from tcc.models_registry import get_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só gera dataset + YAML em outputs/manifests/, sem treinar",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    get_model_spec(cfg, args.model)

    out = run_finetune(cfg, args.model, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] manifest/YAML com stamp em outputs/manifests/finetune_{args.model}_*")
        print(f"checkpoint previsto: {out}")
    else:
        print(f"checkpoint: {out}")


if __name__ == "__main__":
    main()
