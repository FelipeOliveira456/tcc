#!/usr/bin/env python3
"""Setup único do projeto: dados, índice RAG e WorFEval.

Equivalente a:
  python scripts/download_data.py
  python scripts/build_vector_db.py
  python scripts/worfeval.py --setup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.pipeline.setup_pipeline import setup_steps
from tcc.pipeline.steps import run_script


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstrói o índice RAG mesmo se o treino não mudou",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só imprime os comandos, sem executar",
    )
    args = parser.parse_args()

    for label, script, script_args in setup_steps(force_rag=args.force):
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
        run_script(script, *script_args, config=args.config, dry_run=args.dry_run)

    print("\nSetup concluído.", flush=True)


if __name__ == "__main__":
    main()
