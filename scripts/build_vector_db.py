#!/usr/bin/env python3
"""Cria BD vetorial RAG determinístico (somente dados de treino)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.paths import vector_db_dir
from tcc.rag.vector_store import build_deterministic_vector_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstrói mesmo se meta.json bater com o treino atual",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    index_path = build_deterministic_vector_db(cfg, force=args.force)
    meta = json.loads((vector_db_dir(cfg) / "meta.json").read_text(encoding="utf-8"))
    print(f"índice: {index_path}")
    print(f"vetores: {meta['num_vectors']}, seed={meta['seed']}, model={meta['embedding_model']}")


if __name__ == "__main__":
    main()
