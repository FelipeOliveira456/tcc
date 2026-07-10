#!/usr/bin/env python3
"""Clona llama.cpp (shallow) para conversão GGUF no ollama_import.

Usado pelo setup_project. O import só converte se a arquitetura exigir GGUF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.backends.gguf_convert import ensure_llama_cpp, llama_cpp_dir
from tcc.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reclona mesmo se external/llama.cpp já existir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só imprime o destino, sem clonar",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    dest = llama_cpp_dir(cfg)
    if args.dry_run:
        print(f"[dry-run] llama.cpp → {dest}")
        return
    path = ensure_llama_cpp(cfg, force=args.force)
    print(f"llama.cpp: {path}")


if __name__ == "__main__":
    main()
