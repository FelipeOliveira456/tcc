#!/usr/bin/env python3
"""Baixa treino (JSON) e teste (gold por tarefa, sem pasta gold_traj aninhada)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.download.worfbench_data import (
    download_worfbench,
    validate_test_layout,
    validate_train_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    paths = download_worfbench(cfg)
    n_train = validate_train_dataset(paths["train"])
    counts = validate_test_layout(cfg)
    print(f"train: {paths['train']} ({n_train} exemplos)")
    print(f"test:  {paths['test']}/<task>/graph_eval.json")
    for task, n in counts.items():
        print(f"  - {task}: {n}")


if __name__ == "__main__":
    main()
