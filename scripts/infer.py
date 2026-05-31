#!/usr/bin/env python3
"""Inferência no teste: gera workflows (JSON) para o WorFEval.

Combinações (--model obrigatório):
  (default)        I0   — base, sem RAG
  --rag            RAG  — base + RAG
  --finetuned      SFT  — checkpoint, sem RAG
  --finetuned --rag SFT+RAG

Plugue Ollama/LangChain em tcc/backends (ver docs/stack.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.download.worfbench_data import list_test_tasks
from tcc.inference.runner import run_inference
from tcc.models_registry import get_model_spec
from tcc.paths import prediction_path


def _stub_generate(messages, model_id: str, finetuned: bool) -> str:
    raise NotImplementedError(
        "Conecte Ollama/LangChain em infer.py / tcc.backends. "
        "Entrada: messages. Saída: texto do workflow."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só mostra caminhos de saída, sem chamar o modelo",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    get_model_spec(cfg, args.model)

    if args.dry_run:
        tasks = [args.task] if args.task else list_test_tasks(cfg)
        print(f"[dry-run] infer model={args.model} rag={args.rag} finetuned={args.finetuned}")
        for t in tasks:
            print(f"  -> {prediction_path(cfg, args.model, finetuned=args.finetuned, rag=args.rag, task=t)}")
        return

    out = run_inference(
        cfg,
        args.model,
        finetuned=args.finetuned,
        use_rag=args.rag,
        generate_fn=_stub_generate,
        tasks=[args.task] if args.task else None,
        limit=args.limit,
    )
    print(f"predições em: {out.parent}")


if __name__ == "__main__":
    main()
