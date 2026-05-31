#!/usr/bin/env python3
"""Baixa (clone) WorFBench e roda avaliação WorFEval nas predições.

Como funciona o WorFEval
----------------------
1. Você gera predições com infer.py → JSON por tarefa:
     outputs/predictions/<model>/<task>/graph_eval_{i0|rag|sft|sft_rag}.json
   Cada item: {"query": <exemplo gold>, "workflow": "<texto gerado pelo LLM>"}

2. O gold de teste está em:
     data/test/<task>/graph_eval.json

3. Este script chama node_eval.py --task eval_workflow do repo WorFBench:
   - Embute nós/arestas do workflow predito e do gold
   - Usa sentence-transformers (all-mpnet-base-v2) para alinhar nós
   - Calcula precision / recall / F1 (modo node ou graph)

4. Métricas salvas em:
     outputs/eval_results/<model>/<task>/<cenário>_{node|graph}.json

Uso típico
----------
  python scripts/worfeval.py --setup          # só clona WorFBench
  python scripts/worfeval.py --model qwen35-4b                    # avalia I0, todas tarefas
  python scripts/worfeval.py --model qwen35-4b --rag --finetuned # SFT+RAG
  python scripts/worfeval.py --model qwen35-4b --all-scenarios   # i0,rag,sft,sft_rag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.download.worfbench_data import list_test_tasks
from tcc.models_registry import get_model_spec
from tcc.paths import prediction_path
from tcc.worfeval.runner import ensure_worfbench, run_eval_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Apenas clona WorFBench em external/WorFBench",
    )
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Avalia i0, rag, sft, sft_rag (ignora flags --rag/--finetuned)",
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--eval-type", choices=["node", "graph"], default="node")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só imprime o comando node_eval, sem rodar",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    repo = ensure_worfbench(cfg, install_deps=args.install_deps)
    print(f"WorFBench: {repo}")

    if args.setup:
        return

    if not args.model:
        raise SystemExit("Informe --model ou use só --setup")

    get_model_spec(cfg, args.model)
    tasks = [args.task] if args.task else list_test_tasks(cfg)

    if args.all_scenarios:
        scenarios = [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ]
    else:
        scenarios = [(args.finetuned, args.rag)]

    for finetuned, rag in scenarios:
        for task in tasks:
            pred = prediction_path(cfg, args.model, finetuned=finetuned, rag=rag, task=task)
            if not pred.exists():
                print(f"[skip] sem predição: {pred}")
                continue
            out = run_eval_task(
                cfg,
                model_id=args.model,
                task=task,
                finetuned=finetuned,
                rag=rag,
                eval_type=args.eval_type,
                dry_run=args.dry_run,
            )
            print(f"{'[dry-run] ' if args.dry_run else ''}{task} -> {out}")


if __name__ == "__main__":
    main()
