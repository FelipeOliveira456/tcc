#!/usr/bin/env python3
"""Baixa (clone) WorFBench e roda avaliação WorFEval nas predições.

Como funciona o WorFEval
----------------------
1. Você gera predições com infer.py → JSON por tarefa (com stamp):
     outputs/predictions/<model>/<task>/graph_eval_{i0|rag|sft|sft_rag}_YYYYMMDD_HHMMSS.json
   Cada item: {"query": <exemplo gold>, "workflow": "<texto gerado pelo LLM>"}
   O eval usa automaticamente a predição mais recente.

2. O gold de teste está em:
     data/test/<task>/graph_eval.json

3. Este script chama node_eval.py --task eval_workflow do repo WorFBench:
   - Embute nós/arestas do workflow predito e do gold
   - Usa sentence-transformers (all-mpnet-base-v2) para alinhar nós
   - Calcula precision / recall / F1 em modo node (chain/f1chain) e/ou graph (f1graph)

4. Métricas salvas em:
     outputs/eval_results/<model>/<task>/<cenário>_{node|graph}.json

Uso típico
----------
  python scripts/worfeval.py --setup          # só clona WorFBench
  python scripts/worfeval.py --model qwen35-4b                    # I0, chain+graph
  python scripts/worfeval.py --model qwen35-4b --rag --finetuned # SFT+RAG
  python scripts/worfeval.py --model qwen35-4b --all-scenarios   # 4 cenários × chain+graph
  python scripts/worfeval.py --model qwen35-4b --eval-type node  # só chain (f1chain)
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
from tcc.paths import latest_prediction_path
from tcc.worfeval.runner import ensure_worfbench, run_eval_task
from tcc.setup.worfbench_repo import install_worfbench_eval_deps


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
    parser.add_argument(
        "--eval-type",
        choices=["node", "graph", "both"],
        default="both",
        help="node=chain (f1chain), graph=f1graph, both=os dois (padrão)",
    )
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
        install_worfbench_eval_deps()
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

    if args.eval_type == "both":
        eval_types = list(
            cfg.get("worfbench", {}).get("eval_types", ["node", "graph"])
        )
    else:
        eval_types = [args.eval_type]

    for finetuned, rag in scenarios:
        for task in tasks:
            pred = latest_prediction_path(
                cfg, args.model, finetuned=finetuned, rag=rag, task=task
            )
            if not pred.exists():
                print(f"[skip] sem predição: {pred}")
                continue
            for eval_type in eval_types:
                out = run_eval_task(
                    cfg,
                    model_id=args.model,
                    task=task,
                    finetuned=finetuned,
                    rag=rag,
                    eval_type=eval_type,
                    dry_run=args.dry_run,
                )
                print(
                    f"{'[dry-run] ' if args.dry_run else ''}"
                    f"{task}/{eval_type} ({pred.name}) -> {out}"
                )


if __name__ == "__main__":
    main()
