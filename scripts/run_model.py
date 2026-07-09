#!/usr/bin/env python3
"""Pipeline completo de UM modelo: download → SFT → Ollama (base+SFT) → infer paralelo → eval.

Ordem:
  1. download_model.py
  2. finetune.py --export-merged          (pula com --skip-sft)
  3. ollama_import.py (base) + (SFT)
  4. Inferência paralela (2 modelos no Ollama):
       track base: I0 → RAG
       track SFT:  SFT → SFT+RAG
     Com --skip-sft: só track base (sequencial).
  5. WorFEval de todos os cenários gerados

Pré-requisito: `python scripts/setup_project.py` (dados, RAG, WorFBench).
Ollama deve estar rodando antes da inferência.
Recomendado: OLLAMA_MAX_LOADED_MODELS>=2 para manter base e SFT na VRAM.

Exemplos:
  python scripts/run_model.py --model qwen35-0.8b --limit 5
  python scripts/run_model.py --model qwen35-4b --quantize q4_K_M
  python scripts/run_model.py --model qwen35-0.8b --skip-sft
  python scripts/run_model.py --model qwen35-0.8b --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.models_registry import get_model_spec
from tcc.pipeline.model_pipeline import (
    build_ollama_import_args,
    run_all_evals,
    run_infer_track,
    run_parallel_infer_tracks,
)
from tcc.pipeline.steps import run_script
from tcc.worfeval.runner import ensure_worfbench


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", required=True, help="ID em config (ex.: qwen35-0.8b)")
    parser.add_argument("--limit", type=int, default=None, help="Limite de exemplos por tarefa no infer")
    parser.add_argument("--task", default=None, help="Só uma tarefa WorFBench (ex.: wikihow)")
    parser.add_argument(
        "--quantize",
        default=None,
        help="Quantização no ollama create (ex.: q4_K_M)",
    )
    parser.add_argument(
        "--eval-graph",
        action="store_true",
        help="Além de node, roda WorFEval em modo graph",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Não roda ollama_import (modelos já criados no Ollama)",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Não treina nem importa/roda cenários SFT (só base: I0→RAG + eval)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Só imprime os comandos",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    get_model_spec(cfg, args.model)

    eval_types = ["node"]
    if args.eval_graph:
        eval_types.append("graph")

    if not args.dry_run:
        repo = ensure_worfbench(cfg)
        print(f"WorFBench: {repo}")

    if not args.skip_download:
        print(f"\n{'=' * 60}\nDownload HF — {args.model}\n{'=' * 60}", flush=True)
        run_script(
            "download_model.py",
            "--model",
            args.model,
            config=args.config,
            dry_run=args.dry_run,
        )

    if not args.skip_sft:
        print(f"\n{'=' * 60}\nFine-tune QLoRA — {args.model}\n{'=' * 60}", flush=True)
        run_script(
            "finetune.py",
            "--model",
            args.model,
            "--export-merged",
            config=args.config,
            dry_run=args.dry_run,
        )

    if not args.skip_ollama:
        print(f"\n{'=' * 60}\nOllama — modelo base\n{'=' * 60}", flush=True)
        run_script(
            "ollama_import.py",
            *build_ollama_import_args(
                args.model, finetuned=False, quantize=args.quantize, run=True
            ),
            config=args.config,
            dry_run=args.dry_run,
        )
        if not args.skip_sft:
            print(f"\n{'=' * 60}\nOllama — modelo SFT\n{'=' * 60}", flush=True)
            run_script(
                "ollama_import.py",
                *build_ollama_import_args(
                    args.model, finetuned=True, quantize=args.quantize, run=True
                ),
                config=args.config,
                dry_run=args.dry_run,
            )

    infer_common = {
        "model": args.model,
        "config": args.config,
        "limit": args.limit,
        "task": args.task,
        "dry_run": args.dry_run,
    }

    if args.skip_sft:
        run_infer_track(track_label="base", finetuned=False, **infer_common)
    else:
        run_parallel_infer_tracks(**infer_common)

    run_all_evals(
        model=args.model,
        config=args.config,
        task=args.task,
        eval_types=eval_types,
        dry_run=args.dry_run,
        include_sft=not args.skip_sft,
    )

    print(f"\nPipeline concluído para {args.model}.", flush=True)


if __name__ == "__main__":
    main()
