#!/usr/bin/env python3
"""Inferência no teste: gera workflows (JSON) para o WorFEval.

Combinações (--model obrigatório):
  (default)        I0   — base, sem RAG
  --rag            RAG  — base + RAG
  --finetuned      SFT  — checkpoint, sem RAG
  --finetuned --rag SFT+RAG

Requer Ollama com o modelo importado (ver config/default.yaml → inference.ollama).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.backends.ollama_inference import make_generate_fn, resolve_ollama_model_name
from tcc.config import load_config
from tcc.download.worfbench_data import list_test_tasks
from tcc.inference.runner import run_inference
from tcc.models_registry import get_model_spec
from tcc.paths import inference_run_meta_path, prediction_path
from tcc.run_stamp import run_stamp


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
    parser.add_argument(
        "--progress-position",
        type=int,
        default=None,
        help="Linha fixa do tqdm (0=base, 1=sft) para barras paralelas no mesmo terminal",
    )
    parser.add_argument(
        "--progress-prefix",
        default="",
        help="Prefixo do tqdm, ex. base/I0:",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    get_model_spec(cfg, args.model)

    ollama_name = resolve_ollama_model_name(cfg, args.model, args.finetuned)
    stamp = run_stamp()

    if args.dry_run:
        tasks = [args.task] if args.task else list_test_tasks(cfg)
        print(f"[dry-run] infer model={args.model} ollama={ollama_name} stamp={stamp} rag={args.rag}")
        for t in tasks:
            print(
                f"  -> {prediction_path(cfg, args.model, finetuned=args.finetuned, rag=args.rag, task=t, stamp=stamp)}"
            )
        print(f"  meta -> {inference_run_meta_path(cfg, args.model, stamp)}")
        return

    generate_fn = make_generate_fn(cfg)
    out = run_inference(
        cfg,
        args.model,
        finetuned=args.finetuned,
        use_rag=args.rag,
        generate_fn=generate_fn,
        tasks=[args.task] if args.task else None,
        limit=args.limit,
        progress_desc_prefix=f"{args.progress_prefix}: " if args.progress_prefix else "",
        progress_position=args.progress_position,
    )
    print(f"predições em: {out.parent} (ollama={ollama_name}, stamp no nome do arquivo)")


if __name__ == "__main__":
    main()
