#!/usr/bin/env python3
"""Gera Modelfile e comando `ollama create` para pesos HF locais.

Se a arquitetura não for convertível pelo Ollama (ex. GraniteForCausalLM),
converte safetensors → GGUF via llama.cpp (clone em external/llama.cpp) e
usa FROM no .gguf.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.backends.ollama_modelfile import (
    ollama_create_argv,
    ollama_create_command,
    write_modelfile,
)
from tcc.config import load_config
from tcc.models_registry import get_model_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", required=True, help="ID em config (ex.: qwen35-0.8b)")
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Usa checkpoint SFT (merged ou adapter exportado)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Diretório do adapter LoRA (Modelfile com ADAPTER; base em models/<id>/)",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        help=(
            "Só no create safetensors (ex. q4_K_M). Na rota GGUF o outtype "
            "vem de inference.ollama.gguf.outtype (padrão f16, sem quantização)."
        ),
    )
    parser.add_argument(
        "--force-gguf",
        action="store_true",
        help="Força conversão GGUF via llama.cpp (mesmo se a arquitetura for suportada)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Executa ollama create (senão só imprime o comando)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    get_model_spec(cfg, args.model)

    modelfile = write_modelfile(
        cfg,
        args.model,
        finetuned=args.finetuned,
        adapter_dir=args.adapter,
        force_gguf=args.force_gguf,
    )
    cmd = ollama_create_command(
        cfg,
        args.model,
        finetuned=args.finetuned,
        modelfile=modelfile,
        quantize=args.quantize,
    )
    argv = ollama_create_argv(
        cfg,
        args.model,
        finetuned=args.finetuned,
        modelfile=modelfile,
        quantize=args.quantize,
    )
    print(f"Modelfile: {modelfile}")
    print(cmd)
    if args.run:
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
