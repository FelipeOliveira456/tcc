"""Orquestração do pipeline por modelo (infer + eval + SFT)."""

from __future__ import annotations

from pathlib import Path

from tcc.pipeline.steps import run_script


def build_infer_args(
    model: str,
    *,
    rag: bool,
    finetuned: bool,
    limit: int | None,
    task: str | None,
) -> list[str]:
    args = ["--model", model]
    if rag:
        args.append("--rag")
    if finetuned:
        args.append("--finetuned")
    if limit is not None:
        args.extend(["--limit", str(limit)])
    if task:
        args.extend(["--task", task])
    return args


def build_worfeval_args(
    model: str,
    *,
    rag: bool,
    finetuned: bool,
    task: str | None,
    eval_type: str,
) -> list[str]:
    args = ["--model", model]
    if rag:
        args.append("--rag")
    if finetuned:
        args.append("--finetuned")
    if task:
        args.extend(["--task", task])
    args.extend(["--eval-type", eval_type])
    return args


def build_ollama_import_args(
    model: str,
    *,
    finetuned: bool,
    quantize: str | None,
    run: bool,
) -> list[str]:
    args = ["--model", model]
    if finetuned:
        args.append("--finetuned")
    if quantize:
        args.extend(["--quantize", quantize])
    if run:
        args.append("--run")
    return args


def infer_and_eval(
    *,
    label: str,
    model: str,
    rag: bool,
    finetuned: bool,
    config: Path | None,
    limit: int | None,
    task: str | None,
    eval_types: list[str],
    dry_run: bool,
) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
    run_script(
        "infer.py",
        *build_infer_args(model, rag=rag, finetuned=finetuned, limit=limit, task=task),
        config=config,
        dry_run=dry_run,
    )
    for eval_type in eval_types:
        run_script(
            "worfeval.py",
            *build_worfeval_args(
                model,
                rag=rag,
                finetuned=finetuned,
                task=task,
                eval_type=eval_type,
            ),
            config=config,
            dry_run=dry_run,
        )
