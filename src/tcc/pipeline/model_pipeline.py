"""Orquestração do pipeline por modelo (infer + eval + SFT)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tcc.backends.ollama_inference import make_generate_fn
from tcc.config import load_config
from tcc.inference.runner import run_inference
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


def run_infer(
    *,
    label: str,
    model: str,
    rag: bool,
    finetuned: bool,
    config: Path | None,
    limit: int | None,
    task: str | None,
    dry_run: bool,
    progress_desc_prefix: str = "",
    progress_position: int | None = None,
) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
    if dry_run or progress_position is None:
        run_script(
            "infer.py",
            *build_infer_args(model, rag=rag, finetuned=finetuned, limit=limit, task=task),
            config=config,
            dry_run=dry_run,
        )
        return

    cfg = load_config(config)
    generate_fn = make_generate_fn(cfg)
    run_inference(
        cfg,
        model,
        finetuned=finetuned,
        use_rag=rag,
        generate_fn=generate_fn,
        tasks=[task] if task else None,
        limit=limit,
        progress_desc_prefix=progress_desc_prefix,
        progress_position=progress_position,
    )


def run_eval(
    *,
    model: str,
    rag: bool,
    finetuned: bool,
    config: Path | None,
    task: str | None,
    eval_types: list[str],
    dry_run: bool,
) -> None:
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
    """Compat: inferência seguida de eval (cenário único)."""
    run_infer(
        label=label,
        model=model,
        rag=rag,
        finetuned=finetuned,
        config=config,
        limit=limit,
        task=task,
        dry_run=dry_run,
    )
    run_eval(
        model=model,
        rag=rag,
        finetuned=finetuned,
        config=config,
        task=task,
        eval_types=eval_types,
        dry_run=dry_run,
    )


def _scenario_labels(finetuned: bool) -> tuple[str, str]:
    if finetuned:
        return "SFT", "SFT+RAG"
    return "I0", "RAG"


def run_infer_track(
    *,
    track_label: str,
    model: str,
    finetuned: bool,
    config: Path | None,
    limit: int | None,
    task: str | None,
    dry_run: bool,
    progress_position: int | None = None,
) -> None:
    """Uma linha de inferência: (I0 → RAG) ou (SFT → SFT+RAG)."""
    prefix = "SFT" if finetuned else "base"
    print(
        f"\n{'#' * 60}\nTrack {track_label} ({prefix}) — inferência sequencial\n{'#' * 60}",
        flush=True,
    )
    scenario_a, scenario_b = _scenario_labels(finetuned)
    desc_prefix = f"{track_label}/{scenario_a}: " if progress_position is not None else ""
    run_infer(
        label=f"{track_label}: {scenario_a}",
        model=model,
        rag=False,
        finetuned=finetuned,
        config=config,
        limit=limit,
        task=task,
        dry_run=dry_run,
        progress_desc_prefix=desc_prefix,
        progress_position=progress_position,
    )
    desc_prefix = f"{track_label}/{scenario_b}: " if progress_position is not None else ""
    run_infer(
        label=f"{track_label}: {scenario_b}",
        model=model,
        rag=True,
        finetuned=finetuned,
        config=config,
        limit=limit,
        task=task,
        dry_run=dry_run,
        progress_desc_prefix=desc_prefix,
        progress_position=progress_position,
    )


def run_parallel_infer_tracks(
    *,
    model: str,
    config: Path | None,
    limit: int | None,
    task: str | None,
    dry_run: bool,
) -> None:
    """Base (I0→RAG) e SFT (SFT→SFT+RAG) em paralelo (2 threads / 2 modelos Ollama)."""
    print(
        f"\n{'#' * 60}\nInferência paralela: base ‖ SFT\n{'#' * 60}",
        flush=True,
    )
    common = {
        "model": model,
        "config": config,
        "limit": limit,
        "task": task,
        "dry_run": dry_run,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(
                run_infer_track,
                track_label="base",
                finetuned=False,
                progress_position=0,
                **common,
            ): "base",
            pool.submit(
                run_infer_track,
                track_label="sft",
                finetuned=True,
                progress_position=1,
                **common,
            ): "sft",
        }
        errors: list[tuple[str, BaseException]] = []
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                print(f"\n[ok] track {name} concluída.", flush=True)
            except BaseException as exc:
                errors.append((name, exc))
                print(f"\n[erro] track {name}: {exc}", flush=True)
        if errors:
            names = ", ".join(n for n, _ in errors)
            raise RuntimeError(f"Falha na(s) track(s) de inferência: {names}") from errors[0][1]


def run_all_evals(
    *,
    model: str,
    config: Path | None,
    task: str | None,
    eval_types: list[str],
    dry_run: bool,
    include_sft: bool = True,
) -> None:
    """WorFEval para todos os cenários após as inferências."""
    print(f"\n{'=' * 60}\nWorFEval — todos os cenários\n{'=' * 60}", flush=True)
    scenarios: list[tuple[str, bool, bool]] = [
        ("I0", False, False),
        ("RAG", True, False),
    ]
    if include_sft:
        scenarios.extend(
            [
                ("SFT", False, True),
                ("SFT+RAG", True, True),
            ]
        )
    for label, rag, finetuned in scenarios:
        print(f"\n--- eval {label} ---", flush=True)
        run_eval(
            model=model,
            rag=rag,
            finetuned=finetuned,
            config=config,
            task=task,
            eval_types=eval_types,
            dry_run=dry_run,
        )
