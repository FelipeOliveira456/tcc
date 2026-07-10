"""Setup e execução do WorFEval (node_eval.py do WorFBench)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.paths import latest_prediction_path, test_gold
from tcc.setup.worfbench_repo import clone_worfbench, install_worfbench_eval_deps

_EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_workflow.py"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _gold_workflow_count(gold_path: Path) -> int:
    with gold_path.open(encoding="utf-8") as f:
        gold_data = json.load(f)
    return len(gold_data)


def _pred_workflow_count(pred_path: Path) -> int:
    with pred_path.open(encoding="utf-8") as f:
        pred_data = json.load(f)
    missing = sum(1 for row in pred_data if "workflow" not in row)
    if missing:
        raise ValueError(
            f"{pred_path}: {missing} item(ns) sem campo 'workflow' "
            "(formato esperado: {{'query': ..., 'workflow': ...}})"
        )
    return len(pred_data)


def _validate_eval_inputs(*, gold: Path, pred: Path, task: str, tag: str) -> None:
    if not pred.exists():
        raise FileNotFoundError(
            f"Predição não encontrada para {task}/{tag}: {pred}\n"
            "Rode a inferência desse cenário antes do WorFEval."
        )
    if not gold.exists():
        raise FileNotFoundError(f"Gold de teste não encontrado: {gold}")

    n_gold = _gold_workflow_count(gold)
    n_pred = _pred_workflow_count(pred)
    if n_gold != n_pred:
        raise ValueError(
            f"Contagem incompatível em {task}/{tag}: gold={n_gold}, pred={n_pred}\n"
            f"  gold: {gold}\n"
            f"  pred: {pred}\n"
            "O node_eval.py aborta com AssertionError nesse caso. "
            "Causas comuns: inferência interrompida, --limit numa run recente, "
            "ou arquivo de predição errado (stamp antigo)."
        )


def ensure_worfbench(cfg: dict[str, Any], *, install_deps: bool = False) -> Path:
    repo = clone_worfbench(cfg, force=False)
    if install_deps:
        install_worfbench_eval_deps()
    return repo


def run_eval_task(
    cfg: dict[str, Any],
    *,
    model_id: str,
    task: str,
    finetuned: bool,
    rag: bool,
    eval_type: str = "node",
    dry_run: bool = False,
) -> Path:
    repo = resolve_path(cfg, "worfbench_repo")
    wb = cfg.get("worfbench", {})
    pred = latest_prediction_path(cfg, model_id, finetuned=finetuned, rag=rag, task=task)
    gold = test_gold(cfg, task)
    if finetuned and rag:
        tag = "sft_rag"
    elif finetuned:
        tag = "sft"
    elif rag:
        tag = "rag"
    else:
        tag = "i0"
    out = (
        resolve_path(cfg, "outputs_dir")
        / "eval_results"
        / model_id
        / task
        / f"{tag}_{eval_type}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_EVAL_SCRIPT),
        "--gold_path",
        str(gold),
        "--pred_path",
        str(pred),
        "--eval_model",
        wb.get("eval_encoder", "all-mpnet-base-v2"),
        "--eval_output",
        str(out),
        "--eval_type",
        eval_type,
        "--task_type",
        task,
        "--worfbench-repo",
        str(repo),
    ]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return out

    _validate_eval_inputs(gold=gold, pred=pred, task=task, tag=tag)

    if not (repo / "evaluator" / "graph_evaluator.py").is_file():
        raise FileNotFoundError(
            f"WorFBench incompleto em {repo} (falta evaluator/). "
            "Rode: python scripts/worfeval.py --setup"
        )

    if not _EVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"Script de eval não encontrado: {_EVAL_SCRIPT}")

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        hint = ""
        if "AssertionError" in detail and "gold plan and pred plan" in detail:
            hint = (
                "\nDica: gold e pred com tamanhos diferentes — "
                "verifique com: "
                f"python -c \"import json; g=json.load(open('{gold}')); "
                f"p=json.load(open('{pred}')); print(len(g), len(p))\""
            )
        elif "ModuleNotFoundError" in detail:
            hint = (
                "\nDica: instale deps do eval: "
                "python scripts/worfeval.py --setup --install-deps"
            )
        raise RuntimeError(
            f"WorFEval falhou ({task}/{tag}, {eval_type}).\n"
            f"Comando: {' '.join(cmd)}\n"
            f"{detail}{hint}"
        ) from exc

    if proc.stdout.strip():
        print(proc.stdout.strip())
    return out
