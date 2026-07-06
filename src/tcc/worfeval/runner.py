"""Setup e execução do WorFEval (node_eval.py do WorFBench)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.paths import latest_prediction_path, test_gold
from tcc.setup.worfbench_repo import clone_worfbench


def ensure_worfbench(cfg: dict[str, Any], *, install_deps: bool = False) -> Path:
    repo = clone_worfbench(cfg, force=False)
    if install_deps:
        req = repo / "requirements.txt"
        if req.exists():
            subprocess.run(["pip", "install", "-r", str(req)], check=True)
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
    node_eval = repo / "node_eval.py"
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
        "python",
        str(node_eval),
        "--task",
        "eval_workflow",
        "--model_name",
        model_id,
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
    ]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return out
    subprocess.run(cmd, check=True, cwd=str(repo))
    return out
