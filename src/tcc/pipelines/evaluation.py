"""Avaliação WorFEval (node/graph) para os quatro cenários."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.pipelines.inference import gold_path, pred_path
from tcc.scenarios import ScenarioId
from tcc.setup.worfbench_repo import worfeval_node_eval_path


def eval_output_path(
    cfg: dict[str, Any],
    model_id: str,
    task: str,
    scenario: ScenarioId,
    eval_type: str,
) -> Path:
    base = resolve_path(cfg, "outputs_dir") / "eval_results" / model_id
    return base / f"{task}_{scenario.value}_{eval_type}.json"


def run_worfeval(
    cfg: dict[str, Any],
    *,
    model_id: str,
    task: str,
    scenario: ScenarioId,
    eval_type: str = "node",
    dry_run: bool = True,
) -> Path:
    node_eval = worfeval_node_eval_path(cfg)
    wb = cfg.get("worfbench", {})
    out = eval_output_path(cfg, model_id, task, scenario, eval_type)

    cmd = [
        "python",
        str(node_eval),
        "--task",
        "eval_workflow",
        "--model_name",
        model_id,
        "--gold_path",
        str(gold_path(cfg, task)),
        "--pred_path",
        str(pred_path(cfg, model_id, task, scenario)),
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
        manifest_dir = resolve_path(cfg, "outputs_dir") / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        m = manifest_dir / f"eval_{model_id}_{task}_{scenario.value}_{eval_type}.json"
        with m.open("w", encoding="utf-8") as f:
            json.dump({"command": cmd}, f, indent=2)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=str(node_eval.parent))
    return out


def evaluate_matrix(
    cfg: dict[str, Any],
    *,
    model_ids: list[str],
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    Avalia matriz modelo × cenário × tarefa × (node|graph).

    Cenários: i0 (puro), rag, sft, sft_rag.
    """
    tasks = cfg.get("worfbench", {}).get("tasks", [])
    eval_types = cfg.get("worfbench", {}).get("eval_types", ["node", "graph"])
    summary: dict[str, Any] = {"models": {}}

    for model_id in model_ids:
        summary["models"][model_id] = {}
        allowed = {ScenarioId.I0}
        if model_id != "qwen35-27b":
            allowed = set(ScenarioId)

        for sid in allowed:
            summary["models"][model_id][sid.value] = {}
            for task in tasks:
                summary["models"][model_id][sid.value][task] = {}
                for et in eval_types:
                    out = run_worfeval(
                        cfg,
                        model_id=model_id,
                        task=task,
                        scenario=sid,
                        eval_type=et,
                        dry_run=dry_run,
                    )
                    if not dry_run and out.exists():
                        with out.open(encoding="utf-8") as f:
                            summary["models"][model_id][sid.value][task][et] = json.load(f)
                    else:
                        summary["models"][model_id][sid.value][task][et] = {
                            "status": "pending",
                            "expected_output": str(out),
                        }

    agg_path = resolve_path(cfg, "outputs_dir") / "eval_results" / "summary.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
