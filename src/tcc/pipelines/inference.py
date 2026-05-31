"""Geração de workflows por cenário (integra com node_eval.py do WorFBench)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.scenarios import Scenario, ScenarioId, load_scenarios
from tcc.setup.worfbench_repo import worfeval_node_eval_path


def pred_path(
    cfg: dict[str, Any],
    model_id: str,
    task: str,
    scenario: ScenarioId,
) -> Path:
    base = resolve_path(cfg, "outputs_dir") / "predictions" / model_id / task
    return base / f"graph_eval_{scenario.value}.json"


def gold_path(cfg: dict[str, Any], task: str) -> Path:
    repo = resolve_path(cfg, "worfbench_repo")
    return repo / "gold_traj" / task / "graph_eval.json"


def inject_rag_into_messages(
    messages: list[dict[str, str]],
    rag_block: str,
) -> list[dict[str, str]]:
    """Insere bloco RAG antes do turno do usuário final."""
    out: list[dict[str, str]] = []
    inserted = False
    for msg in messages:
        if not inserted and msg.get("role") in ("user", "human"):
            out.append(
                {
                    "role": "system",
                    "content": (
                        "Use os exemplos recuperados do treino como referência estrutural.\n\n"
                        + rag_block
                    ),
                }
            )
            inserted = True
        out.append(msg)
    return out


def run_generation_for_scenario(
    cfg: dict[str, Any],
    *,
    model_id: str,
    task: str,
    scenario_id: ScenarioId,
    dry_run: bool = True,
) -> Path:
    """
    Dispara gen_workflow no WorFBench ou simula (dry_run).

    Para RAG customizado, pré-processa gold e grava JSON temporário com prompts enriquecidos.
    """
    scenarios = load_scenarios(cfg)
    scenario = scenarios[scenario_id]
    node_eval = worfeval_node_eval_path(cfg)
    gold = gold_path(cfg, task)
    pred = pred_path(cfg, model_id, task, scenario_id)
    pred.parent.mkdir(parents=True, exist_ok=True)

    if scenario.use_rag and not dry_run:
        _prepare_rag_augmented_gold(cfg, gold, pred.with_suffix(".rag_prompt.json"), task)

    cmd = [
        "python",
        str(node_eval),
        "--task",
        "gen_workflow",
        "--model_name",
        model_id,
        "--gold_path",
        str(gold),
        "--pred_path",
        str(pred),
        "--task_type",
        task,
    ]
    if scenario.worfbench_few_shot_flag:
        cmd.append("--few_shot")

    if dry_run:
        _write_dry_run_manifest(cfg, cmd, scenario, model_id, task)
        return pred

    subprocess.run(cmd, check=True, cwd=str(node_eval.parent))
    return pred


def _prepare_rag_augmented_gold(
    cfg: dict[str, Any],
    gold: Path,
    out_path: Path,
    task: str,
) -> None:
    from tcc.rag.retriever import TrainRagRetriever

    rag_cfg = cfg.get("rag", {})
    index_path = resolve_path(cfg, "rag_index_dir") / "rag_index.pkl"
    retriever = TrainRagRetriever(index_path, top_k=int(rag_cfg.get("top_k", 2)))

    with gold.open(encoding="utf-8") as f:
        data = json.load(f)

    augmented = []
    for item in data:
        conv = item.get("conversations", [])
        user_text = next(
            (c["content"] for c in conv if c.get("role") in ("user", "human")),
            "",
        )
        retrieved = retriever.retrieve(user_text)
        block = retriever.format_few_shot_block(retrieved)
        new_item = dict(item)
        new_item["conversations"] = inject_rag_into_messages(conv, block)
        augmented.append(new_item)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)


def _write_dry_run_manifest(
    cfg: dict[str, Any],
    cmd: list[str],
    scenario: Scenario,
    model_id: str,
    task: str,
) -> None:
    manifest_dir = resolve_path(cfg, "outputs_dir") / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"gen_{model_id}_{task}_{scenario.id.value}.json"
    payload = {
        "command": cmd,
        "scenario": scenario.id.value,
        "description": scenario.description,
        "use_rag": scenario.use_rag,
        "use_sft_checkpoint": scenario.use_sft_checkpoint,
        "note": "Gerado em dry_run; execute sem --dry-run para inferência real.",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_all_scenarios_for_model(
    cfg: dict[str, Any],
    model_id: str,
    *,
    tasks: list[str] | None = None,
    dry_run: bool = True,
) -> None:
    task_list = tasks or cfg.get("worfbench", {}).get("tasks", [])
    for sid in ScenarioId:
        if model_id == "qwen35-27b" and sid != ScenarioId.I0:
            continue
        for task in task_list:
            run_generation_for_scenario(
                cfg, model_id=model_id, task=task, scenario_id=sid, dry_run=dry_run
            )
