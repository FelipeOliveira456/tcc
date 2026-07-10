"""WorFEval eval_workflow — igual ao upstream, mas não crasha em pred vazia.

O node_eval.py original só faz `continue` quando workflow_to_graph_list retorna [].
Se a pred tem "Node" mas sem linhas `1: ...` ou sem arestas `(START,1)`, o parser pode
gerar grafo só com START/END e o t_eval_nodes quebra com matriz 0×768.

Este módulo pula predições não parseáveis (contam como 0 na média, como no upstream).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


def _real_nodes(graph: dict[str, Any] | list) -> list[str]:
    if not graph or graph == []:
        return []
    nodes = graph.get("nodes", [])
    return [n for n in nodes if n not in ("START", "END")]


def _load_worfbench_eval(repo: Path):
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from evaluator.graph_evaluator import t_eval_graph, t_eval_nodes
    from node_eval import workflow_to_graph_list

    return workflow_to_graph_list, t_eval_nodes, t_eval_graph


def eval_workflow(
    gold_path: str,
    pred_path: str,
    eval_model: str,
    eval_type: str,
    eval_output: str,
    *,
    worfbench_repo: Path,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    workflow_to_graph_list, t_eval_nodes, t_eval_graph = _load_worfbench_eval(worfbench_repo)
    sentence_model = SentenceTransformer(eval_model)

    eval_output_dir = os.path.dirname(eval_output)
    if eval_output_dir:
        os.makedirs(eval_output_dir, exist_ok=True)

    with open(gold_path, encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(pred_path, encoding="utf-8") as f:
        pred_data = json.load(f)

    gold_plan = [d["conversations"][-1]["content"] for d in gold_data]
    pred_plan = [d["workflow"] for d in pred_data]
    if len(gold_plan) != len(pred_plan):
        raise AssertionError(
            "The number of gold plan and pred plan should be the same "
            f"({len(gold_plan)} vs {len(pred_plan)})"
        )

    all_precision = 0.0
    all_recall = 0.0
    all_f1_score = 0.0
    skipped = 0

    with tqdm(total=len(gold_plan)) as pbar:
        for i in range(len(gold_plan)):
            gold_graph_workflow = workflow_to_graph_list(gold_plan[i])
            pred_graph_workflow = workflow_to_graph_list(pred_plan[i])

            if (
                pred_graph_workflow == []
                or not _real_nodes(pred_graph_workflow)
                or not _real_nodes(gold_graph_workflow)
            ):
                skipped += 1
                pbar.update(1)
                continue

            if eval_type == "node":
                eval_result = t_eval_nodes(
                    pred_graph_workflow, gold_graph_workflow, sentence_model
                )
            elif eval_type == "graph":
                eval_result = t_eval_graph(
                    pred_graph_workflow, gold_graph_workflow, sentence_model
                )
            else:
                raise ValueError(f"eval_type inválido: {eval_type}")

            all_precision += eval_result["precision"]
            all_recall += eval_result["recall"]
            all_f1_score += eval_result["f1_score"]
            pbar.update(1)

    n_total = len(gold_plan)
    result = {
        "precision": all_precision / n_total,
        "recall": all_recall / n_total,
        "f1_score": all_f1_score / n_total,
        "n_total": n_total,
        "n_evaluated": n_total - skipped,
        "n_skipped_unparseable": skipped,
    }
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Average Precision:{result['precision']}")
    print(f"Average Recall:{result['recall']}")
    print(f"Average F1_score:{result['f1_score']}")
    if skipped:
        print(
            f"Skipped unparseable predictions: {skipped}/{n_total} "
            "(counted as 0 in averages)"
        )
    print("=========================================")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--eval_model", default="all-mpnet-base-v2")
    parser.add_argument("--eval_type", choices=["node", "graph"], default="node")
    parser.add_argument("--eval_output", required=True)
    parser.add_argument("--task_type", default="wikihow")
    parser.add_argument(
        "--worfbench-repo",
        type=Path,
        default=Path("external/WorFBench"),
        help="Raiz do clone WorFBench (para importar evaluator/)",
    )
    args = parser.parse_args(argv)
    print(args.task_type)
    eval_workflow(
        args.gold_path,
        args.pred_path,
        args.eval_model,
        args.eval_type,
        args.eval_output,
        worfbench_repo=args.worfbench_repo,
    )


if __name__ == "__main__":
    main()
