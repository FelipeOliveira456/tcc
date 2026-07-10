"""Montagem de prompts e gravação de predições (formato WorFBench)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from tcc.download.worfbench_data import list_test_tasks
from tcc.paths import inference_run_meta_path, prediction_path, test_gold
from tcc.rag.vector_store import VectorRetriever
from tcc.run_stamp import run_stamp, utc_now_iso


def _final_user_message(conversations: list[dict]) -> str:
    users = [c["content"] for c in conversations if c.get("role") in ("user", "human")]
    return users[-1] if users else ""


def _system_message(conversations: list[dict]) -> str:
    for c in conversations:
        if c.get("role") == "system":
            return c.get("content", "")
    return ""


def build_prompt_messages(
    gold_item: dict[str, Any],
    *,
    use_rag: bool,
    cfg: dict[str, Any],
    retriever: VectorRetriever | None = None,
) -> list[dict[str, str]]:
    """I0/SFT: system+user. RAG/SFT+RAG: system enriquecido com 2 exemplos recuperados."""
    conv = gold_item.get("conversations") or gold_item.get("messages") or []
    system = _system_message(conv)
    user = _final_user_message(conv)

    if use_rag:
        if retriever is None:
            raise ValueError("use_rag=True exige retriever (carregue VectorRetriever.from_config uma vez)")
        retrieved = retriever.retrieve(user)
        blocks = []
        for i, ex in enumerate(retrieved, 1):
            blocks.append(
                f"### Example {i}\nQuestion:\n{ex['user']}\n\nWorkflow:\n{ex['workflow']}\n"
            )
        system = (
            f"{system}\n\n"
            "Retrieved training examples (use as structural reference):\n\n"
            + "\n".join(blocks)
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def run_inference(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    use_rag: bool,
    generate_fn: Callable[[list[dict[str, str]], str, bool], str],
    tasks: list[str] | None = None,
    limit: int | None = None,
    progress_desc_prefix: str = "",
    progress_position: int | None = None,
) -> Path:
    """
    generate_fn(messages, model_id, finetuned) -> texto do workflow predito.
    Grava um JSON por tarefa no formato esperado pelo WorFEval.
    """
    task_list = tasks or list_test_tasks(cfg)
    stamp = run_stamp()
    started_at = utc_now_iso()
    outputs: dict[str, str] = {}
    retriever = VectorRetriever.from_config(cfg) if use_rag else None

    for task in task_list:
        gold_path = test_gold(cfg, task)
        with gold_path.open(encoding="utf-8") as f:
            gold_data = json.load(f)

        preds = []
        subset = gold_data[:limit] if limit else gold_data
        desc = f"{progress_desc_prefix}{task} ({model_id})".strip()
        tqdm_kwargs: dict[str, Any] = {"desc": desc, "unit": "ex", "dynamic_ncols": True}
        if progress_position is not None:
            tqdm_kwargs["position"] = progress_position
            tqdm_kwargs["leave"] = True
        for item in tqdm(subset, **tqdm_kwargs):
            messages = build_prompt_messages(
                item, use_rag=use_rag, cfg=cfg, retriever=retriever
            )
            workflow = generate_fn(messages, model_id, finetuned)
            preds.append({"query": item, "workflow": workflow})

        out = prediction_path(
            cfg, model_id, finetuned=finetuned, rag=use_rag, task=task, stamp=stamp
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        outputs[task] = str(out)

    finished_at = utc_now_iso()
    meta_path = inference_run_meta_path(cfg, model_id, stamp)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stamp": stamp,
                "started_at": started_at,
                "finished_at": finished_at,
                "model_id": model_id,
                "finetuned": finetuned,
                "rag": use_rag,
                "limit": limit,
                "tasks": task_list,
                "predictions": outputs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return prediction_path(
        cfg, model_id, finetuned=finetuned, rag=use_rag, task=task_list[0], stamp=stamp
    )
