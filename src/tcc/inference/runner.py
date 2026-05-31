"""Montagem de prompts e gravação de predições (formato WorFBench)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from tcc.download.worfbench_data import list_test_tasks
from tcc.paths import prediction_path, test_gold
from tcc.rag.vector_store import retrieve


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
) -> list[dict[str, str]]:
    """I0/SFT: system+user. RAG/SFT+RAG: system enriquecido com 2 exemplos recuperados."""
    conv = gold_item.get("conversations") or gold_item.get("messages") or []
    system = _system_message(conv)
    user = _final_user_message(conv)

    if use_rag:
        retrieved = retrieve(cfg, user)
        blocks = []
        for i, ex in enumerate(retrieved, 1):
            blocks.append(
                f"### Exemplo {i}\nPergunta:\n{ex['user']}\n\nWorkflow:\n{ex['workflow']}\n"
            )
        system = (
            f"{system}\n\n"
            "Exemplos recuperados do treino (use como referência estrutural):\n\n"
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
) -> Path:
    """
    generate_fn(messages, model_id, finetuned) -> texto do workflow predito.
    Grava um JSON por tarefa no formato esperado pelo WorFEval.
    """
    task_list = tasks or list_test_tasks(cfg)

    for task in task_list:
        gold_path = test_gold(cfg, task)
        with gold_path.open(encoding="utf-8") as f:
            gold_data = json.load(f)

        preds = []
        subset = gold_data[:limit] if limit else gold_data
        for item in subset:
            messages = build_prompt_messages(item, use_rag=use_rag, cfg=cfg)
            workflow = generate_fn(messages, model_id, finetuned)
            preds.append({"query": item, "workflow": workflow})

        out = prediction_path(cfg, model_id, finetuned=finetuned, rag=use_rag, task=task)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)

    return prediction_path(cfg, model_id, finetuned=finetuned, rag=use_rag, task=task_list[0])
