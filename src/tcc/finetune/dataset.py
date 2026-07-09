"""Conversão WorFBench → formato ShareGPT para LLaMA-Factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.paths import train_json

SHAREGPT_DATASET_NAME = "worfbench_sharegpt"


def _normalize_role(role: str) -> str:
    if role in ("human",):
        return "user"
    if role in ("gpt",):
        return "assistant"
    return role


def _messages_from_item(item: dict[str, Any]) -> list[dict[str, str]]:
    conv = item.get("messages") or item.get("conversations") or []
    messages: list[dict[str, str]] = []
    for turn in conv:
        role = _normalize_role(turn.get("role", "user"))
        messages.append({"role": role, "content": turn.get("content", "")})
    return messages


def prepare_sharegpt_dataset(cfg: dict[str, Any]) -> Path:
    """JSON ShareGPT (7 turnos) para LLaMA-Factory + mask_history."""
    out = resolve_path(cfg, "data_dir") / "sft" / f"{SHAREGPT_DATASET_NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with train_json(cfg).open(encoding="utf-8") as f:
        data = json.load(f)
    records = [{"messages": _messages_from_item(item)} for item in data]
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return out


def write_dataset_info(cfg: dict[str, Any]) -> Path:
    """Registra dataset em dataset_info.json (exigido pelo LLaMA-Factory).

    O JSON usa formato OpenAI (role/content). Sem ``tags``, o LLaMA-Factory
    assume ShareGPT clássico (from/value) e falha com KeyError: 'from'.
    """
    sft_dir = resolve_path(cfg, "data_dir") / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        SHAREGPT_DATASET_NAME: {
            "file_name": f"{SHAREGPT_DATASET_NAME}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    path = sft_dir / "dataset_info.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
