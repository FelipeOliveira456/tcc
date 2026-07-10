"""Conversão WorFBench → JSON de mensagens para SFT (Unsloth)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.paths import model_dir, train_json

SHAREGPT_DATASET_NAME = "worfbench_sharegpt"
# Fallback quando não há tokenizer: ~4 chars/token (latim).
_CHARS_PER_TOKEN = 4


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


def _messages_plain_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def estimate_token_count(
    messages: list[dict[str, str]],
    *,
    tokenizer: Any | None = None,
) -> int:
    """Conta tokens da conversa (tokenizer se houver; senão chars/4)."""
    text = _messages_plain_text(messages)
    if tokenizer is not None:
        return len(tokenizer.encode(text).ids)
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _try_load_tokenizer(cfg: dict[str, Any], model_id: str | None) -> Any | None:
    """Carrega tokenizer.json local ou do HF; None se indisponível."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None

    candidates: list[Path] = []
    if model_id:
        local = model_dir(cfg, model_id)
        candidates.append(local / "tokenizer.json")
    models_root = resolve_path(cfg, "models_dir")
    if models_root.is_dir():
        candidates.extend(sorted(models_root.glob("*/tokenizer.json")))

    for path in candidates:
        if path.is_file():
            try:
                return Tokenizer.from_file(str(path))
            except Exception:
                continue

    hf_id = None
    if model_id:
        for block in cfg.get("models", {}).get("slm", []):
            if block.get("id") == model_id:
                hf_id = block.get("hf_id")
                break
    if not hf_id:
        return None
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(hf_id, "tokenizer.json")
        return Tokenizer.from_file(path)
    except Exception:
        return None


def fit_messages_to_max_tokens(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> tuple[list[dict[str, str]] | None, str]:
    """Encaixa a conversa em ``max_tokens`` removendo rodadas few-shot antigas.

    Estrutura esperada (WorFBench train):
      system + (user, assistant)* + (user, assistant)
    onde o último par user/assistant é a tarefa real (alvo do SFT).

    Estratégia:
      1. se cabe → full
      2. remove 1 demo (user+assistant) mais antiga, repete
      3. se ainda não cabe só com system + último par → drop (None)
    """
    if estimate_token_count(messages, tokenizer=tokenizer) <= max_tokens:
        return messages, "full"

    has_system = bool(messages) and messages[0]["role"] == "system"
    system = messages[0] if has_system else None
    body = messages[1:] if has_system else list(messages)

    # body = pares user/assistant; último par = tarefa real
    if len(body) < 2 or len(body) % 2 != 0:
        return None, "malformed"

    final_pair = body[-2:]
    demos = body[:-2]
    n_demos = len(demos) // 2

    for drop in range(1, n_demos + 1):
        remaining = demos[2 * drop :]
        cand = ([system] if system else []) + remaining + final_pair
        if estimate_token_count(cand, tokenizer=tokenizer) <= max_tokens:
            return cand, f"drop_{drop}_demo"

    cand = ([system] if system else []) + final_pair
    if estimate_token_count(cand, tokenizer=tokenizer) <= max_tokens:
        return cand, "demos_all_dropped"

    return None, "dropped"


def filter_messages_by_max_tokens(
    records: list[dict[str, Any]],
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Ajusta/remove exemplos longos: corta demos few-shot antes de dropar."""
    kept: list[dict[str, Any]] = []
    outcomes: dict[str, int] = {}
    for rec in records:
        fitted, tag = fit_messages_to_max_tokens(
            rec["messages"], max_tokens=max_tokens, tokenizer=tokenizer
        )
        outcomes[tag] = outcomes.get(tag, 0) + 1
        if fitted is None:
            continue
        kept.append({"messages": fitted})

    dropped = outcomes.get("dropped", 0) + outcomes.get("malformed", 0)
    truncated = sum(
        v
        for k, v in outcomes.items()
        if k.startswith("drop_") or k == "demos_all_dropped"
    )
    stats: dict[str, Any] = {
        "total": len(records),
        "kept": len(kept),
        "dropped": dropped,
        "truncated": truncated,
        "full": outcomes.get("full", 0),
        "max_tokens": max_tokens,
        "outcomes": outcomes,
    }
    return kept, stats


def prepare_sharegpt_dataset(
    cfg: dict[str, Any],
    *,
    model_id: str | None = None,
) -> Path:
    """JSON de conversas para Unsloth (loss no último assistant no treino).

    Exemplos > ``sft.max_example_tokens`` (default = cutoff_len): remove rodadas
    few-shot antigas (user+assistant) até caber; se ainda não couber só com
    system + tarefa final, o exemplo é descartado. Gold em ``data/train/`` e
    índice RAG não são alterados.
    """
    sft = cfg.get("sft", {})
    max_tokens = int(sft.get("max_example_tokens", sft.get("cutoff_len", 2048)))
    out = resolve_path(cfg, "data_dir") / "sft" / f"{SHAREGPT_DATASET_NAME}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with train_json(cfg).open(encoding="utf-8") as f:
        data = json.load(f)
    records = [{"messages": _messages_from_item(item)} for item in data]

    tokenizer = _try_load_tokenizer(cfg, model_id)
    records, stats = filter_messages_by_max_tokens(
        records, max_tokens=max_tokens, tokenizer=tokenizer
    )
    stats["tokenizer"] = "tokenizers" if tokenizer is not None else "chars/4"
    print(
        f"[sft-dataset] max_tokens={max_tokens} "
        f"kept={stats['kept']}/{stats['total']} "
        f"truncated={stats['truncated']} dropped={stats['dropped']} "
        f"({100 * stats['dropped'] / max(stats['total'], 1):.1f}% drop) "
        f"via {stats['tokenizer']}",
        flush=True,
    )
    meta_path = out.with_suffix(".filter.json")
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return out
