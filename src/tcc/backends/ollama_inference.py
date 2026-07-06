"""Inferência via API HTTP do Ollama (/api/chat)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 600.0
    temperature: float = 0.0


def ollama_config_from_cfg(cfg: dict[str, Any]) -> OllamaConfig:
    block = cfg.get("inference", {}).get("ollama", {})
    return OllamaConfig(
        base_url=block.get("base_url", "http://127.0.0.1:11434"),
        timeout_s=float(block.get("timeout_s", 600)),
        temperature=float(block.get("temperature", 0.0)),
    )


def resolve_ollama_model_name(cfg: dict[str, Any], model_id: str, finetuned: bool) -> str:
    """
    Nome do modelo no Ollama.

    Ordem: config inference.ollama.models.<id>.{base|sft} → padrão
    (model_id ou model_id-sft).
    """
    ollama = cfg.get("inference", {}).get("ollama", {})
    per_model = (ollama.get("models") or {}).get(model_id, {})
    if finetuned:
        if name := per_model.get("sft"):
            return name
        suffix = ollama.get("sft_suffix", "-sft")
        return f"{model_id}{suffix}"
    if name := per_model.get("base"):
        return name
    return model_id


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    ollama: OllamaConfig,
) -> str:
    """Chama POST /api/chat (stream=false) e retorna o conteúdo do assistant."""
    url = f"{ollama.base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": ollama.temperature},
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=ollama.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Ollama inacessível em {ollama.base_url}. "
            "Verifique se o serviço está rodando (ollama serve)."
        ) from exc

    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError(f"Resposta vazia do Ollama para o modelo '{model}': {data!r}")
    return content.strip()


def generate_workflow(
    messages: list[dict[str, str]],
    *,
    model: str,
    ollama: OllamaConfig,
) -> str:
    return chat_completion(messages, model=model, ollama=ollama)


def make_generate_fn(cfg: dict[str, Any]):
    """Factory para run_inference: (messages, model_id, finetuned) -> workflow str."""
    ollama = ollama_config_from_cfg(cfg)

    def generate(messages: list[dict[str, str]], model_id: str, finetuned: bool) -> str:
        name = resolve_ollama_model_name(cfg, model_id, finetuned)
        return generate_workflow(messages, model=name, ollama=ollama)

    return generate
