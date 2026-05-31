"""Stub — inferência via Ollama (não executa).

Uso previsto: cenários I0 e SFT puro (system + user → workflow).
Não substitui WorFEval; grava JSON no formato esperado por node_eval --task eval_workflow.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen35-4b"


def build_messages_i0_or_sft(system: str, user: str) -> list[dict[str, str]]:
    """Prompt mínimo (3 turnos lógicos) — alinhado ao gold_traj de teste."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_workflow_stub(
    messages: list[dict[str, str]],
    cfg: OllamaConfig,
) -> str:
    """
    Implementação futura:
      - requests POST /api/chat ou langchain_community.chat_models.ChatOllama
      - retornar content do assistant (workflow em texto)
    """
    raise NotImplementedError(
        "Integrar ChatOllama ou ollama.Client; ver docs/stack.md"
    )


def run_batch_inference_stub(
    gold_path: str,
    pred_path: str,
    cfg: OllamaConfig,
    scenario: str,
) -> None:
    """Itera gold_traj JSON, gera predições, salva lista {query, workflow}."""
    raise NotImplementedError
