"""Passos do setup global (dados, RAG, WorFBench)."""

from __future__ import annotations

SETUP_STEPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("1/3 — baixar treino e teste (HF)", "download_data.py", ()),
    ("2/3 — índice vetorial RAG", "build_vector_db.py", ()),
    ("3/3 — clone WorFBench + deps de eval", "worfeval.py", ("--setup",)),
)


def setup_steps(*, force_rag: bool = False) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Lista de (rótulo, script, args) para setup_project."""
    steps = list(SETUP_STEPS)
    if force_rag:
        label, script, script_args = steps[1]
        steps[1] = (label, script, script_args + ("--force",))
    return tuple(steps)
