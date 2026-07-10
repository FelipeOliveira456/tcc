"""Passos do setup global (dados, RAG, WorFBench, llama.cpp)."""

from __future__ import annotations

SETUP_STEPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("1/4 — baixar treino e teste (HF)", "download_data.py", ()),
    ("2/4 — índice vetorial RAG", "build_vector_db.py", ()),
    ("3/4 — clone WorFBench + deps de eval", "worfeval.py", ("--setup",)),
    ("4/4 — clone llama.cpp (GGUF / Ollama)", "setup_llama_cpp.py", ()),
)


def setup_steps(*, force_rag: bool = False) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Lista de (rótulo, script, args) para setup_project."""
    steps = list(SETUP_STEPS)
    if force_rag:
        label, script, script_args = steps[1]
        steps[1] = (label, script, script_args + ("--force",))
    return tuple(steps)
