"""Clone e preparação do repositório WorFBench (contém o protocolo WorFEval)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from tcc.config import resolve_path

# WorFEval (eval_workflow) usa sentence-transformers + networkx.
# O requirements.txt upstream lista módulos stdlib (collections, re…) — não instalar via pip.
WORFBENCH_EVAL_DEPS = ("networkx",)


def install_worfbench_eval_deps() -> None:
    """Deps pip para node_eval --task eval_workflow (não usa o requirements.txt do upstream)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *WORFBENCH_EVAL_DEPS],
        check=True,
    )


def clone_worfbench(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """
    Clona zjunlp/WorFBench — avaliação via node_eval.py (WorFEval).

    Nota: não existe repositório separado 'WorFEval'; o protocolo está em WorFBench/evaluator/.
    """
    dest = resolve_path(cfg, "worfbench_repo")
    wb = cfg.get("worfbench", {})
    url = wb.get("repo_url", "https://github.com/zjunlp/WorFBench.git")
    ref = wb.get("repo_ref", "main")

    if dest.exists() and any(dest.iterdir()):
        if not force:
            return dest
        subprocess.run(["rm", "-rf", str(dest)], check=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", ref, "--depth", "1", url, str(dest)],
        check=True,
    )
    return dest


def worfeval_node_eval_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "worfbench_repo") / "node_eval.py"
