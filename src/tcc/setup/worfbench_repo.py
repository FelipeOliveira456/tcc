"""Clone e preparação do repositório WorFBench (contém o protocolo WorFEval)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tcc.config import resolve_path


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
    req = dest / "requirements.txt"
    if req.exists():
        subprocess.run(
            ["pip", "install", "-r", str(req)],
            check=True,
            cwd=str(dest),
        )
    return dest


def worfeval_node_eval_path(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "worfbench_repo") / "node_eval.py"
