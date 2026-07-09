"""Executa scripts do projeto em sequência (orquestração)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(
    script: str,
    *args: str,
    config: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Roda `python scripts/<script> [args]` na raiz do projeto."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script)]
    if config is not None:
        cmd.extend(["--config", str(config)])
    cmd.extend(args)
    label = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {label}")
        return
    print(f"\n>>> {label}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
