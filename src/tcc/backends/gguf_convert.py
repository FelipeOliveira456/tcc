"""Conversão HF safetensors → GGUF via llama.cpp (quando ollama create não converte)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.models_registry import get_model_spec

# Arquiteturas que o conversor safetensors do Ollama ainda não cobre bem
# (ex.: Granite — PR ollama#8319 aberto; Nemotron híbrido).
_DEFAULT_FORCE_ARCHITECTURES = (
    "GraniteForCausalLM",
    "NemotronHForCausalLM",
    "NemotronForCausalLM",
)


def gguf_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("inference", {}).get("ollama", {}).get("gguf") or {})


def llama_cpp_dir(cfg: dict[str, Any]) -> Path:
    block = gguf_cfg(cfg)
    raw = block.get("llama_cpp_dir", "external/llama.cpp")
    p = Path(raw)
    if p.is_absolute():
        return p
    return resolve_path(cfg, "project_root") / p


def ensure_llama_cpp(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """Clona llama.cpp (shallow) se necessário; retorna o diretório."""
    dest = llama_cpp_dir(cfg)
    block = gguf_cfg(cfg)
    url = block.get("llama_cpp_repo", "https://github.com/ggerganov/llama.cpp.git")
    ref = block.get("llama_cpp_ref", "master")

    if dest.exists() and (dest / "convert_hf_to_gguf.py").is_file() and not force:
        return dest

    if dest.exists() and force:
        subprocess.run(["rm", "-rf", str(dest)], check=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gguf] clonando llama.cpp → {dest}", flush=True)
    subprocess.run(
        ["git", "clone", "--branch", ref, "--depth", "1", url, str(dest)],
        check=True,
    )
    return dest


def read_hf_architectures(weights_dir: Path) -> list[str]:
    config_path = weights_dir / "config.json"
    if not config_path.is_file():
        return []
    data = json.loads(config_path.read_text(encoding="utf-8"))
    arch = data.get("architectures") or []
    if isinstance(arch, str):
        return [arch]
    return [str(a) for a in arch]


def needs_gguf_conversion(
    cfg: dict[str, Any],
    model_id: str,
    weights_dir: Path,
) -> bool:
    """True se ollama create a partir de safetensors deve ser evitado."""
    block = gguf_cfg(cfg)
    if block.get("enabled", True) is False:
        return False

    try:
        spec = get_model_spec(cfg, model_id)
    except KeyError:
        spec = {}
    if spec.get("ollama_via_gguf") is True:
        return True
    if spec.get("ollama_via_gguf") is False:
        return False

    force = list(block.get("force_architectures") or _DEFAULT_FORCE_ARCHITECTURES)
    arches = read_hf_architectures(weights_dir)
    return any(a in force for a in arches)


def convert_script(llama_cpp: Path) -> Path:
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py não encontrado em {llama_cpp}. "
            "Rode ensure_llama_cpp ou clone o llama.cpp."
        )
    return script


def convert_hf_dir_to_gguf(
    cfg: dict[str, Any],
    weights_dir: Path,
    outfile: Path,
    *,
    outtype: str | None = None,
    force: bool = False,
) -> Path:
    """
    Converte diretório HF (safetensors) → arquivo .gguf.

    Clona llama.cpp se preciso. Reusa outfile se já existir (salvo force=True).
    """
    block = gguf_cfg(cfg)
    outtype = outtype or str(block.get("outtype", "q4_K_M"))
    outfile = outfile.resolve()
    if outfile.is_file() and not force:
        print(f"[gguf] reusando {outfile}", flush=True)
        return outfile

    llama_cpp = ensure_llama_cpp(cfg)
    script = convert_script(llama_cpp)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        str(weights_dir.resolve()),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    print(f"[gguf] convertendo {weights_dir} → {outfile} (outtype={outtype})", flush=True)
    subprocess.run(cmd, check=True, cwd=str(llama_cpp))
    if not outfile.is_file():
        raise RuntimeError(f"Conversão GGUF não gerou arquivo: {outfile}")
    return outfile


def gguf_outfile_for(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    outtype: str | None = None,
) -> Path:
    block = gguf_cfg(cfg)
    outtype = outtype or str(block.get("outtype", "q4_K_M"))
    suffix = "-sft" if finetuned else ""
    name = f"{model_id}{suffix}-{outtype}.gguf"
    return resolve_path(cfg, "models_dir") / "ollama" / "gguf" / name
