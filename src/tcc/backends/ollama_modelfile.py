"""Gera Modelfile para importar pesos HF no Ollama."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.paths import checkpoint_dir, model_dir


def resolve_weights_dir(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
) -> Path:
    if finetuned:
        merged = checkpoint_dir(cfg, model_id) / "merged"
        if merged.is_dir() and any(merged.iterdir()):
            return merged
        ckpt = checkpoint_dir(cfg, model_id)
        if ckpt.is_dir() and any(ckpt.iterdir()):
            return ckpt
        raise FileNotFoundError(
            f"Checkpoint SFT não encontrado em {ckpt}. "
            "Rode finetune.py --export-merged e depois ollama_import.py --finetuned."
        )
    weights = model_dir(cfg, model_id)
    if not weights.is_dir() or not any(weights.iterdir()):
        raise FileNotFoundError(
            f"Pesos HF não encontrados em {weights}. Rode download_model.py --model {model_id}."
        )
    return weights


def build_modelfile(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    adapter_dir: Path | None = None,
    temperature: float | None = None,
) -> str:
    """Conteúdo do Modelfile (FROM pesos locais; ADAPTER opcional para LoRA)."""
    ollama = cfg.get("inference", {}).get("ollama", {})
    temp = temperature if temperature is not None else float(ollama.get("temperature", 0.0))
    weights = resolve_weights_dir(cfg, model_id, finetuned=finetuned and adapter_dir is None)
    lines = [f"# TCC — {model_id}" + (" (SFT)" if finetuned else " (base)")]
    lines.append(f"FROM {weights}")
    if adapter_dir is not None:
        lines.append(f"ADAPTER {adapter_dir}")
    lines.append(f"PARAMETER temperature {temp}")
    return "\n".join(lines) + "\n"


def write_modelfile(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    adapter_dir: Path | None = None,
) -> Path:
    out_dir = resolve_path(cfg, "models_dir") / "ollama"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-sft" if finetuned else ""
    path = out_dir / f"Modelfile.{model_id}{suffix}"
    path.write_text(
        build_modelfile(cfg, model_id, finetuned=finetuned, adapter_dir=adapter_dir),
        encoding="utf-8",
    )
    return path


def resolve_ollama_create_name(cfg: dict[str, Any], model_id: str, *, finetuned: bool) -> str:
    ollama = cfg.get("inference", {}).get("ollama", {})
    per_model = (ollama.get("models") or {}).get(model_id, {})
    if finetuned:
        return per_model.get("sft") or f"{model_id}{ollama.get('sft_suffix', '-sft')}"
    return per_model.get("base") or model_id


def ollama_create_argv(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    modelfile: Path,
    quantize: str | None = None,
) -> list[str]:
    name = resolve_ollama_create_name(cfg, model_id, finetuned=finetuned)
    cmd = ["ollama", "create", name, "-f", str(modelfile)]
    if quantize:
        cmd.extend(["--quantize", quantize])
    return cmd


def ollama_create_command(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    modelfile: Path,
    quantize: str | None = None,
) -> str:
    argv = ollama_create_argv(
        cfg, model_id, finetuned=finetuned, modelfile=modelfile, quantize=quantize
    )
    return " ".join(argv)
