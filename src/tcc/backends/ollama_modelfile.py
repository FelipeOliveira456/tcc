"""Gera Modelfile para importar pesos HF no Ollama."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tcc.config import resolve_path
from tcc.models_registry import get_sft_template
from tcc.paths import checkpoint_dir, model_dir


def _has_lora_adapter(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() or (
        path / "adapter_model.safetensors"
    ).is_file()


def _qwen35_prefers_adapter_import(cfg: dict[str, Any], model_id: str) -> bool:
    """Merge Unsloth pode omitir metadados vision → Ollama quebra (image_mean)."""
    if model_id.startswith("qwen35"):
        return True
    try:
        return get_sft_template(cfg, model_id).startswith("qwen3_5")
    except KeyError:
        return False


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


def resolve_finetuned_ollama_sources(
    cfg: dict[str, Any],
    model_id: str,
    *,
    adapter_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """FROM + ADAPTER opcional para SFT no Ollama.

    Qwen3.5: FROM base HF + ADAPTER LoRA (merge quebra vision no Ollama).
    Outros: FROM merged/ckpt quando não há adapter explícito.
    """
    base = model_dir(cfg, model_id)
    if adapter_dir is not None:
        return base, adapter_dir

    ckpt = checkpoint_dir(cfg, model_id)
    if _qwen35_prefers_adapter_import(cfg, model_id) and _has_lora_adapter(ckpt):
        if not base.is_dir() or not any(base.iterdir()):
            raise FileNotFoundError(
                f"Base HF necessária para ADAPTER em {base}. "
                f"Rode download_model.py --model {model_id}."
            )
        return base, ckpt

    return resolve_weights_dir(cfg, model_id, finetuned=True), None


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
    if finetuned:
        weights, adapter = resolve_finetuned_ollama_sources(
            cfg, model_id, adapter_dir=adapter_dir
        )
    else:
        weights = resolve_weights_dir(cfg, model_id, finetuned=False)
        adapter = adapter_dir
    lines = [f"# TCC — {model_id}" + (" (SFT)" if finetuned else " (base)")]
    lines.append(f"FROM {weights}")
    if adapter is not None:
        lines.append(f"ADAPTER {adapter}")
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
