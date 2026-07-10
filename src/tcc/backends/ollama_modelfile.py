"""Gera Modelfile para importar pesos HF no Ollama."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tcc.backends.gguf_convert import (
    convert_hf_dir_to_gguf,
    gguf_outfile_for,
    needs_gguf_conversion,
)
from tcc.config import resolve_path
from tcc.models_registry import get_sft_template
from tcc.paths import checkpoint_dir, model_dir

# Sidecars multimodais que o merge Unsloth pode omitir (Ollama → image_mean).
_QWEN35_OLLAMA_SIDECARS = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _is_qwen35(cfg: dict[str, Any], model_id: str) -> bool:
    if model_id.startswith("qwen35"):
        return True
    try:
        return get_sft_template(cfg, model_id).startswith("qwen3_5")
    except KeyError:
        return False


def _has_safetensors_weights(path: Path) -> bool:
    return any(path.glob("*.safetensors"))


def _patch_config_vision(base_config: Path, merged_config: Path) -> None:
    if not base_config.is_file() or not merged_config.is_file():
        return
    base = json.loads(base_config.read_text(encoding="utf-8"))
    merged = json.loads(merged_config.read_text(encoding="utf-8"))
    for key in ("vision_config", "image_token_id", "video_token_id"):
        if key in base and key not in merged:
            merged[key] = base[key]
    merged_config.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def build_qwen35_ollama_sft_bundle(cfg: dict[str, Any], model_id: str) -> Path:
    """Merge SFT + sidecars vision da base HF (Ollama não aceita ADAPTER no Qwen3.5)."""
    base = model_dir(cfg, model_id)
    merged = checkpoint_dir(cfg, model_id) / "merged"
    if not merged.is_dir() or not _has_safetensors_weights(merged):
        raise FileNotFoundError(
            f"Merge SFT não encontrado em {merged}. "
            f"Rode finetune.py --model {model_id} --export-merged."
        )
    if not base.is_dir() or not _has_safetensors_weights(base):
        raise FileNotFoundError(
            f"Base HF necessária em {base}. Rode download_model.py --model {model_id}."
        )

    out = checkpoint_dir(cfg, model_id) / "ollama_sft"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(merged, out)
    for name in _QWEN35_OLLAMA_SIDECARS:
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)
    _patch_config_vision(base / "config.json", out / "config.json")
    return out


def resolve_weights_dir(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
) -> Path:
    if finetuned:
        merged = checkpoint_dir(cfg, model_id) / "merged"
        if merged.is_dir() and _has_safetensors_weights(merged):
            return merged
        ckpt = checkpoint_dir(cfg, model_id)
        if ckpt.is_dir() and _has_safetensors_weights(ckpt):
            return ckpt
        raise FileNotFoundError(
            f"Checkpoint SFT não encontrado em {ckpt}. "
            "Rode finetune.py --export-merged e depois ollama_import.py --finetuned."
        )
    weights = model_dir(cfg, model_id)
    if not weights.is_dir() or not _has_safetensors_weights(weights):
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
    if adapter_dir is not None:
        if _is_qwen35(cfg, model_id):
            raise ValueError(
                "ADAPTER HF/PEFT não é suportado no Ollama para Qwen3.5. "
                "Use finetune.py --export-merged (bundle ollama_sft é gerado automaticamente)."
            )
        base = model_dir(cfg, model_id)
        if not base.is_dir() or not _has_safetensors_weights(base):
            raise FileNotFoundError(
                f"Base HF necessária em {base}. Rode download_model.py --model {model_id}."
            )
        return base, adapter_dir
    if _is_qwen35(cfg, model_id):
        return build_qwen35_ollama_sft_bundle(cfg, model_id), None
    return resolve_weights_dir(cfg, model_id, finetuned=True), None


def resolve_ollama_from_path(
    cfg: dict[str, Any],
    model_id: str,
    weights: Path,
    *,
    finetuned: bool,
    adapter: Path | None = None,
    outtype: str | None = None,
    force_gguf: bool = False,
) -> tuple[Path, Path | None, bool]:
    """
    Resolve FROM do Modelfile: diretório safetensors ou arquivo .gguf.

    Retorna (from_path, adapter_or_none, used_gguf).
    Se a arquitetura exigir GGUF (ex. Granite), converte via llama.cpp.
    ADAPTER + GGUF não é suportado — use merge (--export-merged).
    """
    need = force_gguf or needs_gguf_conversion(cfg, model_id, weights)
    if not need:
        return weights, adapter, False
    if adapter is not None:
        raise ValueError(
            f"Modelo {model_id}: Ollama via GGUF não combina com ADAPTER. "
            "Rode finetune.py --export-merged e importe com --finetuned (sem --adapter)."
        )
    outfile = gguf_outfile_for(cfg, model_id, finetuned=finetuned, outtype=outtype)
    gguf = convert_hf_dir_to_gguf(cfg, weights, outfile, outtype=outtype)
    return gguf, None, True


def build_modelfile(
    cfg: dict[str, Any],
    model_id: str,
    *,
    finetuned: bool,
    adapter_dir: Path | None = None,
    temperature: float | None = None,
    outtype: str | None = None,
    force_gguf: bool = False,
) -> str:
    """Conteúdo do Modelfile (FROM safetensors ou GGUF quando necessário)."""
    ollama = cfg.get("inference", {}).get("ollama", {})
    temp = temperature if temperature is not None else float(ollama.get("temperature", 0.0))
    adapter: Path | None = None
    if finetuned:
        weights, adapter = resolve_finetuned_ollama_sources(
            cfg, model_id, adapter_dir=adapter_dir
        )
    else:
        weights = resolve_weights_dir(cfg, model_id, finetuned=False)
    from_path, adapter, used_gguf = resolve_ollama_from_path(
        cfg,
        model_id,
        weights,
        finetuned=finetuned,
        adapter=adapter,
        outtype=outtype,
        force_gguf=force_gguf,
    )
    tag = " (SFT)" if finetuned else " (base)"
    if used_gguf:
        tag += " [GGUF]"
    lines = [f"# TCC — {model_id}{tag}"]
    lines.append(f"FROM {from_path}")
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
    outtype: str | None = None,
    force_gguf: bool = False,
) -> Path:
    out_dir = resolve_path(cfg, "models_dir") / "ollama"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-sft" if finetuned else ""
    path = out_dir / f"Modelfile.{model_id}{suffix}"
    path.write_text(
        build_modelfile(
            cfg,
            model_id,
            finetuned=finetuned,
            adapter_dir=adapter_dir,
            outtype=outtype,
            force_gguf=force_gguf,
        ),
        encoding="utf-8",
    )
    return path


def resolve_ollama_create_name(cfg: dict[str, Any], model_id: str, *, finetuned: bool) -> str:
    ollama = cfg.get("inference", {}).get("ollama", {})
    per_model = (ollama.get("models") or {}).get(model_id, {})
    if finetuned:
        return per_model.get("sft") or f"{model_id}{ollama.get('sft_suffix', '-sft')}"
    return per_model.get("base") or model_id


def modelfile_uses_gguf(modelfile: Path) -> bool:
    """True se o Modelfile aponta FROM para um .gguf (já quantizado na conversão)."""
    for line in modelfile.read_text(encoding="utf-8").splitlines():
        if line.upper().startswith("FROM ") and line.strip().lower().endswith(".gguf"):
            return True
    return False


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
    # GGUF já sai com outtype; --quantize no create seria redundante/conflitante.
    if quantize and not modelfile_uses_gguf(modelfile):
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
