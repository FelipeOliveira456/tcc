"""SFT via Unsloth (bf16 LoRA) — backend principal do TCC.

Alinhado ao WorFBench: loss só na última resposta do assistant (demos = contexto).
Qwen3.5: Unsloth recomenda bf16 LoRA (não QLoRA 4-bit).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tcc.paths import checkpoint_dir, model_dir


def text_tokenizer(tokenizer_or_processor: Any) -> Any:
    """Tokenizer de texto puro (Qwen3.5 retorna Processor multimodal)."""
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and inner is not tokenizer_or_processor:
        return inner
    return tokenizer_or_processor


def encode_text_ids(
    tokenizer_or_processor: Any, text: str, *, max_length: int
) -> list[int]:
    """Tokeniza string sem passar pelo image_processor do Qwen3.5-VL."""
    tok = text_tokenizer(tokenizer_or_processor)
    if hasattr(tok, "encode"):
        return tok.encode(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
    out = tok(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return out["input_ids"]


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_masked_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq_length: int,
) -> dict[str, list[int]] | None:
    """Tokeniza a conversa e mascara tudo exceto o último turno assistant."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None

    full_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    prefix_text = _apply_chat_template(
        tokenizer, messages[:-1], add_generation_prompt=True
    )

    full_ids = encode_text_ids(tokenizer, full_text, max_length=max_seq_length)
    prefix_ids = encode_text_ids(tokenizer, prefix_text, max_length=max_seq_length)

    if not full_ids:
        return None

    n_prefix = len(prefix_ids)
    if n_prefix >= len(full_ids) or full_ids[:n_prefix] != prefix_ids[:n_prefix]:
        n_common = 0
        for a, b in zip(full_ids, prefix_ids):
            if a != b:
                break
            n_common += 1
        n_prefix = n_common
        if n_prefix >= len(full_ids):
            return None

    labels = list(full_ids)
    for i in range(n_prefix):
        labels[i] = -100
    if all(x == -100 for x in labels):
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def load_sharegpt_messages(path: Path) -> list[list[dict[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[list[dict[str, str]]] = []
    for rec in data:
        msgs = rec.get("messages") or []
        if msgs:
            out.append(
                [{"role": m["role"], "content": m.get("content", "")} for m in msgs]
            )
    return out


def _ensure_cuda_alloc_conf() -> None:
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{alloc},expandable_segments:True" if alloc else "expandable_segments:True"
        )


def run_unsloth_train(
    cfg: dict[str, Any],
    model_id: str,
    *,
    dataset_path: Path,
    export_merged: bool = False,
) -> Path:
    """Treina LoRA com Unsloth e opcionalmente merge 16-bit em checkpoints/<id>/merged."""
    _ensure_cuda_alloc_conf()

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth não instalado. No LabRI:\n"
            "  pip install --upgrade unsloth unsloth_zoo\n"
            "Requer transformers v5+ e GPU CUDA."
        ) from exc

    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    sft = cfg.get("sft", {})
    max_seq = int(sft.get("cutoff_len", 2048))
    seed = int(sft.get("seed", 42))
    ckpt = checkpoint_dir(cfg, model_id)
    ckpt.mkdir(parents=True, exist_ok=True)
    merged_dir = ckpt / "merged"

    model_name = str(model_dir(cfg, model_id))
    if not (Path(model_name) / "config.json").is_file():
        from tcc.models_registry import get_model_spec

        model_name = get_model_spec(cfg, model_id)["hf_id"]

    # Qwen3.5: Unsloth recomenda bf16 LoRA (não 4-bit).
    load_in_4bit = bool(sft.get("load_in_4bit", False))
    load_in_16bit = bool(sft.get("load_in_16bit", not load_in_4bit))

    model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        full_finetuning=False,
        dtype=None,
    )
    text_tok = text_tokenizer(tokenizer_or_processor)

    peft_kwargs: dict[str, Any] = {
        "r": int(sft.get("lora_rank", 16)),
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_alpha": int(sft.get("lora_alpha", 32)),
        "lora_dropout": float(sft.get("lora_dropout", 0.0)),
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": seed,
        "max_seq_length": max_seq,
    }
    # Qwen3.5 é multimodal: WorFBench é só texto — não treinar vision tower.
    if "qwen3" in model_name.lower():
        peft_kwargs.update(
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )

    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    conversations = load_sharegpt_messages(dataset_path)
    rows: list[dict[str, list[int]]] = []
    skipped = 0
    for msgs in conversations:
        ex = build_masked_example(tokenizer_or_processor, msgs, max_seq_length=max_seq)
        if ex is None:
            skipped += 1
            continue
        rows.append(ex)

    if not rows:
        raise RuntimeError(
            f"Nenhum exemplo útil após máscara/truncagem (skipped={skipped})."
        )

    print(
        f"[unsloth-sft] examples={len(rows)} skipped={skipped} "
        f"max_seq={max_seq} 4bit={load_in_4bit} 16bit={load_in_16bit}",
        flush=True,
    )
    train_ds = Dataset.from_list(rows)

    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    epochs = float(sft.get("num_train_epochs", 1))
    # Dataset já tokenizado + labels mascarados → Trainer HF (evita re-prep do TRL).
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(ckpt),
            per_device_train_batch_size=int(sft.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(sft.get("gradient_accumulation_steps", 8)),
            num_train_epochs=epochs,
            learning_rate=float(sft.get("learning_rate", 2e-5)),
            lr_scheduler_type="cosine",
            warmup_ratio=float(sft.get("warmup_ratio", 0.1)),
            logging_steps=int(sft.get("logging_steps", 10)),
            save_steps=int(sft.get("save_steps", 500)),
            save_total_limit=2,
            optim=str(sft.get("optim", "adamw_8bit")),
            bf16=True,
            seed=seed,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
        ),
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=text_tok,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        ),
    )

    trainer.train()
    model.save_pretrained(str(ckpt))
    text_tok.save_pretrained(str(ckpt))

    if export_merged:
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            str(merged_dir),
            text_tok,
            save_method="merged_16bit",
        )
        print(f"[unsloth-sft] merged → {merged_dir}", flush=True)

    return ckpt
