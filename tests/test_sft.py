"""Testes de geração de dataset SFT (Unsloth) e Modelfile Ollama."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import importlib.util
import unittest

if importlib.util.find_spec("pytest") is None:
    raise unittest.SkipTest("pytest required — pip install -r requirements-dev.txt")

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.backends.ollama_modelfile import build_modelfile
from tcc.config import load_config
from tcc.finetune.dataset import (
    SHAREGPT_DATASET_NAME,
    prepare_sharegpt_dataset,
)
from tcc.finetune.sft import run_finetune
from tcc.finetune.unsloth_sft import build_masked_example
from tcc.models_registry import get_sft_template


class _FakeTok:
    """Tokenizer mínimo para testar máscara do último assistant."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **_
    ):
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def __call__(
        self, text, truncation=True, max_length=2048, add_special_tokens=False
    ):
        # 1 char ≈ 1 token (só para teste)
        ids = list(range(min(len(text), max_length)))
        return {"input_ids": ids}


@pytest.fixture
def cfg(tmp_path: Path):
    train = tmp_path / "data" / "train" / "worfbench_train.json"
    train.parent.mkdir(parents=True)
    train.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                        {"role": "assistant", "content": "a2"},
                        {"role": "user", "content": "u3"},
                        {"role": "assistant", "content": "gold workflow"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models" / "qwen35-0.8b"
    models.mkdir(parents=True)
    (models / "config.json").write_text("{}", encoding="utf-8")

    return {
        "_project_root": tmp_path,
        "paths": {
            "project_root": ".",
            "data_dir": "data",
            "models_dir": "models",
            "checkpoints_dir": "checkpoints",
            "outputs_dir": "outputs",
        },
        "models": {
            "slm": [
                {
                    "id": "qwen35-0.8b",
                    "hf_id": "Qwen/Qwen3.5-0.8B",
                    "sft_template": "qwen3_5_nothink",
                },
            ]
        },
        "sft": {
            "framework": "unsloth",
            "num_train_epochs": 1,
            "cutoff_len": 2048,
            "max_example_tokens": 2048,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "load_in_4bit": False,
            "load_in_16bit": True,
        },
        "inference": {"ollama": {"temperature": 0.0, "sft_suffix": "-sft"}},
    }


def test_sharegpt_dataset(cfg):
    path = prepare_sharegpt_dataset(cfg)
    assert path.name == f"{SHAREGPT_DATASET_NAME}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data[0]["messages"]) == 7
    assert data[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert "from" not in data[0]["messages"][0]


def test_masked_example_keeps_only_last_assistant():
    tok = _FakeTok()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "gold"},
    ]
    ex = build_masked_example(tok, msgs, max_seq_length=2048)
    assert ex is not None
    assert len(ex["input_ids"]) == len(ex["labels"])
    assert any(x == -100 for x in ex["labels"])
    assert any(x != -100 for x in ex["labels"])
    first_train = next(i for i, v in enumerate(ex["labels"]) if v != -100)
    assert first_train > 0
    assert all(v == -100 for v in ex["labels"][:first_train])


def test_prepare_sharegpt_drops_overlong_examples(tmp_path: Path):
    from tcc.finetune.dataset import (
        filter_messages_by_max_tokens,
        fit_messages_to_max_tokens,
    )

    short = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    long_demo = "D" * 5_000
    long_final = "F" * 200
    long_ok = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long_demo},
            {"role": "assistant", "content": long_demo},
            {"role": "user", "content": long_demo},
            {"role": "assistant", "content": long_demo},
            {"role": "user", "content": long_final},
            {"role": "assistant", "content": "gold"},
        ]
    }
    huge_final = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * 20_000},
            {"role": "assistant", "content": "y" * 20_000},
            {"role": "user", "content": "z" * 20_000},
            {"role": "assistant", "content": "w" * 20_000},
            {"role": "user", "content": "u" * 20_000},
            {"role": "assistant", "content": "a" * 20_000},
        ]
    }

    fitted, tag = fit_messages_to_max_tokens(
        long_ok["messages"], max_tokens=2048, tokenizer=None
    )
    assert fitted is not None
    assert tag.startswith("drop_")
    assert fitted[-2]["content"] == long_final
    assert fitted[-1]["content"] == "gold"

    kept, stats = filter_messages_by_max_tokens(
        [short, long_ok, huge_final], max_tokens=2048, tokenizer=None
    )
    assert stats["kept"] == 2
    assert stats["dropped"] == 1

    train = tmp_path / "data" / "train" / "worfbench_train.json"
    train.parent.mkdir(parents=True)
    train.write_text(json.dumps([short, long_ok, huge_final]), encoding="utf-8")
    cfg = {
        "_project_root": tmp_path,
        "paths": {
            "project_root": ".",
            "data_dir": "data",
            "models_dir": "models",
            "checkpoints_dir": "checkpoints",
            "outputs_dir": "outputs",
        },
        "sft": {"cutoff_len": 2048, "max_example_tokens": 2048},
        "models": {"slm": []},
    }
    path = prepare_sharegpt_dataset(cfg)
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2


def test_default_config_sft_unsloth():
    cfg = load_config()
    sft = cfg["sft"]
    assert sft["framework"] == "unsloth"
    assert sft["per_device_train_batch_size"] == 1
    assert sft["gradient_accumulation_steps"] == 8
    assert sft["cutoff_len"] == 2048
    assert sft["max_example_tokens"] == 2048
    assert sft["num_train_epochs"] == 1
    assert sft["load_in_16bit"] is True
    assert sft["load_in_4bit"] is False
    assert sft["optim"] == "adamw_8bit"


def test_dry_run_finetune(cfg):
    out = run_finetune(cfg, "qwen35-0.8b", dry_run=True)
    assert out == cfg["_project_root"] / "checkpoints" / "qwen35-0.8b"
    manifests = list(
        (cfg["_project_root"] / "outputs" / "manifests").glob(
            "finetune_qwen35-0.8b_*.json"
        )
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["sft_backend"] == "unsloth"
    assert manifest["hparams"]["num_train_epochs"] == 1
    assert manifest["hparams"]["mask"] == "last_assistant_only"


def test_modelfile_generation(cfg):
    text = build_modelfile(cfg, "qwen35-0.8b", finetuned=False)
    assert "FROM" in text
    assert "qwen35-0.8b" in text
    assert "PARAMETER temperature 0.0" in text


def test_ollama_create_argv(cfg, tmp_path: Path):
    from tcc.backends.ollama_modelfile import ollama_create_argv

    mf = tmp_path / "Modelfile"
    mf.write_text("FROM .\n", encoding="utf-8")
    argv = ollama_create_argv(cfg, "qwen35-0.8b", finetuned=False, modelfile=mf)
    assert argv[:3] == ["ollama", "create", "qwen35-0.8b"]
    assert argv[-2:] == ["-f", str(mf)]
    argv_q = ollama_create_argv(
        cfg, "qwen35-0.8b", finetuned=False, modelfile=mf, quantize="q4_K_M"
    )
    assert argv_q == [
        "ollama",
        "create",
        "qwen35-0.8b",
        "-f",
        str(mf),
        "--quantize",
        "q4_K_M",
    ]


def test_registry_templates_from_default_config():
    cfg = load_config()
    assert get_sft_template(cfg, "qwen35-4b") == "qwen3_5_nothink"
    assert get_sft_template(cfg, "granite-3b") == "granite4"
    slm_ids = [s["id"] for s in cfg["models"]["slm"]]
    assert "nemotron-nano-4b" not in slm_ids
