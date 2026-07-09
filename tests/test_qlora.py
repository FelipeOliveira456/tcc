"""Testes de geração de dataset/YAML SFT e Modelfile Ollama."""

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
from tcc.finetune.dataset import SHAREGPT_DATASET_NAME, prepare_sharegpt_dataset, write_dataset_info
from tcc.finetune.qlora import build_llamafactory_yaml, run_finetune
from tcc.models_registry import get_sft_template


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
                {"id": "qwen35-0.8b", "hf_id": "Qwen/Qwen3.5-0.8B", "sft_template": "qwen3_5_nothink"},
            ]
        },
        "sft": {
            "num_train_epochs": 3,
            "cutoff_len": 8192,
            "gradient_accumulation_steps": 4,
        },
        "inference": {"ollama": {"temperature": 0.0, "sft_suffix": "-sft"}},
    }


def test_sharegpt_dataset_and_info(cfg):
    path = prepare_sharegpt_dataset(cfg)
    assert path.name == f"{SHAREGPT_DATASET_NAME}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data[0]["messages"]) == 7
    info_path = write_dataset_info(cfg)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert SHAREGPT_DATASET_NAME in info
    assert info[SHAREGPT_DATASET_NAME]["formatting"] == "sharegpt"


def test_llamafactory_yaml_has_required_fields(cfg):
    yaml_path = build_llamafactory_yaml(cfg, "qwen35-0.8b", "20260706_120000")
    text = yaml_path.read_text(encoding="utf-8")
    assert "stage: sft" in text
    assert "template: qwen3_5_nothink" in text
    assert f"dataset: {SHAREGPT_DATASET_NAME}" in text
    assert "mask_history: true" in text
    assert "quantization_method: bnb" in text
    assert "dataset_dir:" in text


def test_dry_run_finetune(cfg):
    out = run_finetune(cfg, "qwen35-0.8b", dry_run=True)
    assert out == cfg["_project_root"] / "checkpoints" / "qwen35-0.8b"
    manifests = list((cfg["_project_root"] / "outputs" / "manifests").glob("finetune_qwen35-0.8b_*.json"))
    assert manifests
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["sft_backend"] == "llamafactory"


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
    assert argv_q == ["ollama", "create", "qwen35-0.8b", "-f", str(mf), "--quantize", "q4_K_M"]


def test_registry_templates_from_default_config():
    cfg = load_config()
    assert get_sft_template(cfg, "qwen35-4b") == "qwen3_5_nothink"
    assert get_sft_template(cfg, "granite-3b") == "granite4"
    slm_ids = [s["id"] for s in cfg["models"]["slm"]]
    assert "nemotron-nano-4b" not in slm_ids
