"""Testes do WorFEval seguro (skip de predições não parseáveis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.worfeval.eval_workflow import eval_workflow


@pytest.fixture
def worfbench_repo() -> Path:
    repo = ROOT / "external" / "WorFBench"
    if not (repo / "node_eval.py").is_file():
        pytest.skip("WorFBench não clonado")
    return repo


def test_skips_empty_pred_without_crash(tmp_path: Path, worfbench_repo: Path) -> None:
    gold_path = ROOT / "data" / "test" / "alfworld" / "graph_eval.json"
    if not gold_path.is_file():
        pytest.skip("gold alfworld ausente")

    with gold_path.open(encoding="utf-8") as f:
        gold = json.load(f)[:3]
    gold_mini = tmp_path / "gold.json"
    gold_mini.write_text(json.dumps(gold), encoding="utf-8")
    pred = [
        {"query": item, "workflow": "sorry, I cannot help with that."}
        for item in gold
    ]
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps(pred), encoding="utf-8")
    out_path = tmp_path / "out.json"

    mock_model = MagicMock()
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        result = eval_workflow(
            str(gold_mini),
            str(pred_path),
            "all-mpnet-base-v2",
            "node",
            str(out_path),
            worfbench_repo=worfbench_repo,
        )

    assert result["n_total"] == 3
    assert result["n_skipped_unparseable"] == 3
    assert result.get("n_skipped_runtime", 0) == 0
    assert result["n_evaluated"] == 0
    assert result["f1_score"] == 0.0
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["n_skipped_unparseable"] == 3
