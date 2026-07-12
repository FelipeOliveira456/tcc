"""Testes da inferência Ollama (mock HTTP — não exige serviço local)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tcc.backends.ollama_inference import (
    make_generate_fn,
    resolve_ollama_model_name,
)
from tcc.config import load_config
from tcc.inference.runner import build_prompt_messages
from tcc.paths import latest_prediction_path, prediction_path, prediction_scenario_from_filename
from tcc.run_stamp import run_stamp


class OllamaInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()
        self.stamp = "20260706_010203"

    def test_prediction_path_with_stamp(self) -> None:
        p = prediction_path(
            self.cfg, "qwen35-4b", finetuned=False, rag=False, task="wikihow", stamp=self.stamp
        )
        self.assertIn(self.stamp, p.name)
        self.assertTrue(p.name.endswith(".json"))

    def test_run_stamp_format(self) -> None:
        s = run_stamp()
        self.assertEqual(len(s), 15)  # YYYYMMDD_HHMMSS
        self.assertEqual(s[8], "_")

    def test_latest_prediction_path_does_not_confuse_sft_and_sft_rag(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            from tcc.config import load_config

            cfg = load_config()
            task_dir = Path(tmp) / "outputs" / "predictions" / "qwen35-0.8b" / "alfworld"
            task_dir.mkdir(parents=True)
            (task_dir / "graph_eval_sft_20260710_030000.json").write_text("[]", encoding="utf-8")
            (task_dir / "graph_eval_sft_rag_20260710_040651.json").write_text(
                "[]", encoding="utf-8"
            )

            with patch("tcc.paths.resolve_path") as mock_resolve:
                mock_resolve.side_effect = (
                    lambda _c, key: Path(tmp) / "outputs" if key == "outputs_dir" else Path(tmp) / key
                )

                got_sft = latest_prediction_path(
                    cfg, "qwen35-0.8b", finetuned=True, rag=False, task="alfworld"
                )
                got_sft_rag = latest_prediction_path(
                    cfg, "qwen35-0.8b", finetuned=True, rag=True, task="alfworld"
                )
                self.assertEqual(got_sft.name, "graph_eval_sft_20260710_030000.json")
                self.assertEqual(got_sft_rag.name, "graph_eval_sft_rag_20260710_040651.json")

    def test_prediction_scenario_from_filename(self) -> None:
        self.assertEqual(
            prediction_scenario_from_filename(
                Path("graph_eval_sft_rag_20260710_040651.json")
            ),
            "sft_rag",
        )
        self.assertEqual(
            prediction_scenario_from_filename(Path("graph_eval_sft_20260710_030000.json")),
            "sft",
        )
        self.assertIsNone(prediction_scenario_from_filename(Path("graph_eval_sft.json")))

    def test_resolve_model_names(self) -> None:
        self.assertEqual(resolve_ollama_model_name(self.cfg, "qwen35-4b", False), "qwen35-4b")
        self.assertEqual(resolve_ollama_model_name(self.cfg, "qwen35-4b", True), "qwen35-4b-sft")

    @patch("tcc.backends.ollama_inference.urlopen")
    def test_chat_completion(self, mock_urlopen: MagicMock) -> None:
        payload = {"message": {"role": "assistant", "content": "workflow: step1 -> step2"}}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode()
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        generate = make_generate_fn(self.cfg)
        out = generate(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}],
            "qwen35-4b",
            False,
        )
        self.assertEqual(out, "workflow: step1 -> step2")

        call = mock_urlopen.call_args
        body = json.loads(call[0][0].data.decode())
        self.assertEqual(body["model"], "qwen35-4b")
        self.assertFalse(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["options"]["temperature"], 0.0)
        self.assertEqual(body["options"]["num_predict"], 4096)

    def test_rag_prompt_headers_english(self) -> None:
        gold = {
            "conversations": [
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "Do the task"},
                {"role": "assistant", "content": "Node:\n1: x"},
            ]
        }
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"user": "train q", "workflow": "Node:\n1: a"},
        ]
        msgs = build_prompt_messages(gold, use_rag=True, cfg={}, retriever=retriever)
        system = msgs[0]["content"]
        self.assertIn("Retrieved training examples", system)
        self.assertIn("### Example 1", system)
        self.assertIn("Question:", system)
        self.assertNotIn("Exemplos recuperados", system)
        self.assertNotIn("Pergunta:", system)

    def test_run_inference_skips_failed_example(self) -> None:
        import tempfile

        from tcc.inference.runner import run_inference

        gold_item = {
            "conversations": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "Node:\n1: a\nEdge: (START,1) (1,END)"},
            ]
        }

        def generate(messages, model_id, finetuned):
            user = messages[-1]["content"]
            if user == "fail":
                raise RuntimeError("Ollama HTTP 500: peg-native")
            return "Node:\n1: ok\nEdge: (START,1) (1,END)"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "data" / "test" / "toy"
            task_dir.mkdir(parents=True)
            gold = [
                {**gold_item, "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "ok1"},
                    {"role": "assistant", "content": "Node:\n1: a\nEdge: (START,1) (1,END)"},
                ]},
                {**gold_item, "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "fail"},
                    {"role": "assistant", "content": "Node:\n1: a\nEdge: (START,1) (1,END)"},
                ]},
                {**gold_item, "conversations": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "ok2"},
                    {"role": "assistant", "content": "Node:\n1: a\nEdge: (START,1) (1,END)"},
                ]},
            ]
            (task_dir / "graph_eval.json").write_text(
                json.dumps(gold), encoding="utf-8"
            )
            cfg = {
                "_project_root": root,
                "paths": {
                    "data_dir": "data",
                    "outputs_dir": "outputs",
                },
            }
            out = run_inference(
                cfg,
                "toy-model",
                finetuned=False,
                use_rag=False,
                generate_fn=generate,
                tasks=["toy"],
            )
            preds = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(preds), 3)
            self.assertTrue(preds[0]["workflow"])
            self.assertEqual(preds[1]["workflow"], "")
            self.assertTrue(preds[2]["workflow"])
            stamp = out.name.removeprefix("graph_eval_i0_").removesuffix(".json")
            meta_path = root / "outputs" / "predictions" / "toy-model" / f"run_{stamp}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["skipped_errors"], {"toy": 1})


if __name__ == "__main__":
    unittest.main()
