"""Testes da inferência Ollama (mock HTTP — não exige serviço local)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from tcc.backends.ollama_inference import (
    make_generate_fn,
    resolve_ollama_model_name,
)
from tcc.config import load_config
from tcc.paths import prediction_path
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


if __name__ == "__main__":
    unittest.main()
