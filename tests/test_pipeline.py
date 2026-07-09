"""Testes de orquestração (pipeline, WorFBench setup, scripts)."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tcc.backends.ollama_inference import ollama_config_from_cfg
from tcc.config import load_config
from tcc.pipeline.model_pipeline import (
    build_infer_args,
    build_ollama_import_args,
    build_worfeval_args,
)
from tcc.pipeline.setup_pipeline import setup_steps
from tcc.pipeline.steps import PROJECT_ROOT, SCRIPTS_DIR, run_script
from tcc.setup.worfbench_repo import WORFBENCH_EVAL_DEPS, clone_worfbench, install_worfbench_eval_deps

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_setup_steps_default(self) -> None:
        steps = setup_steps()
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0][1], "download_data.py")
        self.assertEqual(steps[1][1], "build_vector_db.py")
        self.assertEqual(steps[1][2], ())
        self.assertEqual(
            steps[2],
            ("3/3 — clone WorFBench + deps de eval", "worfeval.py", ("--setup",)),
        )

    def test_setup_steps_force_rag(self) -> None:
        steps = setup_steps(force_rag=True)
        self.assertEqual(steps[1][2], ("--force",))

    def test_build_infer_args_scenarios(self) -> None:
        self.assertEqual(
            build_infer_args("m", rag=False, finetuned=False, limit=None, task=None),
            ["--model", "m"],
        )
        self.assertEqual(
            build_infer_args("m", rag=True, finetuned=False, limit=5, task="wikihow"),
            ["--model", "m", "--rag", "--limit", "5", "--task", "wikihow"],
        )
        self.assertIn(
            "--finetuned",
            build_infer_args("m", rag=True, finetuned=True, limit=None, task=None),
        )

    def test_build_worfeval_args(self) -> None:
        args = build_worfeval_args(
            "qwen35-0.8b", rag=False, finetuned=True, task="alfworld", eval_type="graph"
        )
        self.assertEqual(
            args,
            [
                "--model",
                "qwen35-0.8b",
                "--finetuned",
                "--task",
                "alfworld",
                "--eval-type",
                "graph",
            ],
        )

    def test_build_ollama_import_args(self) -> None:
        self.assertEqual(
            build_ollama_import_args("m", finetuned=False, quantize=None, run=False),
            ["--model", "m"],
        )
        self.assertEqual(
            build_ollama_import_args("m", finetuned=True, quantize="q4_K_M", run=True),
            ["--model", "m", "--finetuned", "--quantize", "q4_K_M", "--run"],
        )

    def test_ollama_config_num_predict_and_temperature(self) -> None:
        cfg = load_config()
        ollama = ollama_config_from_cfg(cfg)
        self.assertEqual(ollama.temperature, 0.0)
        self.assertEqual(ollama.num_predict, 4096)

    @patch("tcc.pipeline.steps.subprocess.run")
    def test_run_script_invokes_python(self, mock_run: MagicMock) -> None:
        run_script("infer.py", "--model", "x", dry_run=False)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], str(SCRIPTS_DIR / "infer.py"))
        self.assertIn("--model", cmd)
        self.assertIn("x", cmd)
        self.assertEqual(mock_run.call_args[1]["cwd"], str(PROJECT_ROOT))

    @patch("tcc.pipeline.steps.subprocess.run")
    def test_run_script_dry_run_skips_subprocess(self, mock_run: MagicMock) -> None:
        run_script("infer.py", "--model", "x", dry_run=True)
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_install_worfbench_eval_deps_only_networkx(self, mock_run: MagicMock) -> None:
        install_worfbench_eval_deps()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:3], [sys.executable, "-m", "pip"])
        self.assertEqual(list(WORFBENCH_EVAL_DEPS), ["networkx"])
        self.assertNotIn("collections", cmd)

    def test_clone_existing_repo_skips_git_and_pip(self) -> None:
        with self.subTest("tmp"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp) / "external" / "WorFBench"
                repo.mkdir(parents=True)
                (repo / "node_eval.py").write_text("# stub", encoding="utf-8")
                cfg = {
                    "_project_root": Path(tmp),
                    "paths": {
                        "project_root": ".",
                        "worfbench_repo": "external/WorFBench",
                    },
                    "worfbench": {},
                }
                with patch("subprocess.run") as mock_run:
                    out = clone_worfbench(cfg)
                self.assertEqual(out, repo)
                mock_run.assert_not_called()

    def test_setup_project_dry_run(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "setup_project.py"), "--dry-run"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(ROOT),
        )
        self.assertIn("download_data.py", proc.stdout)
        self.assertIn("build_vector_db.py", proc.stdout)
        self.assertIn("worfeval.py", proc.stdout)

    def test_run_model_skip_sft_dry_run(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_model.py"),
                "--model",
                "qwen35-0.8b",
                "--skip-sft",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(ROOT),
        )
        self.assertNotIn("finetune.py", proc.stdout)
        self.assertIn("infer.py", proc.stdout)
        self.assertIn("worfeval.py", proc.stdout)


    @patch("subprocess.run")
    def test_run_eval_uses_sys_executable(self, mock_run: MagicMock) -> None:
        import sys
        import tempfile

        from tcc.config import load_config
        from tcc.worfeval.runner import run_eval_task

        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "WorFBench"
            repo.mkdir()
            (repo / "node_eval.py").write_text("# stub\n", encoding="utf-8")
            pred = root / "pred.json"
            pred.write_text("[]", encoding="utf-8")
            gold = root / "gold.json"
            gold.write_text("[]", encoding="utf-8")

            def fake_resolve(_cfg: dict, key: str) -> Path:
                if key == "worfbench_repo":
                    return repo
                return root / "outputs"

            with patch("tcc.worfeval.runner.resolve_path", side_effect=fake_resolve):
                with patch("tcc.worfeval.runner.latest_prediction_path", return_value=pred):
                    with patch("tcc.worfeval.runner.test_gold", return_value=gold):
                        run_eval_task(
                            cfg,
                            model_id="qwen35-0.8b",
                            task="wikihow",
                            finetuned=False,
                            rag=False,
                        )
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], sys.executable)
            self.assertTrue(str(cmd[1]).endswith("node_eval.py"))


if __name__ == "__main__":
    unittest.main()
