"""Testes da rota GGUF (llama.cpp) para ollama import."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from tcc.backends.gguf_convert import (
    needs_gguf_conversion,
    read_hf_architectures,
)
from tcc.backends.ollama_modelfile import (
    build_modelfile,
    modelfile_uses_gguf,
    ollama_create_argv,
)


class GgufConvertTests(unittest.TestCase):
    def test_read_architectures(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "config.json").write_text(
                json.dumps({"architectures": ["GraniteForCausalLM"]}),
                encoding="utf-8",
            )
            self.assertEqual(read_hf_architectures(d), ["GraniteForCausalLM"])

    def test_needs_gguf_by_architecture(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "models" / "granite-3b"
            weights.mkdir(parents=True)
            (weights / "config.json").write_text(
                json.dumps({"architectures": ["GraniteForCausalLM"]}),
                encoding="utf-8",
            )
            (weights / "model.safetensors").write_text("x", encoding="utf-8")
            cfg = {
                "_project_root": root,
                "paths": {"project_root": str(root), "models_dir": "models"},
                "inference": {
                    "ollama": {
                        "gguf": {
                            "enabled": True,
                            "force_architectures": ["GraniteForCausalLM"],
                        }
                    }
                },
                "models": {
                    "slm": [
                        {
                            "id": "granite-3b",
                            "hf_id": "ibm-granite/granite-4.1-3b",
                            "sft_template": "granite4",
                        }
                    ]
                },
            }
            self.assertTrue(needs_gguf_conversion(cfg, "granite-3b", weights))

    def test_needs_gguf_by_model_flag(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "models" / "x"
            weights.mkdir(parents=True)
            (weights / "config.json").write_text(
                json.dumps({"architectures": ["LlamaForCausalLM"]}),
                encoding="utf-8",
            )
            cfg = {
                "inference": {"ollama": {"gguf": {"enabled": True}}},
                "models": {
                    "slm": [
                        {
                            "id": "x",
                            "hf_id": "x/x",
                            "ollama_via_gguf": True,
                        }
                    ]
                },
            }
            self.assertTrue(needs_gguf_conversion(cfg, "x", weights))

    def test_qwen_does_not_need_gguf(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "models" / "qwen35-0.8b"
            weights.mkdir(parents=True)
            (weights / "config.json").write_text(
                json.dumps({"architectures": ["Qwen3ForCausalLM"]}),
                encoding="utf-8",
            )
            cfg = {
                "inference": {
                    "ollama": {
                        "gguf": {
                            "enabled": True,
                            "force_architectures": ["GraniteForCausalLM"],
                        }
                    }
                },
                "models": {
                    "slm": [
                        {
                            "id": "qwen35-0.8b",
                            "hf_id": "Qwen/Qwen3.5-0.8B",
                            "sft_template": "qwen3_5_nothink",
                        }
                    ]
                },
            }
            self.assertFalse(needs_gguf_conversion(cfg, "qwen35-0.8b", weights))

    @patch("tcc.backends.ollama_modelfile.convert_hf_dir_to_gguf")
    def test_modelfile_granite_uses_gguf(self, mock_convert) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "models" / "granite-3b"
            weights.mkdir(parents=True)
            (weights / "config.json").write_text(
                json.dumps({"architectures": ["GraniteForCausalLM"]}),
                encoding="utf-8",
            )
            (weights / "model.safetensors").write_text("w", encoding="utf-8")
            gguf = root / "models" / "ollama" / "gguf" / "granite-3b-q4_K_M.gguf"
            gguf.parent.mkdir(parents=True)
            gguf.write_bytes(b"gguf")
            mock_convert.return_value = gguf

            cfg = {
                "_project_root": root,
                "paths": {
                    "project_root": str(root),
                    "models_dir": str(root / "models"),
                    "checkpoints_dir": str(root / "checkpoints"),
                },
                "inference": {
                    "ollama": {
                        "temperature": 0.0,
                        "gguf": {
                            "enabled": True,
                            "outtype": "q4_K_M",
                            "force_architectures": ["GraniteForCausalLM"],
                        },
                    }
                },
                "models": {
                    "slm": [
                        {
                            "id": "granite-3b",
                            "hf_id": "ibm-granite/granite-4.1-3b",
                            "sft_template": "granite4",
                            "ollama_via_gguf": True,
                        }
                    ]
                },
            }

            text = build_modelfile(cfg, "granite-3b", finetuned=False)
            self.assertIn(f"FROM {gguf}", text)
            self.assertIn("[GGUF]", text)
            mock_convert.assert_called_once()

    def test_convert_requires_setup_llama_cpp(self) -> None:
        import tempfile

        from tcc.backends.gguf_convert import convert_hf_dir_to_gguf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "w"
            weights.mkdir()
            (weights / "model.safetensors").write_text("x", encoding="utf-8")
            cfg = {
                "_project_root": root,
                "paths": {"project_root": str(root), "models_dir": str(root / "models")},
                "inference": {
                    "ollama": {
                        "gguf": {
                            "llama_cpp_dir": str(root / "missing-llama"),
                            "outtype": "q4_K_M",
                        }
                    }
                },
            }
            out = root / "out.gguf"
            with self.assertRaises(FileNotFoundError) as ctx:
                convert_hf_dir_to_gguf(cfg, weights, out)
            self.assertIn("setup_llama_cpp", str(ctx.exception))

    def test_modelfile_uses_gguf_helper(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            mf = Path(tmp) / "Modelfile"
            mf.write_text("FROM /tmp/x.gguf\nPARAMETER temperature 0\n", encoding="utf-8")
            self.assertTrue(modelfile_uses_gguf(mf))
            mf.write_text("FROM /tmp/weights\n", encoding="utf-8")
            self.assertFalse(modelfile_uses_gguf(mf))

    def test_ollama_create_skips_quantize_for_gguf(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            mf = Path(tmp) / "Modelfile"
            mf.write_text("FROM /tmp/model.gguf\n", encoding="utf-8")
            cfg = {"inference": {"ollama": {}}}
            argv = ollama_create_argv(
                cfg, "granite-3b", finetuned=False, modelfile=mf, quantize="q4_K_M"
            )
            self.assertNotIn("--quantize", argv)


if __name__ == "__main__":
    unittest.main()
