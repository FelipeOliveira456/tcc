#!/usr/bin/env python3
"""Inferência do WorFBench via OpenRouter (por padrão, Moonshot Kimi K3).

Exemplos:
  python scripts/infer_openrouter.py
  python scripts/infer_openrouter.py --rag
  python scripts/infer_openrouter.py --task toolbench --limit 5

Requer OPENROUTER_API_KEY no ambiente ou em .env na raiz do projeto.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcc.config import load_config
from tcc.inference.runner import run_inference

DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_dotenv(path: Path) -> None:
    """Carrega apenas variáveis ausentes, sem substituir o ambiente atual."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _chat_completion(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", ""),
        "X-Title": os.environ.get("OPENROUTER_X_TITLE", "WorFBench TCC"),
    }
    headers = {key: value for key, value in headers.items() if value}
    request = Request(
        os.environ.get("OPENROUTER_BASE_URL", DEFAULT_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "600"))) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter inacessível: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Resposta inesperada do OpenRouter: {data!r}") from exc
    if not content:
        raise RuntimeError(f"Resposta vazia do OpenRouter: {data!r}")
    return str(content).strip()


def main() -> None:
    _load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--task", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY não encontrada. Defina-a no ambiente ou em .env."
        )

    cfg = load_config(args.config)

    def generate(messages: list[dict[str, str]], _model_id: str, _finetuned: bool) -> str:
        return _chat_completion(
            messages,
            api_key=api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    out = run_inference(
        cfg,
        "kimi-k3-openrouter",
        finetuned=False,
        use_rag=args.rag,
        generate_fn=generate,
        tasks=[args.task] if args.task else None,
        limit=args.limit,
    )
    print(f"Predições em: {out.parent} (modelo={args.model})")


if __name__ == "__main__":
    main()
