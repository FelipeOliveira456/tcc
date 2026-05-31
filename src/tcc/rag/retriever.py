"""Recuperação de few-shot examples a partir do índice RAG."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tcc.rag.index import RagExample


class TrainRagRetriever:
    def __init__(self, index_path: Path, top_k: int = 2):
        with index_path.open("rb") as f:
            blob = pickle.load(f)
        self.examples: list[RagExample] = blob["examples"]
        self.top_k = top_k
        self._matrix = np.stack([e.embedding for e in self.examples if e.embedding is not None])
        self._model = SentenceTransformer(blob["model"])

    def retrieve(self, query: str, *, exclude_ids: set[str] | None = None) -> list[RagExample]:
        q_emb = self._model.encode([query], convert_to_numpy=True)[0]
        scores = self._matrix @ q_emb / (
            np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        order = np.argsort(-scores)
        picked: list[RagExample] = []
        for idx in order:
            ex = self.examples[int(idx)]
            if exclude_ids and ex.example_id in exclude_ids:
                continue
            picked.append(ex)
            if len(picked) >= self.top_k:
                break
        return picked

    def format_few_shot_block(self, examples: list[RagExample]) -> str:
        blocks: list[str] = []
        for i, ex in enumerate(examples, 1):
            blocks.append(
                f"### Exemplo recuperado {i}\n"
                f"Pergunta:\n{ex.query_text}\n\n"
                f"Workflow:\n{ex.workflow_text}\n"
            )
        return "\n".join(blocks)
