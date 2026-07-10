"""Testes do retriever RAG em memória."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tcc.rag.vector_store import VectorRetriever, VectorStoreMeta, retrieve


class VectorRetrieverTests(unittest.TestCase):
    def _retriever(self) -> VectorRetriever:
        records = [
            {"id": "a", "user": "q1", "workflow": "w1"},
            {"id": "b", "user": "q2", "workflow": "w2"},
        ]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        meta = VectorStoreMeta(
            embedding_model="test-model",
            seed=42,
            top_k=1,
            chunk_fields=["query"],
            train_sha256="abc",
            num_vectors=2,
        )
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 0.0]])
        return VectorRetriever(records, matrix, meta, model)

    def test_retrieve_uses_cached_model(self) -> None:
        r = self._retriever()
        out1 = r.retrieve("hello")
        r.retrieve("world")
        self.assertEqual(len(out1), 1)
        self.assertEqual(out1[0]["id"], "a")
        self.assertEqual(r._model.encode.call_count, 2)

    @patch("tcc.rag.vector_store.VectorRetriever.from_config")
    def test_retrieve_without_cached_reloader(self, mock_from_config: MagicMock) -> None:
        r = self._retriever()
        mock_from_config.return_value = r
        retrieve({}, "q", retriever=r)
        retrieve({}, "q2", retriever=r)
        mock_from_config.assert_not_called()

    @patch("tcc.rag.vector_store.VectorRetriever.from_config")
    def test_retrieve_without_retriever_loads_once_per_call(self, mock_from_config: MagicMock) -> None:
        r = self._retriever()
        mock_from_config.return_value = r
        retrieve({}, "q")
        retrieve({}, "q2")
        self.assertEqual(mock_from_config.call_count, 2)


if __name__ == "__main__":
    unittest.main()
