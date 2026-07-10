"""BD vetorial determinístico para RAG (somente treino WorFBench)."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tcc.paths import train_json, vector_db_dir
from tcc.rag.index import last_assistant_message, last_user_message, load_train_examples


@dataclass
class VectorStoreMeta:
    embedding_model: str
    seed: int
    top_k: int
    chunk_fields: list[str]
    train_sha256: str
    num_vectors: int


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_deterministic_vector_db(cfg: dict[str, Any], *, force: bool = False) -> Path:
    """
    Índice determinístico:
      - 1 registro por exemplo de treino
      - user/workflow = último par (tarefa real; demos few-shot ignorados)
      - embedding = só a pergunta (último user); workflow só no prompt
      - seed fixo em config.rag.seed
      - metadados versionados em meta.json
    """
    from sentence_transformers import SentenceTransformer

    rag_cfg = cfg.get("rag", {})
    out_dir = vector_db_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.pkl"
    meta_path = out_dir / "meta.json"

    train = train_json(cfg)
    train_hash = _sha256_file(train)
    emb_name = rag_cfg["embedding_model"]
    seed = int(rag_cfg.get("seed", 42))
    # Similaridade só na pergunta; workflow é metadado do prompt.
    fields = list(rag_cfg.get("chunk_fields", ["query"]))
    if fields != ["query"]:
        fields = ["query"]

    if meta_path.exists() and index_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("train_sha256") == train_hash
            and meta.get("embedding_model") == emb_name
            and meta.get("chunk_fields") == fields
        ):
            return index_path

    raw = load_train_examples(train)
    records: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        user_msg = last_user_message(item)
        workflow = last_assistant_message(item)
        if not user_msg or not workflow:
            continue
        ex_id = f"{item.get('source', 'unknown')}::{item.get('id', i)}"
        records.append(
            {
                "id": ex_id,
                "index": i,
                "text": user_msg,  # texto embedado = pergunta
                "user": user_msg,
                "workflow": workflow,
                "source": item.get("source"),
            }
        )

    records.sort(key=lambda r: r["id"])

    np.random.seed(seed)
    model = SentenceTransformer(emb_name)
    texts = [r["text"] for r in records]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64,
    )

    payload = {
        "records": records,
        "embeddings": embeddings.astype(np.float32),
        "embedding_model": emb_name,
        "seed": seed,
    }
    with index_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = VectorStoreMeta(
        embedding_model=emb_name,
        seed=seed,
        top_k=int(rag_cfg.get("top_k", 2)),
        chunk_fields=fields,
        train_sha256=train_hash,
        num_vectors=len(records),
    )
    meta_path.write_text(
        json.dumps(meta.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return index_path


def load_vector_store(cfg: dict[str, Any]) -> tuple[list[dict], np.ndarray, VectorStoreMeta]:
    index_path = vector_db_dir(cfg) / "index.pkl"
    meta_path = vector_db_dir(cfg) / "meta.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Índice não encontrado em {index_path}. Rode build_vector_db.py primeiro."
        )
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    meta = VectorStoreMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    return payload["records"], payload["embeddings"], meta


class VectorRetriever:
    """Índice RAG em memória — carregar uma vez por run de inferência."""

    def __init__(
        self,
        records: list[dict],
        matrix: np.ndarray,
        meta: VectorStoreMeta,
        model: Any,
    ) -> None:
        self._records = records
        self._matrix = matrix
        self._meta = meta
        self._model = model

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VectorRetriever:
        from sentence_transformers import SentenceTransformer

        records, matrix, meta = load_vector_store(cfg)
        model = SentenceTransformer(meta.embedding_model)
        return cls(records, matrix, meta, model)

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[dict]:
        """Recupera top-k por similaridade de cosseno entre perguntas."""
        k = top_k or self._meta.top_k
        q = self._model.encode([query], convert_to_numpy=True)[0]
        scores = self._matrix @ q / (
            np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(q) + 1e-9
        )
        order = np.argsort(-scores)
        return [self._records[int(i)] for i in order[:k]]


def retrieve(
    cfg: dict[str, Any],
    query: str,
    *,
    top_k: int | None = None,
    retriever: VectorRetriever | None = None,
) -> list[dict]:
    """Recupera top-k do treino. Passe `retriever` para evitar recarregar o modelo."""
    if retriever is not None:
        return retriever.retrieve(query, top_k=top_k)
    return VectorRetriever.from_config(cfg).retrieve(query, top_k=top_k)
