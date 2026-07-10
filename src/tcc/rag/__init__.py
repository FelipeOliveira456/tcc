"""RAG sobre o treino WorFBench."""

from tcc.rag.index import build_rag_index, last_assistant_message, last_user_message
from tcc.rag.retriever import TrainRagRetriever

__all__ = [
    "TrainRagRetriever",
    "build_rag_index",
    "last_assistant_message",
    "last_user_message",
]
