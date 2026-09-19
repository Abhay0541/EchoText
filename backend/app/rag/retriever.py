"""
Retrieval layer for EchoText.

Why RAG at all, given we're also fine-tuning with LoRA?
LoRA teaches the model *your general style* (word choice, tone, typical
length). It does NOT reliably memorize specific facts ("what's your address",
"what time does the gym close") or one-off replies you gave months ago.
Retrieval fixes that: before generating, we pull the most similar past
(context -> reply) pairs from your real chat history and hand them to the
model as few-shot examples. Style comes from the fine-tune; specific,
factual, or previously-settled answers come from retrieval.

Implementation note on the embedding backend:
This ships with a TF-IDF vectorizer (scikit-learn) as the default backend.
That's a deliberate choice, not a shortcut: TF-IDF needs no model download,
runs in milliseconds on a laptop CPU, and for short, informal chat messages
(lots of names, slang, inside jokes) it often matches semantic embeddings in
practice while being far cheaper. `EmbeddingBackend` is defined as an
abstract interface specifically so you can swap in a real sentence-transformer
(e.g. `all-MiniLM-L6-v2`) later for better semantic matching without touching
any other file — see `SentenceTransformerBackend` below (kept optional so the
project runs with zero extra downloads out of the box).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.data.whatsapp_parser import TrainingPair


@dataclass
class RetrievedExample:
    context: str
    reply: str
    score: float


class EmbeddingBackend(ABC):
    """Swap-in point for the vectorization strategy. See module docstring."""

    @abstractmethod
    def fit(self, documents: list[str]) -> None: ...

    @abstractmethod
    def transform(self, documents: list[str]) -> np.ndarray: ...


class TfidfBackend(EmbeddingBackend):
    """Default backend. Zero external downloads, runs anywhere."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=20_000,
        )
        self._fitted = False

    def fit(self, documents: list[str]) -> None:
        self._vectorizer.fit(documents)
        self._fitted = True

    def transform(self, documents: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self._vectorizer.transform(documents).toarray()


class SentenceTransformerBackend(EmbeddingBackend):  # pragma: no cover - optional upgrade path
    """
    Optional upgrade path for real semantic embeddings.

    Requires: pip install sentence-transformers
    Swap this in by changing ONE line where ChatIndex is constructed
    (see app/api/routes.py) — nothing else in the retrieval pipeline changes,
    because both backends satisfy the same fit/transform interface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # deferred import
        self._model = SentenceTransformer(model_name)

    def fit(self, documents: list[str]) -> None:
        pass  # sentence-transformers models are pre-trained; no fitting needed

    def transform(self, documents: list[str]) -> np.ndarray:
        return self._model.encode(documents)


class ChatIndex:
    """In-memory vector index over a user's historical (context, reply) pairs."""

    def __init__(self, backend: EmbeddingBackend | None = None):
        self._backend = backend or TfidfBackend()
        self._pairs: list[TrainingPair] = []
        self._matrix: np.ndarray | None = None

    def build(self, pairs: list[TrainingPair]) -> None:
        self._pairs = pairs
        contexts = [self._context_text(p) for p in pairs]
        if not contexts:
            self._matrix = None
            return
        self._backend.fit(contexts)
        self._matrix = self._backend.transform(contexts)

    @staticmethod
    def _context_text(pair: TrainingPair) -> str:
        return "\n".join(f"{m.sender}: {m.text}" for m in pair.context)

    def query(self, incoming_message: str, top_k: int = 3) -> list[RetrievedExample]:
        if self._matrix is None or len(self._pairs) == 0:
            return []
        query_vec = self._backend.transform([incoming_message])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            RetrievedExample(
                context=self._context_text(self._pairs[i]),
                reply=self._pairs[i].reply,
                score=float(sims[i]),
            )
            for i in top_idx
            if sims[i] > 0
        ]
