# embedder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class EmbedderConfig:
    model_name: str = "all-MiniLM-L6-v2"  # used if sentence-transformers exists
    max_features: int = 5000             # used for TF-IDF fallback

class Embedder:
    """
    Tries sentence-transformers first. If unavailable, falls back to TF-IDF.
    Both produce vector embeddings for routing benchmarks.
    """
    def __init__(self, config: Optional[EmbedderConfig] = None):
        self.config = config or EmbedderConfig()
        self.backend = None
        self._tfidf = None

        # Try to use sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.backend = ("sbert", SentenceTransformer(self.config.model_name))
        except Exception:
            self.backend = ("tfidf", None)

    def fit(self, texts: List[str]) -> None:
        """
        Fit is only needed for TF-IDF fallback.
        SBERT doesn't need fitting.
        """
        if self.backend[0] == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(max_features=self.config.max_features)
            self._tfidf.fit(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend[0] == "sbert":
            model = self.backend[1]
            emb = model.encode(texts, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        else:
            if self._tfidf is None:
                raise RuntimeError("TF-IDF embedder not fit. Call fit(texts) first.")
            X = self._tfidf.transform(texts)
            # Normalize to unit length to mimic cosine behavior
            X = X.astype(np.float32)
            # Convert sparse to dense for simplicity
            dense = X.toarray()
            norms = np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
            return dense / norms
