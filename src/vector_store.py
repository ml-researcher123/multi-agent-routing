# vector_store.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

@dataclass
class VectorItem:
    item_id: str          # e.g., chunk id
    agent_id: str
    text: str
    embedding: np.ndarray
    meta: Dict[str, Any]

class InMemoryVectorStore:
    """
    Minimal vector store: keeps embeddings and does cosine KNN using sklearn NearestNeighbors.
    """
    def __init__(self):
        self.items: List[VectorItem] = []
        self._matrix: np.ndarray | None = None
        self._knn = None

    def add_items(self, items: List[VectorItem]) -> None:
        self.items.extend(items)
        self._matrix = None
        self._knn = None

    def build_index(self) -> None:
        if not self.items:
            raise RuntimeError("No items to index.")
        mat = np.stack([it.embedding for it in self.items], axis=0)
        self._matrix = mat.astype(np.float32)

        from sklearn.neighbors import NearestNeighbors
        # cosine distance: 0 = identical, 1 = opposite (for normalized embeddings)
        self._knn = NearestNeighbors(metric="cosine", algorithm="auto")
        self._knn.fit(self._matrix)

    def knn_search(self, query_emb: np.ndarray, top_n: int = 10) -> List[Tuple[VectorItem, float]]:
        """
        Returns list of (VectorItem, similarity_score)
        similarity_score = 1 - cosine_distance
        """
        if self._knn is None:
            self.build_index()
        q = query_emb.reshape(1, -1).astype(np.float32)
        distances, indices = self._knn.kneighbors(q, n_neighbors=min(top_n, len(self.items)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            sim = float(1.0 - dist)
            results.append((self.items[int(idx)], sim))
        return results
