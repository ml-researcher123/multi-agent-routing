# routers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from agents import Agent
from vector_store import InMemoryVectorStore


@dataclass
class RouteResult:
    chosen_agent: str
    candidates: List[Tuple[str, float]]  # (agent_name, score)
    debug: Dict[str, str]


class KNNRouter:
    def __init__(self, agents: List[Agent], store: InMemoryVectorStore, agent_id_to_name: Dict[str, str], top_k: int = 5):
        self.agents = agents
        self.store = store
        self.agent_id_to_name = agent_id_to_name
        self.top_k = top_k

    def route(self, query_emb: np.ndarray) -> RouteResult:
        hits = self.store.knn_search(query_emb, top_n=20)

        scores: Dict[str, float] = {}
        for item, sim in hits:
            scores[item.agent_id] = max(scores.get(item.agent_id, 0.0), sim)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        candidates = [(self.agent_id_to_name[aid], float(score)) for aid, score in ranked]
        chosen = candidates[0][0] if candidates else "UNKNOWN"

        return RouteResult(
            chosen_agent=chosen,
            candidates=candidates,
            debug={"router": "KNNRouter"},
        )


class MLRouter:
    """
    Score-based router on prompt embeddings. Trained externally (train_ml.py).
    Supports multilabel one-vs-rest models by exposing per-agent scores.
    """

    def __init__(self, model, class_names: List[str], top_k: int = 5):
        self.model = model
        self.class_names = class_names
        self.top_k = top_k

    def _scores(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(X)
            return scores[0] if getattr(scores, "ndim", 1) > 1 else np.asarray(scores)
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            scores = scores[0] if getattr(scores, "ndim", 1) > 1 else np.asarray(scores)
            return 1.0 / (1.0 + np.exp(-scores))
        raise ValueError("MLRouter model must provide predict_proba or decision_function")

    def route(self, query_emb: np.ndarray) -> RouteResult:
        X = query_emb.reshape(1, -1)
        scores = self._scores(X)
        n = min(len(self.class_names), len(scores))
        pairs = list(zip(self.class_names[:n], scores[:n]))
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)[: self.top_k]
        candidates = [(name, float(score)) for name, score in ranked]
        chosen = candidates[0][0] if candidates else "UNKNOWN"
        return RouteResult(chosen_agent=chosen, candidates=candidates, debug={"router": "MLRouter"})


class PairwiseMLRouter:
    """
    Binary classifier on (prompt, agent-profile) pair features.
    Scores each agent for a given prompt and ranks by positive class probability.
    """

    def __init__(self, model, class_names: List[str], agent_profile_texts: List[str], embedder, top_k: int = 5):
        self.model = model
        self.class_names = class_names
        self.top_k = top_k
        self.embedder = embedder
        self.agent_profile_texts = agent_profile_texts
        self._agent_embs = self.embedder.encode(agent_profile_texts)
        self.is_pairwise = True
        self._pos_class = 1
        if hasattr(self.model, "classes_") and 1 in list(self.model.classes_):
            self._pos_class = 1
        elif hasattr(self.model, "classes_"):
            self._pos_class = list(self.model.classes_)[-1]

    def _pair_features(self, prompt_emb: np.ndarray) -> np.ndarray:
        p = np.repeat(prompt_emb.reshape(1, -1), self._agent_embs.shape[0], axis=0)
        a = self._agent_embs
        return np.concatenate([p, a, np.abs(p - a), p * a], axis=1)

    def route(self, query_emb: np.ndarray) -> RouteResult:
        X = self._pair_features(query_emb)
        probs = self.model.predict_proba(X)
        if hasattr(self.model, "classes_"):
            classes = list(self.model.classes_)
            pos_idx = classes.index(self._pos_class)
        else:
            pos_idx = 1
        scores = probs[:, pos_idx]
        ranked_idx = np.argsort(-scores)[: self.top_k]
        candidates = [(self.class_names[i], float(scores[i])) for i in ranked_idx]
        chosen = candidates[0][0] if candidates else "UNKNOWN"
        return RouteResult(chosen_agent=chosen, candidates=candidates, debug={"router": "PairwiseMLRouter"})
