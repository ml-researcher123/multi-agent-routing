from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from embedder import Embedder
from vector_store import InMemoryVectorStore, VectorItem
from agents import id_to_name_map


def _camel_to_words(name: str) -> List[str]:
    out: List[str] = []
    buf = ""
    for ch in name:
        if ch.isupper() and buf:
            out.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def build_vector_store(agents, embedder: Embedder) -> Tuple[InMemoryVectorStore, Dict[str, str]]:
    store = InMemoryVectorStore()
    items: List[VectorItem] = []
    for agent in agents:
        name_words = _camel_to_words(agent.name.replace("Agent", ""))
        keywords = " ".join(word.lower() for word in name_words)
        name_hint = " ".join(name_words)
        chunks = [
            agent.description,
            f"Agent name: {agent.name}. Capabilities: {agent.description}",
            f"Use this agent when user asks about: {agent.name.replace('Agent', '').lower()} tasks.",
            f"Keywords: {keywords}",
            f"Task intent: {name_hint}",
        ]
        embs = embedder.encode(chunks)
        for idx, (text, emb) in enumerate(zip(chunks, embs)):
            items.append(
                VectorItem(
                    item_id=f"{agent.agent_id}_c{idx}",
                    agent_id=agent.agent_id,
                    text=text,
                    embedding=emb,
                    meta={"chunk_type": f"c{idx}", "agent_name": agent.name},
                )
            )
    store.add_items(items)
    store.build_index()
    return store, id_to_name_map(agents)


def choose_pred_set(
    candidates: List[Tuple[str, float]],
    threshold: float,
    per_agent_thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    if not candidates:
        return []
    if per_agent_thresholds:
        chosen = [
            name
            for name, score in candidates
            if float(score) >= float(per_agent_thresholds.get(name, threshold))
        ]
    else:
        chosen = [name for name, score in candidates if float(score) >= threshold]
    if not chosen:
        chosen = [candidates[0][0]]
    return list(dict.fromkeys(chosen))


def hard_set_metrics(pred_names: List[str], gold_set: List[str]) -> Tuple[float, float, float, float, float]:
    pred = set(pred_names)
    gold = set(gold_set)
    tp = len(pred & gold)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))
    jacc = tp / max(1, len(pred | gold))
    exact = 1.0 if pred == gold else 0.0
    return precision, recall, f1, jacc, exact


def load_similarity_lookup(similarity_path: str, agent_names: List[str]) -> Dict[str, Dict[str, float]]:
    path = Path(similarity_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("similarity", {})
    lookup: Dict[str, Dict[str, float]] = {}
    expected = set(agent_names)
    for src in agent_names:
        row = matrix.get(src, {})
        if not row:
            raise ValueError(f"Missing similarity row for agent: {src}")
        row_names = set(row.keys())
        if row_names != expected:
            missing = sorted(expected - row_names)
            extra = sorted(row_names - expected)
            raise ValueError(f"Similarity row mismatch for {src}. Missing={missing}, extra={extra}")
        lookup[src] = {dst: float(row[dst]) for dst in agent_names}
    return lookup


def _best_soft_match_score(
    smaller: List[str],
    larger: List[str],
    similarity_lookup: Dict[str, Dict[str, float]],
) -> float:
    if not smaller or not larger:
        return 0.0
    dp: Dict[int, float] = {0: 0.0}
    for src in smaller:
        next_dp: Dict[int, float] = {}
        for mask, score in dp.items():
            for idx, dst in enumerate(larger):
                if mask & (1 << idx):
                    continue
                new_mask = mask | (1 << idx)
                new_score = score + similarity_lookup[src][dst]
                prev = next_dp.get(new_mask)
                if prev is None or new_score > prev:
                    next_dp[new_mask] = new_score
        dp = next_dp
    return max(dp.values()) if dp else 0.0


def soft_set_metrics(
    pred_names: List[str],
    gold_names: List[str],
    similarity_lookup: Dict[str, Dict[str, float]],
) -> Tuple[float, float, float, float]:
    pred = list(dict.fromkeys(pred_names))
    gold = list(dict.fromkeys(gold_names))
    if not pred or not gold:
        return 0.0, 0.0, 0.0, 0.0
    if len(pred) <= len(gold):
        match_score = _best_soft_match_score(pred, gold, similarity_lookup)
    else:
        reverse_lookup = {src: {dst: similarity_lookup[dst][src] for dst in pred} for src in gold}
        match_score = _best_soft_match_score(gold, pred, reverse_lookup)
    soft_precision = match_score / max(1, len(pred))
    soft_recall = match_score / max(1, len(gold))
    soft_f1 = 0.0 if (soft_precision + soft_recall) == 0 else 2.0 * soft_precision * soft_recall / (soft_precision + soft_recall)
    return soft_precision, soft_recall, soft_f1, match_score
