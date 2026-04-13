from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_WAR_COST_TIERS = "agent_cost_tiers_12.json"

from agents import Agent, get_agents as get_all_agents  # type: ignore
from dataset import load_dataset_csv  # type: ignore
from embedder import Embedder, EmbedderConfig  # type: ignore
from routers import KNNRouter, MLRouter  # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC

from routing_utils import build_vector_store, choose_pred_set, hard_set_metrics


AGENT12_NAMES = [
    "TimeSeriesQueryAgent",
    "SQLQueryAgent",
    "APIDataFetchAgent",
    "LogRetrievalAgent",
    "MetadataLookupAgent",
    "StatisticalAnalysisAgent",
    "TrendAnalysisAgent",
    "AnomalyDetectionAgent",
    "ForecastAgent",
    "PlotGenerationAgent",
    "SummaryAgent",
    "ReportWriterAgent",
]


def get_agent12_list() -> List[Agent]:
    names = set(AGENT12_NAMES)
    return [agent for agent in get_all_agents() if agent.name in names]


def _camel_to_words(name: str) -> List[str]:
    out = []
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


def _agent_desc_texts(agents: List[Agent]):
    texts = []
    labels = []
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
        for text in chunks:
            texts.append(text)
            labels.append(agent.name)
    return texts, labels


def _agent_profile_texts(agents: List[Agent]):
    texts = []
    for agent in agents:
        name_words = _camel_to_words(agent.name.replace("Agent", ""))
        keywords = " ".join(word.lower() for word in name_words)
        name_hint = " ".join(name_words)
        texts.append(
            " | ".join(
                [
                    agent.description,
                    f"Agent name: {agent.name}. Capabilities: {agent.description}",
                    f"Use this agent when user asks about: {agent.name.replace('Agent', '').lower()} tasks.",
                    f"Keywords: {keywords}",
                    f"Task intent: {name_hint}",
                ]
            )
        )
    return texts


def _build_profile_examples(agents: List[Agent]):
    texts = []
    labels = []
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
        for text in chunks:
            texts.append(text)
            labels.append(agent.name)
    return texts, labels


@dataclass
class SetMetrics:
    prec: float
    rec: float
    f1: float
    jacc: float
    exact: float
    avg_p: float



def _init_sums() -> Dict[str, float]:
    return {"prec": 0.0, "rec": 0.0, "f1": 0.0, "jacc": 0.0, "exact": 0.0, "avg_p": 0.0}


def _update_sums(sums: Dict[str, float], pred_set: List[str], gold: List[str]) -> None:
    p_s, r_s, f_s, j_s, e_s = hard_set_metrics(pred_set, gold)
    sums["prec"] += p_s
    sums["rec"] += r_s
    sums["f1"] += f_s
    sums["jacc"] += j_s
    sums["exact"] += e_s
    sums["avg_p"] += len(pred_set)


def _finalize_sums(sums: Dict[str, float], n: int) -> SetMetrics:
    if n <= 0:
        return SetMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return SetMetrics(
        prec=sums["prec"] / n,
        rec=sums["rec"] / n,
        f1=sums["f1"] / n,
        jacc=sums["jacc"] / n,
        exact=sums["exact"] / n,
        avg_p=sums["avg_p"] / n,
    )


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _fmt_pct(mean: float, std: float) -> str:
    return f"{100.0 * mean:6.2f}±{100.0 * std:5.2f}%"


def _fmt_val(mean: float, std: float) -> str:
    return f"{mean:5.2f}±{std:4.2f}"


def train_ml_router(
    train_data,
    agents: List[Agent],
    embedder_model: Optional[str],
    use_profile_text: bool,
    seed: int,
) -> Tuple[Embedder, MLRouter]:
    rng = random.Random(seed)
    train_items = list(train_data)
    rng.shuffle(train_items)

    agent_names = [agent.name for agent in agents]
    name_to_idx = {name: i for i, name in enumerate(agent_names)}
    prompts = [ex.prompt for ex in train_items]

    y_multi = np.zeros((len(train_items), len(agent_names)), dtype=int)
    for i, ex in enumerate(train_items):
        for gold in ex.gold_agents:
            y_multi[i, name_to_idx[gold]] = 1

    embedder_cfg = EmbedderConfig()
    if embedder_model:
        embedder_cfg.model_name = embedder_model
    embedder = Embedder(embedder_cfg)

    profile_texts = _agent_profile_texts(agents) if use_profile_text else [agent.name for agent in agents]
    if embedder.backend[0] == "tfidf":
        fit_texts = list(prompts) + [agent.description for agent in agents] + profile_texts
        embedder.fit(fit_texts)

    X_train = embedder.encode(prompts)

    if use_profile_text:
        desc_texts, desc_labels = _agent_desc_texts(agents)
        desc_pairs = list(zip(desc_texts, desc_labels))
        rng.shuffle(desc_pairs)
        desc_texts = [text for text, _ in desc_pairs]
        desc_labels = [label for _, label in desc_pairs]
        X_desc = embedder.encode(desc_texts)
        y_desc = np.zeros((len(desc_labels), len(agent_names)), dtype=int)
        for i, name in enumerate(desc_labels):
            y_desc[i, name_to_idx[name]] = 1
        X_train = np.vstack([X_train, X_desc])
        y_multi = np.vstack([y_multi, y_desc])

    clf = OneVsRestClassifier(
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed)
    )
    clf.fit(X_train, y_multi)

    return embedder, MLRouter(clf, agent_names, top_k=len(agent_names))


def _to_candidates(scores: np.ndarray, class_names: List[str]) -> List[Tuple[str, float]]:
    pairs = list(zip(class_names, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


class MLkNN:
    def __init__(self, k: int = 10, smoothing: float = 1.0):
        self.k = int(k)
        self.smoothing = float(smoothing)
        self._nn: Optional[NearestNeighbors] = None
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._prior_pos: Optional[np.ndarray] = None
        self._prior_neg: Optional[np.ndarray] = None
        self._cond_pos: Optional[np.ndarray] = None
        self._cond_neg: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if Y.ndim != 2:
            raise ValueError("Y must be 2D")
        n, num_labels = Y.shape
        self._X = X
        self._Y = Y
        self._nn = NearestNeighbors(n_neighbors=min(self.k + 1, n), metric="cosine")
        self._nn.fit(X)

        _, neighbors = self._nn.kneighbors(X, return_distance=True)
        if neighbors.shape[1] > 1:
            neighbors = neighbors[:, 1:self.k + 1]
        else:
            neighbors = neighbors[:, :1]

        neighbor_counts = Y[neighbors].sum(axis=1).astype(int)

        s = self.smoothing
        prior_pos = np.zeros(num_labels, dtype=np.float64)
        prior_neg = np.zeros(num_labels, dtype=np.float64)
        cond_pos = np.zeros((num_labels, self.k + 1), dtype=np.float64)
        cond_neg = np.zeros((num_labels, self.k + 1), dtype=np.float64)

        for label_idx in range(num_labels):
            y_l = Y[:, label_idx].astype(int)
            pos_mask = y_l == 1
            neg_mask = ~pos_mask
            pos_count = int(pos_mask.sum())
            neg_count = int(neg_mask.sum())

            prior_pos[label_idx] = (pos_count + s) / (n + 2.0 * s)
            prior_neg[label_idx] = (neg_count + s) / (n + 2.0 * s)

            pos_counts = np.bincount(neighbor_counts[pos_mask, label_idx], minlength=self.k + 1)
            neg_counts = np.bincount(neighbor_counts[neg_mask, label_idx], minlength=self.k + 1)

            cond_pos[label_idx] = (pos_counts + s) / (pos_count + (self.k + 1) * s)
            cond_neg[label_idx] = (neg_counts + s) / (neg_count + (self.k + 1) * s)

        self._prior_pos = prior_pos
        self._prior_neg = prior_neg
        self._cond_pos = cond_pos
        self._cond_neg = cond_neg

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None or self._X is None or self._Y is None:
            raise RuntimeError("MLkNN not fitted.")
        k = min(self.k, self._X.shape[0])
        _, neighbors = self._nn.kneighbors(X, n_neighbors=k, return_distance=True)
        neighbors = neighbors[:, :k]
        neighbor_counts = self._Y[neighbors].sum(axis=1).astype(int)

        prior_pos = self._prior_pos
        prior_neg = self._prior_neg
        cond_pos = self._cond_pos
        cond_neg = self._cond_neg
        if prior_pos is None or prior_neg is None or cond_pos is None or cond_neg is None:
            raise RuntimeError("MLkNN parameters not initialized.")

        n_samples, num_labels = neighbor_counts.shape
        probs = np.zeros((n_samples, num_labels), dtype=np.float64)
        for label_idx in range(num_labels):
            counts = neighbor_counts[:, label_idx].clip(0, self.k)
            p1 = prior_pos[label_idx] * cond_pos[label_idx][counts]
            p0 = prior_neg[label_idx] * cond_neg[label_idx][counts]
            denom = p1 + p0 + 1e-12
            probs[:, label_idx] = p1 / denom
        return probs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PromptDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def _collate_fn(batch):
    texts = [x[0] for x in batch]
    labels = torch.stack([x[1] for x in batch], dim=0)
    return texts, labels


class FineTunedEncoderRouter(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(model_name, local_files_only=True)
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, texts: List[str]) -> torch.Tensor:
        features = self.encoder.tokenize(texts)
        device = next(self.classifier.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}
        outputs = self.encoder(features)
        embeddings = outputs["sentence_embedding"]
        logits = self.classifier(self.dropout(embeddings))
        return logits


def train_encoder(
    train_data,
    agents: List[Agent],
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    use_profile_text: bool,
    seed: int,
    device: torch.device,
):
    _set_seed(seed)
    class_names = [agent.name for agent in agents]
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    texts = [ex.prompt for ex in train_data]
    y = np.zeros((len(texts), len(class_names)), dtype=np.float32)
    for i, ex in enumerate(train_data):
        for gold in ex.gold_agents:
            y[i, name_to_idx[gold]] = 1.0

    if use_profile_text:
        profile_texts, profile_labels = _build_profile_examples(agents)
        y_profile = np.zeros((len(profile_texts), len(class_names)), dtype=np.float32)
        for i, name in enumerate(profile_labels):
            y_profile[i, name_to_idx[name]] = 1.0
        texts = texts + profile_texts
        y = np.vstack([y, y_profile])

    ds = PromptDataset(texts, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)

    model = FineTunedEncoderRouter(model_name=model_name, num_labels=len(class_names))
    model.to(device)

    pos_counts = y.sum(axis=0)
    neg_counts = y.shape[0] - pos_counts
    pos_weight = torch.tensor(
        np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1.0), 1.0),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_texts, batch_labels in loader:
            labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        print(f"[encoder train] seed={seed} epoch {epoch}/{epochs} done", flush=True)

    model.eval()
    return model, class_names


def encoder_predict_probs(
    model: FineTunedEncoderRouter,
    texts: List[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.to(device)
    ds = PromptDataset(texts, np.zeros((len(texts), 1), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    probs_all = []
    with torch.no_grad():
        for batch_texts, _ in loader:
            logits = model(batch_texts)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
    return np.vstack(probs_all)


def encoder_metrics_from_probs(
    class_names: List[str],
    probs_all: np.ndarray,
    gold_sets: List[List[str]],
    threshold: float,
) -> SetMetrics:
    sums = _init_sums()
    for probs, gold in zip(probs_all, gold_sets):
        pred_set = [name for name, p in zip(class_names, probs) if float(p) >= threshold]
        if not pred_set:
            pred_set = [class_names[int(np.argmax(probs))]]
        _update_sums(sums, pred_set, gold)
    return _finalize_sums(sums, len(gold_sets))


def _filter_compatible(data, agent_names: set) -> List:
    return [ex for ex in data if ex.gold_agents and all(g in agent_names for g in ex.gold_agents)]


def _majority_agent(data) -> Optional[str]:
    counts: Dict[str, int] = {}
    for ex in data:
        for gold in ex.gold_agents:
            counts[gold] = counts.get(gold, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]


def _build_label_matrix(data, name_to_idx: Dict[str, int]) -> np.ndarray:
    y = np.zeros((len(data), len(name_to_idx)), dtype=int)
    for i, ex in enumerate(data):
        for gold in ex.gold_agents:
            idx = name_to_idx.get(gold)
            if idx is not None:
                y[i, idx] = 1
    return y


def _build_seed_items(
    test_data,
    embedder: Embedder,
    ml_router: MLRouter,
    knn_router: Optional[KNNRouter] = None,
    extra_cands: Optional[Dict[str, List[List[Tuple[str, float]]]]] = None,
) -> List[Dict[str, object]]:
    prompts = [ex.prompt for ex in test_data]
    embs = embedder.encode(prompts)
    items: List[Dict[str, object]] = []
    for idx, (ex, emb) in enumerate(zip(test_data, embs)):
        r_ml = ml_router.route(emb)
        item = {
            "gold": ex.gold_agents,
            "emb": emb,
            "ml_cands": r_ml.candidates,
        }
        if knn_router is not None:
            r_knn = knn_router.route(emb)
            item["knn_cands"] = r_knn.candidates
        if extra_cands:
            for key, cand_list in extra_cands.items():
                if idx < len(cand_list):
                    item[key] = cand_list[idx]
                else:
                    item[key] = []
        items.append(item)
    return items


def evaluate_threshold(
    items: List[Dict[str, object]],
    threshold: float,
    majority_agent: Optional[str],
    include_knn: bool = True,
    include_cc: bool = True,
    include_mlknn: bool = True,
    include_majority: bool = True,
):
    sums = {
        "knn": _init_sums(),
        "ml": _init_sums(),
        "cc": _init_sums(),
        "mlknn": _init_sums(),
        "maj": _init_sums(),
    }
    for item in items:
        gold = item["gold"]
        if include_knn and "knn_cands" in item:
            knn_pred = choose_pred_set(item["knn_cands"], threshold)
            _update_sums(sums["knn"], knn_pred, gold)

        ml_pred = choose_pred_set(item["ml_cands"], threshold)
        _update_sums(sums["ml"], ml_pred, gold)

        if include_cc and "cc_cands" in item:
            cc_pred = choose_pred_set(item["cc_cands"], threshold)
            _update_sums(sums["cc"], cc_pred, gold)
        if include_mlknn and "mlknn_cands" in item:
            mlknn_pred = choose_pred_set(item["mlknn_cands"], threshold)
            _update_sums(sums["mlknn"], mlknn_pred, gold)

        if include_majority and majority_agent is not None:
            _update_sums(sums["maj"], [majority_agent], gold)

    n = len(items)
    return {
        "knn": _finalize_sums(sums["knn"], n),
        "ml": _finalize_sums(sums["ml"], n),
        "cc": _finalize_sums(sums["cc"], n),
        "mlknn": _finalize_sums(sums["mlknn"], n),
        "maj": _finalize_sums(sums["maj"], n),
    }


def _write_threshold_seed_csv(
    path: Path,
    rows: Iterable[Dict[str, object]],
):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _persist_partial_outputs(
    threshold_csv: Path,
    test_csv: Path,
    threshold_seed_rows: Sequence[Dict[str, object]],
    test_seed_rows: Sequence[Dict[str, object]],
) -> None:
    if threshold_seed_rows:
        _write_threshold_seed_csv(threshold_csv, threshold_seed_rows)
    if test_seed_rows:
        _write_threshold_seed_csv(test_csv, test_seed_rows)


def _aggregate_metrics(metrics_list: Sequence[SetMetrics]) -> Dict[str, Tuple[float, float]]:
    return {
        "prec": _mean_std([m.prec for m in metrics_list]),
        "rec": _mean_std([m.rec for m in metrics_list]),
        "f1": _mean_std([m.f1 for m in metrics_list]),
        "jacc": _mean_std([m.jacc for m in metrics_list]),
        "exact": _mean_std([m.exact for m in metrics_list]),
        "avg_p": _mean_std([m.avg_p for m in metrics_list]),
    }


def _print_method_line(label: str, agg: Dict[str, Tuple[float, float]]) -> None:
    print(
        f"{label:<8} Prec: {_fmt_pct(*agg['prec'])} | Rec: {_fmt_pct(*agg['rec'])} | "
        f"F1: {_fmt_pct(*agg['f1'])} | Jacc: {_fmt_pct(*agg['jacc'])} | "
        f"Exact: {_fmt_pct(*agg['exact'])} | Avg|P|: {_fmt_val(*agg['avg_p'])}"
    )


def _print_method_line_single(label: str, metrics: SetMetrics) -> None:
    agg = {
        "prec": (metrics.prec, 0.0),
        "rec": (metrics.rec, 0.0),
        "f1": (metrics.f1, 0.0),
        "jacc": (metrics.jacc, 0.0),
        "exact": (metrics.exact, 0.0),
        "avg_p": (metrics.avg_p, 0.0),
    }
    _print_method_line(label, agg)


def _metrics_from_seed_row(row: Dict[str, object]) -> SetMetrics:
    return SetMetrics(
        prec=float(row["prec"]),
        rec=float(row["rec"]),
        f1=float(row["f1"]),
        jacc=float(row["jacc"]),
        exact=float(row["exact"]),
        avg_p=float(row["avg_p"]),
    )


def _cleanup_after_seed(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", default=None, help="Optional disjoint dev CSV for threshold selection")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--embedder_model", default="all-mpnet-base-v2")
    ap.add_argument("--no_profile_text", action="store_true")
    ap.add_argument("--encoder", action="store_true")
    ap.add_argument("--encoder_model_name", default="all-mpnet-base-v2")
    ap.add_argument("--encoder_epochs", type=int, default=3)
    ap.add_argument("--encoder_batch_size", type=int, default=16)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--encoder_weight_decay", type=float, default=1e-2)
    ap.add_argument("--encoder_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--skip_knn", action="store_true")
    ap.add_argument("--skip_cc", action="store_true")
    ap.add_argument("--skip_mlknn", action="store_true")
    ap.add_argument("--skip_majority", action="store_true")
    ap.add_argument("--output_prefix", default="agent12_full_experiment")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    use_profile_text = not args.no_profile_text
    output_prefix = Path(args.output_prefix)
    threshold_csv = output_prefix.with_name(output_prefix.name + "_threshold_seed_metrics.csv")
    test_csv = output_prefix.with_name(output_prefix.name + "_test_seed_metrics.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.md")
    run_knn = not args.skip_knn
    run_cc = not args.skip_cc
    run_mlknn = not args.skip_mlknn
    run_majority = not args.skip_majority

    agents = get_agent12_list()
    agent_names = [agent.name for agent in agents]
    agent_name_set = set(agent_names)
    train_data = _filter_compatible(load_dataset_csv(args.train_csv), agent_name_set)
    dev_data = None
    if args.dev_csv:
        dev_data = _filter_compatible(load_dataset_csv(args.dev_csv), agent_name_set)
    test_data = _filter_compatible(load_dataset_csv(args.test_csv), agent_name_set)
    if not train_data or not test_data or (args.dev_csv and not dev_data):
        raise RuntimeError("Train/test data is empty or incompatible with 12-agent inventory.")
    test_gold_sets = [ex.gold_agents for ex in test_data]


    threshold_seed_rows: List[Dict[str, object]] = []
    test_seed_rows: List[Dict[str, object]] = []

    threshold_results: Dict[float, Dict[str, List[SetMetrics]]] = {
        t: {"knn": [], "ml": [], "cc": [], "mlknn": [], "maj": [], "enc": []} for t in thresholds
    }

    seed_items_dev: Dict[int, List[Dict[str, object]]] = {}
    seed_items_test: Dict[int, List[Dict[str, object]]] = {}
    seed_majority_dev: Dict[int, Optional[str]] = {}
    seed_majority_test: Dict[int, Optional[str]] = {}
    seed_encoder_test_probs: Dict[int, np.ndarray] = {}
    seed_encoder_class_names: Dict[int, List[str]] = {}

    print("=== Training + Threshold Sweeps ===", flush=True)
    for seed in seeds:
        print(f"[seed {seed}] start", flush=True)
        embedder, ml_router = train_ml_router(
            train_data=train_data,
            agents=agents,
            embedder_model=args.embedder_model,
            use_profile_text=use_profile_text,
            seed=seed,
        )
        print(f"[seed {seed}] trained embedder + ML router", flush=True)
        store = None
        id_to_name = None
        knn_router = None
        if run_knn:
            store, id_to_name = build_vector_store(agents, embedder)
            knn_router = KNNRouter(agents, store, id_to_name, top_k=len(agents))

        name_to_idx = {name: i for i, name in enumerate(agent_names)}
        train_prompts = [ex.prompt for ex in train_data]
        X_train = np.asarray(embedder.encode(train_prompts))
        if dev_data is not None:
            dev_prompts = [ex.prompt for ex in dev_data]
            X_dev = np.asarray(embedder.encode(dev_prompts))
        else:
            X_dev = None
        test_prompts = [ex.prompt for ex in test_data]
        X_test = np.asarray(embedder.encode(test_prompts))
        y_train = _build_label_matrix(train_data, name_to_idx)

        cc_model = None
        cc_probs_dev = None
        cc_probs = None
        if run_cc:
            cc_base = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
            )
            cc_model = ClassifierChain(cc_base, order="random", random_state=seed)
            cc_model.fit(X_train, y_train)
            print(f"[seed {seed}] trained classifier chain", flush=True)
            if X_dev is not None:
                cc_probs_dev = cc_model.predict_proba(X_dev)
                if isinstance(cc_probs_dev, list):
                    cc_probs_dev = np.vstack([p[:, 1] for p in cc_probs_dev]).T
                elif cc_probs_dev.ndim == 3:
                    cc_probs_dev = cc_probs_dev[:, :, 1]
            cc_probs = cc_model.predict_proba(X_test)
            if isinstance(cc_probs, list):
                cc_probs = np.vstack([p[:, 1] for p in cc_probs]).T
            elif cc_probs.ndim == 3:
                cc_probs = cc_probs[:, :, 1]

        mlknn = None
        mlknn_probs_dev = None
        mlknn_probs = None
        if run_mlknn:
            mlknn = MLkNN(k=10, smoothing=1.0)
            mlknn.fit(X_train, y_train)
            print(f"[seed {seed}] trained MLkNN", flush=True)
            mlknn_probs_dev = mlknn.predict_proba(X_dev) if X_dev is not None else None
            mlknn_probs = mlknn.predict_proba(X_test)

        cc_cands_test = [_to_candidates(scores, agent_names) for scores in cc_probs] if cc_probs is not None else None
        mlknn_cands_test = [_to_candidates(scores, agent_names) for scores in mlknn_probs] if mlknn_probs is not None else None

        cc_cands_dev = None
        mlknn_cands_dev = None
        if cc_probs_dev is not None:
            cc_cands_dev = [_to_candidates(scores, agent_names) for scores in cc_probs_dev]
        if mlknn_probs_dev is not None:
            mlknn_cands_dev = [_to_candidates(scores, agent_names) for scores in mlknn_probs_dev]

        majority_agent_test = _majority_agent(test_data) if run_majority else None
        seed_majority_test[seed] = majority_agent_test
        majority_agent_dev = (_majority_agent(dev_data) if dev_data is not None else majority_agent_test) if run_majority else None
        seed_majority_dev[seed] = majority_agent_dev

        extra_cands_test: Dict[str, List[List[Tuple[str, float]]]] = {}
        if cc_cands_test is not None:
            extra_cands_test["cc_cands"] = cc_cands_test
        if mlknn_cands_test is not None:
            extra_cands_test["mlknn_cands"] = mlknn_cands_test

        items_test = _build_seed_items(
            test_data,
            embedder,
            ml_router,
            knn_router,
            extra_cands=extra_cands_test or None,
        )
        seed_items_test[seed] = items_test

        if dev_data is not None:
            extra_cands_dev: Dict[str, List[List[Tuple[str, float]]]] = {}
            if cc_cands_dev is not None:
                extra_cands_dev["cc_cands"] = cc_cands_dev
            if mlknn_cands_dev is not None:
                extra_cands_dev["mlknn_cands"] = mlknn_cands_dev
            items_dev = _build_seed_items(
                dev_data,
                embedder,
                ml_router,
                knn_router,
                extra_cands=extra_cands_dev or None,
            )
        else:
            items_dev = items_test
        seed_items_dev[seed] = items_dev
        print(f"[seed {seed}] built routed candidate sets", flush=True)

        for t in thresholds:
            metrics = evaluate_threshold(
                items_dev,
                threshold=t,
                majority_agent=majority_agent_dev,
                include_knn=run_knn,
                include_cc=run_cc,
                include_mlknn=run_mlknn,
                include_majority=run_majority,
            )
            if run_knn:
                threshold_results[t]["knn"].append(metrics["knn"])
            threshold_results[t]["ml"].append(metrics["ml"])
            if run_cc:
                threshold_results[t]["cc"].append(metrics["cc"])
            if run_mlknn:
                threshold_results[t]["mlknn"].append(metrics["mlknn"])
            if run_majority:
                threshold_results[t]["maj"].append(metrics["maj"])

            if run_knn:
                threshold_seed_rows.append(
                    {
                        "seed": seed,
                        "threshold": t,
                        "method": "KNN",
                        **vars(metrics["knn"]),
                    }
                )
            threshold_seed_rows.append(
                {
                    "seed": seed,
                    "threshold": t,
                    "method": "ML",
                    **vars(metrics["ml"]),
                }
            )
            if run_cc:
                threshold_seed_rows.append(
                    {
                        "seed": seed,
                        "threshold": t,
                        "method": "CC",
                        **vars(metrics["cc"]),
                    }
                )
            if run_mlknn:
                threshold_seed_rows.append(
                    {
                        "seed": seed,
                        "threshold": t,
                        "method": "MLkNN",
                        **vars(metrics["mlknn"]),
                    }
                )
            if run_majority:
                threshold_seed_rows.append(
                    {
                        "seed": seed,
                        "threshold": t,
                        "method": "Majority",
                        **vars(metrics["maj"]),
                    }
                )

        if args.encoder:
            if args.encoder_device == "cpu":
                enc_device = torch.device("cpu")
            elif args.encoder_device == "cuda":
                enc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if enc_device.type != "cuda":
                    print("[warn] --encoder_device cuda requested but CUDA not available; using CPU.")
            else:
                enc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, class_names = train_encoder(
                train_data=train_data,
                agents=agents,
                model_name=args.encoder_model_name,
                epochs=args.encoder_epochs,
                batch_size=args.encoder_batch_size,
                lr=args.encoder_lr,
                weight_decay=args.encoder_weight_decay,
                use_profile_text=use_profile_text,
                seed=seed,
                device=enc_device,
            )
            dev_texts = [ex.prompt for ex in dev_data] if dev_data is not None else None
            dev_gold = [ex.gold_agents for ex in dev_data] if dev_data is not None else None
            test_texts = [ex.prompt for ex in test_data]
            test_gold = [ex.gold_agents for ex in test_data]

            probs_dev = None
            if dev_texts is not None:
                probs_dev = encoder_predict_probs(
                    model, dev_texts, batch_size=args.encoder_batch_size, device=enc_device
                )
            probs_test = encoder_predict_probs(
                model, test_texts, batch_size=args.encoder_batch_size, device=enc_device
            )
            seed_encoder_test_probs[seed] = probs_test
            seed_encoder_class_names[seed] = class_names
            for t in thresholds:
                enc_probs = probs_dev if probs_dev is not None else probs_test
                enc_gold = dev_gold if dev_gold is not None else test_gold
                enc_metrics = encoder_metrics_from_probs(class_names, enc_probs, enc_gold, threshold=t)
                threshold_results[t]["enc"].append(enc_metrics)
                threshold_seed_rows.append(
                    {
                        "seed": seed,
                        "threshold": t,
                        "method": "Encoder",
                        **vars(enc_metrics),
                    }
                )
            _cleanup_after_seed(model, probs_dev, dev_texts, dev_gold, test_texts, test_gold)

        # Per-seed dev threshold sweep for quick verification
        seed_rows = [r for r in threshold_seed_rows if r["seed"] == seed]
        print(f"\n[seed {seed}] dev threshold sweep", flush=True)
        methods_for_seed = []
        if run_knn:
            methods_for_seed.append("KNN")
        methods_for_seed.append("ML")
        if run_cc:
            methods_for_seed.append("CC")
        if run_mlknn:
            methods_for_seed.append("MLkNN")
        if args.encoder:
            methods_for_seed.append("Encoder")
        if run_majority:
            methods_for_seed.append("Majority")

        for t in thresholds:
            print(f"Threshold = {t:.2f}", flush=True)
            for method in methods_for_seed:
                row = next(
                    (
                        r
                        for r in seed_rows
                        if r["method"] == method and float(r["threshold"]) == float(t)
                    ),
                    None,
                )
                if row is None:
                    continue
                _print_method_line_single(method, _metrics_from_seed_row(row))
        print(f"[seed {seed}] done", flush=True)

        _persist_partial_outputs(
            threshold_csv=threshold_csv,
            test_csv=test_csv,
            threshold_seed_rows=threshold_seed_rows,
            test_seed_rows=test_seed_rows,
        )
        _cleanup_after_seed(
            embedder,
            ml_router,
            knn_router,
            X_train,
            X_dev,
            X_test,
            y_train,
            cc_model,
            mlknn,
            cc_probs_dev,
            cc_probs,
            mlknn_probs_dev,
            mlknn_probs,
            cc_cands_test,
            mlknn_cands_test,
            cc_cands_dev,
            mlknn_cands_dev,
            items_dev,
            items_test,
        )

    # Find best ML threshold by mean F1
    best_threshold = None
    best_f1 = -1.0
    for t in thresholds:
        f1_mean, _ = _mean_std([m.f1 for m in threshold_results[t]["ml"]])
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_threshold = t
    if best_threshold is None:
        best_threshold = thresholds[0]

    sweep_split = "dev" if dev_data is not None else "test"
    print(f"\n=== Threshold Sweep Results ({sweep_split}, mean+-std over seeds) ===", flush=True)
    for t in thresholds:
        print(f"\nThreshold = {t:.2f}")
        if run_knn:
            _print_method_line("KNN", _aggregate_metrics(threshold_results[t]["knn"]))
        _print_method_line("ML", _aggregate_metrics(threshold_results[t]["ml"]))
        if run_cc:
            _print_method_line("CC", _aggregate_metrics(threshold_results[t]["cc"]))
        if run_mlknn:
            _print_method_line("MLkNN", _aggregate_metrics(threshold_results[t]["mlknn"]))
        if args.encoder:
            _print_method_line("Encoder", _aggregate_metrics(threshold_results[t]["enc"]))
        if run_majority:
            _print_method_line("Majority", _aggregate_metrics(threshold_results[t]["maj"]))

    print(f"\nBest ML threshold by mean F1: {best_threshold:.2f} (mean F1={100*best_f1:.2f}%)", flush=True)

    # Final test results at selected threshold
    print(f"\n=== Final Test Results (threshold={best_threshold:.2f}) ===", flush=True)
    test_results: Dict[str, List[SetMetrics]] = {"knn": [], "ml": [], "cc": [], "mlknn": [], "maj": [], "enc": []}
    for seed in seeds:
        items_test = seed_items_test[seed]
        majority_agent_test = seed_majority_test.get(seed)
        metrics = evaluate_threshold(
            items_test,
            threshold=best_threshold,
            majority_agent=majority_agent_test,
            include_knn=run_knn,
            include_cc=run_cc,
            include_mlknn=run_mlknn,
            include_majority=run_majority,
        )
        if run_knn:
            test_results["knn"].append(metrics["knn"])
        test_results["ml"].append(metrics["ml"])
        if run_cc:
            test_results["cc"].append(metrics["cc"])
        if run_mlknn:
            test_results["mlknn"].append(metrics["mlknn"])
        if run_majority:
            test_results["maj"].append(metrics["maj"])

        test_method_pairs = []
        if run_knn:
            test_method_pairs.append(("KNN", "knn"))
        test_method_pairs.append(("ML", "ml"))
        if run_cc:
            test_method_pairs.append(("CC", "cc"))
        if run_mlknn:
            test_method_pairs.append(("MLkNN", "mlknn"))
        if run_majority:
            test_method_pairs.append(("Majority", "maj"))
        for label, key in test_method_pairs:
            test_seed_rows.append(
                {
                    "seed": seed,
                    "threshold": best_threshold,
                    "method": label,
                    **vars(metrics[key]),
                }
            )

        if args.encoder:
            probs = seed_encoder_test_probs.get(seed)
            class_names = seed_encoder_class_names.get(seed, agent_names)
            if probs is not None:
                enc_metrics = encoder_metrics_from_probs(
                    class_names, probs, test_gold_sets, threshold=best_threshold
                )
                test_results["enc"].append(enc_metrics)
                test_seed_rows.append(
                    {
                        "seed": seed,
                        "threshold": best_threshold,
                        "method": "Encoder",
                        **vars(enc_metrics),
                    }
                )

    if run_knn:
        _print_method_line("KNN", _aggregate_metrics(test_results["knn"]))
    _print_method_line("ML", _aggregate_metrics(test_results["ml"]))
    if run_cc:
        _print_method_line("CC", _aggregate_metrics(test_results["cc"]))
    if run_mlknn:
        _print_method_line("MLkNN", _aggregate_metrics(test_results["mlknn"]))
    if args.encoder:
        _print_method_line("Encoder", _aggregate_metrics(test_results["enc"]))
    if run_majority:
        _print_method_line("Majority", _aggregate_metrics(test_results["maj"]))

    # Summary files
    _write_threshold_seed_csv(threshold_csv, threshold_seed_rows)
    _write_threshold_seed_csv(test_csv, test_seed_rows)

    summary_lines = [
        "# Full experiment summary",
        "",
        f"- train_csv: `{args.train_csv}`",
        f"- dev_csv: `{args.dev_csv}`" if args.dev_csv else "- dev_csv: (none)",
        f"- test_csv: `{args.test_csv}`",
        f"- seeds: {seeds}",
        f"- thresholds: {thresholds}",
        f"- best_ml_threshold: {best_threshold}",
        f"- run_knn: {run_knn}",
        f"- run_cc: {run_cc}",
        f"- run_mlknn: {run_mlknn}",
        f"- run_majority: {run_majority}",
        "",
        f"## Threshold sweep ({sweep_split}, mean±std)",
    ]
    for t in thresholds:
        summary_lines.append(f"\n### Threshold = {t:.2f}")
        summary_method_pairs = []
        if run_knn:
            summary_method_pairs.append(("KNN", "knn"))
        summary_method_pairs.append(("ML", "ml"))
        if run_cc:
            summary_method_pairs.append(("CC", "cc"))
        if run_mlknn:
            summary_method_pairs.append(("MLkNN", "mlknn"))
        if run_majority:
            summary_method_pairs.append(("Majority", "maj"))
        for label, key in summary_method_pairs:
            agg = _aggregate_metrics(threshold_results[t][key])
            summary_lines.append(
                f"- {label} F1: {100*agg['f1'][0]:.2f}% ± {100*agg['f1'][1]:.2f} | "
                f"Jacc: {100*agg['jacc'][0]:.2f}% ± {100*agg['jacc'][1]:.2f}"
            )
        if args.encoder:
            agg = _aggregate_metrics(threshold_results[t]["enc"])
            summary_lines.append(
                f"- Encoder F1: {100*agg['f1'][0]:.2f}% ± {100*agg['f1'][1]:.2f} | "
                f"Jacc: {100*agg['jacc'][0]:.2f}% ± {100*agg['jacc'][1]:.2f}"
            )

    summary_lines.append("\n## Final test results @ best threshold (mean+/-std)")
    final_summary_pairs = []
    if run_knn:
        final_summary_pairs.append(("KNN", "knn"))
    final_summary_pairs.append(("ML", "ml"))
    if run_cc:
        final_summary_pairs.append(("CC", "cc"))
    if run_mlknn:
        final_summary_pairs.append(("MLkNN", "mlknn"))
    if run_majority:
        final_summary_pairs.append(("Majority", "maj"))
    for label, key in final_summary_pairs:
        agg = _aggregate_metrics(test_results[key])
        summary_lines.append(
            f"- {label} F1: {100*agg['f1'][0]:.2f}% ± {100*agg['f1'][1]:.2f} | "
            f"Jacc: {100*agg['jacc'][0]:.2f}% ± {100*agg['jacc'][1]:.2f}"
        )
    if args.encoder:
        agg = _aggregate_metrics(test_results["enc"])
        summary_lines.append(
            f"- Encoder F1: {100*agg['f1'][0]:.2f}% ± {100*agg['f1'][1]:.2f} | "
            f"Jacc: {100*agg['jacc'][0]:.2f}% ± {100*agg['jacc'][1]:.2f}"
        )

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\nSaved threshold seed metrics to {threshold_csv}")
    print(f"Saved test seed metrics to {test_csv}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
