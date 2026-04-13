from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import load_dataset_csv  # type: ignore
from routers import KNNRouter  # type: ignore
from routing_utils import build_vector_store, choose_pred_set, hard_set_metrics  # type: ignore

from agent12_full_experiment import (  # type: ignore
    MLkNN,
    _build_label_matrix,
    _build_seed_items,
    _filter_compatible,
    _majority_agent,
    _set_seed,
    _to_candidates,
    encoder_predict_probs,
    get_agent12_list,
    train_encoder,
    train_ml_router,
)


SUPPORTED_METHODS = {"majority", "knn", "ml", "cc", "mlknn", "encoder"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export per-prompt prediction CSVs for downstream utility simulation."
    )
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--methods", default="majority,knn,ml,cc,mlknn,encoder")
    ap.add_argument("--embedder_model", default="all-mpnet-base-v2")
    ap.add_argument("--no_profile_text", action="store_true")
    ap.add_argument("--encoder_model_name", default="all-mpnet-base-v2")
    ap.add_argument("--encoder_epochs", type=int, default=3)
    ap.add_argument("--encoder_batch_size", type=int, default=8)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--encoder_weight_decay", type=float, default=1e-2)
    ap.add_argument("--encoder_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output_dir", default="outputs/predictions_agent12")
    return ap.parse_args()


def parse_methods(raw: str) -> List[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in methods if m not in SUPPORTED_METHODS]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Supported: {sorted(SUPPORTED_METHODS)}")
    return methods


def parse_agent_set(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    sep = "|" if "|" in raw else ","
    return [part.strip() for part in raw.split(sep) if part.strip()]


def compatible_raw_rows(path: str, agent_name_set: set[str]) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    filtered: List[Dict[str, str]] = []
    for row in rows:
        gold = parse_agent_set(row.get("gold_agents", ""))
        if gold and all(g in agent_name_set for g in gold):
            filtered.append(row)
    return filtered


def normalize_proba_matrix(probs) -> np.ndarray:
    arr = probs
    if isinstance(arr, list):
        arr = np.vstack([p[:, 1] for p in arr]).T
    else:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
    return np.asarray(arr, dtype=float)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_rows(
    seed: int,
    raw_rows: Sequence[Dict[str, str]],
    gold_sets: Sequence[Sequence[str]],
    pred_sets: Sequence[Sequence[str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row, gold, pred in zip(raw_rows, gold_sets, pred_sets):
        precision, recall, f1, jaccard, exact = hard_set_metrics(list(pred), list(gold))
        rows.append(
            {
                "seed": str(seed),
                "prompt_id": row.get("prompt_id", ""),
                "prompt": row.get("prompt", ""),
                "gold_agents": "|".join(gold),
                "pred_agents": "|".join(pred),
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
                "jaccard": f"{jaccard:.6f}",
                "exact": f"{exact:.6f}",
            }
        )
    return rows


def write_prediction_file(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    fieldnames = [
        "seed",
        "prompt_id",
        "prompt",
        "gold_agents",
        "pred_agents",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "exact",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def from_candidates(items: Sequence[Dict[str, object]], key: str, threshold: float) -> List[List[str]]:
    return [choose_pred_set(item.get(key, []), threshold) for item in items]


def encoder_predictions(
    probs: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> List[List[str]]:
    preds: List[List[str]] = []
    for row in probs:
        pred = [name for name, score in zip(class_names, row) if float(score) >= threshold]
        if not pred:
            pred = [class_names[int(np.argmax(row))]]
        preds.append(list(dict.fromkeys(pred)))
    return preds


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = parse_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agents = get_agent12_list()
    agent_names = [agent.name for agent in agents]
    agent_name_set = set(agent_names)
    use_profile_text = not args.no_profile_text

    train_all = _filter_compatible(load_dataset_csv(args.train_csv), agent_name_set)
    test_all = _filter_compatible(load_dataset_csv(args.test_csv), agent_name_set)
    raw_rows = compatible_raw_rows(args.test_csv, agent_name_set)
    if len(raw_rows) != len(test_all):
        raise RuntimeError("Filtered raw rows and compatible test examples differ in length.")

    combined_rows: Dict[str, List[Dict[str, str]]] = {method: [] for method in methods}

    for seed in seeds:
        _set_seed(seed)
        print(f"[seed {seed}] train ML/KNN backbone", flush=True)
        embedder, ml_router = train_ml_router(
            train_data=train_all,
            agents=agents,
            embedder_model=args.embedder_model,
            use_profile_text=use_profile_text,
            seed=seed,
        )
        store, id_to_name = build_vector_store(agents, embedder)
        knn_router = KNNRouter(agents, store, id_to_name, top_k=len(agents))

        X_train = np.asarray(embedder.encode([ex.prompt for ex in train_all]))
        X_test = np.asarray(embedder.encode([ex.prompt for ex in test_all]))
        name_to_idx = {name: i for i, name in enumerate(agent_names)}
        y_train = _build_label_matrix(train_all, name_to_idx)

        extra_cands: Dict[str, List[List[Tuple[str, float]]]] = {}
        if "cc" in methods:
            print(f"[seed {seed}] train CC", flush=True)
            cc_base = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
            )
            cc_model = ClassifierChain(cc_base, order="random", random_state=seed)
            cc_model.fit(X_train, y_train)
            cc_probs = normalize_proba_matrix(cc_model.predict_proba(X_test))
            extra_cands["cc_cands"] = [_to_candidates(scores, agent_names) for scores in cc_probs]

        if "mlknn" in methods:
            print(f"[seed {seed}] train MLkNN", flush=True)
            mlknn = MLkNN(k=10, smoothing=1.0)
            mlknn.fit(X_train, y_train)
            mlknn_probs = mlknn.predict_proba(X_test)
            extra_cands["mlknn_cands"] = [_to_candidates(scores, agent_names) for scores in mlknn_probs]

        items = _build_seed_items(
            test_all,
            embedder,
            ml_router,
            knn_router,
            extra_cands=extra_cands or None,
        )
        gold_sets = [ex.gold_agents for ex in test_all]

        if "majority" in methods:
            majority = _majority_agent(test_all)
            preds = [[majority] if majority else [] for _ in test_all]
            combined_rows["majority"].extend(prediction_rows(seed, raw_rows, gold_sets, preds))

        if "knn" in methods:
            preds = from_candidates(items, "knn_cands", args.threshold)
            combined_rows["knn"].extend(prediction_rows(seed, raw_rows, gold_sets, preds))

        if "ml" in methods:
            preds = from_candidates(items, "ml_cands", args.threshold)
            combined_rows["ml"].extend(prediction_rows(seed, raw_rows, gold_sets, preds))

        if "cc" in methods:
            preds = from_candidates(items, "cc_cands", args.threshold)
            combined_rows["cc"].extend(prediction_rows(seed, raw_rows, gold_sets, preds))

        if "mlknn" in methods:
            preds = from_candidates(items, "mlknn_cands", args.threshold)
            combined_rows["mlknn"].extend(prediction_rows(seed, raw_rows, gold_sets, preds))

        if "encoder" in methods:
            print(f"[seed {seed}] train Encoder", flush=True)
            device = pick_device(args.encoder_device)
            model, class_names = train_encoder(
                train_data=train_all,
                agents=agents,
                model_name=args.encoder_model_name,
                epochs=args.encoder_epochs,
                batch_size=args.encoder_batch_size,
                lr=args.encoder_lr,
                weight_decay=args.encoder_weight_decay,
                use_profile_text=use_profile_text,
                seed=seed,
                device=device,
            )
            probs = encoder_predict_probs(
                model,
                [ex.prompt for ex in test_all],
                batch_size=args.encoder_batch_size,
                device=device,
            )
            preds = encoder_predictions(probs, class_names, args.threshold)
            combined_rows["encoder"].extend(prediction_rows(seed, raw_rows, gold_sets, preds))

    for method, rows in combined_rows.items():
        path = output_dir / f"{method}_predictions.csv"
        write_prediction_file(path, rows)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
