from __future__ import annotations

import argparse
import csv
import sys
import time
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
from routing_utils import build_vector_store, choose_pred_set  # type: ignore

from agent12_full_experiment import (  # type: ignore
    MLkNN,
    _build_label_matrix,
    _filter_compatible,
    _set_seed,
    _to_candidates,
    encoder_predict_probs,
    get_agent12_list,
    train_encoder,
    train_ml_router,
)


SUPPORTED_METHODS = {"majority", "knn", "ml", "cc", "mlknn", "encoder"}


def parse_methods(raw: str) -> List[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in methods if m not in SUPPORTED_METHODS]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Supported: {sorted(SUPPORTED_METHODS)}")
    return methods


def summarize(times_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def time_loop(fn, items, warmup: int = 5) -> List[float]:
    for i in range(min(warmup, len(items))):
        fn(items[i])
    times: List[float] = []
    for item in items:
        t0 = time.perf_counter()
        fn(item)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure per-query routing latency for the 12-agent benchmark."
    )
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--seeds", default=None, help="Comma-separated seeds. Overrides --seed.")
    ap.add_argument("--seed", type=int, default=42, help="Single seed used when --seeds is omitted.")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--methods", default="majority,knn,ml,cc,mlknn,encoder")
    ap.add_argument("--embedder_model", default="all-mpnet-base-v2")
    ap.add_argument("--no_profile_text", action="store_true")
    ap.add_argument("--encoder_model_name", default="all-mpnet-base-v2")
    ap.add_argument("--encoder_epochs", type=int, default=3)
    ap.add_argument("--encoder_batch_size", type=int, default=16)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--encoder_weight_decay", type=float, default=1e-2)
    ap.add_argument("--encoder_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output_csv", default="outputs/latency_agent12_results.csv")
    args = ap.parse_args()

    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else [args.seed]
    )
    methods = parse_methods(args.methods)

    agents = get_agent12_list()
    agent_names = [agent.name for agent in agents]
    agent_name_set = set(agent_names)

    train_data = _filter_compatible(load_dataset_csv(args.train_csv), agent_name_set)
    test_data = _filter_compatible(load_dataset_csv(args.test_csv), agent_name_set)
    if not train_data or not test_data:
        raise RuntimeError("Train/test data is empty or incompatible with 12-agent inventory.")

    use_profile_text = not args.no_profile_text
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        _set_seed(seed)
        print(f"[seed {seed}] train ML/KNN backbone", flush=True)
        embedder, ml_router = train_ml_router(
            train_data=train_data,
            agents=agents,
            embedder_model=args.embedder_model,
            use_profile_text=use_profile_text,
            seed=seed,
        )
        store, id_to_name = build_vector_store(agents, embedder)
        knn_router = KNNRouter(agents, store, id_to_name, top_k=len(agents))

        X_train = np.asarray(embedder.encode([ex.prompt for ex in train_data]))
        name_to_idx = {name: i for i, name in enumerate(agent_names)}
        y_train = _build_label_matrix(train_data, name_to_idx)

        cc_model = None
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

        mlknn_model = None
        if "mlknn" in methods:
            print(f"[seed {seed}] train MLkNN", flush=True)
            mlknn_model = MLkNN(k=10, smoothing=1.0)
            mlknn_model.fit(X_train, y_train)

        def add_result(method: str, times: Sequence[float]) -> None:
            stats = summarize(times)
            rows.append({"seed": seed, "method": method, **stats})
            print(
                f"{method:10s} mean={stats['mean_ms']:.2f} ms | "
                f"p50={stats['p50_ms']:.2f} | p95={stats['p95_ms']:.2f} | "
                f"std={stats['std_ms']:.2f}",
                flush=True,
            )

        if "majority" in methods:
            def run_majority(ex):
                return None

            add_result("Majority", time_loop(run_majority, test_data))

        if "knn" in methods:
            def run_knn(ex):
                emb = embedder.encode([ex.prompt])[0]
                result = knn_router.route(emb)
                _ = choose_pred_set(result.candidates, args.threshold)

            add_result("KNN", time_loop(run_knn, test_data))

        if "ml" in methods:
            def run_ml(ex):
                emb = embedder.encode([ex.prompt])[0]
                result = ml_router.route(emb)
                _ = choose_pred_set(result.candidates, args.threshold)

            add_result("ML", time_loop(run_ml, test_data))

        if "cc" in methods and cc_model is not None:
            def run_cc(ex):
                emb = embedder.encode([ex.prompt])[0].reshape(1, -1)
                probs = normalize_proba_matrix(cc_model.predict_proba(emb))[0]
                _ = choose_pred_set(_to_candidates(probs, agent_names), args.threshold)

            add_result("CC", time_loop(run_cc, test_data))

        if "mlknn" in methods and mlknn_model is not None:
            def run_mlknn(ex):
                emb = embedder.encode([ex.prompt])[0].reshape(1, -1)
                probs = mlknn_model.predict_proba(emb)[0]
                _ = choose_pred_set(_to_candidates(probs, agent_names), args.threshold)

            add_result("MLkNN", time_loop(run_mlknn, test_data))

        if "encoder" in methods:
            print(f"[seed {seed}] train Encoder", flush=True)
            device = pick_device(args.encoder_device)
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
                device=device,
            )

            texts = [ex.prompt for ex in test_data]
            times: List[float] = []
            batch_size = args.encoder_batch_size
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                t0 = time.perf_counter()
                _ = encoder_predict_probs(model, batch, batch_size=batch_size, device=device)
                t1 = time.perf_counter()
                per_item = (t1 - t0) * 1000.0 / max(1, len(batch))
                times.extend([per_item] * len(batch))
            add_result("Encoder", times)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "method", "n", "mean_ms", "std_ms", "p50_ms", "p95_ms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
