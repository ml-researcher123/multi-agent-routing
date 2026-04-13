import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from simulate_agent12_execution import (
    capability_coverage,
    load_cost_tiers,
    selected_cost,
    task_success,
    utility,
)
from weighted_war import choose_weighted_pred_sets

from agent12_full_experiment import (
    _build_seed_items,
    _filter_compatible,
    _set_seed,
    _to_candidates,
    choose_pred_set,
    encoder_predict_probs,
    get_agent12_list,
    hard_set_metrics,
    load_dataset_csv,
    train_encoder,
    train_ml_router,
)


METHOD_LABELS = {
    "ml": "ML",
    "encoder": "Encoder",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate deterministic weighted WAR on top of score-based routers."
    )
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--methods", default="ml,encoder")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--war_lambdas", default="0,0.02,0.05,0.1,0.15")
    ap.add_argument("--war_cost_tiers", default="agent_cost_tiers_12.json")
    ap.add_argument("--utility_cost_lambda", type=float, default=0.10)
    ap.add_argument("--utility_extra_lambda", type=float, default=0.05)
    ap.add_argument("--embedder_model", default="all-mpnet-base-v2")
    ap.add_argument("--no_profile_text", action="store_true")
    ap.add_argument("--encoder_model_name", default="all-mpnet-base-v2")
    ap.add_argument("--encoder_epochs", type=int, default=3)
    ap.add_argument("--encoder_batch_size", type=int, default=8)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--encoder_weight_decay", type=float, default=1e-2)
    ap.add_argument("--encoder_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--output_prefix", default="weighted_war_methods")
    return ap.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_methods(raw: str) -> List[str]:
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHOD_LABELS]
    if bad:
        raise ValueError(f"Unsupported methods: {bad}. Supported: {sorted(METHOD_LABELS)}")
    return methods


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_pred_sets(
    pred_sets: Sequence[Sequence[str]],
    gold_sets: Sequence[Sequence[str]],
    cost_tiers: Dict[str, int],
    utility_cost_lambda: float,
    utility_extra_lambda: float,
) -> Dict[str, float]:
    n = len(gold_sets)
    if n == 0:
        raise ValueError("No examples to evaluate.")

    prec = rec = f1 = jacc = exact = avg_p = 0.0
    success = coverage = avg_cost = avg_extra = avg_utility = 0.0

    for pred, gold in zip(pred_sets, gold_sets):
        pred_list = list(pred)
        gold_list = list(gold)
        p, r, f, j, e = hard_set_metrics(pred_list, gold_list)
        pred_set = set(pred_list)
        gold_set = set(gold_list)

        prec += p
        rec += r
        f1 += f
        jacc += j
        exact += e
        avg_p += len(pred_list)
        success += task_success(pred_set, gold_set)
        coverage += capability_coverage(pred_set, gold_set)
        avg_cost += selected_cost(pred_set, cost_tiers)
        avg_extra += len(pred_set - gold_set)
        avg_utility += utility(
            pred_set,
            gold_set,
            cost_tiers=cost_tiers,
            cost_lambda=utility_cost_lambda,
            extra_lambda=utility_extra_lambda,
        )

    return {
        "prec": prec / n,
        "rec": rec / n,
        "f1": f1 / n,
        "jacc": jacc / n,
        "exact": exact / n,
        "avg_p": avg_p / n,
        "success": success / n,
        "coverage": coverage / n,
        "avg_cost": avg_cost / n,
        "avg_extra": avg_extra / n,
        "avg_utility": avg_utility / n,
    }


def pred_sets_from_candidates(
    candidate_lists: Sequence[Sequence[Tuple[str, float]]],
    threshold: float,
) -> List[List[str]]:
    return [choose_pred_set(cands, threshold) for cands in candidate_lists]


def pick_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] --encoder_device cuda requested but CUDA is not available; using CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encoder_candidate_lists(
    probs_all: np.ndarray,
    class_names: Sequence[str],
) -> List[List[Tuple[str, float]]]:
    return [_to_candidates(np.asarray(scores), list(class_names)) for scores in probs_all]


def select_best_base_threshold(
    candidate_lists: Sequence[Sequence[Tuple[str, float]]],
    gold_sets: Sequence[Sequence[str]],
    thresholds: Sequence[float],
    cost_tiers: Dict[str, int],
    utility_cost_lambda: float,
    utility_extra_lambda: float,
    method_label: str,
    seed: int,
    dev_rows: List[Dict[str, object]],
) -> Tuple[float, float]:
    best_threshold = None
    best_utility = -1e18
    for threshold in thresholds:
        preds = pred_sets_from_candidates(candidate_lists, threshold)
        metrics = evaluate_pred_sets(
            preds,
            gold_sets,
            cost_tiers=cost_tiers,
            utility_cost_lambda=utility_cost_lambda,
            utility_extra_lambda=utility_extra_lambda,
        )
        dev_rows.append(
            {
                "seed": seed,
                "method": method_label,
                "variant": "base",
                "threshold": threshold,
                "lambda": "",
                **metrics,
            }
        )
        if metrics["avg_utility"] > best_utility:
            best_utility = metrics["avg_utility"]
            best_threshold = threshold
    if best_threshold is None:
        raise RuntimeError(f"Failed to select base threshold for {method_label}.")
    return best_threshold, best_utility


def select_best_weighted_war(
    candidate_lists: Sequence[Sequence[Tuple[str, float]]],
    gold_sets: Sequence[Sequence[str]],
    thresholds: Sequence[float],
    war_lambdas: Sequence[float],
    cost_tiers: Dict[str, int],
    utility_cost_lambda: float,
    utility_extra_lambda: float,
    method_label: str,
    seed: int,
    dev_rows: List[Dict[str, object]],
) -> Tuple[float, float, float]:
    best_threshold = None
    best_lambda = None
    best_utility = -1e18
    for threshold in thresholds:
        for cost_lambda in war_lambdas:
            preds = choose_weighted_pred_sets(
                candidate_lists,
                threshold=threshold,
                cost_lambda=cost_lambda,
                cost_tiers=cost_tiers,
            )
            metrics = evaluate_pred_sets(
                preds,
                gold_sets,
                cost_tiers=cost_tiers,
                utility_cost_lambda=utility_cost_lambda,
                utility_extra_lambda=utility_extra_lambda,
            )
            dev_rows.append(
                {
                    "seed": seed,
                    "method": method_label,
                    "variant": "war",
                    "threshold": threshold,
                    "lambda": cost_lambda,
                    **metrics,
                }
            )
            if metrics["avg_utility"] > best_utility:
                best_utility = metrics["avg_utility"]
                best_threshold = threshold
                best_lambda = cost_lambda
    if best_threshold is None or best_lambda is None:
        raise RuntimeError(f"Failed to select weighted WAR settings for {method_label}.")
    return best_threshold, best_lambda, best_utility


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    thresholds = parse_float_list(args.thresholds)
    war_lambdas = parse_float_list(args.war_lambdas)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    use_profile_text = not args.no_profile_text
    output_prefix = Path(args.output_prefix)
    dev_csv = output_prefix.with_name(output_prefix.name + "_dev_sweep.csv")
    test_csv = output_prefix.with_name(output_prefix.name + "_test_summary.csv")
    summary_md = output_prefix.with_name(output_prefix.name + "_summary.md")

    agents = get_agent12_list()
    agent_names = [a.name for a in agents]
    agent_name_set = set(agent_names)
    cost_tiers = load_cost_tiers(Path(args.war_cost_tiers))

    train_data = _filter_compatible(load_dataset_csv(args.train_csv), agent_name_set)
    dev_data = _filter_compatible(load_dataset_csv(args.dev_csv), agent_name_set)
    test_data = _filter_compatible(load_dataset_csv(args.test_csv), agent_name_set)
    dev_gold = [ex.gold_agents for ex in dev_data]
    test_gold = [ex.gold_agents for ex in test_data]

    dev_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    enc_device = pick_device(args.encoder_device)

    for seed in seeds:
        _set_seed(seed)

        method_dev_candidates: Dict[str, List[List[Tuple[str, float]]]] = {}
        method_test_candidates: Dict[str, List[List[Tuple[str, float]]]] = {}

        if "ml" in methods:
            embedder, ml_router = train_ml_router(
                train_data=train_data,
                agents=agents,
                embedder_model=args.embedder_model,
                use_profile_text=use_profile_text,
                seed=seed,
            )
            items_dev = _build_seed_items(dev_data, embedder, ml_router, knn_router=None)
            items_test = _build_seed_items(test_data, embedder, ml_router, knn_router=None)
            method_dev_candidates["ML"] = [item["ml_cands"] for item in items_dev]
            method_test_candidates["ML"] = [item["ml_cands"] for item in items_test]

        if "encoder" in methods:
            print(f"[seed {seed}] train encoder", flush=True)
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
            dev_probs = encoder_predict_probs(
                model,
                [ex.prompt for ex in dev_data],
                batch_size=args.encoder_batch_size,
                device=enc_device,
            )
            test_probs = encoder_predict_probs(
                model,
                [ex.prompt for ex in test_data],
                batch_size=args.encoder_batch_size,
                device=enc_device,
            )
            method_dev_candidates["Encoder"] = encoder_candidate_lists(dev_probs, class_names)
            method_test_candidates["Encoder"] = encoder_candidate_lists(test_probs, class_names)

        for method_label in [METHOD_LABELS[m] for m in methods]:
            dev_cands = method_dev_candidates[method_label]
            test_cands = method_test_candidates[method_label]

            best_base_t, best_base_u = select_best_base_threshold(
                candidate_lists=dev_cands,
                gold_sets=dev_gold,
                thresholds=thresholds,
                cost_tiers=cost_tiers,
                utility_cost_lambda=args.utility_cost_lambda,
                utility_extra_lambda=args.utility_extra_lambda,
                method_label=method_label,
                seed=seed,
                dev_rows=dev_rows,
            )
            base_test_preds = pred_sets_from_candidates(test_cands, best_base_t)
            base_test_metrics = evaluate_pred_sets(
                base_test_preds,
                test_gold,
                cost_tiers=cost_tiers,
                utility_cost_lambda=args.utility_cost_lambda,
                utility_extra_lambda=args.utility_extra_lambda,
            )
            test_rows.append(
                {
                    "seed": seed,
                    "method": method_label,
                    "variant": "base",
                    "selected_threshold": best_base_t,
                    "selected_lambda": "",
                    **base_test_metrics,
                }
            )

            best_war_t, best_war_lambda, best_war_u = select_best_weighted_war(
                candidate_lists=dev_cands,
                gold_sets=dev_gold,
                thresholds=thresholds,
                war_lambdas=war_lambdas,
                cost_tiers=cost_tiers,
                utility_cost_lambda=args.utility_cost_lambda,
                utility_extra_lambda=args.utility_extra_lambda,
                method_label=method_label,
                seed=seed,
                dev_rows=dev_rows,
            )
            war_test_preds = choose_weighted_pred_sets(
                test_cands,
                threshold=best_war_t,
                cost_lambda=best_war_lambda,
                cost_tiers=cost_tiers,
            )
            war_test_metrics = evaluate_pred_sets(
                war_test_preds,
                test_gold,
                cost_tiers=cost_tiers,
                utility_cost_lambda=args.utility_cost_lambda,
                utility_extra_lambda=args.utility_extra_lambda,
            )
            test_rows.append(
                {
                    "seed": seed,
                    "method": method_label,
                    "variant": "war",
                    "selected_threshold": best_war_t,
                    "selected_lambda": best_war_lambda,
                    **war_test_metrics,
                }
            )

            selected_rows.append(
                {
                    "seed": seed,
                    "method": method_label,
                    "base_threshold_by_dev_utility": best_base_t,
                    "base_dev_utility": best_base_u,
                    "war_threshold_by_dev_utility": best_war_t,
                    "war_lambda_by_dev_utility": best_war_lambda,
                    "war_dev_utility": best_war_u,
                }
            )

    write_csv(dev_csv, dev_rows)
    write_csv(test_csv, test_rows)

    lines: List[str] = []
    lines.append("# Weighted WAR Methods Summary")
    lines.append("")
    lines.append(f"- methods: {[METHOD_LABELS[m] for m in methods]}")
    lines.append(f"- utility_cost_lambda: {args.utility_cost_lambda}")
    lines.append(f"- utility_extra_lambda: {args.utility_extra_lambda}")
    lines.append(f"- war_cost_tiers: {args.war_cost_tiers}")
    lines.append("")
    lines.append("## Selected dev-optimal settings")
    for row in selected_rows:
        lines.append(
            f"- seed {row['seed']} {row['method']}: "
            f"base threshold={row['base_threshold_by_dev_utility']:.2f} "
            f"(dev utility={row['base_dev_utility']:.3f}), "
            f"WAR threshold={row['war_threshold_by_dev_utility']:.2f}, "
            f"WAR lambda={row['war_lambda_by_dev_utility']:.2f} "
            f"(dev utility={row['war_dev_utility']:.3f})"
        )

    lines.append("")
    lines.append("## Test summary (mean+-std over seeds)")
    for method_label in [METHOD_LABELS[m] for m in methods]:
        lines.append(f"\n### {method_label}")
        for variant in ["base", "war"]:
            rows = [row for row in test_rows if row["method"] == method_label and row["variant"] == variant]
            if not rows:
                continue
            lines.append(f"\n#### {variant}")
            for key in ["f1", "jacc", "exact", "success", "coverage", "avg_cost", "avg_extra", "avg_utility", "avg_p"]:
                mean, std = mean_std([float(row[key]) for row in rows])
                lines.append(f"- {key}: {mean:.4f} +- {std:.4f}")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved dev sweep to {dev_csv}")
    print(f"Saved test summary to {test_csv}")
    print(f"Saved markdown summary to {summary_md}")


if __name__ == "__main__":
    main()
