from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for local_path in (ROOT / "src", ROOT / "scripts"):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from agent12_full_experiment import (  # noqa: E402
    _build_seed_items,
    _filter_compatible,
    _majority_agent,
    _set_seed,
    _to_candidates,
    encoder_predict_probs,
    get_agent12_list,
    load_dataset_csv,
    train_encoder,
    train_ml_router,
)
from evaluate_weighted_war_methods import (  # noqa: E402
    encoder_candidate_lists,
    evaluate_pred_sets,
    load_cost_tiers,
    pick_device,
    pred_sets_from_candidates,
)
from weighted_war import choose_weighted_pred_sets  # noqa: E402


METHODS = ("ML", "Encoder")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate Linear SVM and encoder routers on a leakage-resistant semantic-group split, "
            "including F1-selected, utility-selected, and WAR policies."
        )
    )
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--split_manifest", default="")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--war_lambdas", default="0,0.02,0.05,0.1,0.15")
    ap.add_argument(
        "--war_cost_tiers",
        default=str(ROOT / "config" / "agent_cost_tiers_12.json"),
    )
    ap.add_argument("--utility_cost_lambda", type=float, default=0.10)
    ap.add_argument("--utility_extra_lambda", type=float, default=0.05)
    ap.add_argument("--embedder_model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--encoder_model_name", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--encoder_epochs", type=int, default=3)
    ap.add_argument("--encoder_batch_size", type=int, default=8)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--encoder_weight_decay", type=float, default=1e-2)
    ap.add_argument("--encoder_device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--no_profile_text", action="store_true")
    ap.add_argument("--skip_encoder", action="store_true")
    ap.add_argument("--output_dir", default="outputs/semantic_group_robustness")
    return ap.parse_args()


def float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_sets_from_items(items: Sequence[Dict[str, object]]) -> List[List[Tuple[str, float]]]:
    return [list(item["ml_cands"]) for item in items]


def evaluate_base_sweep(
    candidate_lists: Sequence[Sequence[Tuple[str, float]]],
    gold_sets: Sequence[Sequence[str]],
    thresholds: Sequence[float],
    cost_tiers: Dict[str, int],
    utility_cost_lambda: float,
    utility_extra_lambda: float,
) -> List[Tuple[float, Dict[str, float]]]:
    rows: List[Tuple[float, Dict[str, float]]] = []
    for threshold in thresholds:
        predictions = pred_sets_from_candidates(candidate_lists, threshold)
        metrics = evaluate_pred_sets(
            predictions,
            gold_sets,
            cost_tiers=cost_tiers,
            utility_cost_lambda=utility_cost_lambda,
            utility_extra_lambda=utility_extra_lambda,
        )
        rows.append((threshold, metrics))
    return rows


def evaluate_war_sweep(
    candidate_lists: Sequence[Sequence[Tuple[str, float]]],
    gold_sets: Sequence[Sequence[str]],
    thresholds: Sequence[float],
    war_lambdas: Sequence[float],
    cost_tiers: Dict[str, int],
    utility_cost_lambda: float,
    utility_extra_lambda: float,
) -> List[Tuple[float, float, Dict[str, float]]]:
    rows: List[Tuple[float, float, Dict[str, float]]] = []
    for threshold in thresholds:
        for war_lambda in war_lambdas:
            predictions = choose_weighted_pred_sets(
                candidate_lists,
                threshold=threshold,
                cost_lambda=war_lambda,
                cost_tiers=cost_tiers,
            )
            metrics = evaluate_pred_sets(
                predictions,
                gold_sets,
                cost_tiers=cost_tiers,
                utility_cost_lambda=utility_cost_lambda,
                utility_extra_lambda=utility_extra_lambda,
            )
            rows.append((threshold, war_lambda, metrics))
    return rows


def choose_base(
    sweep: Sequence[Tuple[float, Dict[str, float]]],
    objective: str,
) -> Tuple[float, Dict[str, float]]:
    if objective == "f1":
        key = lambda item: (item[1]["f1"], item[1]["avg_utility"], -item[0])
    elif objective == "utility":
        key = lambda item: (item[1]["avg_utility"], item[1]["f1"], -item[0])
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return max(sweep, key=key)


def choose_war(
    sweep: Sequence[Tuple[float, float, Dict[str, float]]],
) -> Tuple[float, float, Dict[str, float]]:
    return max(
        sweep,
        key=lambda item: (
            item[2]["avg_utility"],
            item[2]["f1"],
            -item[1],
            -item[0],
        ),
    )


def evaluate_test_policy(
    variant: str,
    candidate_lists: Sequence[Sequence[Tuple[str, float]]],
    gold_sets: Sequence[Sequence[str]],
    threshold: float,
    war_lambda: float,
    cost_tiers: Dict[str, int],
    utility_cost_lambda: float,
    utility_extra_lambda: float,
) -> Dict[str, float]:
    if variant == "war":
        predictions = choose_weighted_pred_sets(
            candidate_lists,
            threshold=threshold,
            cost_lambda=war_lambda,
            cost_tiers=cost_tiers,
        )
    else:
        predictions = pred_sets_from_candidates(candidate_lists, threshold)
    return evaluate_pred_sets(
        predictions,
        gold_sets,
        cost_tiers=cost_tiers,
        utility_cost_lambda=utility_cost_lambda,
        utility_extra_lambda=utility_extra_lambda,
    )


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    metric_names = (
        "prec",
        "rec",
        "f1",
        "jacc",
        "exact",
        "avg_p",
        "success",
        "coverage",
        "avg_cost",
        "avg_extra",
        "avg_utility",
    )
    keys = sorted({(str(row["method"]), str(row["variant"])) for row in rows})
    for method, variant in keys:
        selected = [row for row in rows if row["method"] == method and row["variant"] == variant]
        aggregate: Dict[str, object] = {
            "method": method,
            "variant": variant,
            "seeds": "|".join(str(row["seed"]) for row in selected),
            "n_seeds": len(selected),
            "selected_thresholds": "|".join(str(row["selected_threshold"]) for row in selected),
            "selected_lambdas": "|".join(str(row["selected_lambda"]) for row in selected),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(aggregate)
    return output


def markdown_summary(
    aggregate_rows_data: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    split_sizes: Dict[str, int],
) -> str:
    lines = [
        "# Semantic-group robustness summary",
        "",
        "Near-neighbor semantic components are kept intact across train, development, and test.",
        "All thresholds and WAR parameters are selected on the grouped development split.",
        "",
        f"- split sizes: {split_sizes}",
        f"- seeds: {args.seeds}",
        f"- thresholds: {args.thresholds}",
        f"- WAR lambdas: {args.war_lambdas}",
        f"- encoder model: `{args.encoder_model_name}`",
        "",
        "| Method | Policy | F1 | Exact | Coverage | Cost | Utility |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows_data:
        lines.append(
            "| {method} | {variant} | {f1:.3f}+-{f1s:.3f} | {exact:.3f}+-{exacts:.3f} | "
            "{coverage:.3f}+-{coverages:.3f} | {cost:.3f}+-{costs:.3f} | "
            "{utility:.3f}+-{utilitys:.3f} |".format(
                method=row["method"],
                variant=row["variant"],
                f1=row["f1_mean"],
                f1s=row["f1_std"],
                exact=row["exact_mean"],
                exacts=row["exact_std"],
                coverage=row["coverage_mean"],
                coverages=row["coverage_std"],
                cost=row["avg_cost_mean"],
                costs=row["avg_cost_std"],
                utility=row["avg_utility_mean"],
                utilitys=row["avg_utility_std"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = int_list(args.seeds)
    thresholds = float_list(args.thresholds)
    war_lambdas = float_list(args.war_lambdas)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agents = get_agent12_list()
    agent_names = [agent.name for agent in agents]
    compatible_names = set(agent_names)
    train_data = _filter_compatible(load_dataset_csv(args.train_csv), compatible_names)
    dev_data = _filter_compatible(load_dataset_csv(args.dev_csv), compatible_names)
    test_data = _filter_compatible(load_dataset_csv(args.test_csv), compatible_names)
    if not train_data or not dev_data or not test_data:
        raise RuntimeError("A semantic-group split is empty or incompatible with the catalog.")

    dev_gold = [example.gold_agents for example in dev_data]
    test_gold = [example.gold_agents for example in test_data]
    cost_tiers = load_cost_tiers(Path(args.war_cost_tiers))
    encoder_device = pick_device(args.encoder_device)
    use_profile_text = not args.no_profile_text

    dev_sweep_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    majority = _majority_agent(train_data)
    majority_predictions = [[majority] for _ in test_data] if majority else [[] for _ in test_data]
    majority_metrics = evaluate_pred_sets(
        majority_predictions,
        test_gold,
        cost_tiers=cost_tiers,
        utility_cost_lambda=args.utility_cost_lambda,
        utility_extra_lambda=args.utility_extra_lambda,
    )
    test_rows.append(
        {
            "seed": seeds[0],
            "method": "Majority",
            "variant": "reference",
            "selected_threshold": "",
            "selected_lambda": "",
            **majority_metrics,
        }
    )

    for seed in seeds:
        print(f"[robustness] seed {seed}", flush=True)
        _set_seed(seed)
        method_candidates: Dict[str, Tuple[List[List[Tuple[str, float]]], List[List[Tuple[str, float]]]]] = {}

        embedder, ml_router = train_ml_router(
            train_data=train_data,
            agents=agents,
            embedder_model=args.embedder_model,
            use_profile_text=use_profile_text,
            seed=seed,
        )
        ml_dev = candidate_sets_from_items(_build_seed_items(dev_data, embedder, ml_router))
        ml_test = candidate_sets_from_items(_build_seed_items(test_data, embedder, ml_router))
        method_candidates["ML"] = (ml_dev, ml_test)

        model = None
        if not args.skip_encoder:
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
                device=encoder_device,
            )
            dev_probs = encoder_predict_probs(
                model,
                [example.prompt for example in dev_data],
                batch_size=args.encoder_batch_size,
                device=encoder_device,
            )
            test_probs = encoder_predict_probs(
                model,
                [example.prompt for example in test_data],
                batch_size=args.encoder_batch_size,
                device=encoder_device,
            )
            method_candidates["Encoder"] = (
                encoder_candidate_lists(dev_probs, class_names),
                encoder_candidate_lists(test_probs, class_names),
            )

        for method, (dev_candidates, test_candidates) in method_candidates.items():
            base_sweep = evaluate_base_sweep(
                dev_candidates,
                dev_gold,
                thresholds=thresholds,
                cost_tiers=cost_tiers,
                utility_cost_lambda=args.utility_cost_lambda,
                utility_extra_lambda=args.utility_extra_lambda,
            )
            war_sweep = evaluate_war_sweep(
                dev_candidates,
                dev_gold,
                thresholds=thresholds,
                war_lambdas=war_lambdas,
                cost_tiers=cost_tiers,
                utility_cost_lambda=args.utility_cost_lambda,
                utility_extra_lambda=args.utility_extra_lambda,
            )
            for threshold, metrics in base_sweep:
                dev_sweep_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "variant": "base",
                        "threshold": threshold,
                        "lambda": "",
                        **metrics,
                    }
                )
            for threshold, war_lambda, metrics in war_sweep:
                dev_sweep_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "variant": "war",
                        "threshold": threshold,
                        "lambda": war_lambda,
                        **metrics,
                    }
                )

            f1_threshold, _ = choose_base(base_sweep, objective="f1")
            utility_threshold, _ = choose_base(base_sweep, objective="utility")
            war_threshold, war_lambda, _ = choose_war(war_sweep)
            policies = (
                ("f1_selected", f1_threshold, 0.0),
                ("utility_selected", utility_threshold, 0.0),
                ("war", war_threshold, war_lambda),
            )
            for variant, threshold, selected_lambda in policies:
                metrics = evaluate_test_policy(
                    variant=variant,
                    candidate_lists=test_candidates,
                    gold_sets=test_gold,
                    threshold=threshold,
                    war_lambda=selected_lambda,
                    cost_tiers=cost_tiers,
                    utility_cost_lambda=args.utility_cost_lambda,
                    utility_extra_lambda=args.utility_extra_lambda,
                )
                test_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "variant": variant,
                        "selected_threshold": threshold,
                        "selected_lambda": selected_lambda,
                        **metrics,
                    }
                )

        del embedder, ml_router, ml_dev, ml_test, method_candidates
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        write_csv(output_dir / "semantic_group_dev_sweep.csv", dev_sweep_rows)
        write_csv(output_dir / "semantic_group_test_seed_metrics.csv", test_rows)

    aggregate = aggregate_rows(test_rows)
    write_csv(output_dir / "semantic_group_test_aggregate.csv", aggregate)
    summary = markdown_summary(
        aggregate,
        args=args,
        split_sizes={"train": len(train_data), "dev": len(dev_data), "test": len(test_data)},
    )
    (output_dir / "semantic_group_robustness_summary.md").write_text(summary, encoding="utf-8")

    manifest: Dict[str, object] = {
        "train_csv": args.train_csv,
        "dev_csv": args.dev_csv,
        "test_csv": args.test_csv,
        "split_sizes": {"train": len(train_data), "dev": len(dev_data), "test": len(test_data)},
        "split_manifest": args.split_manifest,
        "seeds": seeds,
        "thresholds": thresholds,
        "war_lambdas": war_lambdas,
        "utility_cost_lambda": args.utility_cost_lambda,
        "utility_extra_lambda": args.utility_extra_lambda,
        "embedder_model": args.embedder_model,
        "encoder_model_name": args.encoder_model_name,
        "encoder_epochs": args.encoder_epochs,
        "encoder_batch_size": args.encoder_batch_size,
        "encoder_lr": args.encoder_lr,
        "encoder_weight_decay": args.encoder_weight_decay,
        "encoder_device": str(encoder_device),
        "profile_text_augmentation": use_profile_text,
    }
    if args.split_manifest:
        manifest_path = Path(args.split_manifest)
        if manifest_path.exists():
            manifest["semantic_split"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    (output_dir / "semantic_group_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary, flush=True)
    print(f"[robustness] outputs saved under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
