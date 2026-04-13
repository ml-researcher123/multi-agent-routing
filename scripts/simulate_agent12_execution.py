from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

DEFAULT_COST_TIERS_JSON = Path(__file__).resolve().parents[1] / "config" / "agent_cost_tiers_12.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Simulate downstream execution metrics from predicted agent sets."
    )
    ap.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="METHOD=predictions.csv ; CSV must contain gold_agents and pred_agents columns.",
    )
    ap.add_argument(
        "--cost_lambda",
        type=float,
        default=0.10,
        help="Penalty weight applied to total selected-agent cost.",
    )
    ap.add_argument(
        "--extra_lambda",
        type=float,
        default=0.05,
        help="Penalty weight applied to extra agents beyond the gold set.",
    )
    ap.add_argument(
        "--cost_tiers_json",
        default=str(DEFAULT_COST_TIERS_JSON),
        help="JSON file containing canonical ordinal cost tiers.",
    )
    ap.add_argument(
        "--output_csv",
        default="simulated_execution_summary.csv",
        help="Where to save the per-method aggregate summary.",
    )
    ap.add_argument(
        "--output_md",
        default="simulated_execution_summary.md",
        help="Where to save a human-readable markdown summary.",
    )
    return ap.parse_args()


def parse_input_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --input value: {spec!r}. Expected METHOD=path.csv")
    method, path = spec.split("=", 1)
    method = method.strip()
    csv_path = Path(path.strip())
    if not method:
        raise ValueError(f"Missing method name in --input {spec!r}")
    return method, csv_path


def load_cost_tiers(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers") if isinstance(payload, dict) else payload
    return {str(k): int(v) for k, v in tiers.items()}


def parse_agent_set(text: str) -> Set[str]:
    if text is None:
        return set()
    raw = str(text).strip()
    if not raw:
        return set()
    sep = "|" if "|" in raw else ","
    return {part.strip() for part in raw.split(sep) if part.strip()}


def capability_coverage(pred: Set[str], gold: Set[str]) -> float:
    if not gold:
        return 1.0
    return len(pred & gold) / len(gold)


def task_success(pred: Set[str], gold: Set[str]) -> float:
    return 1.0 if gold.issubset(pred) else 0.0


def selected_cost(pred: Set[str], cost_tiers: Dict[str, int]) -> float:
    return float(sum(cost_tiers.get(agent, 2) for agent in pred))


def utility(
    pred: Set[str],
    gold: Set[str],
    cost_tiers: Dict[str, int],
    cost_lambda: float,
    extra_lambda: float,
) -> float:
    coverage = capability_coverage(pred, gold)
    cost = selected_cost(pred, cost_tiers)
    extra = len(pred - gold)
    return coverage - cost_lambda * cost - extra_lambda * extra


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_file(
    method: str,
    csv_path: Path,
    cost_tiers: Dict[str, int],
    cost_lambda: float,
    extra_lambda: float,
) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    required = {"gold_agents", "pred_agents"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    successes: List[float] = []
    coverages: List[float] = []
    costs: List[float] = []
    extras: List[float] = []
    utilities: List[float] = []
    latencies: List[float] = []

    for row in rows:
        gold = parse_agent_set(row.get("gold_agents", ""))
        pred = parse_agent_set(row.get("pred_agents", ""))

        successes.append(task_success(pred, gold))
        coverages.append(capability_coverage(pred, gold))
        costs.append(selected_cost(pred, cost_tiers))
        extras.append(float(len(pred - gold)))
        utilities.append(utility(pred, gold, cost_tiers, cost_lambda, extra_lambda))

        latency_val = row.get("latency_ms", "")
        if latency_val not in ("", None):
            try:
                latencies.append(float(latency_val))
            except ValueError:
                pass

    summary = {
        "method": method,
        "n": len(rows),
        "task_success": mean(successes),
        "coverage": mean(coverages),
        "avg_cost": mean(costs),
        "avg_extra_agents": mean(extras),
        "avg_utility": mean(utilities),
    }
    if latencies:
        summary["avg_latency_ms"] = mean(latencies)
    return summary


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Dict[str, float]], cost_lambda: float, extra_lambda: float) -> None:
    lines: List[str] = []
    lines.append("# Simulated Execution Summary")
    lines.append("")
    lines.append(f"- Task success: `1` iff the predicted agent set fully covers the gold capability set.")
    lines.append(f"- Capability coverage: `|S intersect G| / |G|`.")
    lines.append(f"- Execution cost: sum of ordinal agent tiers over selected agents.")
    lines.append(f"- Utility: `coverage - {cost_lambda:.2f} * cost - {extra_lambda:.2f} * extra_agents`.")
    lines.append("")
    lines.append("| Method | N | Task Success | Coverage | Avg Cost | Avg Extra | Avg Utility | Avg Latency (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        latency = row.get("avg_latency_ms", "")
        latency_str = f"{latency:.2f}" if latency != "" else "-"
        lines.append(
            f"| {row['method']} | {int(row['n'])} | "
            f"{100 * row['task_success']:.2f}% | {100 * row['coverage']:.2f}% | "
            f"{row['avg_cost']:.2f} | {row['avg_extra_agents']:.2f} | "
            f"{row['avg_utility']:.3f} | {latency_str} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cost_tiers = load_cost_tiers(Path(args.cost_tiers_json))
    summaries: List[Dict[str, float]] = []
    for spec in args.inputs:
        method, csv_path = parse_input_spec(spec)
        summaries.append(
            evaluate_file(
                method=method,
                csv_path=csv_path,
                cost_tiers=cost_tiers,
                cost_lambda=args.cost_lambda,
                extra_lambda=args.extra_lambda,
            )
        )

    summaries.sort(key=lambda row: row["avg_utility"], reverse=True)
    write_csv(Path(args.output_csv), summaries)
    write_markdown(Path(args.output_md), summaries, args.cost_lambda, args.extra_lambda)

    print("Saved:", args.output_csv)
    print("Saved:", args.output_md)
    for row in summaries:
        latency = f" | latency={row['avg_latency_ms']:.2f}ms" if "avg_latency_ms" in row else ""
        print(
            f"{row['method']:<10} "
            f"success={100*row['task_success']:.2f}% "
            f"coverage={100*row['coverage']:.2f}% "
            f"cost={row['avg_cost']:.2f} "
            f"extra={row['avg_extra_agents']:.2f} "
            f"utility={row['avg_utility']:.3f}{latency}"
        )


if __name__ == "__main__":
    main()
