from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["Majority", "KNN", "ML", "CC", "MLkNN", "Encoder", "LLM (zero-shot)"]
METHOD_COLORS = {
    "Majority": "#7a7a7a",
    "KNN": "#4c78a8",
    "ML": "#f58518",
    "CC": "#54a24b",
    "MLkNN": "#e45756",
    "Encoder": "#72b7b2",
    "LLM (zero-shot)": "#9d755d",
}
WAR_METHOD_COLORS = {
    "ML": "#f58518",
    "Encoder": "#72b7b2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate agent12 paper figures from aggregate CSVs.")
    parser.add_argument("--threshold_csv", default="manual_aggregate_threshold_3seed.csv")
    parser.add_argument("--war_dev_csv", default="weighted_war_methods_dev_sweep.csv")
    parser.add_argument("--war_test_csv", default="weighted_war_methods_test_summary.csv")
    parser.add_argument("--selected_threshold", type=float, default=0.60)
    parser.add_argument("--llm_method", default="LLM (zero-shot)")
    parser.add_argument("--llm_precision", type=float, default=46.22)
    parser.add_argument("--llm_recall", type=float, default=42.33)
    parser.add_argument("--llm_avg_set_size", type=float, default=1.40)
    parser.add_argument("--out_threshold_png", default="fig_threshold_sweep.png")
    parser.add_argument("--out_pr_png", default="fig_pr_scatter.png")
    parser.add_argument("--out_setsize_png", default="fig_avg_set_size.png")
    parser.add_argument("--out_war_png", default="fig_war_tradeoff.png")
    return parser.parse_args()


def load_threshold_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "threshold",
        "method",
        "prec_mean",
        "prec_std",
        "rec_mean",
        "rec_std",
        "f1_mean",
        "f1_std",
        "avg_p_mean",
        "avg_p_std",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def add_llm_selected_row(
    selected: pd.DataFrame,
    selected_threshold: float,
    method: str,
    precision_pct: float,
    recall_pct: float,
    avg_set_size: float,
) -> pd.DataFrame:
    if method in set(selected["method"]):
        return selected
    llm_row = {
        "threshold": selected_threshold,
        "method": method,
        "prec_mean": precision_pct / 100.0,
        "prec_std": 0.0,
        "rec_mean": recall_pct / 100.0,
        "rec_std": 0.0,
        "f1_mean": float("nan"),
        "f1_std": 0.0,
        "avg_p_mean": avg_set_size,
        "avg_p_std": 0.0,
    }
    return pd.concat([selected, pd.DataFrame([llm_row])], ignore_index=True)


def load_war_dev_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"seed", "method", "variant", "threshold", "lambda", "avg_utility"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_war_test_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"seed", "method", "variant", "selected_threshold", "selected_lambda", "avg_utility"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def plot_threshold_sweep(summary: pd.DataFrame, selected_threshold: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    methods = [m for m in METHOD_ORDER if m in set(summary["method"])]
    for method in methods:
        part = summary[summary["method"] == method].dropna(subset=["threshold", "f1_mean"]).sort_values("threshold")
        if part.empty:
            continue
        ax.plot(
            part["threshold"],
            part["f1_mean"] * 100.0,
            marker="o",
            linewidth=2.2,
            markersize=5,
            label=method,
            color=METHOD_COLORS.get(method),
        )
        lower = (part["f1_mean"] - part["f1_std"]).clip(lower=0) * 100.0
        upper = (part["f1_mean"] + part["f1_std"]).clip(upper=1) * 100.0
        if (part["f1_std"] > 0).any():
            ax.fill_between(part["threshold"], lower, upper, alpha=0.12, color=METHOD_COLORS.get(method))

    ax.axvline(selected_threshold, linestyle="--", linewidth=1.4, color="#333333", alpha=0.8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Dev Threshold Sweep")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pr_scatter(threshold_df: pd.DataFrame, selected_threshold: float, out_path: str) -> None:
    selected = threshold_df[threshold_df["threshold"].round(6) == round(selected_threshold, 6)].copy()
    if selected.empty:
        raise ValueError(f"No aggregate threshold rows found at threshold={selected_threshold}")

    selected["label_order"] = selected["method"].apply(lambda method: METHOD_ORDER.index(method) if method in METHOD_ORDER else 999)
    selected = selected.sort_values("label_order")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    label_offsets = {
        "LLM (zero-shot)": (6, -12),
        "KNN": (6, 6),
        "Majority": (6, 6),
        "Encoder": (6, 6),
    }
    for _, row in selected.iterrows():
        method = row["method"]
        ax.scatter(
            row["rec_mean"] * 100.0,
            row["prec_mean"] * 100.0,
            s=90 if method != "Encoder" else 120,
            color=METHOD_COLORS.get(method, "#333333"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            method,
            (row["rec_mean"] * 100.0, row["prec_mean"] * 100.0),
            xytext=label_offsets.get(method, (6, 4)),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision-Recall by Method")
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_avg_set_size(threshold_df: pd.DataFrame, selected_threshold: float, out_path: str) -> None:
    selected = threshold_df[threshold_df["threshold"].round(6) == round(selected_threshold, 6)].copy()
    if selected.empty:
        raise ValueError(f"No aggregate threshold rows found at threshold={selected_threshold}")

    selected["label_order"] = selected["method"].apply(lambda method: METHOD_ORDER.index(method) if method in METHOD_ORDER else 999)
    selected = selected.sort_values("label_order")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(
        selected["method"],
        selected["avg_p_mean"],
        color=[METHOD_COLORS.get(method, "#666666") for method in selected["method"]],
        alpha=0.9,
    )
    for bar, val in zip(bars, selected["avg_p_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Avg |P|")
    ax.set_title("Average Predicted Set Size")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_war_tradeoff(war_dev_df: pd.DataFrame, war_test_df: pd.DataFrame, out_path: str) -> None:
    war_only = war_dev_df[war_dev_df["variant"] == "war"].copy()
    if war_only.empty:
        raise ValueError("WAR dev sweep CSV does not contain any WAR rows.")

    selected = war_test_df[war_test_df["variant"] == "war"][["method", "selected_threshold", "selected_lambda"]].drop_duplicates()
    selected["selected_lambda"] = pd.to_numeric(selected["selected_lambda"], errors="coerce")
    selected["selected_threshold"] = pd.to_numeric(selected["selected_threshold"], errors="coerce")
    if selected.empty:
        raise ValueError("WAR test summary CSV does not contain selected WAR settings.")

    merged = war_only.merge(
        selected.rename(columns={"selected_threshold": "threshold"}),
        on=["method", "threshold"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Could not align WAR dev sweep with selected thresholds from the test summary.")

    grouped = (
        merged.groupby(["method", "lambda"], as_index=False)
        .agg(avg_utility_mean=("avg_utility", "mean"), avg_utility_std=("avg_utility", "std"))
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for method in ["ML", "Encoder"]:
        part = grouped[grouped["method"] == method].sort_values("lambda")
        if part.empty:
            continue

        ax.plot(
            part["lambda"],
            part["avg_utility_mean"],
            marker="o",
            linewidth=2.2,
            color=WAR_METHOD_COLORS[method],
            label=method,
        )
        if (part["avg_utility_std"] > 0).any():
            lower = part["avg_utility_mean"] - part["avg_utility_std"]
            upper = part["avg_utility_mean"] + part["avg_utility_std"]
            ax.fill_between(part["lambda"], lower, upper, color=WAR_METHOD_COLORS[method], alpha=0.12)

        selected_row = selected[selected["method"] == method].iloc[0]
        best_lambda = float(selected_row["selected_lambda"])
        best_point = part[part["lambda"].round(6) == round(best_lambda, 6)]
        if not best_point.empty:
            best_utility = float(best_point["avg_utility_mean"].iloc[0])
            ax.scatter([best_lambda], [best_utility], color=WAR_METHOD_COLORS[method], s=65, zorder=4)
            ax.annotate(
                f"{method} best (t={float(selected_row['selected_threshold']):.2f})",
                (best_lambda, best_utility),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Cost penalty lambda")
    ax.set_ylabel("Utility")
    ax.set_title("WAR Utility Trade-off")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    threshold_df = load_threshold_metrics(args.threshold_csv)
    war_dev_df = load_war_dev_metrics(args.war_dev_csv)
    war_test_df = load_war_test_summary(args.war_test_csv)
    threshold_df = add_llm_selected_row(
        threshold_df,
        selected_threshold=args.selected_threshold,
        method=args.llm_method,
        precision_pct=args.llm_precision,
        recall_pct=args.llm_recall,
        avg_set_size=args.llm_avg_set_size,
    )

    plot_threshold_sweep(threshold_df, args.selected_threshold, args.out_threshold_png)
    plot_pr_scatter(threshold_df, args.selected_threshold, args.out_pr_png)
    plot_avg_set_size(threshold_df, args.selected_threshold, args.out_setsize_png)
    plot_war_tradeoff(war_dev_df, war_test_df, args.out_war_png)

    print(f"saved {args.out_threshold_png}")
    print(f"saved {args.out_pr_png}")
    print(f"saved {args.out_setsize_png}")
    print(f"saved {args.out_war_png}")


if __name__ == "__main__":
    main()
