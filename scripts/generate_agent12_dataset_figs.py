from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUTS = ",".join(
    [
        "wildchat_agent12_balanced_3000_equalized_train.csv",
        "wildchat_agent12_balanced_3000_equalized_dev.csv",
        "wildchat_agent12_balanced_3000_equalized_test.csv",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset figures for the WildChat-12 benchmark.")
    parser.add_argument(
        "--input_csvs",
        default=DEFAULT_INPUTS,
        help="Comma-separated CSV paths. Defaults to the train/dev/test WildChat-12 splits.",
    )
    parser.add_argument("--out_setsize_png", default="fig_gold_setsize_donut.png")
    parser.add_argument("--out_agent_png", default="fig_agent_pct_lollipop.png")
    return parser.parse_args()


def load_counts(paths: list[str]) -> tuple[Counter, Counter, int]:
    set_counts = Counter()
    agent_counts = Counter()
    total_rows = 0

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gold = [a.strip() for a in (row.get("gold_agents") or "").split("|") if a.strip()]
                if not gold:
                    continue
                total_rows += 1
                set_counts[len(gold)] += 1
                agent_counts.update(gold)

    if total_rows == 0:
        raise ValueError("No labeled rows found across input CSVs.")
    return set_counts, agent_counts, total_rows


def plot_setsize_donut(set_counts: Counter, out_path: str) -> None:
    labels = ["1 agent", "2 agents", "3 agents"]
    values = [set_counts.get(1, 0), set_counts.get(2, 0), set_counts.get(3, 0)]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    wedges, _, autotexts = ax.pie(
        values,
        startangle=90,
        labels=None,
        autopct=lambda pct: f"{pct:.0f}%",
        pctdistance=0.82,
        wedgeprops={"width": 0.42},
        textprops={"fontsize": 10},
    )
    for text in autotexts:
        text.set_color("#222222")

    ax.legend(
        wedges,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    ax.set_title("Gold Set Size Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_agent_lollipop(agent_counts: Counter, total_rows: int, out_path: str) -> None:
    agents = sorted(agent_counts.keys())
    rates = [(agent_counts[a] / total_rows) * 100.0 for a in agents]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hlines(y=agents, xmin=0, xmax=rates, color="#5B8FF9", linewidth=1.8)
    ax.plot(rates, agents, "o", color="#5B8FF9", markersize=7)
    ax.set_xlabel("Appearance Rate (%)")
    ax.set_ylabel("Agent")
    ax.set_title("Per-Agent Appearance Rate")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    input_paths = [part.strip() for part in args.input_csvs.split(",") if part.strip()]
    set_counts, agent_counts, total_rows = load_counts(input_paths)
    plot_setsize_donut(set_counts, args.out_setsize_png)
    plot_agent_lollipop(agent_counts, total_rows, args.out_agent_png)
    print(f"Wrote {args.out_setsize_png} and {args.out_agent_png}")
