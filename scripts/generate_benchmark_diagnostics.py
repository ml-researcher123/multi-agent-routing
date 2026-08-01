from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark cardinality and consistency diagnostics.")
    parser.add_argument("--input_csvs", required=True, help="Comma-separated processed benchmark splits.")
    parser.add_argument("--consistency_csv", required=True)
    parser.add_argument("--output_png", default="fig_benchmark_diagnostics.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path.strip()) for path in args.input_csvs.split(",") if path.strip()]
    data = pd.concat(frames, ignore_index=True)
    set_counts = data["gold_agent_count"].value_counts().sort_index()
    consistency = pd.read_csv(args.consistency_csv).iloc[0]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.35),
        gridspec_kw={"width_ratios": [0.9, 1.65]},
    )

    labels = [f"{int(size)} agent" + ("" if int(size) == 1 else "s") for size in set_counts.index]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    wedges, _, autotexts = ax_left.pie(
        set_counts.values,
        labels=None,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 1.0},
        textprops={"fontsize": 9},
    )
    for wedge, hatch in zip(wedges, ["", "//", "xx"]):
        wedge.set_hatch(hatch)
    for text in autotexts:
        text.set_color("#1f1f1f")
    ax_left.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
    ax_left.set_title("Reference-Set Cardinality", fontsize=11)

    metrics = ["Set Jaccard", "Any-label overlap", "Exact-set match"]
    nearest = [
        consistency["rank1_mean_jaccard"],
        consistency["rank1_share_any_rate"],
        consistency["rank1_exact_rate"],
    ]
    random = [
        consistency["random_mean_jaccard"],
        consistency["random_share_any_rate"],
        consistency["random_exact_rate"],
    ]
    width = 0.34
    x = range(len(metrics))
    bars_near = ax_right.bar([i - width / 2 for i in x], nearest, width, label="Rank-1 neighbor", color="#4c78a8")
    bars_rand = ax_right.bar(
        [i + width / 2 for i in x],
        random,
        width,
        label="Random pair",
        color="#b8b8b8",
        hatch="//",
        edgecolor="#666666",
    )
    for bars in (bars_near, bars_rand):
        for bar in bars:
            value = bar.get_height()
            ax_right.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    ax_right.set_xticks(list(x), metrics)
    ax_right.set_ylim(0, 0.86)
    ax_right.set_ylabel("Rate")
    ax_right.set_title("Inter-Prompt Route Consistency", fontsize=11)
    ax_right.grid(axis="y", alpha=0.25, linestyle=":")
    ax_right.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(args.output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output_png}")


if __name__ == "__main__":
    main()
