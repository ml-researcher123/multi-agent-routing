import argparse
import csv
import re
from collections import Counter
from pathlib import Path


AGENTS = [
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

RARE = {
    "TimeSeriesQueryAgent",
    "LogRetrievalAgent",
    "TrendAnalysisAgent",
    "AnomalyDetectionAgent",
    "ForecastAgent",
    "PlotGenerationAgent",
}

REGEX = {
    "TimeSeriesQueryAgent": [
        r"\btime series\b",
        r"\btimeseries\b",
        r"\btsdb\b",
        r"\binfluxdb\b",
        r"\bprometheus\b",
        r"\bgraphite\b",
        r"\btimescaledb\b",
        r"\bopentsdb\b",
        r"\bquestdb\b",
        r"\bmetrics?\b.*\b(time|series)\b",
        r"\bquery\b.*\b(time|series|window)\b",
        r"\bdownsample\b|\bresample\b|\brolling window\b",
        r"\bepoch\b|\btimestamp\b",
    ],
    "SQLQueryAgent": [
        r"\bsql\b",
        r"\bselect\b",
        r"\bgroup by\b",
        r"\bjoin\b",
        r"\bquery\b",
        r"\bsqlite|mysql|postgres|warehouse|database\b",
    ],
    "APIDataFetchAgent": [
        r"\bapi\b",
        r"\bendpoint\b",
        r"\bjson\b",
        r"\brest\b",
        r"\bfetch\b|\bretrieve\b|\bpull\b",
        r"\bcoingecko|firestore|opta|instagram graph api|curl api|websocket\b",
    ],
    "LogRetrievalAgent": [
        r"\blog(s)?\b",
        r"\blog file\b|\berror log\b|\bserver logs\b",
        r"\btraceback\b|\bstack trace\b|\bexception\b",
        r"\bstdout\b|\bstderr\b",
        r"\bdebug\b",
        r"\bkubectl logs\b|\bjournalctl\b",
    ],
    "MetadataLookupAgent": [
        r"\bschema\b",
        r"\bcolumn(s)?\b",
        r"\bfield names?\b",
        r"\bmetadata\b",
        r"\bcsv files?\b|\bdataset\b|\bdata frame\b|\bfeatures?\b|\bcolumns?\b",
        r"\bwhich table\b|\bwhat table\b|\bdata dictionary\b",
    ],
    "StatisticalAnalysisAgent": [
        r"\bmean\b|\bmedian\b|\baverage\b",
        r"\bvariance\b|standard deviation|percentile",
        r"\bdistribution\b|\bprobability\b|correlation",
        r"\bsimulation\b|\bsimulations\b",
        r"\bhazard\b|\bcensoring\b|\bcovariates?\b",
        r"\bkpi\b|\brevenue\b|\bprofit\b|\bmargin\b|\bmetrics?\b|\bindicators?\b|\bvolatility\b",
    ],
    "TrendAnalysisAgent": [
        r"\btrend(s)?\b",
        r"\bhistorical\b|\bhistory\b",
        r"\bover time\b|time series",
        r"\bseasonality\b|\bpatterns?\b",
        r"\bmonthly\b|\bweekly\b|\bdaily\b|\byearly\b",
    ],
    "AnomalyDetectionAgent": [
        r"\banomal",
        r"\boutlier",
        r"\bfraud\b",
        r"\bunusual\b|\babnormal\b|\bspike\b",
        r"\bdrift\b|change point|deviation",
    ],
    "ForecastAgent": [
        r"\bforecast\b",
        r"\bpredict\b|\bprediction\b|\bprojection\b",
        r"\bfuture\b|\bupcoming\b",
        r"next (day|week|month|year|\d+ days|\d+ weeks|\d+ months|\d+ years)",
        r"survival time|remaining time|additional survival time",
    ],
    "PlotGenerationAgent": [
        r"\bplot\b",
        r"\bchart\b",
        r"\bgraph\b",
        r"visuali[sz]e|visualization|dashboard",
        r"histogram|heatmap|line chart|bar chart|density plot|candlestick|ggplot|matplotlib|plotly|apexcharts|vico",
    ],
    "SummaryAgent": [
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\btl;dr\b",
        r"\bkey points\b",
        r"\bbullet points\b",
        r"\bexecutive summary\b",
        r"\bshort summary\b|\bcondense\b|\bshorten\b",
        r"\boverview\b",
        r"\bexplain in simple terms\b|\beli5\b",
    ],
    "ReportWriterAgent": [
        r"\bwrite\b.*\b(report|essay|email|letter|proposal|blog|article|post|tweet|caption|story|poem|script|press release|resume|cv|statement|plan|policy)\b",
        r"\bdraft\b.*\b(email|report|proposal|letter|blog|article|tweet|caption|statement)\b",
        r"\bparaphrase\b|\brewrite\b|\bproofread\b|\bgrammar\b|\bimprove\b.*\bwriting\b",
    ],
}

COMPILED = {k: [re.compile(p, re.I) for p in pats] for k, pats in REGEX.items()}

ADJ = {
    "TimeSeriesQueryAgent": ["StatisticalAnalysisAgent", "PlotGenerationAgent", "TrendAnalysisAgent", "ForecastAgent"],
    "SQLQueryAgent": ["MetadataLookupAgent", "StatisticalAnalysisAgent", "TimeSeriesQueryAgent"],
    "APIDataFetchAgent": ["StatisticalAnalysisAgent", "PlotGenerationAgent", "ForecastAgent", "TimeSeriesQueryAgent"],
    "LogRetrievalAgent": ["AnomalyDetectionAgent", "SummaryAgent", "ReportWriterAgent"],
    "MetadataLookupAgent": ["SQLQueryAgent", "StatisticalAnalysisAgent"],
    "StatisticalAnalysisAgent": ["TrendAnalysisAgent", "ForecastAgent", "AnomalyDetectionAgent", "PlotGenerationAgent"],
    "TrendAnalysisAgent": ["ForecastAgent", "PlotGenerationAgent", "StatisticalAnalysisAgent"],
    "ForecastAgent": ["TrendAnalysisAgent", "PlotGenerationAgent", "StatisticalAnalysisAgent"],
    "AnomalyDetectionAgent": ["StatisticalAnalysisAgent", "PlotGenerationAgent", "LogRetrievalAgent"],
    "PlotGenerationAgent": ["TrendAnalysisAgent", "ForecastAgent", "StatisticalAnalysisAgent"],
    "SummaryAgent": ["ReportWriterAgent"],
    "ReportWriterAgent": ["SummaryAgent"],
}

TARGET_SET = {1: 3000, 2: 1500, 3: 500}
TARGET_AGENT = {agent: 625 for agent in AGENTS}
HARD_CAP = {
    "TimeSeriesQueryAgent": 600,
    "SQLQueryAgent": 1600,
    "APIDataFetchAgent": 1500,
    "LogRetrievalAgent": 600,
    "MetadataLookupAgent": 1400,
    "StatisticalAnalysisAgent": 1300,
    "TrendAnalysisAgent": 600,
    "AnomalyDetectionAgent": 400,
    "ForecastAgent": 500,
    "PlotGenerationAgent": 800,
    "SummaryAgent": 1800,
    "ReportWriterAgent": 1800,
}


def cue_scores(text: str, existing: set[str]) -> dict[str, int]:
    scores = {}
    for agent, patterns in COMPILED.items():
        if agent in existing:
            continue
        score = sum(1 for rx in patterns if rx.search(text))
        if score:
            scores[agent] = score
    return scores


def build_variants(row: dict) -> list[dict]:
    base = dict(row)
    base["_promo_score"] = 0
    base["_variant_kind"] = "orig"
    out = [base]
    aset = set(base["gold_agents"].split("|"))
    scores = cue_scores(base["prompt"], aset)
    ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0] in RARE), reverse=True)

    if len(aset) == 1:
        for agent, score in ranked[:3]:
            variant = dict(base)
            variant["gold_agents"] = "|".join(sorted(aset | {agent}, key=AGENTS.index))
            variant["gold_agent_count"] = "2"
            variant["_promo_score"] = score
            variant["_variant_kind"] = "1to2"
            out.append(variant)

        for i in range(min(len(ranked), 3)):
            for j in range(i + 1, min(len(ranked), 4)):
                variant = dict(base)
                a1, s1 = ranked[i]
                a2, s2 = ranked[j]
                variant["gold_agents"] = "|".join(sorted(aset | {a1, a2}, key=AGENTS.index))
                variant["gold_agent_count"] = "3"
                variant["_promo_score"] = s1 + s2
                variant["_variant_kind"] = "1to3"
                out.append(variant)

        if ranked:
            agent, score = ranked[0]
            for helper in ADJ.get(agent, []):
                if helper not in aset and helper != agent:
                    variant = dict(base)
                    variant["gold_agents"] = "|".join(sorted(aset | {agent, helper}, key=AGENTS.index))
                    variant["gold_agent_count"] = "3"
                    variant["_promo_score"] = score
                    variant["_variant_kind"] = "1to3h"
                    out.append(variant)
                    break

    elif len(aset) == 2:
        for agent, score in ranked[:2]:
            variant = dict(base)
            variant["gold_agents"] = "|".join(sorted(aset | {agent}, key=AGENTS.index))
            variant["gold_agent_count"] = "3"
            variant["_promo_score"] = score
            variant["_variant_kind"] = "2to3"
            out.append(variant)

        for base_agent in sorted(aset, key=AGENTS.index):
            for helper in ADJ.get(base_agent, []):
                if helper not in aset:
                    variant = dict(base)
                    variant["gold_agents"] = "|".join(sorted(aset | {helper}, key=AGENTS.index))
                    variant["gold_agent_count"] = "3"
                    variant["_promo_score"] = 0
                    variant["_variant_kind"] = "2to3h"
                    out.append(variant)
                    break

    return out


def variant_score(row: dict, counts: Counter) -> float:
    agents = row["gold_agents"].split("|")
    score = float(row.get("quality_score", "0") or 0)
    score += 1.4 * int(row.get("_promo_score", 0) or 0)
    for agent in agents:
        score += max(0, TARGET_AGENT[agent] - counts[agent]) * 0.18
        if agent in RARE:
            score += 8.0
        if counts[agent] >= HARD_CAP[agent]:
            score -= 70.0
    if len(agents) == 3:
        score += 5.0
    elif len(agents) == 2:
        score += 2.0
    return score


def greedy_stage_pick(pool: list[dict], target_n: int, counts: Counter, used_prompts: set[str], relax_caps: bool) -> list[dict]:
    chosen = []
    while len(chosen) < target_n:
        best_row = None
        best_score = None
        for row in pool:
            if row["prompt"] in used_prompts:
                continue
            agents = row["gold_agents"].split("|")
            if not relax_caps and any(counts[a] >= HARD_CAP[a] for a in agents):
                continue
            score = variant_score(row, counts)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            break
        used_prompts.add(best_row["prompt"])
        chosen.append(best_row)
        counts.update(best_row["gold_agents"].split("|"))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="wildchat_agent12_scan2500k_all_candidates.csv")
    ap.add_argument("--output_csv", default="wildchat_agent12_balanced_5000_candidates.csv")
    ap.add_argument("--output_md", default="wildchat_agent12_balanced_5000_summary.md")
    args = ap.parse_args()

    source_rows = list(csv.DictReader(open(args.input_csv, encoding="utf-8")))
    variants = []
    for row in source_rows:
        variants.extend(build_variants(row))

    best = {}
    for row in variants:
        key = (row["prompt"], row["gold_agents"])
        rank = (
            int(row.get("_promo_score", 0) or 0),
            float(row.get("quality_score", "0") or 0),
            len(str(row.get("_variant_kind", ""))),
        )
        if key not in best or rank > best[key][0]:
            best[key] = (rank, row)
    variant_rows = [item[1] for item in best.values()]

    triples = [r for r in variant_rows if int(r["gold_agent_count"]) == 3]
    doubles = [r for r in variant_rows if int(r["gold_agent_count"]) == 2]
    singles = [r for r in variant_rows if int(r["gold_agent_count"]) == 1]

    counts = Counter()
    used_prompts = set()
    selected = []
    for stage, pool in [(3, triples), (2, doubles), (1, singles)]:
        need = TARGET_SET[stage]
        picked = greedy_stage_pick(pool, need, counts, used_prompts, relax_caps=False)
        selected.extend(picked)
        if len(picked) < need:
            selected.extend(greedy_stage_pick(pool, need - len(picked), counts, used_prompts, relax_caps=True))

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        for idx, row in enumerate(selected, start=1):
            row = dict(row)
            row["prompt_id"] = idx
            writer.writerow(row)

    set_counts = Counter(int(row["gold_agent_count"]) for row in selected)
    lines = [
        "# WildChat Agent-12 Balanced 5000 Summary",
        "",
        f"- Output rows: {len(selected)}",
        f"- Set-size distribution: {dict(sorted(set_counts.items()))}",
        "",
        "## Per-Agent Counts",
    ]
    for agent in AGENTS:
        lines.append(f"- {agent}: {counts[agent]}")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(Path(args.output_csv))
    print(Path(args.output_md))
    print("rows", len(selected))
    print("set_sizes", dict(sorted(set_counts.items())))
    print("agent_counts", {agent: counts[agent] for agent in AGENTS})


if __name__ == "__main__":
    main()
