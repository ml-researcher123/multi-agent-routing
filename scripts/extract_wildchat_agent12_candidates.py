import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset


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

RARE_AGENTS = {
    "TimeSeriesQueryAgent",
    "LogRetrievalAgent",
    "TrendAnalysisAgent",
    "AnomalyDetectionAgent",
    "ForecastAgent",
    "PlotGenerationAgent",
}

NEGATIVE_PATTERNS = [
    r"\b(create|generate|make)\b.*\b(image|photo|logo|wallpaper|picture|artwork)\b",
    r"\bstable diffusion\b|\bmidjourney\b|\bdall[- ]?e\b",
]
NEGATIVE_RE = re.compile("|".join(f"(?:{p})" for p in NEGATIVE_PATTERNS), re.I)

CODE_TASK_RE = re.compile(
    r"\b(write|create|build|generate|implement|code|script)\b.*\b(code|script|function|class|api|endpoint|sql|query|program|app)\b",
    re.I,
)

PATTERNS = {
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
        r"\btime range\b|\bwindow\b|\binterval\b|\bgranularity\b",
        r"\bdownsample\b|\bresample\b|\brolling window\b",
        r"\bepoch\b|\btimestamp\b",
    ],
    "SQLQueryAgent": [
        r"\bsql\b",
        r"\bselect\b",
        r"\bquery\b.*\b(table|database|warehouse)\b",
        r"\b(database|postgres|mysql|sqlite|warehouse)\b",
        r"\bjoin\b",
        r"\bgroup by\b",
        r"\bwhere clause\b",
        r"\bnot exists\b",
        r"\btable\b.*\b(rows?|records?)\b",
    ],
    "APIDataFetchAgent": [
        r"\bapi\b",
        r"\bendpoint\b",
        r"\brest\b",
        r"\bhttp client\b",
        r"\bpost json\b",
        r"\bjson\b.*\b(api|endpoint|request|response|post|get)\b",
        r"\b(fetch|get|retrieve|pull)\b.*\b(json|response|weather|prices?|rates?|endpoint|api)\b",
        r"\bcall\b.*\bapi\b",
        r"\bcheapest flight\b",
        r"\bflight from\b",
        r"\bexchange rates?\b",
        r"\bcurrent weather\b",
        r"\btoday'?s weather\b",
    ],
    "LogRetrievalAgent": [
        r"\blog(s)?\b",
        r"\blog file\b|\berror log\b|\bserver logs\b",
        r"\btraceback\b|\bstack trace\b|\bexception\b",
        r"\bsegmentation fault\b|\bcore dump\b",
        r"\bstdout\b|\bstderr\b",
        r"\bdebug\b",
        r"\bkubectl logs\b|\bjournalctl\b|\bsyslog\b",
    ],
    "MetadataLookupAgent": [
        r"\bschema\b",
        r"\bcolumns?\b",
        r"\bfield names?\b",
        r"\bdatabase fields?\b",
        r"which table",
        r"what table",
        r"which endpoint",
        r"what endpoint",
        r"where .* stored",
        r"\bmetadata\b",
        r"\ber diagram\b",
        r"\bdata dictionary\b",
    ],
    "StatisticalAnalysisAgent": [
        r"\bmean\b",
        r"\bmedian\b",
        r"standard deviation",
        r"\bvariance\b",
        r"\bdistribution\b",
        r"\bpercentiles?\b",
        r"\bcorrelation\b",
        r"\bstatistics\b",
        r"\bvolatility\b",
        r"\bdescriptive statistics\b",
        r"\baverage\b",
        r"\bprobability\b",
        r"\bregression analysis\b",
        r"\bcompare\b.*\baverage\b",
    ],
    "TrendAnalysisAgent": [
        r"trend direction",
        r"overall trend",
        r"long[- ]term trend",
        r"change over time",
        r"rising or falling",
        r"\btrend\b",
        r"\bover time\b",
        r"\bseasonality\b",
        r"\bincreas(?:e|ing)\b.*\bdecreas(?:e|ing)\b",
        r"\bgrowth trend\b",
    ],
    "AnomalyDetectionAgent": [
        r"\banomal",
        r"\boutlier",
        r"\babnormal",
        r"\bunusual",
        r"\bspike",
        r"\bsudden drop\b",
        r"\bdeviation\b",
        r"\bdetect .* unusual\b",
        r"\bfraud\b|\bdrift\b|change point",
    ],
    "ForecastAgent": [
        r"\bforecast\b",
        r"\bforecasted\b",
        r"\bprojection\b",
        r"\bproject\b.*\b(next|future)\b",
        r"\bpredict(?:ion|ed)?\b.*\b(next|future|tomorrow|upcoming)\b",
        r"\bestimate\b.*\b(next|future)\b",
        r"\bwhat will\b.*\b(next|future|tomorrow)\b",
        r"\bpredict\b.*\b(survival time|additional survival time|remaining time|future price|next \d+ (days?|weeks?|months?|years?))\b",
        r"\bhistorical\b.*\bprice\b.*\bpredict\b",
    ],
    "PlotGenerationAgent": [
        r"\bvisuali[sz]e\b",
        r"\bvisualization\b",
        r"\bhistogram\b",
        r"\bscatter plot\b",
        r"\bline chart\b",
        r"\bbar chart\b",
        r"\bline plot\b",
        r"\bbar plot\b",
        r"\bdensity plot\b",
        r"\bkm plot\b",
        r"\bcandlestick\b",
        r"\bggplot2\b",
        r"\bmatplotlib\b",
        r"\bplotly\b",
        r"\bchart library\b",
        r"\bplot\b.*\b(data|curve|variable|variables|density|scatter|line|bar|survival|km|histogram|points)\b",
        r"\b(chart|graph)\b.*\b(data|distribution|trend|series|values|points|variables|analytics|expenses)\b",
        r"\bdraw\b.*\b(chart|graph)\b",
        r"\b(show|create|make|generate)\b.*\b(chart|graph|plot)\b",
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

COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}

STRONG = {
    "TimeSeriesQueryAgent": re.compile(r"\b(time series|timeseries|tsdb|influxdb|prometheus|graphite|timescaledb|opentsdb|questdb)\b", re.I),
    "SQLQueryAgent": re.compile(r"\b(sql|select|join|group by|where clause|database|warehouse|postgres|mysql)\b", re.I),
    "APIDataFetchAgent": re.compile(r"\b(api|endpoint|http client|post json|json .* (api|endpoint|request|response|post|get)|rest|flight from|exchange rates?|current weather|today'?s weather)\b", re.I),
    "LogRetrievalAgent": re.compile(r"\b(logs?|traceback|stack trace|exception|stderr|stdout|journalctl|kubectl logs)\b", re.I),
    "MetadataLookupAgent": re.compile(r"\b(schema|column|field names?|database fields?|which table|what table|which endpoint|what endpoint|metadata|er diagram)\b", re.I),
    "StatisticalAnalysisAgent": re.compile(r"\b(mean|median|variance|distribution|statistics|statistical|volatility|percentile|correlation|average|probability|regression)\b", re.I),
    "TrendAnalysisAgent": re.compile(r"\b(trend|change over time|over time|seasonality|rising|falling|growth)\b", re.I),
    "AnomalyDetectionAgent": re.compile(r"\b(anomal|outlier|abnormal|unusual|spike|fraud|drift|change point|deviation)\b", re.I),
    "ForecastAgent": re.compile(r"\b(forecast|projection|predict|future|upcoming|tomorrow|next \d+|survival time|remaining time)\b", re.I),
    "PlotGenerationAgent": re.compile(r"\b(visuali|histogram|scatter plot|line chart|bar chart|line plot|bar plot|density plot|km plot|candlestick|ggplot2|matplotlib|plotly|chart library|plot .* (data|curve|variable|density|scatter|line|bar|survival|histogram|points)|chart .* (data|distribution|trend|series|values|points|variables|analytics|expenses)|graph .* (data|distribution|trend|series|values|points|variables|analytics|expenses)|draw .* chart|show .* graph|create .* plot|generate .* chart)\b", re.I),
    "SummaryAgent": re.compile(r"\b(summarize|summary|tl;dr|key points|bullet points|executive summary|overview|eli5)\b", re.I),
    "ReportWriterAgent": re.compile(r"\b(write|draft|paraphrase|rewrite|proofread|grammar)\b", re.I),
}

QUERY_ACTION_RE = re.compile(r"\b(query|select|get|return|find|show me|retrieve|pull)\b", re.I)
FETCH_ACTION_RE = re.compile(r"\b(fetch|get|retrieve|pull|call)\b", re.I)
PLOT_SIGNAL_RE = re.compile(r"\b(visuali[sz]e|visualization|histogram|scatter plot|line chart|bar chart|line plot|bar plot|density plot|km plot|candlestick|ggplot2|matplotlib|plotly|chart library|plot .* (data|curve|variable|variables|density|scatter|line|bar|survival|histogram|points)|chart .* (data|distribution|trend|series|values|points|variables|analytics|expenses)|graph .* (data|distribution|trend|series|values|points|variables|analytics|expenses)|draw .* (chart|graph)|show .* (chart|graph)|create .* (chart|graph|plot)|generate .* (chart|graph|plot))\b", re.I)
STATS_SIGNAL_RE = re.compile(r"\b(mean|median|variance|distribution|statistics|statistical|volatility|percentile|correlation|average|probability|regression|simulation|confidence interval|hazard|piecewise|covariates?)\b", re.I)
TREND_SIGNAL_RE = re.compile(r"\b(trend|patterns?|over time|seasonality|rising|falling|growth|historical)\b", re.I)
FORECAST_SIGNAL_RE = re.compile(r"\b(forecast|projection|predict|future|upcoming|tomorrow|next \d+|survival time|remaining time)\b", re.I)
ANOMALY_SIGNAL_RE = re.compile(r"\b(anomal|outlier|abnormal|unusual|spike|fraud|drift|change point|deviation)\b", re.I)
SUMMARY_SIGNAL_RE = re.compile(r"\b(summarize|summary|tl;dr|key points|bullet points|executive summary|overview|eli5)\b", re.I)
REPORT_SIGNAL_RE = re.compile(r"\b(write|draft|paraphrase|rewrite|proofread|grammar|report|essay|email|letter|proposal|blog|article|post|tweet|caption|story|poem)\b", re.I)
LOG_SIGNAL_RE = re.compile(r"\b(logs?|traceback|stack trace|exception|stderr|stdout|journalctl|kubectl logs)\b", re.I)
TIMESERIES_SIGNAL_RE = re.compile(r"\b(time series|timeseries|tsdb|influxdb|prometheus|graphite|timescaledb|opentsdb|questdb|time range|interval|granularity)\b", re.I)

MULTI_STEP_RE = re.compile(r"\b(and then|then|also|plus|along with|first .* then|after that|determine .* and .* fetch)\b", re.I)


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def template_signature(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+", " <url> ", t)
    t = re.sub(r"\b[0-9]+(?:\.[0-9]+)?\b", " <num> ", t)
    t = re.sub(r"\b[a-f0-9]{8,}\b", " <id> ", t)
    t = re.sub(r"[^a-z0-9<> ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def coarse_topic_key(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    words = [w for w in t.split() if len(w) > 2 and w not in {
        "the", "and", "for", "that", "with", "from", "into", "then", "also", "this",
        "have", "been", "were", "will", "your", "please", "make", "create", "show",
        "give", "using", "used", "what", "how", "can", "you", "are", "about",
    }]
    return " ".join(words[:8])


def detect_agents(text: str):
    if NEGATIVE_RE.search(text):
        return []

    matched = []
    strong_hits = {}
    for agent, regs in COMPILED.items():
        hits = sum(1 for rx in regs if rx.search(text))
        if hits:
            matched.append(agent)
            strong_hits[agent] = STRONG[agent].search(text) is not None

    if not matched:
        return []

    if "ReportWriterAgent" in matched and CODE_TASK_RE.search(text):
        matched = [a for a in matched if a != "ReportWriterAgent"]

    # Disambiguation: pure metadata questions should not become SQL/API fetch.
    if "MetadataLookupAgent" in matched and "SQLQueryAgent" in matched:
        if not QUERY_ACTION_RE.search(text):
            matched = [a for a in matched if a != "SQLQueryAgent"]
    if "MetadataLookupAgent" in matched and "APIDataFetchAgent" in matched:
        if not FETCH_ACTION_RE.search(text):
            matched = [a for a in matched if a != "APIDataFetchAgent"]

    # Explicit combo enrichment.
    if "MetadataLookupAgent" in matched and QUERY_ACTION_RE.search(text):
        matched.append("SQLQueryAgent")
    if "SQLQueryAgent" in matched and re.search(r"\b(schema|column|field|which table|what table|metadata)\b", text, re.I):
        matched.append("MetadataLookupAgent")
    if "APIDataFetchAgent" in matched and STATS_SIGNAL_RE.search(text):
        matched.append("StatisticalAnalysisAgent")
    if "APIDataFetchAgent" in matched and PLOT_SIGNAL_RE.search(text):
        matched.append("PlotGenerationAgent")
    if "APIDataFetchAgent" in matched and FORECAST_SIGNAL_RE.search(text):
        matched.append("ForecastAgent")
    if "StatisticalAnalysisAgent" in matched and PLOT_SIGNAL_RE.search(text):
        matched.append("PlotGenerationAgent")
    if "TrendAnalysisAgent" in matched and PLOT_SIGNAL_RE.search(text):
        matched.append("PlotGenerationAgent")
    if "AnomalyDetectionAgent" in matched and PLOT_SIGNAL_RE.search(text):
        matched.append("PlotGenerationAgent")
    if "ForecastAgent" in matched and TREND_SIGNAL_RE.search(text):
        matched.append("TrendAnalysisAgent")
    if "ForecastAgent" in matched and STATS_SIGNAL_RE.search(text):
        matched.append("StatisticalAnalysisAgent")
    if "ForecastAgent" in matched and PLOT_SIGNAL_RE.search(text):
        matched.append("PlotGenerationAgent")
    if "TimeSeriesQueryAgent" in matched and STATS_SIGNAL_RE.search(text):
        matched.append("StatisticalAnalysisAgent")
    if "TimeSeriesQueryAgent" in matched and PLOT_SIGNAL_RE.search(text):
        matched.append("PlotGenerationAgent")
    if "TimeSeriesQueryAgent" in matched and FORECAST_SIGNAL_RE.search(text):
        matched.append("ForecastAgent")
    if "LogRetrievalAgent" in matched and ANOMALY_SIGNAL_RE.search(text):
        matched.append("AnomalyDetectionAgent")
    if "LogRetrievalAgent" in matched and SUMMARY_SIGNAL_RE.search(text):
        matched.append("SummaryAgent")
    if "LogRetrievalAgent" in matched and REPORT_SIGNAL_RE.search(text):
        matched.append("ReportWriterAgent")

    # Multi-agent labels should have either multiple strong signals or explicit multi-step wording.
    if len(matched) >= 2:
        strong_count = sum(1 for a in matched if strong_hits.get(a))
        explicit_combo = False
        pair_checks = [
            ({"MetadataLookupAgent", "SQLQueryAgent"}, re.compile(r"\b(schema|column|field|which table|what table|metadata)\b.*\b(query|select|get|return|find|show me|retrieve)\b|\b(query|select|get|return|find|show me|retrieve)\b.*\b(schema|column|field|which table|what table|metadata)\b", re.I)),
            ({"APIDataFetchAgent", "StatisticalAnalysisAgent"}, re.compile(r"\b(fetch|get|retrieve|pull|api|endpoint)\b.*\b(mean|median|average|distribution|statistics|correlation|variance|probability|simulation)\b|\b(mean|median|average|distribution|statistics|correlation|variance|probability|simulation)\b.*\b(fetch|get|retrieve|pull|api|endpoint)\b", re.I)),
            ({"APIDataFetchAgent", "PlotGenerationAgent"}, re.compile(r"\b(fetch|get|retrieve|pull|api|endpoint)\b.*\b(plot|chart|graph|visuali[sz]e)\b|\b(plot|chart|graph|visuali[sz]e)\b.*\b(fetch|get|retrieve|pull|api|endpoint)\b", re.I)),
            ({"StatisticalAnalysisAgent", "PlotGenerationAgent"}, re.compile(r"\b(mean|median|average|distribution|statistics|correlation|variance|probability)\b.*\b(plot|chart|graph|visuali[sz]e)\b|\b(plot|chart|graph|visuali[sz]e)\b.*\b(mean|median|average|distribution|statistics|correlation|variance|probability)\b", re.I)),
            ({"TrendAnalysisAgent", "ForecastAgent"}, re.compile(r"\b(trend|pattern|historical|over time)\b.*\b(forecast|predict|future|projection)\b|\b(forecast|predict|future|projection)\b.*\b(trend|pattern|historical|over time)\b", re.I)),
            ({"TrendAnalysisAgent", "PlotGenerationAgent"}, re.compile(r"\b(trend|pattern|historical|over time)\b.*\b(plot|chart|graph|visuali[sz]e)\b|\b(plot|chart|graph|visuali[sz]e)\b.*\b(trend|pattern|historical|over time)\b", re.I)),
            ({"AnomalyDetectionAgent", "PlotGenerationAgent"}, re.compile(r"\b(anomal|outlier|abnormal|unusual|spike|fraud)\b.*\b(plot|chart|graph|visuali[sz]e)\b|\b(plot|chart|graph|visuali[sz]e)\b.*\b(anomal|outlier|abnormal|unusual|spike|fraud)\b", re.I)),
            ({"TimeSeriesQueryAgent", "PlotGenerationAgent"}, re.compile(r"\b(time series|timeseries|tsdb)\b.*\b(plot|chart|graph|visuali[sz]e)\b|\b(plot|chart|graph|visuali[sz]e)\b.*\b(time series|timeseries|tsdb)\b", re.I)),
            ({"TimeSeriesQueryAgent", "StatisticalAnalysisAgent"}, re.compile(r"\b(time series|timeseries|tsdb)\b.*\b(mean|median|distribution|statistics|correlation)\b|\b(mean|median|distribution|statistics|correlation)\b.*\b(time series|timeseries|tsdb)\b", re.I)),
            ({"LogRetrievalAgent", "SummaryAgent"}, re.compile(r"\b(logs?|traceback|stack trace|exception)\b.*\b(summary|summarize|tl;dr|key points)\b|\b(summary|summarize|tl;dr|key points)\b.*\b(logs?|traceback|stack trace|exception)\b", re.I)),
            ({"ReportWriterAgent", "SummaryAgent"}, re.compile(r"\b(summary|summarize|executive summary)\b.*\b(report|write|draft|proposal|essay)\b|\b(report|write|draft|proposal|essay)\b.*\b(summary|summarize|executive summary)\b", re.I)),
        ]
        for pair, rx in pair_checks:
            if pair.issubset(set(matched)) and rx.search(text):
                explicit_combo = True
                break
        if strong_count < 2 and not MULTI_STEP_RE.search(text) and not explicit_combo:
            matched = [next((a for a in matched if strong_hits.get(a)), matched[0])]

    # Keep max 3 agents, preferring strong and rarer ones.
    if len(matched) > 3:
        order = sorted(
            matched,
            key=lambda a: (
                0 if a in RARE_AGENTS else 1,
                0 if strong_hits.get(a) else 1,
                AGENTS.index(a),
            ),
        )
        matched = order[:3]

    return sorted(set(matched), key=lambda a: AGENTS.index(a))


def quality_score(text: str, agents):
    score = 10.0
    if len(text) < 35:
        score -= 3.0
    if len(text) > 650:
        score -= 2.5
    if len(text) > 900:
        score -= 5.0
    if MULTI_STEP_RE.search(text):
        score += 2.0
    for agent in agents:
        score += 2.0 if STRONG[agent].search(text) else 0.0
        if agent in RARE_AGENTS:
            score += 1.5
    if len(agents) == 2:
        score += 1.0
    elif len(agents) == 3:
        score += 2.0
    if re.search(r"\b(latest|current|today|real[- ]time|historical)\b", text, re.I):
        score += 1.0
    return score


def load_candidates(dataset_name: str, split: str, max_scan: int, min_score: float):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    best_by_key = {}
    raw_agent_counts = Counter()
    raw_set_counts = Counter()
    seen = 0
    kept = 0

    for row in ds:
        seen += 1
        if max_scan and seen > max_scan:
            break
        if seen % 100000 == 0:
            print(f"[progress] seen={seen:,} kept={kept:,}")

        language = str(row.get("language", "") or "")
        if not language.lower().startswith("english"):
            continue
        if row.get("redacted") or row.get("toxic"):
            continue

        conv = row.get("conversation") or []
        if not conv:
            continue
        first = conv[0]
        if str(first.get("role", "")).lower() != "user":
            continue
        if first.get("redacted") or first.get("toxic"):
            continue

        prompt = clean_text(str(first.get("content", "")))
        if len(prompt) < 30 or len(prompt) > 900:
            continue

        agents = detect_agents(prompt)
        if not agents:
            continue

        score = quality_score(prompt, agents)
        if score < min_score:
            continue

        combo = tuple(agents)
        key = (template_signature(prompt), combo)
        row_out = {
            "prompt": prompt,
            "source_record_id": row.get("conversation_hash") or row.get("conversation_id") or "",
            "model": row.get("model", ""),
            "timestamp": str(row.get("timestamp", "")),
            "country": row.get("country", "") or (first.get("country", "") if isinstance(first, dict) else ""),
            "agents": combo,
            "score": score,
        }
        prev = best_by_key.get(key)
        if prev is None or row_out["score"] > prev["score"] or (
            row_out["score"] == prev["score"] and len(row_out["prompt"]) < len(prev["prompt"])
        ):
            best_by_key[key] = row_out
        kept += 1
        raw_agent_counts.update(combo)
        raw_set_counts[len(combo)] += 1

    deduped = sorted(best_by_key.values(), key=lambda r: (-r["score"], len(r["prompt"])))
    print(f"[done] seen={seen:,} kept={kept:,} deduped={len(deduped):,}")
    return deduped, raw_agent_counts, raw_set_counts, seen


def write_outputs(rows, out_csv: Path, out_md: Path, meta: dict):
    set_counts = Counter(len(row["agents"]) for row in rows)
    agent_counts = Counter()
    for row in rows:
        agent_counts.update(row["agents"])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "prompt",
                "gold_agents",
                "gold_agent_count",
                "source",
                "source_record_id",
                "source_model",
                "source_timestamp",
                "source_country",
                "quality_score",
                "label_source",
                "status",
                "notes",
            ],
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n",
        )
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            prompt = str(row["prompt"]).replace("\r", " ").replace("\n", " ")
            writer.writerow(
                {
                    "prompt_id": i,
                    "prompt": prompt,
                    "gold_agents": "|".join(row["agents"]),
                    "gold_agent_count": len(row["agents"]),
                    "source": "wildchat_4.8m",
                    "source_record_id": row["source_record_id"],
                    "source_model": row["model"],
                    "source_timestamp": row["timestamp"],
                    "source_country": row["country"],
                    "quality_score": f"{row['score']:.2f}",
                    "label_source": "heuristic_filtered_v12",
                    "status": "candidate",
                    "notes": "",
                }
            )

    lines = [
        "# WildChat Agent-12 Candidate Pool",
        "",
        f"- Dataset: `{meta['dataset_name']}`",
        f"- Split: `{meta['split']}`",
        f"- Records scanned: {meta['seen']:,}",
        f"- Heuristic candidates before final selection: {meta['candidate_count']:,}",
        f"- Output rows: {len(rows):,}",
        "- Important: `gold_agents` are schema-guided reference candidates from deterministic rule-based mining, not universal intent annotations.",
        "",
        f"- Set-size distribution: {dict(sorted(set_counts.items()))}",
        "",
        "## Per-Agent Counts",
    ]
    for agent in AGENTS:
        lines.append(f"- {agent}: {agent_counts[agent]}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="allenai/WildChat-4.8M")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_scan", type=int, default=1200000)
    ap.add_argument("--min_score", type=float, default=11.0)
    ap.add_argument("--out_csv", default="wildchat_agent12_candidates.csv")
    ap.add_argument("--out_md", default="wildchat_agent12_candidates_summary.md")
    args = ap.parse_args()

    rows, raw_agent_counts, raw_set_counts, seen = load_candidates(
        dataset_name=args.dataset_name,
        split=args.split,
        max_scan=args.max_scan,
        min_score=args.min_score,
    )
    write_outputs(
        rows=rows,
        out_csv=Path(args.out_csv),
        out_md=Path(args.out_md),
        meta={
            "dataset_name": args.dataset_name,
            "split": args.split,
            "seen": seen,
            "candidate_count": len(rows),
            "raw_agent_counts": dict(raw_agent_counts),
            "raw_set_counts": dict(raw_set_counts),
        },
    )
    print(Path(args.out_csv))
    print(Path(args.out_md))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
