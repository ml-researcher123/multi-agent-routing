# dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import random

@dataclass
class PromptExample:
    prompt_id: int
    prompt: str
    gold_agents: List[str]  # agent names (not IDs)
    template_id: int | None = None

def load_dataset_csv(path: str) -> List[PromptExample]:
    df = pd.read_csv(path)
    out: List[PromptExample] = []
    for _, row in df.iterrows():
        gold = str(row["gold_agents"]).split("|")
        template_id = None
        if "template_id" in df.columns:
            try:
                template_id = int(row["template_id"])
            except Exception:
                template_id = None
        out.append(PromptExample(
            prompt_id=int(row["prompt_id"]),
            prompt=str(row["prompt"]),
            gold_agents=[g.strip() for g in gold if g.strip()],
            template_id=template_id,
        ))
    return out

def generate_dataset_300(seed: int = 42, n_samples: int = 1000, balanced: bool = True) -> List[PromptExample]:
    """
    Offline generator for a 300-sample dataset similar to what we produced earlier.
    Useful if you don't want to rely on a CSV file.
    """
    random.seed(seed)

    base = [
        (0, "Fetch wind speed data for the last {n} hours", ["TimeSeriesQueryAgent","APIDataFetchAgent"]),
        (1, "Get rows where customer_id equals {value}", ["SQLQueryAgent"]),
        (2, "Pull recent logs related to {topic}", ["LogRetrievalAgent"]),
        (3, "What columns exist in the {dataset} dataset?", ["MetadataLookupAgent"]),
        (4, "Compute mean and variance for the past {n} days of {metric}", ["StatisticalAnalysisAgent"]),
        (5, "Identify the trend of {metric} over the past {n} days", ["TrendAnalysisAgent"]),
        (6, "Detect unusual spikes or drops in {metric} for {period}", ["AnomalyDetectionAgent"]),
        (7, "Predict {metric} for {horizon}", ["ForecastAgent"]),
        (8, "Plot {metric} for {period}", ["PlotGenerationAgent"]),
        (9, "Summarize what happened in {metric} during {period}", ["SummaryAgent"]),
        (10, "Write a detailed report about {topic} for {period}", ["ReportWriterAgent"]),
        (11, "Forecast {metric} for {horizon} and plot it", ["ForecastAgent","PlotGenerationAgent"]),
        (12, "Plot and summarize {metric} for {period}", ["PlotGenerationAgent","SummaryAgent"]),
        (13, "Retrieve logs and summarize errors for {service}", ["LogRetrievalAgent","SummaryAgent"]),
        (14, "Fetch {metric} for {period} and describe the trend", ["TimeSeriesQueryAgent","TrendAnalysisAgent"]),
        (15, "Get {metric} for {period} and detect trend visually", ["TimeSeriesQueryAgent","PlotGenerationAgent","TrendAnalysisAgent"]),
        (16, "Give me a quick sanity check on the {metric} numbers for {period}", ["StatisticalAnalysisAgent","SummaryAgent"]),
        (17, "Trace anomalies and then draft a brief report for {period}", ["AnomalyDetectionAgent","ReportWriterAgent"]),
        (18, "Forecast {metric} for {horizon} and summarize risks", ["ForecastAgent","SummaryAgent"]),
        (19, "Check if {metric} has outliers in {period}", ["AnomalyDetectionAgent"]),
        (20, "Summarize log patterns for {service} over {period}", ["LogRetrievalAgent","SummaryAgent"]),
        (21, "Generate a report with charts for {metric} in {period}", ["ReportWriterAgent","PlotGenerationAgent"]),
        (22, "Compute correlations between {metric} and {metric_b} for {period}", ["StatisticalAnalysisAgent"]),
        (23, "Plot {metric} and identify its trend for {period}", ["PlotGenerationAgent","TrendAnalysisAgent"]),
        (24, "Forecast {metric} for {horizon} and explain the trend drivers", ["ForecastAgent","TrendAnalysisAgent"]),
        (25, "Summarize and report anomalies in {metric} for {period}", ["AnomalyDetectionAgent","ReportWriterAgent","SummaryAgent"]),
        (26, "Extract API data for {metric} during {period}", ["APIDataFetchAgent","TimeSeriesQueryAgent"]),
        (27, "Write a concise report of {metric} behavior in {period}", ["ReportWriterAgent","SummaryAgent"]),
        (28, "Plot and analyze {metric} for {period}", ["PlotGenerationAgent","StatisticalAnalysisAgent"]),
        (29, "Call the {api} endpoint to fetch {metric} as JSON", ["APIDataFetchAgent"]),
        (30, "Paginate the {api} API to collect {metric} records", ["APIDataFetchAgent"]),
        (31, "Use the {api} REST API with auth to get {metric} for {period}", ["APIDataFetchAgent"]),
        (32, "Fetch external {metric} from {api} and return raw payloads", ["APIDataFetchAgent"]),
        (33, "Retrieve {metric} from internal data warehouse for {period}", ["TimeSeriesQueryAgent"]),
        (34, "Run a SQL query to find {metric} rows with value {value}", ["SQLQueryAgent"]),
        (35, "Filter the {dataset} table where {metric} > {value}", ["SQLQueryAgent"]),
        (36, "List columns and schema for the {dataset} dataset", ["MetadataLookupAgent"]),
        (37, "What fields are available in the {dataset} table?", ["MetadataLookupAgent"]),
        (38, "Fetch logs for {service} with error code {value}", ["LogRetrievalAgent"]),
        (39, "Show WARN and ERROR logs for {service} in {period}", ["LogRetrievalAgent"]),
        (40, "Compute median and std dev for {metric} over {period}", ["StatisticalAnalysisAgent"]),
        (41, "Calculate summary stats for {metric} in {period}", ["StatisticalAnalysisAgent"]),
    ]

    wrappers = [
        lambda s: s,
        lambda s: "Can you " + s[0].lower() + s[1:] + "?",
        lambda s: "Please " + s[0].lower() + s[1:] + ".",
        lambda s: "I need you to " + s[0].lower() + s[1:] + ".",
        lambda s: "Help me " + s[0].lower() + s[1:] + ".",
    ]
    paraphrases = [
        lambda s: s.replace("Fetch", "Pull").replace("Get", "Retrieve"),
        lambda s: s.replace("Summarize", "Give me a summary of"),
        lambda s: s.replace("Plot", "Graph").replace("dashboard", "overview"),
        lambda s: s.replace("Predict", "Forecast"),
        lambda s: s.replace("Compare", "Contrast"),
    ]

    datasets = ["energy","wind","transactions","weather","system_events","ops_metrics","sensor_streams"]
    topics = ["wind performance","energy production","system reliability","forecast accuracy","turbine efficiency","latency spikes","capacity planning"]
    metrics = ["wind speed","wind energy output","power output","temperature","turbine RPM","latency","error rate","throughput"]
    metrics_b = ["temperature","pressure","humidity","power output","latency","error rate"]
    periods = ["today","yesterday","the last 24 hours","the past 7 days","this week","last week","this month","the last 3 days","the last 30 days"]
    horizons = ["tomorrow","next 24 hours","next 72 hours","next week","tomorrow evening","next 2 weeks"]
    period_a = ["this week","last week","this month","yesterday"]
    period_b = ["last week","the week before","last month","the previous day"]
    tasks = ["submit my report","call my supervisor","backup the dataset","run the nightly pipeline"]
    times = ["9pm","10:30pm","tomorrow 8am","next Monday 9am"]
    sites = ["a weather site","a news site","a movies listing site","an online store"]
    apis = ["metrics/v1", "weather/api", "energy/telemetry", "alerts/v2", "sensors/stream"]
    things = ["top headlines","movies","latest releases","pricing"]
    conditions = ["goes above 25 m/s","drops below 1 kW","spikes unusually high","falls 30% below baseline"]
    prefs = ["I prefer charts over tables","Always include confidence intervals","Use metric units","Summaries should be short"]
    values = ["1023","A17","TX-8821","failed","E500"]
    noise_phrases = [
        "FYI", "btw", "asap", "not urgent", "for my notes", "for the weekly review"
    ]
    decoy_clauses = [
        "and include a quick chart",
        "and write a brief summary",
        "and draft a short report",
        "and add a dashboard view",
        "and set an alert if it changes",
    ]

    out: List[PromptExample] = []
    agent_names = sorted({g for _, _, gold in base for g in gold})
    counts = {a: 0 for a in agent_names}

    def choose_template():
        if not balanced:
            return random.choice(base)
        # Prefer templates whose gold agents are currently underrepresented.
        weights = []
        for _, _, gold in base:
            min_count = min(counts[g] for g in gold)
            weights.append(1.0 / (1.0 + min_count))
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for w, tmpl in zip(weights, base):
            acc += w
            if acc >= r:
                return tmpl
        return base[-1]

    for i in range(n_samples):
        template_id, tmpl, gold = choose_template()
        text = tmpl.format(
            n=random.choice([6,12,24,48,72,168]),
            value=random.choice(values),
            topic=random.choice(topics),
            dataset=random.choice(datasets),
            metric=random.choice(metrics),
            metric_b=random.choice(metrics_b),
            period=random.choice(periods),
            horizon=random.choice(horizons),
            period_a=random.choice(period_a),
            period_b=random.choice(period_b),
            task=random.choice(tasks),
            time=random.choice(times),
            site=random.choice(sites),
            thing=random.choice(things),
            condition=random.choice(conditions),
            pref=random.choice(prefs),
            service=random.choice(["API Gateway","Lambda","ETL job","scheduler","frontend"]),
            api=random.choice(apis),
        )
        text = random.choice(wrappers)(text)
        if random.random() < 0.35:
            text = random.choice(paraphrases)(text)
        if random.random() < 0.30:
            text = f"{random.choice(noise_phrases)}: {text}"
        if random.random() < 0.25:
            text = f"{text} {random.choice(decoy_clauses)}."
        for g in gold:
            counts[g] += 1
        out.append(PromptExample(prompt_id=i+1, prompt=text, gold_agents=gold, template_id=template_id))
    return out
