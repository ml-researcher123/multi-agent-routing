# agents.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class Agent:
    agent_id: str
    name: str
    description: str
    cost: float = 0.0  # optional: for RL reward shaping (lower is better)

def get_agents() -> List[Agent]:
    """
    12 agents (1-12). These are dummy agents for routing benchmarks.
    """
    agents = [
        Agent("A01", "TimeSeriesQueryAgent",
              "Retrieves time-series numerical data from internal data stores over a time range and interval; supports hourly/daily windows, filtering by metric name, and returning arrays ready for analysis or plotting. Example: 'get wind speed for last 24 hours'.",
              cost=1.0),
        Agent("A02", "SQLQueryAgent",
              "Executes structured SQL-like queries over tabular data: filters, joins, aggregations, grouping, ID lookups, and simple transformations on rows/columns. Example: 'select rows where customer_id = 1023'.",
              cost=1.5),
        Agent("A03", "APIDataFetchAgent",
              "Fetches live or recent data from REST APIs using endpoints and query parameters; handles auth, pagination, JSON payloads, and external services. Example: 'call /metrics/v1 to fetch telemetry JSON'.",
              cost=2.0),
        Agent("A04", "LogRetrievalAgent",
              "Retrieves and filters system/application logs by keywords, severity, timestamps, patterns, and service names; returns relevant log slices for debugging. Example: 'show ERROR logs for API Gateway'.",
              cost=1.2),
        Agent("A05", "MetadataLookupAgent",
              "Finds schema info, column descriptions, dataset metadata, and what data exists; answers 'what fields are available' and data catalog questions. Example: 'list columns in transactions table'.",
              cost=0.8),
        Agent("A06", "StatisticalAnalysisAgent",
              "Computes statistics on numeric data: mean, variance, correlations, distributions, summary metrics, and quick sanity checks for ranges or outliers. Example: 'compute mean and std dev for temperature'.",
              cost=1.0),
        Agent("A07", "TrendAnalysisAgent",
              "Detects trends and long-term patterns in time series; can identify seasonality, slope changes, and high-level direction over a period. Example: 'identify trend over past 7 days'.",
              cost=1.1),
        Agent("A08", "AnomalyDetectionAgent",
              "Detects spikes, drops, outliers, and unusual patterns in numeric/time-series signals; highlights events worth investigating. Example: 'find anomalies in power output'.",
              cost=1.3),
        Agent("A09", "ForecastAgent",
              "Predicts future values from historical time-series using forecasting logic; supports horizons, projections, and confidence-style summaries. Example: 'forecast next 72 hours of wind speed'.",
              cost=2.2),
        Agent("A10", "PlotGenerationAgent",
              "Creates plots/charts (line, bar, scatter, histogram) to visualize numeric or time-series data; chooses appropriate chart type for the request. Example: 'plot temperature over last week'.",
              cost=1.6),
        Agent("A11", "SummaryAgent",
              "Summarizes data and insights in concise natural language; focuses on key points, changes, conclusions, and next-step takeaways. Example: 'summarize what happened today'.",
              cost=0.9),
        Agent("A12", "ReportWriterAgent",
              "Writes structured detailed reports with sections, narrative, and formatted explanations of analysis; suitable for stakeholders or documentation. Example: 'write a detailed report about last week'.",
              cost=1.8),
    ]
    return agents

def name_to_id_map(agents: List[Agent]) -> Dict[str, str]:
    return {a.name: a.agent_id for a in agents}

def id_to_name_map(agents: List[Agent]) -> Dict[str, str]:
    return {a.agent_id: a.name for a in agents}
