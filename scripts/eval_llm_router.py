import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI


def load_inventory(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    agents = data["agents"]
    names = [agent["name"] for agent in agents]
    return data, agents, names


def build_system_prompt(inventory: dict, agents: list[dict]) -> str:
    parts = []
    parts.append("You are evaluating prompts for a fixed semantic agent routing benchmark.")
    parts.append("Choose the minimal sufficient set of agents required to answer the prompt.")
    parts.append("Rules:")
    parts.append("- Choose only from the fixed agent list below.")
    parts.append("- Return 1 to 3 agents.")
    parts.append("- Use multiple agents only when the prompt explicitly requires multiple distinct capabilities.")
    parts.append("- Do not add optional downstream steps that are not explicitly requested.")
    parts.append("- Be conservative.")
    if inventory.get("extra_guidance"):
        parts.append(f"Extra guidance: {inventory['extra_guidance']}")
    parts.append("")
    parts.append("Fixed agents:")
    for agent in agents:
        parts.append(f"- {agent['name']}: {agent['description']}")
    parts.append("")
    parts.append("Return valid JSON only.")
    return "\n".join(parts)


def schema(agent_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "gold_agents": {
                "type": "array",
                "items": {"type": "string", "enum": agent_names},
                "minItems": 1,
                "maxItems": 3,
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["gold_agents", "confidence", "rationale"],
        "additionalProperties": False,
    }


def set_metrics(pred_names: list[str], gold_names: list[str]):
    pred = set(pred_names)
    gold = set(gold_names)
    tp = len(pred & gold)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    jacc = tp / max(1, len(pred | gold))
    exact = 1.0 if pred == gold else 0.0
    return precision, recall, f1, jacc, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with prompt and gold_agents")
    ap.add_argument("--inventory_json", required=True, help="Fixed agent inventory JSON")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--output_csv", default="llm_router_predictions.csv")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    inventory_path = Path(args.inventory_json)
    output_path = Path(args.output_csv)

    inventory, agents, agent_names = load_inventory(inventory_path)
    system_prompt = build_system_prompt(inventory, agents)
    response_schema = schema(agent_names)

    client = OpenAI()
    rows = list(csv.DictReader(dataset_path.open(encoding="utf-8", newline="")))
    out_rows = []
    sums = Counter()

    latencies_ms = []
    for row in rows:
        prompt = row["prompt"].strip()
        gold = [a.strip() for a in row["gold_agents"].split("|") if a.strip()]

        t0 = time.perf_counter()
        resp = client.responses.create(
            model=args.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Prompt:\n{prompt}"},
            ],
            temperature=0,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "llm_router_eval",
                    "strict": True,
                    "schema": response_schema,
                }
            },
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)
        parsed = json.loads(resp.output_text)
        pred = []
        seen = set()
        for agent in parsed["gold_agents"]:
            if agent in agent_names and agent not in seen:
                pred.append(agent)
                seen.add(agent)

        precision, recall, f1, jacc, exact = set_metrics(pred, gold)
        sums["prec"] += precision
        sums["rec"] += recall
        sums["f1"] += f1
        sums["jacc"] += jacc
        sums["exact"] += exact
        sums["pred_size"] += len(pred)

        out_rows.append(
            {
                "prompt_id": row.get("prompt_id", ""),
                "prompt": prompt,
                "gold_agents": "|".join(gold),
                "pred_agents": "|".join(pred),
                "confidence": parsed["confidence"],
                "rationale": parsed["rationale"],
                "precision": f"{precision:.6f}",
                "recall": f"{recall:.6f}",
                "f1": f"{f1:.6f}",
                "jaccard": f"{jacc:.6f}",
                "exact": f"{exact:.6f}",
                "latency_ms": f"{latency_ms:.3f}",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "prompt",
                "gold_agents",
                "pred_agents",
                "confidence",
                "rationale",
                "precision",
                "recall",
                "f1",
                "jaccard",
                "exact",
                "latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    n = len(rows)
    print("\n=== LLM Router Metrics ===")
    print(f"Samples: {n}")
    print(
        "LLM      Prec: "
        f"{100.0 * sums['prec'] / n:6.2f}% | "
        f"Rec: {100.0 * sums['rec'] / n:6.2f}% | "
        f"F1: {100.0 * sums['f1'] / n:6.2f}% | "
        f"Jacc: {100.0 * sums['jacc'] / n:6.2f}% | "
        f"Exact: {100.0 * sums['exact'] / n:6.2f}% | "
        f"Avg|P|: {sums['pred_size'] / n:5.2f}"
    )
    if latencies_ms:
        lat_sorted = sorted(latencies_ms)
        def pct(vals, p):
            if not vals:
                return 0.0
            k = (p / 100.0) * (len(vals) - 1)
            f = int(k)
            c = min(f + 1, len(vals) - 1)
            if f == c:
                return vals[f]
            return vals[f] + (vals[c] - vals[f]) * (k - f)

        mean_ms = sum(latencies_ms) / len(latencies_ms)
        p50 = pct(lat_sorted, 50)
        p95 = pct(lat_sorted, 95)
        print(f"LLM latency: mean={mean_ms:.2f} ms | p50={p50:.2f} | p95={p95:.2f}")
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
