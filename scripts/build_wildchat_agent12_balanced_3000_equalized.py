import argparse
import csv
import io
from collections import Counter
from pathlib import Path

from build_wildchat_agent12_balanced_5000 import AGENTS, build_variants


TARGET_SET = {1: 1800, 2: 900, 3: 300}
TOTAL_APPEARANCES = sum(size * count for size, count in TARGET_SET.items())


def load_variant_rows(input_csv: str) -> list[dict]:
    raw = Path(input_csv).read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="replace")
    source_rows = list(csv.DictReader(io.StringIO(text)))
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
    return [item[1] for item in best.values()]


def compute_support_ceiling(variant_rows: list[dict]) -> dict[str, int]:
    support = {agent: set() for agent in AGENTS}
    for row in variant_rows:
        for agent in row["gold_agents"].split("|"):
            support[agent].add(row["prompt"])
    return {agent: len(prompts) for agent, prompts in support.items()}


def feasible_target(ceiling: dict[str, int]) -> dict[str, int]:
    ideal = int(round(TOTAL_APPEARANCES / len(AGENTS)))
    target = {agent: min(ideal, ceiling[agent]) for agent in AGENTS}
    remaining = TOTAL_APPEARANCES - sum(target.values())

    if remaining <= 0:
        return target

    # Distribute remaining appearances to agents with spare capacity.
    caps = {agent: max(0, ceiling[agent] - target[agent]) for agent in AGENTS}
    while remaining > 0:
        progressed = False
        for agent, cap in sorted(caps.items(), key=lambda kv: kv[1], reverse=True):
            if cap <= 0 or remaining <= 0:
                continue
            target[agent] += 1
            caps[agent] -= 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return target


def hard_caps(target: dict[str, int], ceiling: dict[str, int]) -> dict[str, int]:
    caps = {}
    for agent in AGENTS:
        slack = 120 if agent in {"SQLQueryAgent", "APIDataFetchAgent", "MetadataLookupAgent", "StatisticalAnalysisAgent", "SummaryAgent", "ReportWriterAgent"} else 80
        caps[agent] = min(ceiling[agent], target[agent] + slack)
    return caps


def row_score(row: dict, counts: Counter, target: dict[str, int], caps: dict[str, int]) -> float:
    agents = row["gold_agents"].split("|")
    score = float(row.get("quality_score", "0") or 0)
    score += 1.1 * int(row.get("_promo_score", 0) or 0)

    for agent in agents:
        deficit = max(0, target[agent] - counts[agent])
        over = max(0, counts[agent] - target[agent])
        score += deficit * 0.18
        score -= over * 0.22
        if counts[agent] >= caps[agent]:
            score -= 140.0

    if len(agents) == 3:
        score += 4.5
    elif len(agents) == 2:
        score += 1.5
    return score


def greedy_pick(pool: list[dict], needed: int, counts: Counter, used_prompts: set[str], target: dict[str, int], caps: dict[str, int], relax_caps: bool) -> list[dict]:
    chosen = []
    while len(chosen) < needed:
        best_row = None
        best_score = None
        for row in pool:
            if row["prompt"] in used_prompts:
                continue
            agents = row["gold_agents"].split("|")
            if not relax_caps and any(counts[a] >= caps[a] for a in agents):
                continue
            score = row_score(row, counts, target, caps)
            if best_score is None or score > best_score:
                best_score = score
                best_row = row
        if best_row is None:
            break
        used_prompts.add(best_row["prompt"])
        chosen.append(best_row)
        counts.update(best_row["gold_agents"].split("|"))
    return chosen


def initial_selection(variant_rows: list[dict], target: dict[str, int], caps: dict[str, int]) -> tuple[list[dict], Counter]:
    triples = [r for r in variant_rows if int(r["gold_agent_count"]) == 3]
    doubles = [r for r in variant_rows if int(r["gold_agent_count"]) == 2]
    singles = [r for r in variant_rows if int(r["gold_agent_count"]) == 1]

    counts = Counter()
    used_prompts = set()
    selected = []
    for stage, pool in [(3, triples), (2, doubles), (1, singles)]:
        need = TARGET_SET[stage]
        picked = greedy_pick(pool, need, counts, used_prompts, target, caps, relax_caps=False)
        selected.extend(picked)
        if len(picked) < need:
            selected.extend(greedy_pick(pool, need - len(picked), counts, used_prompts, target, caps, relax_caps=True))
    return selected, counts


def objective(counts: Counter, target: dict[str, int]) -> float:
    return sum((counts[agent] - target[agent]) ** 2 for agent in AGENTS)


def local_search(selected: list[dict], variant_rows: list[dict], target: dict[str, int]) -> tuple[list[dict], Counter]:
    by_stage = {1: [], 2: [], 3: []}
    counts = Counter()
    for row in selected:
        stage = int(row["gold_agent_count"])
        by_stage[stage].append(row)
        counts.update(row["gold_agents"].split("|"))

    pool_by_stage = {
        1: [r for r in variant_rows if int(r["gold_agent_count"]) == 1],
        2: [r for r in variant_rows if int(r["gold_agent_count"]) == 2],
        3: [r for r in variant_rows if int(r["gold_agent_count"]) == 3],
    }
    selected_prompt_counts = Counter(row["prompt"] for row in selected)
    current = objective(counts, target)

    for _ in range(4):
        improved = 0
        for stage in [3, 2, 1]:
            deficits = sorted(AGENTS, key=lambda agent: target[agent] - counts[agent], reverse=True)
            candidates = []
            for agent in deficits[:6]:
                if target[agent] - counts[agent] <= 0:
                    continue
                candidates.extend(
                    row for row in pool_by_stage[stage]
                    if selected_prompt_counts[row["prompt"]] == 0 and agent in row["gold_agents"].split("|")
                )

            dedup = []
            seen = set()
            for row in candidates:
                key = (row["prompt"], row["gold_agents"])
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(row)

            dedup = sorted(
                dedup,
                key=lambda row: (
                    float(row.get("quality_score", "0") or 0),
                    int(row.get("_promo_score", 0) or 0),
                ),
                reverse=True,
            )[:1400]

            for cand in dedup:
                cand_agents = cand["gold_agents"].split("|")
                best_swap = None
                best_value = current
                for sel in by_stage[stage]:
                    sel_agents = sel["gold_agents"].split("|")
                    new_counts = counts.copy()
                    new_counts.subtract(sel_agents)
                    new_counts.update(cand_agents)
                    candidate_value = objective(new_counts, target)
                    if candidate_value < best_value:
                        best_value = candidate_value
                        best_swap = sel
                if best_swap is None:
                    continue

                swap_agents = best_swap["gold_agents"].split("|")
                counts.subtract(swap_agents)
                counts.update(cand_agents)
                selected_prompt_counts[best_swap["prompt"]] -= 1
                if selected_prompt_counts[best_swap["prompt"]] <= 0:
                    del selected_prompt_counts[best_swap["prompt"]]
                selected_prompt_counts[cand["prompt"]] += 1
                by_stage[stage].remove(best_swap)
                by_stage[stage].append(cand)
                current = best_value
                improved += 1

        if improved == 0:
            break

    final_rows = by_stage[3] + by_stage[2] + by_stage[1]
    final_rows = sorted(final_rows, key=lambda row: (int(row["gold_agent_count"]), float(row.get("quality_score", "0") or 0)), reverse=True)
    final_counts = Counter()
    for row in final_rows:
        final_counts.update(row["gold_agents"].split("|"))
    return final_rows, final_counts


def write_outputs(rows: list[dict], output_csv: str, output_md: str, ceiling: dict[str, int], target: dict[str, int], counts: Counter) -> None:
    set_counts = Counter(int(row["gold_agent_count"]) for row in rows)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id", "prompt", "gold_agents", "gold_agent_count", "source", "source_record_id",
                "source_model", "source_timestamp", "source_country", "quality_score", "label_source",
                "status", "notes",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = {k: row.get(k, "") for k in writer.fieldnames}
            out["prompt_id"] = idx
            out["label_source"] = "heuristic_balanced_3000_equalized_v12"
            prev = out.get("notes", "")
            extra = f"variant={row.get('_variant_kind', 'orig')}; promo_score={row.get('_promo_score', 0)}"
            out["notes"] = f"{prev}; {extra}".strip("; ")
            writer.writerow(out)

    lines = [
        "# WildChat Agent-12 Balanced 3000 Equalized Summary",
        "",
        f"- Output rows: {len(rows)}",
        f"- Set-size distribution: {dict(sorted(set_counts.items()))}",
        "- Target set-size mix was enforced at 1800 single-agent / 900 two-agent / 300 three-agent prompts.",
        "",
        "## Unique-Prompt Ceiling in Current Variant Pool",
    ]
    for agent in AGENTS:
        lines.append(f"- {agent}: {ceiling[agent]}")

    lines.extend(["", "## Feasible Balancing Target Used in Local Search"])
    for agent in AGENTS:
        lines.append(f"- {agent}: {target[agent]}")

    lines.extend(["", "## Achieved Per-Agent Counts"])
    for agent in AGENTS:
        lines.append(f"- {agent}: {counts[agent]}")

    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="wildchat_agent12_scan2500k_all_candidates.csv")
    ap.add_argument("--output_csv", default="wildchat_agent12_balanced_3000_equalized_candidates.csv")
    ap.add_argument("--output_md", default="wildchat_agent12_balanced_3000_equalized_summary.md")
    args = ap.parse_args()

    variant_rows = load_variant_rows(args.input_csv)
    ceiling = compute_support_ceiling(variant_rows)
    target = feasible_target(ceiling)
    caps = hard_caps(target, ceiling)
    selected, counts = initial_selection(variant_rows, target, caps)
    final_rows, final_counts = local_search(selected, variant_rows, target)

    write_outputs(final_rows, args.output_csv, args.output_md, ceiling, target, final_counts)

    set_counts = Counter(int(row["gold_agent_count"]) for row in final_rows)
    print(Path(args.output_csv))
    print(Path(args.output_md))
    print("rows", len(final_rows))
    print("set_sizes", dict(sorted(set_counts.items())))
    print("agent_counts", {agent: final_counts[agent] for agent in AGENTS})


if __name__ == "__main__":
    main()
