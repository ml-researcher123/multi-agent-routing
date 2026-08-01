from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "dev", "test")


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Create a leakage-resistant train/dev/test split by keeping all prompt "
            "pairs above a frozen-embedding cosine threshold in the same component."
        )
    )
    ap.add_argument(
        "--input_csvs",
        required=True,
        help="Comma-separated CSVs whose union forms the complete benchmark.",
    )
    ap.add_argument(
        "--embedding_model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    ap.add_argument("--similarity_threshold", type=float, default=0.82)
    ap.add_argument("--train_fraction", type=float, default=0.80)
    ap.add_argument("--dev_fraction", type=float, default=0.10)
    ap.add_argument("--test_fraction", type=float, default=0.10)
    ap.add_argument("--assignment_trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encode_batch_size", type=int, default=64)
    ap.add_argument("--similarity_block_size", type=int, default=256)
    ap.add_argument("--output_dir", default="outputs/semantic_group_split")
    return ap.parse_args()


def normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def parse_agents(raw: object) -> List[str]:
    text = str(raw).strip()
    if not text:
        return []
    separator = "|" if "|" in text else ","
    return [part.strip() for part in text.split(separator) if part.strip()]


def load_union(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = {"prompt_id", "prompt", "gold_agents"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["_normalized_prompt"] = data["prompt"].map(normalized_prompt)
    duplicate_mask = data["_normalized_prompt"].duplicated(keep="first")
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        print(f"[split] dropping {duplicate_count} normalized duplicate prompts", flush=True)
        data = data.loc[~duplicate_mask].copy()
    return data.reset_index(drop=True)


def build_similarity_components(
    embeddings: np.ndarray,
    threshold: float,
    block_size: int,
) -> np.ndarray:
    n = embeddings.shape[0]
    uf = UnionFind(n)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        similarities = embeddings[start:stop] @ embeddings.T
        for local_idx, row in enumerate(similarities):
            idx = start + local_idx
            neighbors = np.flatnonzero(row[idx + 1 :] >= threshold) + idx + 1
            for neighbor in neighbors.tolist():
                uf.union(idx, int(neighbor))
        print(f"[split] similarity graph {stop}/{n}", flush=True)

    roots = np.asarray([uf.find(i) for i in range(n)], dtype=np.int32)
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids.astype(np.int32)


def component_features(
    data: pd.DataFrame,
    group_ids: np.ndarray,
) -> Tuple[np.ndarray, List[str], List[int]]:
    agent_names = sorted({agent for raw in data["gold_agents"] for agent in parse_agents(raw)})
    cardinalities = sorted({len(parse_agents(raw)) for raw in data["gold_agents"]})
    agent_to_idx = {name: i for i, name in enumerate(agent_names)}
    card_to_idx = {value: i for i, value in enumerate(cardinalities)}
    num_groups = int(group_ids.max()) + 1
    features = np.zeros(
        (num_groups, 1 + len(agent_names) + len(cardinalities)),
        dtype=np.float64,
    )
    for row_idx, raw in enumerate(data["gold_agents"]):
        group_idx = int(group_ids[row_idx])
        agents = parse_agents(raw)
        features[group_idx, 0] += 1.0
        for agent in agents:
            features[group_idx, 1 + agent_to_idx[agent]] += 1.0
        features[group_idx, 1 + len(agent_names) + card_to_idx[len(agents)]] += 1.0
    return features, agent_names, cardinalities


def assignment_score(
    totals: np.ndarray,
    targets: np.ndarray,
    overall: np.ndarray,
    num_agents: int,
) -> float:
    sizes = totals[:, 0]
    target_sizes = targets[:, 0]
    if np.any(sizes <= 0):
        return float("inf")

    size_error = float(np.mean(np.abs(sizes - target_sizes) / np.maximum(target_sizes, 1.0)))

    label_totals = totals[:, 1 : 1 + num_agents]
    overall_label_prevalence = overall[1 : 1 + num_agents] / max(overall[0], 1.0)
    label_prevalence = label_totals / sizes[:, None]
    label_error = float(np.mean(np.abs(label_prevalence - overall_label_prevalence[None, :])))

    card_totals = totals[:, 1 + num_agents :]
    overall_card_prevalence = overall[1 + num_agents :] / max(overall[0], 1.0)
    card_prevalence = card_totals / sizes[:, None]
    card_error = float(np.mean(np.abs(card_prevalence - overall_card_prevalence[None, :])))

    missing_label_penalty = float(np.sum(label_totals == 0))
    return 8.0 * size_error + 4.0 * label_error + 2.0 * card_error + missing_label_penalty


def select_component_assignment(
    features: np.ndarray,
    fractions: np.ndarray,
    trials: int,
    seed: int,
    num_agents: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    overall = features.sum(axis=0)
    targets = fractions[:, None] * overall[None, :]
    best_assignment: np.ndarray | None = None
    best_totals: np.ndarray | None = None
    best_score = float("inf")

    for trial in range(trials):
        assignment = rng.choice(3, size=features.shape[0], p=fractions)
        totals = np.zeros((3, features.shape[1]), dtype=np.float64)
        for split_idx in range(3):
            mask = assignment == split_idx
            if np.any(mask):
                totals[split_idx] = features[mask].sum(axis=0)
        score = assignment_score(totals, targets, overall, num_agents=num_agents)
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_totals = totals.copy()
        if (trial + 1) % 5000 == 0:
            print(f"[split] assignment search {trial + 1}/{trials}; best={best_score:.6f}", flush=True)

    if best_assignment is None or best_totals is None:
        raise RuntimeError("Failed to assign semantic components to splits.")
    return best_assignment, best_score, best_totals


def max_cross_split_similarity(
    embeddings: np.ndarray,
    split_indices: Sequence[np.ndarray],
    block_size: int,
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for left_idx in range(3):
        for right_idx in range(left_idx + 1, 3):
            left = split_indices[left_idx]
            right = split_indices[right_idx]
            maximum = -1.0
            for start in range(0, len(left), block_size):
                block = embeddings[left[start : start + block_size]]
                maximum = max(maximum, float(np.max(block @ embeddings[right].T)))
            key = f"{SPLIT_NAMES[left_idx]}_{SPLIT_NAMES[right_idx]}"
            output[key] = maximum
    return output


def sha256_rows(data: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{row.prompt_id}\t{normalized_prompt(row.prompt)}\t{row.gold_agents}"
        for row in data.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_distribution(
    data: pd.DataFrame,
    split_names: np.ndarray,
    output_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    agents = sorted({agent for raw in data["gold_agents"] for agent in parse_agents(raw)})
    for split_name in SPLIT_NAMES:
        subset = data.loc[split_names == split_name]
        for agent in agents:
            count = sum(agent in parse_agents(raw) for raw in subset["gold_agents"])
            rows.append(
                {
                    "split": split_name,
                    "feature": f"agent:{agent}",
                    "count": count,
                    "fraction": count / max(len(subset), 1),
                }
            )
        for cardinality in sorted({len(parse_agents(raw)) for raw in data["gold_agents"]}):
            count = sum(len(parse_agents(raw)) == cardinality for raw in subset["gold_agents"])
            rows.append(
                {
                    "split": split_name,
                    "feature": f"cardinality:{cardinality}",
                    "count": count,
                    "fraction": count / max(len(subset), 1),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    fractions = np.asarray(
        [args.train_fraction, args.dev_fraction, args.test_fraction],
        dtype=np.float64,
    )
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Train/dev/test fractions must sum to 1.")
    if not 0.0 < args.similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1.")

    input_paths = [Path(item.strip()) for item in args.input_csvs.split(",") if item.strip()]
    data = load_union(input_paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    print(f"[split] loading frozen grouping model: {args.embedding_model}", flush=True)
    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(
        data["prompt"].astype(str).tolist(),
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    group_ids = build_similarity_components(
        embeddings,
        threshold=args.similarity_threshold,
        block_size=args.similarity_block_size,
    )
    features, agent_names, cardinalities = component_features(data, group_ids)
    assignment, score, totals = select_component_assignment(
        features,
        fractions=fractions,
        trials=args.assignment_trials,
        seed=args.seed,
        num_agents=len(agent_names),
    )

    row_split_indices = assignment[group_ids]
    row_split_names = np.asarray([SPLIT_NAMES[idx] for idx in row_split_indices], dtype=object)
    split_row_indices = [np.flatnonzero(row_split_indices == idx) for idx in range(3)]
    cross_similarity = max_cross_split_similarity(
        embeddings,
        split_indices=split_row_indices,
        block_size=args.similarity_block_size,
    )

    export = data.drop(columns=["_normalized_prompt"]).copy()
    export["semantic_group_id"] = group_ids
    export["semantic_split"] = row_split_names
    for split_name in SPLIT_NAMES:
        path = output_dir / f"semantic_group_{split_name}.csv"
        export.loc[row_split_names == split_name].to_csv(path, index=False)
        print(f"[split] wrote {path} ({int(np.sum(row_split_names == split_name))} rows)")

    write_distribution(export, row_split_names, output_dir / "semantic_group_distribution.csv")
    np.save(output_dir / "semantic_group_embeddings.npy", embeddings)

    group_sizes = np.bincount(group_ids)
    manifest = {
        "input_csvs": [str(path) for path in input_paths],
        "input_rows_after_normalized_deduplication": int(len(data)),
        "input_sha256": sha256_rows(export),
        "embedding_model": args.embedding_model,
        "embedding_dimension": int(embeddings.shape[1]),
        "similarity_threshold": args.similarity_threshold,
        "grouping_rule": (
            "Connected components of the exact all-pairs cosine graph; every pair at or "
            "above the threshold is forced into the same split."
        ),
        "num_semantic_groups": int(len(group_sizes)),
        "largest_group_rows": int(group_sizes.max()),
        "singleton_groups": int(np.sum(group_sizes == 1)),
        "requested_fractions": dict(zip(SPLIT_NAMES, fractions.tolist())),
        "actual_rows": {
            name: int(np.sum(row_split_names == name)) for name in SPLIT_NAMES
        },
        "assignment_trials": args.assignment_trials,
        "assignment_seed": args.seed,
        "assignment_score": score,
        "agent_names": agent_names,
        "cardinalities": cardinalities,
        "max_cross_split_cosine": cross_similarity,
        "max_cross_split_below_threshold": all(
            value < args.similarity_threshold + 1e-6 for value in cross_similarity.values()
        ),
        "component_feature_totals": totals.tolist(),
    }
    manifest_path = output_dir / "semantic_group_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[split] wrote {manifest_path}")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
