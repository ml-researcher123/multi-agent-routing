import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze inter-prompt consistency via prompt similarity and label overlap.")
    ap.add_argument("--input_csvs", required=True, help="Comma-separated CSV paths.")
    ap.add_argument("--output_prefix", required=True)
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--labels_col", default="gold_agents")
    ap.add_argument("--backend", choices=["sentence_transformer", "tfidf"], default="tfidf")
    ap.add_argument("--model_name", default="all-mpnet-base-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap for debugging; 0 means all rows.")
    return ap.parse_args()


def parse_label_set(value: object) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    sep = "|" if "|" in text else ","
    return {part.strip() for part in text.split(sep) if part.strip()}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def exact_match(a: Set[str], b: Set[str]) -> int:
    return int(a == b)


def any_overlap(a: Set[str], b: Set[str]) -> int:
    return int(len(a & b) > 0)


def load_frames(paths: Sequence[str], prompt_col: str, labels_col: str, max_rows: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        df = pd.read_csv(path)
        if prompt_col not in df.columns or labels_col not in df.columns:
            raise ValueError(f"{path} is missing required columns {prompt_col!r} and/or {labels_col!r}")
        df = df.copy()
        df["_source_file"] = path.name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if max_rows > 0:
        full = full.head(max_rows).copy()
    full["_row_uid"] = np.arange(len(full))
    return full


def embed_prompts_sentence_transformer(texts: Sequence[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32, copy=False)


def embed_prompts_tfidf(texts: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )
    x = vectorizer.fit_transform(list(texts))
    x = normalize(x, norm="l2", copy=False)
    return x.astype(np.float32)


def build_neighbor_pairs(df: pd.DataFrame, embeddings, top_k: int) -> pd.DataFrame:
    label_sets = df["_label_set"].tolist()
    n = len(df)
    if top_k >= n:
        top_k = max(1, n - 1)

    if hasattr(embeddings, "toarray"):
        sim_matrix = (embeddings @ embeddings.T).toarray().astype(np.float32, copy=False)
    else:
        sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -np.inf)

    neighbor_rows = []
    for i in range(n):
        idxs = np.argpartition(-sim_matrix[i], top_k)[:top_k]
        idxs = idxs[np.argsort(-sim_matrix[i, idxs])]
        src_labels = label_sets[i]
        for rank, j in enumerate(idxs, start=1):
            tgt_labels = label_sets[j]
            neighbor_rows.append(
                {
                    "src_uid": int(df.iloc[i]["_row_uid"]),
                    "tgt_uid": int(df.iloc[j]["_row_uid"]),
                    "src_source_file": df.iloc[i]["_source_file"],
                    "tgt_source_file": df.iloc[j]["_source_file"],
                    "src_prompt_id": df.iloc[i]["prompt_id"] if "prompt_id" in df.columns else "",
                    "tgt_prompt_id": df.iloc[j]["prompt_id"] if "prompt_id" in df.columns else "",
                    "neighbor_rank": rank,
                    "cosine": float(sim_matrix[i, j]),
                    "jaccard": float(jaccard(src_labels, tgt_labels)),
                    "share_any": int(any_overlap(src_labels, tgt_labels)),
                    "exact_match": int(exact_match(src_labels, tgt_labels)),
                    "src_label_count": len(src_labels),
                    "tgt_label_count": len(tgt_labels),
                    "src_labels": "|".join(sorted(src_labels)),
                    "tgt_labels": "|".join(sorted(tgt_labels)),
                }
            )
    return pd.DataFrame(neighbor_rows)


def build_random_pairs(df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label_sets = df["_label_set"].tolist()
    n = len(df)
    rows = []
    for _ in range(count):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        a = label_sets[i]
        b = label_sets[j]
        rows.append(
            {
                "src_uid": int(df.iloc[i]["_row_uid"]),
                "tgt_uid": int(df.iloc[j]["_row_uid"]),
                "jaccard": float(jaccard(a, b)),
                "share_any": int(any_overlap(a, b)),
                "exact_match": int(exact_match(a, b)),
                "src_label_count": len(a),
                "tgt_label_count": len(b),
            }
        )
    return pd.DataFrame(rows)


def summarize(neighbors: pd.DataFrame, random_pairs: pd.DataFrame) -> dict:
    rank1 = neighbors[neighbors["neighbor_rank"] == 1].copy()
    high_cut = float(neighbors["cosine"].quantile(0.9))
    low_cut = float(neighbors["cosine"].quantile(0.1))
    high = neighbors[neighbors["cosine"] >= high_cut]
    low = neighbors[neighbors["cosine"] <= low_cut]

    return {
        "num_prompts": int(neighbors["src_uid"].nunique()),
        "top_k": int(neighbors["neighbor_rank"].max()),
        "neighbor_pairs": int(len(neighbors)),
        "random_pairs": int(len(random_pairs)),
        "rank1_mean_cosine": float(rank1["cosine"].mean()),
        "rank1_mean_jaccard": float(rank1["jaccard"].mean()),
        "rank1_share_any_rate": float(rank1["share_any"].mean()),
        "rank1_exact_rate": float(rank1["exact_match"].mean()),
        "topk_mean_cosine": float(neighbors["cosine"].mean()),
        "topk_mean_jaccard": float(neighbors["jaccard"].mean()),
        "topk_share_any_rate": float(neighbors["share_any"].mean()),
        "topk_exact_rate": float(neighbors["exact_match"].mean()),
        "random_mean_jaccard": float(random_pairs["jaccard"].mean()),
        "random_share_any_rate": float(random_pairs["share_any"].mean()),
        "random_exact_rate": float(random_pairs["exact_match"].mean()),
        "high_sim_threshold": high_cut,
        "high_sim_mean_jaccard": float(high["jaccard"].mean()),
        "high_sim_share_any_rate": float(high["share_any"].mean()),
        "low_sim_threshold": low_cut,
        "low_sim_mean_jaccard": float(low["jaccard"].mean()),
        "low_sim_share_any_rate": float(low["share_any"].mean()),
        "cosine_jaccard_corr": float(neighbors["cosine"].corr(neighbors["jaccard"], method="pearson")),
    }


def write_summary_md(summary: dict, output_path: Path, backend: str) -> None:
    lines = [
        "# Inter-Prompt Consistency Analysis",
        "",
        f"- prompts: `{summary['num_prompts']}`",
        f"- top-k neighbors: `{summary['top_k']}`",
        f"- similarity backend: `{backend}`",
        f"- neighbor pairs analyzed: `{summary['neighbor_pairs']}`",
        f"- random baseline pairs: `{summary['random_pairs']}`",
        "",
        "## Main results",
        "",
        f"- rank-1 nearest neighbor mean cosine: `{summary['rank1_mean_cosine']:.4f}`",
        f"- rank-1 nearest neighbor mean Jaccard: `{summary['rank1_mean_jaccard']:.4f}`",
        f"- rank-1 share-any-label rate: `{summary['rank1_share_any_rate']:.4f}`",
        f"- rank-1 exact label-set match rate: `{summary['rank1_exact_rate']:.4f}`",
        "",
        f"- top-k mean Jaccard: `{summary['topk_mean_jaccard']:.4f}`",
        f"- top-k share-any-label rate: `{summary['topk_share_any_rate']:.4f}`",
        f"- top-k exact label-set match rate: `{summary['topk_exact_rate']:.4f}`",
        "",
        f"- random-pair mean Jaccard: `{summary['random_mean_jaccard']:.4f}`",
        f"- random-pair share-any-label rate: `{summary['random_share_any_rate']:.4f}`",
        f"- random-pair exact label-set match rate: `{summary['random_exact_rate']:.4f}`",
        "",
        "## Similarity gradient",
        "",
        f"- top 10% cosine threshold: `{summary['high_sim_threshold']:.4f}`",
        f"- top 10% cosine mean Jaccard: `{summary['high_sim_mean_jaccard']:.4f}`",
        f"- top 10% cosine share-any-label rate: `{summary['high_sim_share_any_rate']:.4f}`",
        "",
        f"- bottom 10% cosine threshold: `{summary['low_sim_threshold']:.4f}`",
        f"- bottom 10% cosine mean Jaccard: `{summary['low_sim_mean_jaccard']:.4f}`",
        f"- bottom 10% cosine share-any-label rate: `{summary['low_sim_share_any_rate']:.4f}`",
        "",
        f"- Pearson correlation between cosine similarity and label-set Jaccard: `{summary['cosine_jaccard_corr']:.4f}`",
        "",
        "## Interpretation",
        "",
        "Higher label overlap among nearest prompts under the chosen text representation than among random pairs suggests that",
        "the routing labels are systematic rather than arbitrary. This is an internal consistency signal: it supports learnable",
        "structure under the benchmark protocol, but it does not by itself establish unique human-level correctness for each prompt.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csvs = [p.strip() for p in args.input_csvs.split(",") if p.strip()]
    df = load_frames(input_csvs, args.prompt_col, args.labels_col, args.max_rows)
    df["_label_set"] = df[args.labels_col].apply(parse_label_set)

    texts = df[args.prompt_col].astype(str).tolist()
    if args.backend == "sentence_transformer":
        embeddings = embed_prompts_sentence_transformer(texts, args.model_name, args.batch_size)
    else:
        embeddings = embed_prompts_tfidf(texts)
    neighbors = build_neighbor_pairs(df, embeddings, args.top_k)
    random_pairs = build_random_pairs(df, len(neighbors), args.random_seed)
    summary = summarize(neighbors, random_pairs)

    prefix = Path(args.output_prefix)
    neighbors.to_csv(prefix.with_name(prefix.name + "_neighbor_pairs.csv"), index=False)
    random_pairs.to_csv(prefix.with_name(prefix.name + "_random_pairs.csv"), index=False)
    pd.DataFrame([summary]).to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    write_summary_md(summary, prefix.with_name(prefix.name + "_summary.md"), args.backend)

    print("Saved:")
    print(prefix.with_name(prefix.name + "_neighbor_pairs.csv"))
    print(prefix.with_name(prefix.name + "_random_pairs.csv"))
    print(prefix.with_name(prefix.name + "_summary.csv"))
    print(prefix.with_name(prefix.name + "_summary.md"))
    print("Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
