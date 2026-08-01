import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _dedup_by_prompt(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    dup = 0
    for row in rows:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        if prompt in seen:
            dup += 1
            continue
        seen[prompt] = row
    if dup:
        print(f"[warn] dropped {dup} duplicate prompts during split")
    return list(seen.values())


def _adjust_allocations(
    groups: dict[str, list[dict]],
    alloc: dict[str, list[int]],
    target_idx: int,
    target_total: int,
):
    current = sum(v[target_idx] for v in alloc.values())
    diff = target_total - current
    if diff == 0:
        return
    keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    step = 1 if diff > 0 else -1
    i = 0
    max_iters = len(keys) * 10
    while diff != 0 and i < max_iters:
        key = keys[i % len(keys)]
        counts = alloc[key]
        new_val = counts[target_idx] + step
        if 0 <= new_val <= len(groups[key]):
            counts[target_idx] = new_val
            remainder = len(groups[key]) - counts[0] - counts[1]
            if remainder >= 0:
                counts[2] = remainder
                diff -= step
            else:
                counts[target_idx] -= step
        i += 1


def stratified_split(
    rows: list[dict],
    train_size: int,
    dev_size: int,
    test_size: int,
    seed: int,
    stratify_by: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    total = train_size + dev_size + test_size
    if total <= 0:
        raise ValueError("Total split size must be positive.")

    if stratify_by == "none":
        rng.shuffle(rows)
        return rows[:train_size], rows[train_size : train_size + dev_size], rows[train_size + dev_size : total]

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(stratify_by, "unknown")
        groups[str(key)].append(row)

    for key in groups:
        rng.shuffle(groups[key])

    ratio_train = train_size / total
    ratio_dev = dev_size / total

    alloc: dict[str, list[int]] = {}
    for key, group in groups.items():
        n = len(group)
        train_n = int(round(n * ratio_train))
        dev_n = int(round(n * ratio_dev))
        if train_n + dev_n > n:
            dev_n = max(0, n - train_n)
        test_n = n - train_n - dev_n
        alloc[key] = [train_n, dev_n, test_n]

    _adjust_allocations(groups, alloc, 0, train_size)
    _adjust_allocations(groups, alloc, 1, dev_size)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    test_rows: list[dict] = []
    for key, group in groups.items():
        train_n, dev_n, _ = alloc[key]
        train_rows.extend(group[:train_n])
        dev_rows.extend(group[train_n : train_n + dev_n])
        test_rows.extend(group[train_n + dev_n :])

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)
    return train_rows, dev_rows, test_rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _count_gold(rows: list[dict]) -> Counter:
    counts = Counter()
    for row in rows:
        gcount = row.get("gold_agent_count", "")
        counts[str(gcount)] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_dev", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_size", type=int, default=2400)
    ap.add_argument("--dev_size", type=int, default=300)
    ap.add_argument("--test_size", type=int, default=300)
    ap.add_argument(
        "--stratify_by",
        default="gold_agent_count",
        choices=["gold_agent_count", "none"],
    )
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    rows = _load_rows(Path(args.input_csv))
    if args.dedup:
        rows = _dedup_by_prompt(rows)

    train_rows, dev_rows, test_rows = stratified_split(
        rows=rows,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        seed=args.seed,
        stratify_by=args.stratify_by,
    )

    _write_rows(Path(args.out_train), train_rows)
    _write_rows(Path(args.out_dev), dev_rows)
    _write_rows(Path(args.out_test), test_rows)

    print(f"train: {len(train_rows)} | dev: {len(dev_rows)} | test: {len(test_rows)}")
    print(f"gold_agent_count (train): {_count_gold(train_rows)}")
    print(f"gold_agent_count (dev): {_count_gold(dev_rows)}")
    print(f"gold_agent_count (test): {_count_gold(test_rows)}")


if __name__ == "__main__":
    main()
