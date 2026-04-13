from typing import Dict, Iterable, List, Sequence, Tuple


CandidateList = Sequence[Tuple[str, float]]


def adjusted_candidates(
    candidates: CandidateList,
    cost_tiers: Dict[str, int],
    cost_lambda: float,
) -> List[Tuple[str, float]]:
    adjusted: List[Tuple[str, float]] = []
    default_cost = max(cost_tiers.values()) if cost_tiers else 2
    for name, score in candidates:
        tier_cost = float(cost_tiers.get(name, default_cost))
        adjusted_score = float(score) - float(cost_lambda) * tier_cost
        adjusted.append((name, adjusted_score))
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def choose_weighted_pred_set(
    candidates: CandidateList,
    threshold: float,
    cost_lambda: float,
    cost_tiers: Dict[str, int],
) -> List[str]:
    ranked = adjusted_candidates(candidates, cost_tiers=cost_tiers, cost_lambda=cost_lambda)
    pred = [name for name, score in ranked if score >= threshold]
    if not pred and ranked:
        pred = [ranked[0][0]]
    return pred


def choose_weighted_pred_sets(
    candidate_lists: Iterable[CandidateList],
    threshold: float,
    cost_lambda: float,
    cost_tiers: Dict[str, int],
) -> List[List[str]]:
    return [
        choose_weighted_pred_set(candidates, threshold=threshold, cost_lambda=cost_lambda, cost_tiers=cost_tiers)
        for candidates in candidate_lists
    ]
