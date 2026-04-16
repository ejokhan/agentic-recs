"""
Retrieval metrics for WANDS with graded relevance (Exact=2, Partial=1, Irrelevant=0).
"""
from typing import List, Tuple
import numpy as np


def dcg(relevances):
    rels = np.asarray(relevances, dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(ranked_rels, all_query_rels, k=10):
    ranked = ranked_rels[:k]
    ideal = sorted(all_query_rels, reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(ranked) / idcg


def mrr(ranked_rels):
    for i, r in enumerate(ranked_rels, start=1):
        if r >= 1:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked_rels, k=10, threshold=2):
    return float(any(r >= threshold for r in ranked_rels[:k]))


def bootstrap_ci(scores, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = scores[idx].mean()
    mean = float(scores.mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return mean, lo, hi
