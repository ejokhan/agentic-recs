"""
Simulated A/B Test — statistical comparison between retrieval systems.
Uses paired tests on the same 480 WANDS queries.

This is what you'd run in production after an online A/B test,
but applied to offline evaluation data.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import bootstrap_ci

EVAL_DIR = Path("data/eval")


def run_ab_test(system_a: str, system_b: str, metric: str = "ndcg@10"):
    """
    Paired A/B test between two retrieval systems.
    Returns: mean difference, 95% CI, p-value, effect size (Cohen's d).
    """
    df_a = pd.read_parquet(EVAL_DIR / f"results_{system_a}.parquet")
    df_b = pd.read_parquet(EVAL_DIR / f"results_{system_b}.parquet")

    # Merge on query_id to ensure paired comparison
    merged = df_a[["query_id", metric]].merge(
        df_b[["query_id", metric]],
        on="query_id",
        suffixes=("_a", "_b")
    )

    scores_a = merged[f"{metric}_a"].to_numpy()
    scores_b = merged[f"{metric}_b"].to_numpy()
    deltas = scores_b - scores_a

    # Paired t-test
    t_stat, p_value_t = stats.ttest_rel(scores_b, scores_a)

    # Wilcoxon signed-rank (non-parametric, more robust)
    try:
        w_stat, p_value_w = stats.wilcoxon(deltas, alternative="two-sided")
    except ValueError:
        w_stat, p_value_w = 0, 1.0

    # Effect size (Cohen's d for paired samples)
    cohens_d = deltas.mean() / deltas.std() if deltas.std() > 0 else 0

    # Bootstrap CI on the difference
    mean_diff, ci_lo, ci_hi = bootstrap_ci(deltas)

    # Win/tie/lose counts
    wins = int(np.sum(deltas > 0.001))
    losses = int(np.sum(deltas < -0.001))
    ties = int(len(deltas) - wins - losses)

    return {
        "system_a": system_a,
        "system_b": system_b,
        "metric": metric,
        "n_queries": len(deltas),
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
        "mean_diff": float(mean_diff),
        "ci_95": (float(ci_lo), float(ci_hi)),
        "p_value_ttest": float(p_value_t),
        "p_value_wilcoxon": float(p_value_w),
        "cohens_d": float(cohens_d),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def print_ab_result(result):
    """Pretty-print A/B test result."""
    r = result
    sig_t = "***" if r["p_value_ttest"] < 0.001 else "**" if r["p_value_ttest"] < 0.01 else "*" if r["p_value_ttest"] < 0.05 else "ns"
    sig_w = "***" if r["p_value_wilcoxon"] < 0.001 else "**" if r["p_value_wilcoxon"] < 0.01 else "*" if r["p_value_wilcoxon"] < 0.05 else "ns"

    print(f"\n{'='*65}")
    print(f"  A/B TEST: {r['system_a'].upper()} (A) vs {r['system_b'].upper()} (B)")
    print(f"  Metric: {r['metric']}  |  N = {r['n_queries']} queries")
    print(f"{'='*65}")
    print(f"  System A ({r['system_a']}):  {r['mean_a']:.4f}")
    print(f"  System B ({r['system_b']}):  {r['mean_b']:.4f}")
    print(f"  Difference (B-A):     {r['mean_diff']:+.4f}  95% CI: [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
    print(f"")
    print(f"  Paired t-test:        p = {r['p_value_ttest']:.6f}  {sig_t}")
    print(f"  Wilcoxon signed-rank: p = {r['p_value_wilcoxon']:.6f}  {sig_w}")
    print(f"  Effect size (d):      {r['cohens_d']:.3f}  {'(small)' if abs(r['cohens_d']) < 0.5 else '(medium)' if abs(r['cohens_d']) < 0.8 else '(large)'}")
    print(f"")
    print(f"  B wins: {r['wins']}  |  A wins: {r['losses']}  |  Ties: {r['ties']}")
    print(f"  Win rate: {r['wins']/r['n_queries']*100:.1f}%")

    if r['p_value_ttest'] < 0.05 and r['mean_diff'] > 0:
        print(f"\n  CONCLUSION: {r['system_b']} is significantly better than {r['system_a']} (p < 0.05)")
    elif r['p_value_ttest'] < 0.05 and r['mean_diff'] < 0:
        print(f"\n  CONCLUSION: {r['system_a']} is significantly better than {r['system_b']} (p < 0.05)")
    else:
        print(f"\n  CONCLUSION: No significant difference (p = {r['p_value_ttest']:.4f})")


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SIMULATED A/B TESTS ON WANDS (480 queries)")
    print("  Statistical comparison of retrieval systems")
    print("="*65)

    # Test 1: BM25 vs Dense
    r1 = run_ab_test("bm25", "dense", "ndcg@10")
    print_ab_result(r1)

    # Test 2: BM25 vs Hybrid
    r2 = run_ab_test("bm25", "hybrid", "ndcg@10")
    print_ab_result(r2)

    # Test 3: Dense vs Hybrid
    r3 = run_ab_test("dense", "hybrid", "ndcg@10")
    print_ab_result(r3)

    # Also test MRR
    print("\n\n" + "="*65)
    print("  SAME TESTS ON MRR")
    print("="*65)

    for a, b in [("bm25", "dense"), ("bm25", "hybrid"), ("dense", "hybrid")]:
        r = run_ab_test(a, b, "mrr")
        print_ab_result(r)
