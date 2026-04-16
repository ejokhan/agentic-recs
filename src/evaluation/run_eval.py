"""Evaluation harness — score any retriever against WANDS human labels."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import ndcg_at_k, mrr, hit_at_k, bootstrap_ci

PROCESSED = Path("data/processed")
K = 10


def score_retriever(retriever, name, k=K):
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")

    labels_by_query = {
        qid: dict(zip(g["product_id"].to_numpy(), g["relevance"].to_numpy()))
        for qid, g in labels.groupby("query_id")
    }

    rows = []
    t0 = time.time()
    for _, row in queries.iterrows():
        qid, q = row["query_id"], row["query"]
        rel_map = labels_by_query.get(qid, {})
        all_rels = list(rel_map.values())

        top_ids, _ = retriever.retrieve(q, k=k)
        ranked_rels = [int(rel_map.get(int(pid), 0)) for pid in top_ids]

        rows.append({
            "query_id": qid,
            "query": q,
            "n_labeled": len(all_rels),
            "n_exact": sum(1 for r in all_rels if r == 2),
            "ndcg@10": ndcg_at_k(ranked_rels, all_rels, k=k),
            "mrr": mrr(ranked_rels),
            "hit@10_exact": hit_at_k(ranked_rels, k=k, threshold=2),
            "hit@10_any": hit_at_k(ranked_rels, k=k, threshold=1),
        })

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)

    print(f"\n=== {name.upper()} — evaluated {len(df)} queries in {elapsed:.1f}s ===\n")

    for metric in ["ndcg@10", "mrr", "hit@10_exact", "hit@10_any"]:
        mean, lo, hi = bootstrap_ci(df[metric].to_numpy())
        print(f"  {metric:15s}  {mean:.4f}   95% CI: [{lo:.4f}, {hi:.4f}]")

    hard = df[df["n_exact"] >= 1]
    print(f"\n  On {len(hard)} queries with >=1 Exact match:")
    for metric in ["ndcg@10", "hit@10_exact"]:
        mean, lo, hi = bootstrap_ci(hard[metric].to_numpy())
        print(f"    {metric:15s}  {mean:.4f}   95% CI: [{lo:.4f}, {hi:.4f}]")

    out_path = Path("data/eval") / f"results_{name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nPer-query results saved -> {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("retriever", choices=["bm25", "dense"])
    args = parser.parse_args()

    if args.retriever == "bm25":
        from src.retrieval.bm25_baseline import BM25Retriever
        r = BM25Retriever.load_or_build()
        score_retriever(r, name="bm25")
    elif args.retriever == "dense":
        print("Dense retriever not yet implemented — waiting for embeddings job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
