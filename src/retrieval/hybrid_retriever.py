"""
Hybrid retriever — combines BM25 (lexical) + Dense (semantic) via RRF fusion.
RRF = Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across both retrievers.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.bm25_baseline import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever

PROCESSED = Path("data/processed")
RRF_K = 60  # standard RRF constant


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = RRF_K):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    @classmethod
    def load_or_build(cls) -> "HybridRetriever":
        print("Loading BM25...")
        bm25 = BM25Retriever.load_or_build()
        print("Loading Dense...")
        dense = DenseRetriever.load_or_build()
        return cls(bm25=bm25, dense=dense)

    def retrieve(self, query: str, k: int = 10, n_candidates: int = 50):
        """
        Retrieve top-k by fusing BM25 + Dense results via RRF.
        n_candidates: how many to pull from each retriever before fusing.
        """
        bm25_ids, _ = self.bm25.retrieve(query, k=n_candidates)
        dense_ids, _ = self.dense.retrieve(query, k=n_candidates)

        # RRF scoring
        scores = defaultdict(float)
        for rank, pid in enumerate(bm25_ids, start=1):
            scores[int(pid)] += 1.0 / (self.rrf_k + rank)
        for rank, pid in enumerate(dense_ids, start=1):
            scores[int(pid)] += 1.0 / (self.rrf_k + rank)

        # Sort by fused score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        top_ids = np.array([pid for pid, _ in sorted_items], dtype=np.int64)
        top_scores = np.array([s for _, s in sorted_items], dtype=np.float64)

        return top_ids, top_scores


if __name__ == "__main__":
    print("=== Building Hybrid Retriever ===\n")
    r = HybridRetriever.load_or_build()

    print("\n=== Sanity check: same 3 queries ===\n")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    products = pd.read_parquet(PROCESSED / "products.parquet").set_index("product_id")

    for q in queries["query"].head(3):
        ids, scores = r.retrieve(q, k=5)
        print(f"Query: '{q}'")
        for pid, score in zip(ids, scores):
            name = products.loc[pid, "product_name"]
            print(f"  [{score:.4f}]  {name}")
        print()
