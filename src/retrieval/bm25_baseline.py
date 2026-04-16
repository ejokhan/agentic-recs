"""
BM25 baseline retriever over WANDS products.
The simplest classical text-retrieval method — this is what the agent has to beat.

Usage:
    from src.retrieval.bm25_baseline import BM25Retriever
    r = BM25Retriever.load_or_build()
    top_ids, top_scores = r.retrieve("salon chair", k=10)
"""
import pickle
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

PROCESSED = Path("data/processed")
INDEX_PATH = Path("data/embeddings/bm25.pkl")


def simple_tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric tokens. Good enough for BM25 baseline."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, bm25: BM25Okapi, product_ids: np.ndarray):
        self.bm25 = bm25
        self.product_ids = product_ids

    @classmethod
    def build(cls, products_path: Path = PROCESSED / "products.parquet") -> "BM25Retriever":
        print(f"Loading products from {products_path}...")
        df = pd.read_parquet(products_path)
        print(f"  {len(df)} products")

        print("Tokenizing...")
        t0 = time.time()
        tokenized = [simple_tokenize(t) for t in df["embedding_text"].fillna("")]
        print(f"  tokenized in {time.time()-t0:.1f}s")

        print("Building BM25 index...")
        t0 = time.time()
        bm25 = BM25Okapi(tokenized)
        print(f"  built in {time.time()-t0:.1f}s")

        return cls(bm25=bm25, product_ids=df["product_id"].to_numpy(dtype=np.int64))

    def save(self, path: Path = INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "product_ids": self.product_ids}, f)
        print(f"Saved -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(bm25=data["bm25"], product_ids=data["product_ids"])

    @classmethod
    def load_or_build(cls, path: Path = INDEX_PATH) -> "BM25Retriever":
        if path.exists():
            print(f"Loading existing BM25 index from {path}")
            return cls.load(path)
        r = cls.build()
        r.save(path)
        return r

    def retrieve(self, query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-k (product_ids, scores) for a query."""
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return self.product_ids[top_idx], scores[top_idx]


if __name__ == "__main__":
    print("=== Building BM25 baseline ===\n")
    r = BM25Retriever.load_or_build()

    print("\n=== Sanity check: 3 example queries ===\n")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    products = pd.read_parquet(PROCESSED / "products.parquet").set_index("product_id")

    for q in queries["query"].head(3):
        ids, scores = r.retrieve(q, k=5)
        print(f"Query: '{q}'")
        for pid, score in zip(ids, scores):
            name = products.loc[pid, "product_name"]
            print(f"  [{score:6.2f}]  {name}")
        print()
