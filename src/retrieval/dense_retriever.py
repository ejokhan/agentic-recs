"""
Dense retriever using BGE embeddings + FAISS index.
Same .retrieve(query, k) interface as BM25 so we can swap them freely.
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMB_DIR = Path("data/embeddings")
PROCESSED = Path("data/processed")
INDEX_PATH = EMB_DIR / "faiss.index"
HF_CACHE = "/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache"


class DenseRetriever:
    def __init__(self, index: faiss.Index, product_ids: np.ndarray, model: SentenceTransformer):
        self.index = index
        self.product_ids = product_ids
        self.model = model

    @classmethod
    def build(cls) -> "DenseRetriever":
        print("Loading embeddings...")
        embeddings = np.load(EMB_DIR / "product_embeddings.npy")
        product_ids = np.load(EMB_DIR / "product_ids.npy")
        meta = json.loads((EMB_DIR / "meta.json").read_text())
        print(f"  {embeddings.shape[0]} products, dim={embeddings.shape[1]}")

        print("Building FAISS index...")
        t0 = time.time()
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
        index.add(embeddings)
        print(f"  built in {time.time()-t0:.1f}s, total vectors: {index.ntotal}")

        print("Saving FAISS index...")
        faiss.write_index(index, str(INDEX_PATH))
        print(f"  saved -> {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")

        print(f"Loading embedding model: {meta['model']}")
        model = SentenceTransformer(meta["model"], cache_folder=HF_CACHE, device="cpu")
        model.max_seq_length = meta.get("max_seq_length", 512)

        return cls(index=index, product_ids=product_ids, model=model)

    @classmethod
    def load(cls) -> "DenseRetriever":
        print("Loading FAISS index...")
        index = faiss.read_index(str(INDEX_PATH))
        product_ids = np.load(EMB_DIR / "product_ids.npy")
        meta = json.loads((EMB_DIR / "meta.json").read_text())
        model = SentenceTransformer(meta["model"], cache_folder=HF_CACHE, device="cpu")
        model.max_seq_length = meta.get("max_seq_length", 512)
        print(f"  {index.ntotal} vectors, dim={meta['dim']}")
        return cls(index=index, product_ids=product_ids, model=model)

    @classmethod
    def load_or_build(cls) -> "DenseRetriever":
        if INDEX_PATH.exists():
            return cls.load()
        return cls.build()

    def retrieve(self, query: str, k: int = 10):
        """Return top-k (product_ids, scores) — same interface as BM25."""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), k)
        top_ids = self.product_ids[indices[0]]
        return top_ids, scores[0]


if __name__ == "__main__":
    print("=== Building Dense Retriever ===\n")
    r = DenseRetriever.build()

    print("\n=== Sanity check: same 3 queries as BM25 ===\n")
    queries = pd.read_parquet(PROCESSED / "queries.parquet")
    products = pd.read_parquet(PROCESSED / "products.parquet").set_index("product_id")

    for q in queries["query"].head(3):
        ids, scores = r.retrieve(q, k=5)
        print(f"Query: '{q}'")
        for pid, score in zip(ids, scores):
            name = products.loc[pid, "product_name"]
            print(f"  [{score:.4f}]  {name}")
        print()
