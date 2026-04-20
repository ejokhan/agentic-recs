"""
Item-to-item recommendations — "You might also like"
Uses FAISS nearest neighbors on product embeddings.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import faiss

EMB_DIR = Path("data/embeddings")
PROCESSED = Path("data/processed")


class SimilarProducts:
    def __init__(self, index, product_ids, products_df):
        self.index = index
        self.product_ids = product_ids
        self.products = products_df

    @classmethod
    def load(cls):
        index = faiss.read_index(str(EMB_DIR / "faiss.index"))
        product_ids = np.load(EMB_DIR / "product_ids.npy")
        products = pd.read_parquet(PROCESSED / "products.parquet")
        return cls(index, product_ids, products)

    def recommend(self, product_id: int, k: int = 5):
        """Find k most similar products to a given product."""
        idx = np.where(self.product_ids == product_id)[0]
        if len(idx) == 0:
            return [], []

        idx = idx[0]
        embeddings = np.load(EMB_DIR / "product_embeddings.npy")
        query_vec = embeddings[idx:idx+1]

        scores, indices = self.index.search(query_vec.astype(np.float32), k + 1)

        # Skip self (first result)
        result_ids = []
        result_scores = []
        for i, s in zip(indices[0], scores[0]):
            pid = int(self.product_ids[i])
            if pid != product_id:
                result_ids.append(pid)
                result_scores.append(float(s))
            if len(result_ids) >= k:
                break

        return result_ids, result_scores


if __name__ == "__main__":
    print("=== Similar Products (You Might Also Like) ===\n")
    rec = SimilarProducts.load()
    products = rec.products.set_index("product_id")

    # Test with a few products
    test_ids = rec.product_ids[:3]
    for pid in test_ids:
        name = products.loc[pid, "product_name"]
        cat = products.loc[pid, "product_class"]
        print(f"Product: {name} [{cat}]")

        sim_ids, sim_scores = rec.recommend(int(pid), k=5)
        for sid, score in zip(sim_ids, sim_scores):
            sname = products.loc[sid, "product_name"]
            scat = products.loc[sid, "product_class"]
            print(f"  [{score:.4f}] {sname} [{scat}]")
        print()
