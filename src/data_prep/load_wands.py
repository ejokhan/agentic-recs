"""
Load WANDS (Wayfair ANnotation Dataset) — 3 tab-separated CSVs → clean parquets.

Input:
    data/raw/wands/dataset/product.csv   (42,994 products)
    data/raw/wands/dataset/query.csv     (480 queries)
    data/raw/wands/dataset/label.csv     (233,448 judgments)

Output:
    data/processed/products.parquet   (products + embedding_text)
    data/processed/queries.parquet
    data/processed/labels.parquet     (with relevance as int: Exact=2, Partial=1, Irrelevant=0)
"""
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/wands/dataset")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def clean_features(feat: str) -> str:
    """Convert 'color : chocolate|warrantylength:5 years|...' → readable sentence."""
    if not isinstance(feat, str) or not feat.strip():
        return ""
    parts = []
    for kv in feat.split("|"):
        if ":" not in kv:
            continue
        k, _, v = kv.partition(":")
        k, v = k.strip(), v.strip()
        if not v:
            continue
        if v.lower() in {"no", "yes"} and len(kv) < 25:
            continue
        parts.append(f"{k}: {v}")
    return "; ".join(parts[:30])


def build_embedding_text(row) -> str:
    pieces = [
        str(row.get("product_name") or "").strip(),
        f"Category: {row.get('product_class') or ''}".strip(),
        f"Taxonomy: {row.get('category hierarchy') or ''}".strip(),
        str(row.get("product_description") or "").strip()[:1500],
        f"Features: {clean_features(row.get('product_features'))}",
    ]
    return " | ".join(p for p in pieces if p and not p.endswith(": "))


# --- Products ---
print("Loading products.csv...")
products = pd.read_csv(RAW / "product.csv", sep="\t")
print(f"  raw rows: {len(products)}")
print(f"  columns: {list(products.columns)}")

products["embedding_text"] = products.apply(build_embedding_text, axis=1)
products["text_len"] = products["embedding_text"].str.len()
print(f"  embedding_text length — mean: {products['text_len'].mean():.0f}, "
      f"p95: {products['text_len'].quantile(0.95):.0f}, "
      f"max: {products['text_len'].max()}")

products.to_parquet(OUT / "products.parquet", index=False)
print(f"  saved -> {OUT / 'products.parquet'}")

# --- Queries ---
print("\nLoading query.csv...")
queries = pd.read_csv(RAW / "query.csv", sep="\t")
print(f"  rows: {len(queries)}")
print(f"  columns: {list(queries.columns)}")
queries.to_parquet(OUT / "queries.parquet", index=False)
print(f"  saved -> {OUT / 'queries.parquet'}")

# --- Labels ---
print("\nLoading label.csv...")
labels = pd.read_csv(RAW / "label.csv", sep="\t")
print(f"  rows: {len(labels)}")
print(f"  label distribution:")
print(labels["label"].value_counts().to_string())

label_map = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
labels["relevance"] = labels["label"].map(label_map)
assert labels["relevance"].notna().all(), "Unmapped label found!"

labels.to_parquet(OUT / "labels.parquet", index=False)
print(f"  saved -> {OUT / 'labels.parquet'}")

# --- Sanity check ---
print("\n" + "=" * 60)
print("SANITY CHECK — one example query and its labeled products:")
print("=" * 60)
q = queries.iloc[0]
print(f"Query: '{q['query']}' (class: {q['query_class']})")
lbl_sample = labels[labels["query_id"] == q["query_id"]].head(3)
for _, lbl in lbl_sample.iterrows():
    prod = products[products["product_id"] == lbl["product_id"]]
    if len(prod) == 0:
        continue
    p = prod.iloc[0]
    print(f"\n  → {lbl['label']} (relevance={lbl['relevance']}): {p['product_name']}")
    print(f"    embedding_text[:200]: {p['embedding_text'][:200]}...")

print("\nDone.")
