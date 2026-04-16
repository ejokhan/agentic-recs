"""
Embed WANDS products with BGE-large-en-v1.5 on GPU.

Writes:
    data/embeddings/product_embeddings.npy   (float32, [N, 1024])
    data/embeddings/product_ids.npy          (int64, [N])   — same order as above
    data/embeddings/meta.json                (model name, dim, count, timestamps)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PROCESSED = Path("data/processed")
OUT = Path("data/embeddings")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64  # fits easily on A100 40GB
CACHE_DIR = Path(".hf_cache").absolute()
CACHE_DIR.mkdir(exist_ok=True)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "Need GPU — don't run on login node!"
print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\nLoading products...")
df = pd.read_parquet(PROCESSED / "products.parquet")
texts = df["embedding_text"].fillna("").tolist()
ids = df["product_id"].to_numpy(dtype=np.int64)
print(f"  {len(texts)} products to embed")

print(f"\nLoading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, cache_folder=str(CACHE_DIR), device="cuda")
model.max_seq_length = 512  # BGE's native max

print(f"\nEncoding (batch_size={BATCH_SIZE})...")
t0 = time.time()
emb = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # unit-norm → dot product = cosine
)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s  ({len(texts)/elapsed:.1f} items/sec)")
print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")

np.save(OUT / "product_embeddings.npy", emb.astype(np.float32))
np.save(OUT / "product_ids.npy", ids)

meta = {
    "model": MODEL_NAME,
    "dim": int(emb.shape[1]),
    "count": int(emb.shape[0]),
    "normalized": True,
    "max_seq_length": 512,
    "batch_size": BATCH_SIZE,
    "elapsed_seconds": round(elapsed, 1),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=2))

print(f"\nSaved:")
print(f"  {OUT/'product_embeddings.npy'}  ({(OUT/'product_embeddings.npy').stat().st_size / 1e6:.1f} MB)")
print(f"  {OUT/'product_ids.npy'}")
print(f"  {OUT/'meta.json'}")
print(f"\nMeta: {json.dumps(meta, indent=2)}")
