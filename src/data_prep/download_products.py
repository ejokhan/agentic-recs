"""
Download Amazon Reviews 2023 Home_and_Kitchen product metadata.
Source: McAuley Lab direct JSONL.GZ (no HuggingFace script loader).

Strategy:
    - Stream-download the gzipped JSONL
    - Parse line-by-line (file is ~1GB compressed, we stop early)
    - Keep products with >= 10 reviews, non-empty title + description
    - Save filtered subset as parquet
"""
import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_Home_and_Kitchen.jsonl.gz"
)

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GZ_PATH = OUT_DIR / "meta_Home_and_Kitchen.jsonl.gz"
PARQUET_PATH = OUT_DIR / "home_kitchen_raw.parquet"

MAX_KEEP = 50_000  # we'll subsample to 10K later; pull extra for filtering

# --- 1. Download the .jsonl.gz if we don't have it ---
if not GZ_PATH.exists():
    print(f"Downloading {URL} ...")
    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = 100.0 * block_num * block_size / total_size
            print(f"  {pct:5.1f}% ({block_num * block_size / 1e6:.0f} MB)", end="\r")
    urllib.request.urlretrieve(URL, GZ_PATH, reporthook=_hook)
    print(f"\nSaved -> {GZ_PATH} ({GZ_PATH.stat().st_size / 1e6:.1f} MB)")
else:
    print(f"Already downloaded -> {GZ_PATH}")

# --- 2. Parse and filter ---
print("\nParsing + filtering...")
rows = []
with gzip.open(GZ_PATH, "rt", encoding="utf-8") as f:
    for line in tqdm(f):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        title = (item.get("title") or "").strip()
        description = item.get("description") or []
        if isinstance(description, list):
            description = " ".join(description).strip()
        rating_number = item.get("rating_number") or 0

        if not title or not description or rating_number < 10:
            continue

        rows.append({
            "parent_asin": item.get("parent_asin"),
            "title": title,
            "description": description[:2000],
            "categories": item.get("categories") or [],
            "price": item.get("price"),
            "average_rating": item.get("average_rating"),
            "rating_number": rating_number,
            "store": item.get("store"),
            "features": item.get("features") or [],
        })

        if len(rows) >= MAX_KEEP:
            break

df = pd.DataFrame(rows)
print(f"\nFiltered products: {len(df)}")
print(df.head(3))

df.to_parquet(PARQUET_PATH, index=False)
print(f"\nSaved -> {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1e6:.1f} MB)")
