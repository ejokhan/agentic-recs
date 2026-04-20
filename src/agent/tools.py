"""
Tools the LLM agent can autonomously call.
Each tool is a Python function with a description the LLM reads.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.tools import tool
import pandas as pd
import numpy as np


@tool
def search_products(query: str, k: int = 10) -> str:
    """Search 42,994 Wayfair products by text query. Returns top-k product names and IDs.
    Use this when you need to find products matching a user's search."""
    from src.retrieval.bm25_baseline import BM25Retriever
    retriever = BM25Retriever.load_or_build()
    products = pd.read_parquet("data/processed/products.parquet").set_index("product_id")

    ids, scores = retriever.retrieve(query, k=k)
    results = []
    for pid, score in zip(ids, scores):
        if pid in products.index:
            name = products.loc[pid, "product_name"]
            cat = products.loc[pid, "product_class"]
            results.append(f"ID:{pid} | {name} | Category: {cat} | Score: {score:.2f}")
    return "\n".join(results)


@tool
def filter_by_category(category: str) -> str:
    """Filter products to a specific category. Returns available categories matching the input.
    Use this when the user specifies a product type like 'office chairs' or 'coffee tables'."""
    products = pd.read_parquet("data/processed/products.parquet")
    matches = products[products["product_class"].str.contains(category, case=False, na=False)]
    cats = matches["product_class"].value_counts().head(10)
    return f"Found {len(matches)} products in matching categories:\n" + \
           "\n".join(f"  {cat}: {count} products" for cat, count in cats.items())


@tool
def get_similar_products(product_id: int) -> str:
    """Find products similar to a given product ID. Returns 5 most similar items.
    Use this for 'you might also like' recommendations after showing results."""
    from src.retrieval.similar_products import SimilarProducts
    rec = SimilarProducts.load()
    products = rec.products.set_index("product_id")

    sim_ids, sim_scores = rec.recommend(product_id, k=5)
    results = []
    for sid, score in zip(sim_ids, sim_scores):
        if sid in products.index:
            name = products.loc[sid, "product_name"]
            cat = products.loc[sid, "product_class"]
            results.append(f"ID:{sid} | {name} | {cat} | Similarity: {score:.3f}")
    return "\n".join(results) if results else "No similar products found."


@tool
def ask_clarification(question: str) -> str:
    """Ask the user a clarifying question when the search query is ambiguous.
    Use this when the query is too vague — e.g., 'chair' without specifying type/room/style.
    Only ask ONE question. The response will be simulated for offline evaluation."""
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=0.0,
    )
    prompt = f"""You are a shopper on a furniture website.
A search assistant asked you: "{question}"
Give a brief, natural answer (1 sentence) that adds useful constraints.
Just the answer, nothing else."""

    response = llm.invoke(prompt)
    return response.content.strip()


# List of all available tools
ALL_TOOLS = [search_products, filter_by_category, get_similar_products, ask_clarification]

if __name__ == "__main__":
    print("=== Testing Tools ===\n")
    print("1. search_products:")
    print(search_products.invoke({"query": "office chair", "k": 3}))
    print("\n2. filter_by_category:")
    print(filter_by_category.invoke({"category": "Office Chairs"}))
    print("\n3. get_similar_products:")
    # Use first product ID from products
    products = pd.read_parquet("data/processed/products.parquet")
    first_id = int(products["product_id"].iloc[0])
    print(get_similar_products.invoke({"product_id": first_id}))
    print("\nAll tools registered successfully!")
