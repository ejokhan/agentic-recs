"""
Test the agent on a few WANDS queries and print the full reasoning trace.
This is the "does it actually work" sanity check.
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.agent.graph import build_agent
import pandas as pd


def run_single(agent, query: str):
    """Run the agent on one query, print everything."""
    print("=" * 70)
    print(f"INPUT: {query}")
    print("=" * 70)

    result = agent.invoke({"query": query, "reasoning_trace": []})

    print(result.get("response", "No response"))
    print()


def main():
    # Verify Groq key
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Set GROQ_API_KEY first")
        print("  export GROQ_API_KEY='gsk_...'")
        sys.exit(1)

    print("Building agent...\n")
    agent = build_agent()
    print("Agent ready!\n")

    # Load WANDS queries
    queries = pd.read_parquet("data/processed/queries.parquet")

    # Test on 3 carefully chosen queries:
    # 1. Simple/clear query (agent should skip clarifier)
    # 2. Ambiguous query (agent should clarify)
    # 3. Multi-intent query (agent should decompose)
    test_queries = [
        "4 drawer dresser",           # simple, specific
        "chair",                       # ambiguous — what kind?
        "modern pet-friendly sofa",    # multi-constraint
    ]

    for q in test_queries:
        run_single(agent, q)

    print("\n" + "=" * 70)
    print("Also testing 2 real WANDS queries:")
    print("=" * 70 + "\n")

    for q in queries["query"].head(2):
        run_single(agent, q)


if __name__ == "__main__":
    main()
