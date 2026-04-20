"""
Agent nodes — each function is one step in the LangGraph state machine.
Every node takes AgentState, does one job, returns updated fields.
"""
import json
import os
import sys
from pathlib import Path

from langchain_groq import ChatGroq

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.agent.state import AgentState

# --- LLM setup ---
def get_llm(temperature=0.0):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=temperature,
    )


# ============================================================
# NODE 1: PLANNER — decides if the query is ambiguous
# ============================================================
def planner_node(state: AgentState) -> dict:
    """
    Analyze the query for ambiguity.
    Output: is_ambiguous, missing_info, ambiguity_score
    """
    query = state["query"]
    trace = state.get("reasoning_trace", [])

    llm = get_llm()
    prompt = f"""You are a product search assistant for a home furniture store.

Analyze this search query and determine if it is ambiguous or underspecified.

Query: "{query}"

A query is ambiguous if:
- It could refer to multiple product categories (e.g., "chair" could be office, dining, accent, outdoor)
- It lacks important constraints a shopper would normally specify (budget, room, size, material, style)
- It uses vague terms that need clarification

Respond in JSON only, no other text:
{{
    "is_ambiguous": true/false,
    "ambiguity_score": 0.0 to 1.0,
    "missing_info": ["list", "of", "missing", "details"],
    "reasoning": "one sentence explaining your decision"
}}"""

    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # If LLM doesn't return clean JSON, default to not ambiguous
        result = {"is_ambiguous": False, "ambiguity_score": 0.0,
                  "missing_info": [], "reasoning": "parse error, defaulting"}

    trace.append(f"PLANNER: ambiguous={result['is_ambiguous']}, "
                 f"score={result.get('ambiguity_score', 0):.2f}, "
                 f"missing={result.get('missing_info', [])}")

    return {
        "is_ambiguous": result.get("is_ambiguous", False),
        "ambiguity_score": result.get("ambiguity_score", 0.0),
        "missing_info": result.get("missing_info", []),
        "retrieval_query": query,  # default: use original query
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 2: CLARIFIER — asks one question, simulates user answer
# ============================================================
def clarifier_node(state: AgentState) -> dict:
    """
    Generate a clarifying question and simulate the user's response.
    In production this would be an actual user interaction.
    For evaluation, we simulate based on the query context.
    """
    query = state["query"]
    missing = state.get("missing_info", [])
    trace = state.get("reasoning_trace", [])

    llm = get_llm()

    # Step 1: Generate clarifying question
    question_prompt = f"""You are a helpful product search assistant. The user searched for: "{query}"

This query is missing: {missing}

Ask ONE short clarifying question to help narrow the search.
Keep it under 15 words. Just the question, nothing else."""

    q_response = llm.invoke(question_prompt)
    clarifying_question = q_response.content.strip()

    # Step 2: Simulate user response (in production, this waits for real input)
    answer_prompt = f"""You are a shopper searching for "{query}" on a furniture website.

A search assistant asked you: "{clarifying_question}"

Give a brief, natural answer (1 sentence) that adds useful constraints.
Just the answer, nothing else."""

    a_response = llm.invoke(answer_prompt)
    clarification_response = a_response.content.strip()

    # Step 3: Reformulate query with the new info
    reform_prompt = f"""Original search query: "{query}"
Clarification: "{clarifying_question}" -> "{clarification_response}"

Write an improved search query that includes the original intent plus the clarification.
Just the query, nothing else. Keep it under 20 words."""

    r_response = llm.invoke(reform_prompt)
    refined_query = r_response.content.strip().strip('"')

    trace.append(f"CLARIFIER: Q='{clarifying_question}' "
                 f"A='{clarification_response}' "
                 f"refined='{refined_query}'")

    return {
        "clarifying_question": clarifying_question,
        "clarification_response": clarification_response,
        "refined_query": refined_query,
        "retrieval_query": refined_query,  # override retrieval query
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 3: RETRIEVER — finds candidate products
# ============================================================
def retriever_node(state: AgentState) -> dict:
    """
    Retrieve top-K products using BM25 (swaps to dense/hybrid later).
    """
    from src.retrieval.hybrid_retriever import HybridRetriever
    import pandas as pd

    retrieval_query = state.get("retrieval_query", state["query"])
    trace = state.get("reasoning_trace", [])

    retriever = HybridRetriever.load_or_build()
    products = pd.read_parquet("data/processed/products.parquet").set_index("product_id")

    top_ids, top_scores = retriever.retrieve(retrieval_query, k=20)

    names = []
    for pid in top_ids:
        if pid in products.index:
            names.append(str(products.loc[pid, "product_name"]))
        else:
            names.append("unknown")

    trace.append(f"RETRIEVER: query='{retrieval_query}', got {len(top_ids)} results")

    return {
        "retrieved_ids": top_ids.tolist(),
        "retrieved_scores": top_scores.tolist(),
        "retrieved_names": names,
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 4: RERANKER — LLM scores and reorders products
# ============================================================
def reranker_node(state: AgentState) -> dict:
    """
    Use LLM to rerank retrieved products based on relevance to the query.
    Returns top 10 reranked products.
    """
    query = state.get("retrieval_query", state["query"])
    ids = state["retrieved_ids"][:20]
    names = state["retrieved_names"][:20]
    trace = state.get("reasoning_trace", [])

    llm = get_llm()

    # Build product list for the prompt
    product_list = "\n".join(f"{i+1}. [ID:{pid}] {name}"
                             for i, (pid, name) in enumerate(zip(ids, names)))

    prompt = f"""You are a product search relevance expert for a home furniture store.

Query: "{query}"

Products retrieved:
{product_list}

Rank the top 10 most relevant products for this query. Consider:
- Does the product match what the shopper is looking for?
- Is the product category correct?
- Would a shopper be satisfied finding this product?

Respond in JSON only, no other text:
{{
    "ranked_ids": [list of 10 product IDs in order of relevance, most relevant first],
    "reasoning": "one sentence explaining your ranking logic"
}}"""

    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
        reranked_ids = [int(x) for x in result.get("ranked_ids", ids[:10])]
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        # Fallback: keep original order
        reranked_ids = ids[:10]
        reasoning = "parse error, kept original order"

    trace.append(f"RERANKER: reordered top {len(reranked_ids)}, "
                 f"reason='{reasoning[:80]}'")

    return {
        "reranked_ids": reranked_ids,
        "reranked_scores": list(range(len(reranked_ids), 0, -1)),  # rank scores
        "rerank_strategy": "llm_relevance",
        "reasoning_trace": trace,
    }


# ============================================================
# NODE 5: RESPONDER — formats the final output
# ============================================================
def responder_node(state: AgentState) -> dict:
    """
    Format the final response with ranked products and reasoning trace.
    """
    import pandas as pd

    query = state["query"]
    reranked_ids = state.get("reranked_ids", state.get("retrieved_ids", []))[:10]
    trace = state.get("reasoning_trace", [])

    products = pd.read_parquet("data/processed/products.parquet").set_index("product_id")

    lines = [f"Query: '{query}'", ""]
    if state.get("clarifying_question"):
        lines.append(f"Clarification: {state['clarifying_question']}")
        lines.append(f"Answer: {state.get('clarification_response', 'N/A')}")
        lines.append(f"Refined query: {state.get('refined_query', query)}")
        lines.append("")

    lines.append("Top recommendations:")
    for rank, pid in enumerate(reranked_ids, 1):
        if pid in products.index:
            p = products.loc[pid]
            name = p["product_name"]
            cat = p.get("product_class", "")
            lines.append(f"  {rank}. {name} [{cat}]")
        else:
            lines.append(f"  {rank}. Product ID {pid} (not found)")

    lines.append("")
    lines.append("--- Reasoning Trace ---")
    for step in trace:
        lines.append(f"  {step}")

    response = "\n".join(lines)
    trace.append("RESPONDER: formatted output")

    return {
        "response": response,
        "reasoning_trace": trace,
    }
