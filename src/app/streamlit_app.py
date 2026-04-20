"""
Agentic Recs — Streamlit Demo
Product search with LLM-powered planning, clarification, and reranking.
Evaluated on Wayfair's WANDS dataset (42,994 products, 233K human labels).
"""
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# --- Page config ---
st.set_page_config(
    page_title="Agentic Recs",
    page_icon="🛋️",
    layout="wide",
)

# --- Load data ---
@st.cache_data
def load_products():
    return pd.read_parquet("data/processed/products.parquet")

@st.cache_data
def load_queries():
    return pd.read_parquet("data/processed/queries.parquet")

@st.cache_resource
def load_bm25():
    from src.retrieval.bm25_baseline import BM25Retriever
    return BM25Retriever.load_or_build()

def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", "")),
        temperature=0.0,
    )

# --- Agent functions (simplified for demo) ---
def analyze_query(llm, query):
    """Planner: detect ambiguity."""
    prompt = f"""You are a product search assistant for a home furniture store.
Analyze this search query for ambiguity.

Query: "{query}"

A query is ambiguous if it could refer to multiple product categories or lacks important constraints.

Respond in JSON only:
{{"is_ambiguous": true/false, "ambiguity_score": 0.0 to 1.0, "missing_info": ["list"], "reasoning": "one sentence"}}"""
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return {"is_ambiguous": False, "ambiguity_score": 0.0, "missing_info": [], "reasoning": "parse error"}

def reformulate_query(llm, original_query, clarification):
    """Reformulate query with user's clarification."""
    prompt = f"""Original search: "{original_query}"
User clarification: "{clarification}"

Write an improved search query combining both. Just the query, under 20 words."""
    
    response = llm.invoke(prompt)
    return response.content.strip().strip('"')

def rerank_products(llm, query, products_df):
    """LLM reranker: score and reorder products."""
    product_list = "\n".join(
        f"{i+1}. [ID:{row['product_id']}] {row['product_name']} ({row.get('product_class', 'N/A')})"
        for i, (_, row) in enumerate(products_df.head(15).iterrows())
    )
    
    prompt = f"""You are a product search relevance expert.

Query: "{query}"

Products:
{product_list}

Rank the top 10 most relevant products. Consider category match and shopper intent.

Respond in JSON only:
{{"ranked_ids": [list of product IDs], "reasoning": "one sentence"}}"""
    
    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
        return result.get("ranked_ids", []), result.get("reasoning", "")
    except:
        return products_df["product_id"].head(10).tolist(), "kept original order"

# --- UI ---
st.title("🛋️ Agentic Recs")
st.markdown("**LLM-powered product search** with clarifying questions and intelligent reranking")
st.markdown("*Built on [Wayfair's WANDS dataset](https://github.com/wayfair/WANDS) — 42,994 products, 233K human relevance labels*")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This demo shows an **agentic recommendation system** that:
    
    1. **Plans** — detects ambiguous queries
    2. **Clarifies** — asks you one question when needed
    3. **Retrieves** — searches 42,994 products
    4. **Reranks** — LLM scores relevance
    
    Built with LangGraph, Groq (Llama 3.3 70B), and FAISS.
    """)
    
    st.header("Evaluation Results")
    st.markdown("""
    | Retriever | NDCG@10 |
    |-----------|---------|
    | BM25 | 0.685 |
    | Dense (BGE) | 0.744 |
    | Hybrid | **0.757** |
    """)
    
    st.header("Try these queries")
    example_queries = [
        "chair",
        "modern pet-friendly sofa",
        "smart coffee table",
        "4 drawer dresser",
        "dinosaur",
        "salon chair",
    ]
    for q in example_queries:
        if st.button(q, key=f"ex_{q}"):
            st.session_state["query_input"] = q

# Initialize session state
if "stage" not in st.session_state:
    st.session_state["stage"] = "search"
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

# Load resources
products = load_products()
retriever = load_bm25()

# Search bar
query = st.text_input("🔍 What are you looking for?", 
                       value=st.session_state.get("query_input", ""),
                       placeholder="e.g., modern sofa for small living room")

if query and st.session_state.get("last_query") != query:
    st.session_state["last_query"] = query
    st.session_state["stage"] = "analyzing"
    st.session_state["clarification"] = None

if query and st.session_state["stage"] == "analyzing":
    llm = get_llm()
    
    with st.status("🤔 Analyzing your query...", expanded=True):
        # Step 1: Planner
        st.write("**Step 1: Planning** — checking if your query needs clarification...")
        analysis = analyze_query(llm, query)
        
        is_ambiguous = analysis.get("is_ambiguous", False)
        score = analysis.get("ambiguity_score", 0)
        missing = analysis.get("missing_info", [])
        
        st.write(f"Ambiguity score: {score:.2f}")
        
        if is_ambiguous and score > 0.6:
            st.write(f"Missing info: {', '.join(missing)}")
            st.session_state["stage"] = "clarifying"
            st.session_state["analysis"] = analysis
            st.session_state["missing"] = missing
        else:
            st.write("Query is clear — searching directly!")
            st.session_state["stage"] = "retrieving"
            st.session_state["search_query"] = query

# Clarification step
if query and st.session_state["stage"] == "clarifying":
    missing = st.session_state.get("missing", [])
    
    st.info(f"🤔 Your search for **\"{query}\"** could be more specific. "
            f"Missing: {', '.join(missing)}")
    
    clarification = st.text_input(
        "Help me narrow it down:",
        placeholder=f"e.g., specify {missing[0] if missing else 'details'}...",
        key="clarify_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Search with clarification", disabled=not clarification):
            llm = get_llm()
            refined = reformulate_query(llm, query, clarification)
            st.session_state["search_query"] = refined
            st.session_state["clarification"] = clarification
            st.session_state["refined_query"] = refined
            st.session_state["stage"] = "retrieving"
            st.rerun()
    with col2:
        if st.button("⏭️ Skip — search as-is"):
            st.session_state["search_query"] = query
            st.session_state["stage"] = "retrieving"
            st.rerun()

# Retrieval + Reranking
if query and st.session_state["stage"] == "retrieving":
    search_query = st.session_state.get("search_query", query)
    llm = get_llm()
    
    with st.status("🔍 Finding products...", expanded=True):
        # Show clarification if it happened
        if st.session_state.get("clarification"):
            st.write(f"**Clarification:** {st.session_state['clarification']}")
            st.write(f"**Refined query:** {st.session_state.get('refined_query', search_query)}")
        
        # Step 2: Retrieve
        st.write(f"**Step 2: Retrieving** — searching 42,994 products for: *\"{search_query}\"*")
        top_ids, top_scores = retriever.retrieve(search_query, k=20)
        retrieved = products[products["product_id"].isin(top_ids)].copy()
        
        # Maintain retrieval order
        id_order = {pid: i for i, pid in enumerate(top_ids)}
        retrieved["rank"] = retrieved["product_id"].map(id_order)
        retrieved = retrieved.sort_values("rank")
        
        st.write(f"Found {len(retrieved)} candidates")
        
        # Step 3: Rerank
        st.write("**Step 3: Reranking** — LLM scoring relevance...")
        reranked_ids, reasoning = rerank_products(llm, search_query, retrieved)
        st.write(f"Reranker logic: *{reasoning}*")
    
    # Display results
    st.subheader("🛋️ Top Recommendations")
    
    reranked_products = []
    for pid in reranked_ids[:10]:
        match = products[products["product_id"] == int(pid)]
        if len(match) > 0:
            reranked_products.append(match.iloc[0])
    
    for rank, prod in enumerate(reranked_products, 1):
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### #{rank}")
            with col2:
                st.markdown(f"**{prod['product_name']}**")
                category = prod.get('product_class', 'N/A')
                rating = prod.get('average_rating', 'N/A')
                reviews = prod.get('rating_count', 'N/A')
                st.caption(f"📁 {category} · ⭐ {rating} · 📝 {int(reviews) if pd.notna(reviews) else 'N/A'} reviews")
                
                desc = str(prod.get('product_description', ''))[:200]
                if desc and desc != 'nan':
                    st.markdown(f"*{desc}...*")
            st.divider()
    
    # Reasoning trace
    with st.expander("🔍 Reasoning Trace"):
        st.markdown(f"**Original query:** {query}")
        if st.session_state.get("clarification"):
            st.markdown(f"**Clarification:** {st.session_state['clarification']}")
            st.markdown(f"**Refined query:** {st.session_state.get('refined_query', '')}")
        st.markdown(f"**Search query:** {search_query}")
        st.markdown(f"**Candidates retrieved:** {len(retrieved)}")
        st.markdown(f"**Reranker reasoning:** {reasoning}")
    
    st.session_state["stage"] = "done"

# Footer
st.markdown("---")
st.markdown(
    "Built by [Ijaz Ul Haq, PhD](https://github.com/ejokhan) · "
    "[GitHub Repo](https://github.com/ejokhan/agentic-recs) · "
    "Powered by LangGraph + Groq + WANDS"
)
