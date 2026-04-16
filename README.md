# Agentic Recs — LLM Agent for Product Recommendation

An end-to-end agentic recommendation system that combines retrieval-augmented generation, multi-armed bandit exploration, and LLM-based reasoning to deliver personalized product recommendations for e-commerce.

## Why this project
Classical recommender systems optimize a fixed objective (clicks, conversions) but struggle with ambiguous natural-language queries like *"modern living room, small budget, pet-friendly"*. This project pairs a deterministic retrieval baseline with an LLM agent that plans, decomposes, and reranks — then uses Thompson sampling to learn which strategy works when.

## Architecture
User query
│
▼
┌──────────────────────────────────────────┐
│ LangGraph Agent                          │
│  Planner → Retriever → Reranker → Bandit │
└──────────────────────────────────────────┘
│
▼
Ranked products + reasoning trace
## Tech stack
- **Embeddings:** sentence-transformers (BGE)
- **Vector store:** FAISS
- **Agent framework:** LangGraph
- **LLM:** Llama 3.3 70B via Groq (free-tier, low-latency)
- **Bandit:** Thompson sampling (Beta priors)
- **Evaluation:** NDCG@10, MRR, LLM-as-judge, bootstrap A/B
- **Demo:** Streamlit

## Dataset
Amazon Reviews 2023 (McAuley Lab) — Home & Kitchen subset, 10K products across 4 categories.

## Results
_To be filled in._

## Author
Ijaz Ul Haq, PhD — [GitHub](https://github.com/ejokhan) · [Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ)
