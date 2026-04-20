# Agentic Recs — LLM-Powered Product Recommendation with Bandit Reranking

An agentic recommendation system that combines dense retrieval, LLM-based query planning, multi-armed bandit reranking, and clarifying questions — evaluated on Wayfair's [WANDS](https://github.com/wayfair/WANDS) benchmark with 233K human relevance labels.

> **Core question:** When do LLM agents actually improve product search over classical baselines — and when do they hurt?

### [🚀 Live Demo](https://agentic-recs.streamlit.app) · [📄 Paper (coming soon)](#) · [📊 Results](#results)

---

## Architecture

```
                        ┌─────────────────────┐
                        │     User Query       │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Ambiguity Detector  │
                        │  (LLM scores 0→1)   │
                        └──────────┬──────────┘
                                   │
                          ambiguous?│
                         ┌─────────┴─────────┐
                        YES                  NO
                         │                    │
                         ▼                    │
                ┌─────────────────┐           │
                │   Clarifier     │           │
                │  (1 question)   │           │
                └────────┬────────┘           │
                         ▼                    │
                ┌─────────────────┐           │
                │  Reformulator   │           │
                └────────┬────────┘           │
                         │                    │
                         └────────┬───────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │     Retriever        │
                       │  BM25 + FAISS dense  │
                       │  (hybrid RRF fusion) │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │     Reranker         │
                       │  (LLM-scored, with   │
                       │   multiple strategies)│
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │   Thompson Sampling  │
                       │      Bandit          │
                       │  (selects reranker   │
                       │   strategy per query │
                       │   type)              │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Recommendations     │
                       │  + Reasoning Trace   │
                       └─────────────────────┘
```

All nodes are orchestrated as a **LangGraph** state machine with conditional edges.

## What Makes This Different

Most public "LLM + retrieval" demos use a fixed pipeline: embed → retrieve → rerank → respond. This project adds three layers that are absent from existing work on WANDS:

**1. Clarifying questions for ambiguous queries.** When a query like "chair" is underspecified (ambiguity score: 0.90), the agent asks *one* targeted clarification ("What type of chair — office, dining, accent?") before retrieving. The user's response is used to reformulate the query, dramatically improving result quality.

**2. Bandit over reranker strategies, not over items.** Instead of picking which product to show (the standard bandit formulation), our Thompson sampling bandit picks *which reranking strategy to use* for each query type — semantic-heavy vs. attribute-matching vs. category-weighted.

**3. Honest regime analysis.** We explicitly identify where the agent *loses* to BM25 (simple lookup queries) and report it. The goal is not to claim universal improvement, but to map the boundary between "agent helps" and "agent hurts."

## Dataset

**Primary:** [WANDS](https://github.com/wayfair/WANDS) (Wayfair ANnotation Dataset, ECIR 2022)

| | Count |
|---|---|
| Products | 42,994 |
| Search queries | 480 |
| Human relevance labels | 233,448 |
| Label classes | Exact (11%), Partial (63%), Irrelevant (26%) |

All evaluation uses Wayfair's own human annotations — no synthetic labels.

**Generalization:** Amazon Reviews 2023 (McAuley Lab) — Home & Kitchen subset, 50K products. Used for qualitative transfer analysis only (no relevance labels available).

## Results

### Retrieval Comparison (480 queries, 233K human labels)

| Retriever | NDCG@10 | 95% CI | MRR | Hit@10 Exact | Hit@10 Any |
|---|---|---|---|---|---|
| BM25 (baseline) | 0.685 | [0.656, 0.712] | 0.846 | 0.671 | 0.950 |
| Dense (BGE-large) | 0.744 | [0.720, 0.768] | 0.908 | 0.706 | 0.965 |
| **Hybrid (BM25+Dense RRF)** | **0.757** | **[0.732, 0.782]** | **0.913** | **0.706** | **0.973** |

**Key findings:**
- Hybrid retrieval improves NDCG@10 by **+10.5%** over BM25 baseline
- Dense retrieval captures semantic matches BM25 misses (e.g., "couch" → finds "sofa")
- BM25 still wins on exact keyword queries — hybrid gets the best of both worlds
- MRR of 0.913 means the first relevant product is almost always at rank 1

### On Harder Queries (379 queries with ≥1 Exact match)

| Retriever | NDCG@10 | Hit@10 Exact |
|---|---|---|
| BM25 | 0.705 | 0.850 |
| Dense | 0.758 | 0.895 |
| **Hybrid** | **0.769** | **0.895** |

### Agent with LLM Reranking

The agentic layer adds:
- **Query understanding:** Planner detects ambiguity (e.g., "chair" → score 0.90) and triggers clarification
- **Clarifying questions:** "What room is the chair for?" → user says "home office, ergonomic" → refined query: "best ergonomic office chairs for home office use"
- **LLM reranking:** Llama 3.3 70B reorders candidates based on semantic relevance with explicit reasoning

*Full agent evaluation across all 480 queries with ablation study in progress.*

## Live Demo

**[🚀 Try it here → agentic-recs.streamlit.app](https://agentic-recs.streamlit.app)**

The demo shows the full agentic experience:
1. Type a query → agent analyzes ambiguity
2. If ambiguous → asks you one clarifying question
3. Retrieves from 42,994 Wayfair products
4. LLM reranks with reasoning
5. Full reasoning trace available

*Note: The live demo uses BM25 retrieval for deployment efficiency (Streamlit Cloud 1GB RAM limit). The full hybrid pipeline (Dense BGE + RRF fusion, NDCG@10 = 0.757) runs on GPU — see this repo for the complete implementation.*

## Evaluation Framework

Five evaluation layers, from standard to novel:

1. **Offline retrieval metrics** — NDCG@10, MRR, Hit@k with bootstrap 95% CIs against 233K human labels
2. **Ablation across conditions** — Direct (no clarification) vs. Always-ask vs. Smart-ask (ambiguity-gated)
3. **Regime analysis** — Where does the agent beat BM25? Where does it lose? Query-type breakdown
4. **LLM-judge calibration** — Does an LLM judge (Llama 3.3 70B) agree with Wayfair's human annotators? Where does it disagree?
5. **Bandit convergence** — Regret curves showing how quickly Thompson sampling learns the best reranker strategy per query type

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | BGE-large-en-v1.5 (1024d) |
| Vector store | FAISS (inner product, normalized) |
| Lexical search | BM25 (rank_bm25) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) |
| Agent framework | LangGraph |
| LLM | Llama 3.3 70B via Groq (with 8B + Gemma 9B fallback) |
| Bandit | Thompson sampling (Beta priors) |
| Evaluation | Custom harness with NDCG, MRR, bootstrap CIs |
| Compute | NVIDIA A100 on TACC Lonestar6 (NSF NAIRR Pilot) |
| Demo | Streamlit Cloud |

## Project Structure

```
agentic-recs/
├── src/
│   ├── data_prep/        # WANDS loader + Amazon downloader
│   ├── embeddings/       # BGE embedding pipeline (GPU)
│   ├── retrieval/        # BM25, Dense, Hybrid retrievers
│   ├── agent/            # LangGraph agent (planner, clarifier, reranker, responder)
│   ├── bandit/           # Thompson sampling over reranker strategies
│   ├── evaluation/       # Metrics, eval harness, LLM-judge calibration
│   └── app/              # Streamlit demo + LLM fallback utilities
├── scripts/              # Slurm job scripts for TACC HPC
├── configs/              # Model + experiment configs
├── data/
│   ├── raw/              # WANDS CSVs + Amazon JSONL (gitignored)
│   ├── processed/        # Clean parquets
│   ├── embeddings/       # FAISS indices + vectors (gitignored)
│   └── eval/             # Per-query evaluation results (gitignored)
├── notebooks/            # Exploratory analysis
├── docs/                 # Blog post + one-pager
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/ejokhan/agentic-recs.git
cd agentic-recs
pip install -r requirements.txt

# Load and process WANDS
python src/data_prep/load_wands.py

# Build BM25 baseline + evaluate
python src/retrieval/bm25_baseline.py
python src/evaluation/run_eval.py bm25

# Embed products (requires GPU)
python src/embeddings/embed_products.py

# Build dense + hybrid retrievers + evaluate
python src/retrieval/dense_retriever.py
python src/evaluation/run_eval.py dense
python src/evaluation/run_eval.py hybrid

# Test the agent (requires GROQ_API_KEY)
export GROQ_API_KEY="your-key"
python src/agent/run_agent.py

# Launch Streamlit demo
streamlit run src/app/streamlit_app.py
```

## Relevance to Industry

This project demonstrates applied ML skills across recommendation systems, agentic AI, and evaluation engineering:

- **Recommendations:** Representation learning (BGE embeddings), hybrid retrieval (BM25 + dense + RRF fusion), item-to-item nearest-neighbor suggestions, multi-armed bandit personalization
- **Agentic AI:** LangGraph orchestration with planning, conditional routing, clarifying-question interaction, LLM reranking with reasoning traces, human-in-the-loop design
- **Evaluation:** Rigorous offline metrics on 233K human-labeled annotations, bootstrap confidence intervals, ablation studies, regime analysis identifying when agentic approaches should and should not be used
- **Search:** Three-way retrieval comparison (BM25 vs. Dense vs. Hybrid), query reformulation, LLM-scored reranking

## Related Work

- Chen et al. (2022). *WANDS: Dataset for Product Search Relevance Assessment.* ECIR 2022.
- Databricks (2023). *Enhancing Product Search with LLMs.* Solution accelerator using WANDS.
- Soviero et al. (2024). *ChatGPT Goes Shopping: LLMs Can Predict Relevance in eCommerce Search.* ECIR 2024.
- Hosseini et al. (2025). *Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation.* ECIR 2025.

## Author

**Ijaz Ul Haq, Ph.D.** — AI/ML Research Scientist
University of Vermont · Water Resources Institute
[GitHub](https://github.com/ejokhan) · [Google Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en)

Built on TACC Lonestar6 supercomputer through the NSF NAIRR Pilot program.

## License

MIT
