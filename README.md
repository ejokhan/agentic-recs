# Agentic Recs — LLM-Powered Product Recommendation with Bandit Reranking

An agentic recommendation system that combines dense retrieval, LLM-based query planning, multi-armed bandit reranking, and clarifying questions — evaluated on Wayfair's [WANDS](https://github.com/wayfair/WANDS) benchmark with 233K human relevance labels.

> **Core question:** When do LLM agents actually improve product search over classical baselines — and when do they hurt?

### [🚀 Live Demo](https://agentic-recs.streamlit.app) · [📄 Paper](docs/agentic_recs_paper.pdf) · [📊 Results](#results)

---

## Architecture

![System Architecture](docs/architecture_diagram.png)

*The Planner routes to the Clarifier only when ambiguity score exceeds 0.6 (dashed orange path). All other paths are deterministic. Each (LLM) annotation indicates a Groq API call to Llama 3.3 70B.*

All nodes are orchestrated as a **LangGraph** state machine with conditional edges. The agent exposes four tools for autonomous selection: `search_products`, `filter_by_category`, `get_similar_products`, and `ask_clarification`.

## What Makes This Different

Most public "LLM + retrieval" demos use a fixed pipeline: embed → retrieve → rerank → respond. This project adds three layers that are absent from existing work on WANDS:

**1. Clarifying questions for ambiguous queries.** When a query like "chair" is underspecified (ambiguity score: 0.90), the agent asks *one* targeted clarification ("What type of chair — office, dining, accent?") before retrieving. The user's response is used to reformulate the query, dramatically improving result quality.

**2. Bandit over reranker strategies, not over items.** Instead of picking which product to show (the standard bandit formulation), our Thompson sampling bandit picks *which reranking strategy to use* for each query type — semantic-heavy vs. attribute-matching vs. category-weighted. The bandit learns "short queries benefit from semantic reranking; feature-heavy queries benefit from attribute matching."

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

### Statistical A/B Tests (Paired, 480 queries)

| Comparison | Δ NDCG@10 | p-value (t-test) | Cohen's d | B wins / A wins |
|---|---|---|---|---|
| BM25 → Dense | +0.059 | <0.001 *** | 0.280 (small) | 235 / 141 |
| BM25 → Hybrid | +0.072 | <0.001 *** | 0.500 (medium) | 251 / 96 |
| Dense → Hybrid | +0.014 | 0.022 * | 0.105 (small) | 191 / 163 |

**Key finding:** Dense→Hybrid MRR difference is *not* significant (p=0.62), meaning BM25's contribution to hybrid is in diversifying lower ranks, not improving top-1. If latency is critical, dense alone captures 99.5% of the MRR gain.

### On Harder Queries (379 queries with ≥1 Exact match)

| Retriever | NDCG@10 | Hit@10 Exact |
|---|---|---|
| BM25 | 0.705 | 0.850 |
| Dense | 0.758 | 0.895 |
| **Hybrid** | **0.769** | **0.895** |

### Thompson Sampling Bandit Convergence (100 rounds)

| Query Type | Best Strategy | Posterior Mean | Pulls |
|---|---|---|---|
| Short queries | Semantic | 0.772 | 38 |
| Long queries | Attribute-matching | 0.894 | 40 |

The bandit correctly learns the optimal reranking strategy per query type within ~50 rounds.

### Agent with LLM Reranking

The agentic layer adds:
- **Query understanding:** Planner detects ambiguity (e.g., "chair" → score 0.90) and triggers clarification
- **Clarifying questions:** "What room is the chair for?" → user says "home office, ergonomic" → refined query: "best ergonomic office chairs for home office use"
- **LLM reranking:** Llama 3.3 70B reorders candidates based on semantic relevance with explicit reasoning

## Live Demo

**[🚀 Try it here → agentic-recs.streamlit.app](https://agentic-recs.streamlit.app)**

The demo shows the full agentic experience:
1. Type a query → agent analyzes ambiguity
2. If ambiguous → asks you one clarifying question
3. Retrieves from 42,994 Wayfair products
4. LLM reranks with reasoning
5. "You might also like" recommendations for each result
6. Full reasoning trace available

*Note: The live demo uses BM25 retrieval for deployment efficiency (Streamlit Cloud 1GB RAM limit). The full hybrid pipeline (Dense BGE + RRF fusion, NDCG@10 = 0.757) runs on GPU — see this repo for the complete implementation.*

## Evaluation Framework

Five evaluation layers, from standard to novel:

1. **Offline retrieval metrics** — NDCG@10, MRR, Hit@k with bootstrap 95% CIs against 233K human labels
2. **Statistical A/B tests** — Paired t-tests, Wilcoxon signed-rank, Cohen's d effect sizes, win/loss/tie counts
3. **Regime analysis** — Where does the agent beat BM25? Where does it lose? Query-type breakdown
4. **Bandit convergence** — Thompson sampling learns optimal reranker strategy per query type
5. **LLM-judge calibration** — Does Llama 3.3 70B agree with Wayfair's human annotators? (in progress)

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | BGE-large-en-v1.5 (1024d) |
| Vector store | FAISS (inner product, normalized) |
| Lexical search | BM25 (rank_bm25) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Agent framework | LangGraph |
| LLM | Llama 3.3 70B via Groq (with 8B + Gemma 9B fallback) |
| Bandit | Thompson sampling (Beta priors, 3 strategies) |
| Recommendations | FAISS kNN item-to-item |
| Evaluation | Custom harness: NDCG, MRR, bootstrap CIs, paired t-test, Wilcoxon, Cohen's d |
| Compute | NVIDIA A100 on TACC Lonestar6 |
| Demo | Streamlit Cloud |

## Project Structure

```
agentic-recs/
├── src/
│   ├── data_prep/        # WANDS loader + Amazon downloader
│   ├── embeddings/       # BGE embedding pipeline (GPU)
│   ├── retrieval/        # BM25, Dense, Hybrid retrievers + Similar Products
│   ├── agent/            # LangGraph agent (state, nodes, graph, tools)
│   ├── bandit/           # Thompson sampling over reranker strategies
│   ├── evaluation/       # Metrics, eval harness, A/B testing framework
│   └── app/              # Streamlit demo + LLM fallback utilities
├── scripts/              # Slurm job scripts for TACC HPC
├── docs/                 # Paper, architecture diagram, one-pager
├── data/
│   ├── raw/              # WANDS CSVs + Amazon JSONL (gitignored)
│   ├── processed/        # Clean parquets
│   ├── embeddings/       # FAISS indices + vectors (gitignored)
│   └── eval/             # Per-query evaluation results (gitignored)
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

# Run A/B tests
python src/evaluation/ab_test.py

# Test the agent (requires GROQ_API_KEY)
export GROQ_API_KEY="your-key"
python src/agent/run_agent.py

# Test similar products
python src/retrieval/similar_products.py

# Test bandit
python src/bandit/thompson.py

# Launch Streamlit demo
streamlit run src/app/streamlit_app.py
```

## Relevance to Industry

This project demonstrates applied ML skills across recommendation systems, agentic AI, and evaluation engineering:

- **Recommendations:** Representation learning (BGE embeddings), hybrid retrieval (BM25 + dense + RRF fusion), item-to-item nearest-neighbor suggestions, Thompson sampling multi-armed bandit personalization, session-aware query context
- **Agentic AI:** LangGraph orchestration with planning, conditional routing, clarifying-question interaction, LLM reranking with reasoning traces, four callable tools (search, filter, similar, clarify), human-in-the-loop design
- **Evaluation:** Rigorous offline metrics on 233K human-labeled annotations, bootstrap confidence intervals, paired A/B tests (t-test + Wilcoxon + Cohen's d), ablation studies, regime analysis identifying when agentic approaches should and should not be used
- **Search:** Three-way retrieval comparison (BM25 vs. Dense vs. Hybrid), query reformulation, LLM-scored reranking, statistical significance testing

## Related Work

- Chen et al. (2022). *WANDS: Dataset for Product Search Relevance Assessment.* ECIR 2022.
- Databricks (2023). *Enhancing Product Search with LLMs.* Solution accelerator using WANDS.
- Soviero et al. (2024). *ChatGPT Goes Shopping: LLMs Can Predict Relevance in eCommerce Search.* ECIR 2024.
- Hosseini et al. (2025). *Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation.* ECIR 2025.
- Cormack et al. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
- Thompson (1933). *On the Likelihood that One Unknown Probability Exceeds Another.* Biometrika.

## Author

**Ijaz Ul Haq, Ph.D.** — AI/ML Research Scientist
University of Vermont · Department of Computer Science and Water Resources Institute
[GitHub](https://github.com/ejokhan) · [Google Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en)

Built on TACC Lonestar6 supercomputer using NVIDIA A100 GPUs.

## License

MIT
