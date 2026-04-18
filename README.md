# Agentic Recs — LLM-Powered Product Recommendation with Bandit Reranking

An agentic recommendation system that combines dense retrieval, LLM-based query planning, multi-armed bandit reranking, and clarifying questions — evaluated on Wayfair's [WANDS](https://github.com/wayfair/WANDS) benchmark with 233K human relevance labels.

> **Core question:** When do LLM agents actually improve product search over classical baselines — and when do they hurt?

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

## What makes this different

Most public "LLM + retrieval" demos use a fixed pipeline: embed → retrieve → rerank → respond. This project adds three layers that are absent from existing work on WANDS:

**1. Clarifying questions for ambiguous queries.** When a query like "pet-friendly chair" is underspecified, the agent asks *one* targeted clarification ("living room or home office?") before retrieving. We measure whether this friction improves end-task relevance or hurts it — a question nobody has answered on WANDS.

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

### Baseline: BM25 (480 queries)

| Metric | Score | 95% CI |
|---|---|---|
| NDCG@10 | 0.685 | [0.656, 0.712] |
| MRR | 0.846 | [0.818, 0.872] |
| Hit@10 (Exact) | 0.671 | [0.627, 0.710] |
| Hit@10 (Any relevant) | 0.950 | [0.929, 0.969] |

On the 379 queries with ≥1 Exact match: NDCG@10 = 0.705, Hit@10 Exact = 0.850.

### Dense, Hybrid, Agent, Bandit, Clarifier

_Experiments in progress._

## Evaluation framework

Five evaluation layers, from standard to novel:

1. **Offline retrieval metrics** — NDCG@10, MRR, Hit@k with bootstrap 95% CIs against 233K human labels
2. **Ablation across conditions** — Direct (no clarification) vs. Always-ask vs. Smart-ask (ambiguity-gated)
3. **Regime analysis** — Where does the agent beat BM25? Where does it lose? Query-type breakdown
4. **LLM-judge calibration** — Does an LLM judge (Llama 3.3 70B) agree with Wayfair's human annotators? Where does it disagree?
5. **Bandit convergence** — Regret curves showing how quickly Thompson sampling learns the best reranker strategy per query type

## Tech stack

| Component | Tool |
|---|---|
| Embeddings | BGE-large-en-v1.5 (1024d) |
| Vector store | FAISS (inner product, normalized) |
| Lexical search | BM25 (rank_bm25) |
| Agent framework | LangGraph |
| LLM | Llama 3.3 70B via Groq |
| Bandit | Thompson sampling (Beta priors) |
| Evaluation | Custom harness with NDCG, MRR, bootstrap CIs |
| Compute | NVIDIA A100 on TACC Lonestar6 (NSF NAIRR Pilot) |
| Demo | Streamlit |

## Project structure

```
agentic-recs/
├── src/
│   ├── data_prep/        # WANDS loader + Amazon downloader
│   ├── embeddings/       # BGE embedding pipeline (GPU)
│   ├── retrieval/        # BM25 baseline + dense + hybrid
│   ├── agent/            # LangGraph agent (planner, clarifier, reformulator)
│   ├── bandit/           # Thompson sampling over reranker strategies
│   ├── evaluation/       # Metrics, eval harness, LLM-judge calibration
│   └── app/              # Streamlit demo
├── scripts/              # Slurm job scripts for TACC HPC
├── configs/              # Model + experiment configs
├── data/
│   ├── raw/              # WANDS CSVs + Amazon JSONL (gitignored)
│   ├── processed/        # Clean parquets (gitignored)
│   ├── embeddings/       # FAISS indices + vectors (gitignored)
│   └── eval/             # Per-query evaluation results (gitignored)
├── notebooks/            # Exploratory analysis
├── docs/                 # Blog post + one-pager
└── README.md
```

## Quick start

```bash
git clone https://github.com/ejokhan/agentic-recs.git
cd agentic-recs
pip install -r requirements.txt

# Load and process WANDS
python src/data_prep/load_wands.py

# Build BM25 baseline
python src/retrieval/bm25_baseline.py

# Evaluate BM25
python src/evaluation/run_eval.py bm25

# Embed products (requires GPU)
python src/embeddings/embed_products.py

# (Agent, bandit, Streamlit — coming soon)
```

## Relevance to industry

This project is designed to demonstrate applied ML skills across recommendation systems, agentic AI, and evaluation engineering:

- **Recommendations:** Representation learning (BGE embeddings), item-to-item nearest-neighbor suggestions, session-aware reranking, multi-armed bandit personalization
- **Agentic AI:** LangGraph orchestration with planning, tool use, conditional routing, clarifying-question interaction, human-in-the-loop design
- **Evaluation:** Rigorous offline metrics on human-labeled data, LLM-judge calibration study, regime analysis identifying when agentic approaches should and should not be used
- **Search:** Hybrid retrieval (BM25 + dense), query decomposition, reranking with LLM scoring

## Related work

- Chen et al. (2022). *WANDS: Dataset for Product Search Relevance Assessment.* ECIR 2022.
- Databricks (2023). *Enhancing Product Search with LLMs.* Solution accelerator using WANDS.
- Soviero et al. (2024). *ChatGPT Goes Shopping: LLMs Can Predict Relevance in eCommerce Search.* ECIR 2024.
- Hosseini et al. (2025). *Retrieve, Annotate, Evaluate, Repeat: Leveraging Multimodal LLMs for Large-Scale Product Retrieval Evaluation.* ECIR 2025.

## Author

**Ijaz Ul Haq, Ph.D.** — AI/ML Research Scientist
University of Vermont · Water Resources Institute
[GitHub](https://github.com/ejokhan) · [Google Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en)

Built on TACC Lonestar6 supercomputer.

## License

MIT
