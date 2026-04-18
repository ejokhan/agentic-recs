"""
Agent state — the data structure that flows through every LangGraph node.
Each node reads from state, does its work, and returns updated fields.
"""
from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # Input
    query: str                        # original user query
    query_class: Optional[str]        # WANDS product class (if known)

    # Ambiguity detection
    is_ambiguous: bool                # does the query need clarification?
    missing_info: list[str]           # what's underspecified?
    ambiguity_score: float            # 0-1 confidence

    # Clarification
    clarifying_question: Optional[str]
    clarification_response: Optional[str]  # simulated user answer
    refined_query: Optional[str]           # query after clarification

    # Retrieval
    retrieval_query: str              # what actually goes to the retriever
    retrieved_ids: list[int]          # product IDs from retriever
    retrieved_scores: list[float]     # retriever scores
    retrieved_names: list[str]        # product names (for display)

    # Reranking
    rerank_strategy: str              # which strategy the bandit picked
    reranked_ids: list[int]           # product IDs after reranking
    reranked_scores: list[float]

    # Response
    response: str                     # final formatted response
    reasoning_trace: list[str]        # log of what each node did
