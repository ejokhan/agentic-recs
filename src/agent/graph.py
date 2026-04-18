"""
LangGraph agent — wires nodes into a state machine with conditional routing.

The graph:
    START → planner → [ambiguous?] → clarifier → retriever → reranker → responder → END
                           │
                           └── (not ambiguous) → retriever → reranker → responder → END
"""
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    planner_node,
    clarifier_node,
    retriever_node,
    reranker_node,
    responder_node,
)


def should_clarify(state: AgentState) -> str:
    """Conditional edge: route to clarifier if query is ambiguous."""
    if state.get("is_ambiguous", False) and state.get("ambiguity_score", 0) > 0.6:
        return "clarifier"
    return "retriever"


def build_agent():
    """Build and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("responder", responder_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Add edges
    graph.add_conditional_edges("planner", should_clarify,
                                {"clarifier": "clarifier", "retriever": "retriever"})
    graph.add_edge("clarifier", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "responder")
    graph.add_edge("responder", END)

    return graph.compile()


if __name__ == "__main__":
    agent = build_agent()
    # Print the graph structure
    print("Agent graph built successfully!")
    print(f"Nodes: {list(agent.get_graph().nodes.keys())}")
