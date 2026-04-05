from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import guard_node, context_node, answer_node, refusal_node
from utils.logger import Logger

logger = Logger("graph")

NODES_LIST = [("guard_node", guard_node), ("context_node", context_node), ("answer_node",  answer_node), ("refusal_node", refusal_node)]


def route_after_guard(state: AgentState) -> str:
    """
    Routes to context_node if in scope, refusal_node if not.

    Args:
        state (AgentState): The current graph state containing messages.

    Returns:
        str: "in_scope" if the guard classified the message as property-related,
             "off_topic" otherwise.
    """
    last = state["messages"][-1].content.strip().lower()
    return "in_scope" if "in_scope" in last else "off_topic"


def build_graph():
    """Builds and compiles the LangGraph agent."""
    graph = StateGraph(AgentState)


    for node_name , node_func in NODES_LIST:
        graph.add_node(node_name, node_func)

    graph.add_edge(START, "guard_node")

    graph.add_conditional_edges(
        "guard_node",
        route_after_guard,
        {
            "in_scope":  "context_node",
            "off_topic": "refusal_node",
        }
    )

    graph.add_edge("context_node", "answer_node")
    graph.add_edge("answer_node",  END)
    graph.add_edge("refusal_node", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Agent graph compiled successfully")
    return compiled


casino_agent = build_graph()