import json
from pathlib import Path
from functools import lru_cache

from langchain_core.messages import AIMessage, SystemMessage

from agent.state import AgentState
from agent.llm_client import LLMClient
from agent.prompts import GUARD_SYSTEM, REFUSAL_MESSAGE, build_system_prompt
from utils.logger import Logger

logger = Logger("nodes")

DATA_PATH = Path(__file__).resolve().parent.parent / "casino_data.json" # security measures against SSRF


@lru_cache(maxsize=1)
def _load_casino_data() -> dict:
    """
    Loads casino_data.json into memory.

    Returns:
        dict: The parsed casino property data.

    Raises:
        FileNotFoundError: If casino_data.json does not exist.
    """
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def guard_node(state: AgentState) -> dict:
    """
    Classifies whether the user's question is about the property.

    Sends the last message to the LLM with the guard prompt.
    Defaults to "off_topic" on empty response or error.

    Args:
        state (AgentState): The current graph state containing messages.

    Returns:
        dict: State update with the guard's classification message.
    """
    try:
        response = LLMClient().invoke([SystemMessage(content=GUARD_SYSTEM), state["messages"][-1]])
        if not response or not response.content:
            logger.warning("Guard returned empty response — defaulting to off_topic")
            return {"messages": [AIMessage(content="off_topic")]}
        logger.info(f"Guard decision: {response.content.strip()}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Guard node failed: {e}")
        return {"messages": [AIMessage(content="off_topic")]}


def context_node(state: AgentState) -> dict:
    """
    Injects the property data as a SystemMessage (once per thread).

    Skips injection if the property data is already present in the
    message history to avoid duplicate context on multi-turn conversations.

    Args:
        state (AgentState): The current graph state containing messages.

    Returns:
        dict: State update with the SystemMessage, or empty if already injected.

    Raises:
        Exception: If casino_data.json cannot be loaded or parsed.
    """
    for msg in state["messages"]:
        if isinstance(msg, SystemMessage) and "knowledgeable concierge" in msg.content:
            return {"messages": []}

    try:
        data = _load_casino_data()
        system_prompt = build_system_prompt(data)
        return {"messages": [SystemMessage(content=system_prompt)]}
    except Exception as e:
        logger.error(f"Context node failed to load casino data: {e}")
        raise


def answer_node(state: AgentState) -> dict:
    """
    Generates the final answer by invoking the LLM with the full message history.

    Returns a fallback message on empty response or error.

    Args:
        state (AgentState): The current graph state containing messages.

    Returns:
        dict: State update with the LLM's response message.
    """
    try:
        response = LLMClient().invoke(state["messages"])
        if not response or not response.content:
            logger.warning("Answer LLM returned empty response")
            return {"messages": [AIMessage(content="I'm sorry, I couldn't generate a response. Please try again.")]}
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Answer node failed: {e}")
        return {"messages": [AIMessage(content="I'm sorry, something went wrong. Please try again.")]}


def refusal_node(_state: AgentState) -> dict:
    """
    Returns a fixed refusal message for off-topic questions.

    Args:
        _state (AgentState): LangGraph node signature.

    Returns:
        dict: State update with the refusal message.
    """
    return {"messages": [AIMessage(content=REFUSAL_MESSAGE)]}
