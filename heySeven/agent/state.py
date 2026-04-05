from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    State shared across all LangGraph nodes.
    messages: conversation history — add_messages appends
    instead of replacing, so every node builds on the previous.
    """
    messages: Annotated[list[BaseMessage], add_messages]
