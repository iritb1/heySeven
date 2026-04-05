"""
Tests that the guard node correctly classifies in-scope and off-topic questions.
LLM calls are mocked — no real OpenAI requests are made.
"""

from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
from agent.nodes import guard_node

############ off-topic ############


@patch("agent.nodes.LLMClient")
def test_off_topic_weather_question(mock_client):
    """Verifies that a general knowledge question (weather) is classified as off-topic."""
    mock_client.return_value.invoke.return_value = AIMessage(content="off_topic")
    state = {"messages": [HumanMessage(content="What is the weather in Tel Aviv?")]}
    result = guard_node(state)
    assert "off_topic" in result["messages"][-1].content


@patch("agent.nodes.LLMClient")
def test_off_topic_booking_request(mock_client):
    """Verifies that an action request (booking a flight) is classified as off-topic."""
    mock_client.return_value.invoke.return_value = AIMessage(content="off_topic")
    state = {"messages": [HumanMessage(content="Book me a flight to Vegas")]}
    result = guard_node(state)
    assert "off_topic" in result["messages"][-1].content


############ in-scope ############


@patch("agent.nodes.LLMClient")
def test_in_scope_restaurant_question(mock_client):
    """Verifies that a restaurant question is classified as in-scope."""
    mock_client.return_value.invoke.return_value = AIMessage(content="in_scope")
    state = {"messages": [HumanMessage(content="What restaurants are open late?")]}
    result = guard_node(state)
    assert "in_scope" in result["messages"][-1].content


@patch("agent.nodes.LLMClient")
def test_in_scope_hotel_question(mock_client):
    """Verifies that a hotel amenity question is classified as in-scope."""
    mock_client.return_value.invoke.return_value = AIMessage(content="in_scope")
    state = {"messages": [HumanMessage(content="Do you have a heated pool?")]}
    result = guard_node(state)
    assert "in_scope" in result["messages"][-1].content
