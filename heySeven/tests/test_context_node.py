"""
Tests that the context node builds a correct system prompt from casino_data.json.
File I/O is mocked — no real JSON file required.
"""

from unittest.mock import patch
from langchain_core.messages import HumanMessage, SystemMessage
from agent.nodes import context_node

# Minimal mock data representing the structure of casino_data.json.
MOCK_CASINO_DATA = {
    "property": {
        "name": "Twin Arrows Navajo Casino Resort",
        "location": "22181 Resort Boulevard, Flagstaff, AZ 86004",
        "phone": "928-856-7200",
        "website": "https://www.twinarrows.com",
        "description": "Northern Arizona's premier casino resort.",
        "casino_hours": "Open 24 hours, 7 days a week",
        "directions": "Off I-40 at Exit 219",
    },
    "restaurants": [{"name": "Zenith Steakhouse", "type": "Fine Dining"}],
    "hotel": {"room_types": [{"type": "King", "description": "King bed"}]},
    "casino": {"slots": {"count": "1,100+"}, "table_games": ["Blackjack"]},
    "players_club": {"name": "Navajo Players Club"},
    "amenities": {"pool": "Heated indoor pool"},
    "entertainment": {"venues": [{"name": "Event Center"}]},
    "promotions": [{"name": "Romance Package"}],
}


@patch("agent.nodes._load_casino_data", return_value=MOCK_CASINO_DATA)
def test_context_node_returns_system_message(_mock_load):
    """Verifies that the context node returns exactly one SystemMessage."""
    state = {"messages": [HumanMessage(content="Do you have a pool?")]}
    result = context_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], SystemMessage)


@patch("agent.nodes._load_casino_data", return_value=MOCK_CASINO_DATA)
def test_context_node_prompt_contains_property_name(_mock_load):
    """Verifies that the property name appears in the generated system prompt."""
    state = {"messages": [HumanMessage(content="Do you have a pool?")]}
    result = context_node(state)

    assert "Twin Arrows Navajo Casino Resort" in result["messages"][0].content


@patch("agent.nodes._load_casino_data", return_value=MOCK_CASINO_DATA)
def test_context_node_prompt_contains_all_sections(_mock_load):
    """Verifies that key property data (restaurants, games, amenities, promotions)
    is present in the system prompt."""
    state = {"messages": [HumanMessage(content="Do you have a pool?")]}
    result = context_node(state)

    prompt = result["messages"][0].content
    for keyword in ["Zenith Steakhouse", "Blackjack", "Heated indoor pool", "Romance Package"]:
        assert keyword in prompt


@patch("agent.nodes._load_casino_data", return_value=MOCK_CASINO_DATA)
def test_context_node_prompt_contains_instructions(_mock_load):
    """Verifies that the guardrail instructions (answer only from data, no fabrication)
    are included in the system prompt."""
    state = {"messages": [HumanMessage(content="Do you have a pool?")]}
    result = context_node(state)

    prompt = result["messages"][0].content
    assert "Answer ONLY based on the property information below" in prompt
    assert "Never make up facts" in prompt
