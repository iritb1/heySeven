"""
Tests for the FastAPI endpoints.
LLM and data loading are mocked — no real OpenAI requests or files needed.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from api.main import app

client = TestClient(app)

# Minimal mock data used across chat endpoint tests.
MOCK_DATA = {
    "property": {"name": "Test Casino", "location": "Test Location"},
    "restaurants": [], "hotel": {}, "casino": {},
    "players_club": {}, "amenities": {}, "entertainment": {}, "promotions": [],
}


def test_health_endpoint():
    """Verifies GET /health returns 200 with {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("agent.nodes._load_casino_data", return_value=MOCK_DATA)
@patch("agent.nodes.LLMClient")
def test_chat_endpoint_returns_answer(mock_client, _mock_load):
    """Verifies POST /chat returns the agent's answer for an in-scope question.

    Mocks the guard to classify as in_scope, then the answer node to return
    a fixed response. Asserts the response contains the expected answer.
    """
    mock_client.return_value.invoke.side_effect = [
        AIMessage(content="in_scope"),
        AIMessage(content="We have six restaurants!"),
    ]

    response = client.post("/chat", json={"message": "What restaurants do you have?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "We have six restaurants!"


@patch("agent.nodes._load_casino_data", return_value=MOCK_DATA)
@patch("agent.nodes.LLMClient")
def test_chat_endpoint_off_topic(mock_client, _mock_load):
    """Verifies that an off-topic question returns the refusal message."""
    mock_client.return_value.invoke.return_value = AIMessage(content="off_topic")

    response = client.post("/chat", json={"message": "What is the weather?"})
    assert response.status_code == 200
    assert "I'm best with questions" in response.json()["answer"]


@patch("agent.nodes._load_casino_data", return_value=MOCK_DATA)
@patch("agent.nodes.LLMClient")
def test_chat_returns_answer_for_in_scope(mock_client, _mock_load):
    """Verifies that an in-scope question about amenities returns a relevant answer."""
    mock_client.return_value.invoke.side_effect = [
        AIMessage(content="in_scope"),
        AIMessage(content="We have a heated indoor pool!"),
    ]

    response = client.post("/chat", json={"message": "Tell me about the pool"})
    assert response.status_code == 200
    assert "pool" in response.json()["answer"].lower()


def test_chat_rejects_empty_message():
    """Verifies that an empty message returns 422 (Pydantic min_length=1)."""
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_rejects_oversized_message():
    """Verifies that a message over 1000 chars returns 422 (Pydantic max_length=1000)."""
    response = client.post("/chat", json={"message": "x" * 1001})
    assert response.status_code == 422
