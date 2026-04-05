import json
from datetime import date
from typing import Any, Dict

from pydantic import BaseModel, Field

from agent.llm_client import LLMClient
from langchain_core.messages import SystemMessage, HumanMessage
from utils.logger import Logger

logger = Logger("parser")

PROPERTY_NAME = "Twin Arrows Navajo Casino Resort"

EXTRACTION_PROMPT = f"""\
You are a casino property data generator. Generate a comprehensive JSON object
about {PROPERTY_NAME} containing everything a guest might want to know before
or during a visit — including its restaurants, entertainment, amenities, rooms,
promotions, and any other useful guest information.

Return ONLY valid JSON — no markdown fences, no commentary.
Be as accurate and detailed as possible. Use null for anything you're unsure about.
"""


class Meta(BaseModel):
    """Metadata attached to every generated casino data file."""
    property: str = PROPERTY_NAME
    generated_date: str = Field(default_factory=lambda: date.today().isoformat())
    strategy: str = "LLM-generated from model knowledge."


def build_casino_data() -> Dict[str, Any]:
    """Generates structured casino property data via the LLM.

    Sends EXTRACTION_PROMPT to the LLM, parses the JSON response,
    attaches metadata, and returns the result.

    Returns:
        Dict[str, Any]: The generated property data with an '_meta' key.

    Raises:
        ValueError: If the LLM returns invalid JSON.
    """
    logger.info(f"Requesting LLM to generate data for: {PROPERTY_NAME}")

    response = LLMClient().invoke([
        SystemMessage(content=EXTRACTION_PROMPT),
        HumanMessage(content=f"Generate the full property data JSON for: {PROPERTY_NAME}"),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    try:
        raw_data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}")
        raise ValueError(f"LLM data generation failed — invalid JSON: {e}")

    raw_data["_meta"] = Meta().model_dump()

    logger.info("casino_data structure built successfully")
    return raw_data


def save(data: Dict[str, Any], path: str = "casino_data.json") -> None:
    """Saves the casino data dict to a JSON file.

    Args:
        data (Dict[str, Any]): The casino property data to save.
        path (str): Output file path. Defaults to 'casino_data.json'.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        size_kb = f.tell() / 1024
    logger.info(f"Saved {path} ({size_kb:.1f} KB)")
