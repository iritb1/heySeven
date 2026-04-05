import sys

from utils.logger import Logger
from ingestion.parser import build_casino_data, save, PROPERTY_NAME

logger = Logger("ingest")


def main() -> None:
    logger.info(f"Starting data generation for: {PROPERTY_NAME}")

    data = build_casino_data()
    save(data, "casino_data.json")

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
