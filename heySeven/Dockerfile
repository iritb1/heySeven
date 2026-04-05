FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Run ingestion to generate casino_data.json, then start the API
CMD ["sh", "-c", "python ingestion/ingest.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
