FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY setup.py .

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]