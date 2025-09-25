# syntax=docker/dockerfile:1
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy reqs first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Environment defaults
ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
    KAFKA_TOPIC=iot-sensors \
    KAFKA_GROUP_ID=iot-anomaly \
    MODEL_PATH=models/autoencoder_best.keras \
    THRESHOLD_PATH=models/threshold.txt \
    PYTHONUNBUFFERED=1

# Entrypoint to start streaming consumer
CMD ["python", "streaming.py"]
