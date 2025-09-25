"""
IoT Anomaly Detection - Kafka Streaming Consumer

This script consumes IoT sensor data from a Kafka topic, applies the
preprocessing pipeline and trained autoencoder model to compute anomaly
scores in real time, and prints/forwards alerts when the score exceeds
an adaptive threshold.

Environment variables:
  KAFKA_BOOTSTRAP_SERVERS: comma-separated brokers, e.g. localhost:9092
  KAFKA_TOPIC: topic name to consume from (default: iot-sensors)
  KAFKA_GROUP_ID: consumer group id (default: iot-anomaly)
  THRESHOLD_PATH: path to threshold file (default: models/threshold.txt)
  MODEL_PATH: path to saved Keras model (default: models/autoencoder_best.keras)

Usage:
  python streaming.py
"""

import os
import json
import signal
import sys
import time
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from tensorflow import keras

from data_preprocessing import IoTDataPreprocessor
from model import load_threshold, reconstruction_errors

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "iot-sensors")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "iot-anomaly")
MODEL_PATH = os.getenv("MODEL_PATH", "models/autoencoder_best.keras")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.txt")

SENSOR_COLUMNS = os.getenv("SENSOR_COLUMNS", "temperature,humidity,pressure,vibration").split(",")
TIMESTAMP_COL = os.getenv("TIMESTAMP_COL", "timestamp")
SEQUENCE_LEN = int(os.getenv("SEQUENCE_LEN", "10"))

running = True

def handle_sigterm(signum, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_sigterm)
signal.signal(signal.SIGTERM, handle_sigterm)


def build_consumer():
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP.split(","),
        group_id=GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        auto_offset_reset="latest",
        consumer_timeout_ms=1000,
    )


def main():
    print("Loading model and threshold...")
    model = keras.models.load_model(MODEL_PATH)
    threshold = load_threshold(THRESHOLD_PATH)
    print(f"Loaded threshold: {threshold}")

    pre = IoTDataPreprocessor(scaler_type="standard", window_size=SEQUENCE_LEN)
    # Warm up preprocessor feature list with empty frame
    pre.feature_columns = None

    buffer_rows = []

    consumer = build_consumer()
    print(f"Consuming from {TOPIC} on {BOOTSTRAP}")

    while running:
        polled = consumer.poll(timeout_ms=1000)
        any_msg = False
        for tp, messages in polled.items():
            for m in messages:
                any_msg = True
                payload = m.value  # expected dict with timestamp + sensors
                buffer_rows.append(payload)

        if not any_msg:
            time.sleep(0.1)
            continue

        # Keep last N rows for sequence/rolling features stability
        if len(buffer_rows) < SEQUENCE_LEN + 5:
            continue
        df = pd.DataFrame(buffer_rows[-(SEQUENCE_LEN + 128):])

        # Ensure columns exist
        for col in SENSOR_COLUMNS + [TIMESTAMP_COL]:
            if col not in df.columns:
                df[col] = np.nan

        try:
            # Fit on initial window if not yet fitted, then transform
            if pre.feature_columns is None:
                X_scaled, _ = pre.fit_transform(df, SENSOR_COLUMNS, timestamp_col=TIMESTAMP_COL)
            else:
                X_scaled = pre.transform(df, SENSOR_COLUMNS, timestamp_col=TIMESTAMP_COL)

            # Use last window for scoring
            X_window = X_scaled[-SEQUENCE_LEN:]
            # Model expects 2D; compute reconstruction on window rows and average
            errors = reconstruction_errors(model, X_window)
            score = float(np.mean(errors))
            is_anomaly = score > threshold

            ts = df[TIMESTAMP_COL].iloc[-1]
            print(json.dumps({
                "timestamp": str(ts),
                "score": score,
                "threshold": threshold,
                "anomaly": is_anomaly,
            }))
        except Exception as e:
            print(f"Processing error: {e}")
            continue

    consumer.close()
    print("Shutting down consumer")


if __name__ == "__main__":
    main()
