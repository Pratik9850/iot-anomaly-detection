"""
IoT Anomaly Detection - Autoencoder Model

This module defines and trains an autoencoder for unsupervised anomaly detection
on IoT sensor data. It includes utilities to:
- Build a dense autoencoder given input dimensionality and layer widths
- Train with early stopping and model checkpointing
- Compute reconstruction error and select anomaly thresholds
- Save and load trained models

Usage:
    from model import build_autoencoder, train_autoencoder, reconstruction_errors
    model = build_autoencoder(input_dim=X_train.shape[1])
    history = train_autoencoder(model, X_train, X_val)
    errors = reconstruction_errors(model, X_val)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int,
                      encoding_dims=(128, 64, 32),
                      activation="relu",
                      bottleneck_activation="relu",
                      output_activation=None,
                      l2=1e-6,
                      dropout=0.0) -> keras.Model:
    """
    Build a symmetrical dense autoencoder.

    Args:
        input_dim: Number of input features
        encoding_dims: Tuple of encoder layer sizes (decoder is mirrored)
        activation: Activation for hidden layers
        bottleneck_activation: Activation for bottleneck layer
        output_activation: Optional activation for output layer
        l2: L2 regularization factor
        dropout: Dropout rate applied after each hidden layer

    Returns:
        Compiled Keras autoencoder model
    """
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs

    # Encoder
    for i, units in enumerate(encoding_dims):
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=keras.regularizers.l2(l2),
                         name=f"enc_dense_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"enc_dropout_{i}")(x)

    # Bottleneck
    bottleneck = layers.Dense(encoding_dims[-1] // 2 if len(encoding_dims) > 0 else 16,
                              activation=bottleneck_activation,
                              kernel_regularizer=keras.regularizers.l2(l2),
                              name="bottleneck")(x)

    # Decoder (mirror)
    y = bottleneck
    for i, units in enumerate(reversed(encoding_dims)):
        y = layers.Dense(units, activation=activation,
                         kernel_regularizer=keras.regularizers.l2(l2),
                         name=f"dec_dense_{i}")(y)
        if dropout > 0:
            y = layers.Dropout(dropout, name=f"dec_dropout_{i}")(y)

    outputs = layers.Dense(input_dim, activation=output_activation, name="output")(y)
    autoencoder = keras.Model(inputs, outputs, name="iot_autoencoder")

    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        loss="mse")
    return autoencoder


def train_autoencoder(model: keras.Model,
                      X_train: np.ndarray,
                      X_val: np.ndarray = None,
                      epochs: int = 50,
                      batch_size: int = 256,
                      patience: int = 5,
                      model_dir: str = "models"):
    """
    Train autoencoder with early stopping and checkpointing.

    Args:
        model: Keras model to train
        X_train: Training data (normal samples)
        X_val: Validation data (preferably normal samples)
        epochs: Max training epochs
        batch_size: Batch size
        patience: Early stopping patience
        model_dir: Directory to save best model

    Returns:
        Training History
    """
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, "autoencoder_best.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                        save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    return history


def reconstruction_errors(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Compute sample-wise reconstruction MSE for anomaly scoring.
    """
    X_pred = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - X_pred), axis=1)
    return errors


def select_threshold(errors: np.ndarray, quantile: float = 0.99) -> float:
    """
    Select anomaly threshold as a high quantile of training errors.
    """
    return float(np.quantile(errors, quantile))


def save_threshold(threshold: float, path: str = "models/threshold.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(threshold))


def load_threshold(path: str = "models/threshold.txt") -> float:
    with open(path, "r") as f:
        return float(f.read().strip())


if __name__ == "__main__":
    # Minimal demo with random data
    np.random.seed(42)
    X_train = np.random.normal(size=(5000, 32)).astype("float32")
    X_val = np.random.normal(size=(1000, 32)).astype("float32")

    model = build_autoencoder(input_dim=X_train.shape[1])
    hist = train_autoencoder(model, X_train, X_val, epochs=5)

    errs = reconstruction_errors(model, X_val)
    th = select_threshold(errs)
    print("Validation error quantiles:", np.quantile(errs, [0.5, 0.9, 0.99]))
    print("Selected threshold:", th)
    save_threshold(th)
