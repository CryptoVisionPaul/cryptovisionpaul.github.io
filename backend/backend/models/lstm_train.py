"""
lstm_train.py
-------------

Basic LSTM training pipeline for CryptoVision.

IMPORTANT:
- This is the first simple version of the model.
- Later we can improve architecture, add more features, tune hyperparameters.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Importăm dataset-ul de features (preț + indicatori)
# Observă: aici folosim calea completă, conform structurii actuale a proiectului.
from backend.backend.data.prepare_dataset import build_feature_dataframe


def create_sequences(
    data: np.ndarray,
    target_index: int,
    seq_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transformă un array 2D (num_samples x num_features)
    în secvențe 3D pentru LSTM (num_seq x seq_length x num_features),
    plus vectorul țintă (prețul viitor).

    :param data: array 2D cu features
    :param target_index: indexul coloanei care reprezintă prețul (target)
    :param seq_length: lungimea secvenței (număr de timesteps)
    """
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        window = data[i : i + seq_length]
        target = data[i + seq_length, target_index]  # prețul de după fereastră
        xs.append(window)
        ys.append(target)

    return np.array(xs), np.array(ys)


def train_lstm_model(
    coin_id: str = "bitcoin",
    days: int = 180,
    seq_length: int = 30,
    epochs: int = 10,
    batch_size: int = 32,
    save_path: str = "lstm_model.h5",
) -> None:
    """
    Antrenează un model LSTM simplu pe baza:
    - prețului
    - indicatorilor tehnici (RSI, MACD, Bollinger, ADX)

    și salvează modelul într-un fișier .h5.
    """
    print(f"[LSTM TRAIN] Building feature dataset for {coin_id} / last {days} days...")
    df: pd.DataFrame = build_feature_dataframe(coin_id=coin_id, days=days)

    # Alegem coloanele de features (inclusiv prețul)
    feature_columns = [
        "price",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "adx",
    ]

    df_features = df[feature_columns].copy().dropna()

    if df_features.shape[0] < seq_length + 10:
        raise ValueError(
            f"Not enough data after indicators for training. "
            f"Have {df_features.shape[0]} rows, need at least {seq_length + 10}."
        )

    # Targetul nostru este prețul (index 0 în feature_columns)
    target_index = feature_columns.index("price")

    # Normalizăm features cu MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features.values)

    # Generăm secvențele pentru LSTM
    X, y = create_sequences(scaled_data, target_index=target_index, seq_length=seq_length)

    print(f"[LSTM TRAIN] Dataset shape: X={X.shape}, y={y.shape}")

    # Împărțim în train / validation (simplu: 80% / 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"[LSTM TRAIN] Train: {X_train.shape}, Val: {X_val.shape}")

    # Definim modelul LSTM
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_length, len(feature_columns))))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))  # Predicție preț

    model.compile(optimizer="adam", loss="mse")

    print("[LSTM TRAIN] Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    print("[LSTM TRAIN] Training finished. Saving model...")
    model.save(save_path)
    print(f"[LSTM TRAIN] Model saved to {save_path}")


if __name__ == "__main__":
    # Exemplu de rulare manuală pe un PC/server:
    # python -m backend.models.lstm_train
    train_lstm_model(
        coin_id="bitcoin",
        days=180,
        seq_length=30,
        epochs=5,       # pentru teste inițiale, menținem mic
        batch_size=32,
        save_path="lstm_model_bitcoin.h5",
    )
