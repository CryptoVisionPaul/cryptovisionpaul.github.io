"""
lstm_predict.py
---------------

Prediction utilities for the CryptoVision LSTM model.

We keep two functions:

1) predict_next_price(prices)
   - simple fallback used currently by the demo endpoint
   - if something nu merge cu AI-ul, măcar nu crapă aplicația

2) predict_next_price_lstm(...)
   - folosește modelul LSTM antrenat (.h5)
   - reconstruiește features (price + indicatori)
   - face o predicție de preț viitor
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from backend.backend.data.prepare_dataset import build_feature_dataframe


def predict_next_price(prices: List[float]) -> float:
    """
    Fallback simplu: dacă nu avem model încă sau vrem ceva super-rapid,
    returnăm pur și simplu ultimul preț.

    Aceasta este funcția folosită acum de endpoint-ul /demo/prediction.
    """
    if not prices:
        raise ValueError("Price list is empty.")
    return float(prices[-1])


def predict_next_price_lstm(
    coin_id: str = "bitcoin",
    days: int = 180,
    seq_length: int = 30,
    model_path: str = "lstm_model_bitcoin.h5",
) -> float:
    """
    Folosește modelul LSTM antrenat pentru a prezice următorul preț.

    Pași:
    - încarcă modelul .h5
    - construiește dataframe cu features (price + indicatori)
    - scalează datele la fel ca în training (MinMaxScaler)
    - formează ultima fereastră de 'seq_length' timesteps
    - rulează modelul și inversează scaling-ul pentru a obține prețul real
    """
    try:
        model = load_model(model_path)
    except Exception as e:
        # Dacă modelul nu există sau nu poate fi încărcat, nu stricăm aplicația
        print(f"[LSTM PREDICT] Could not load model {model_path}: {e}")
        raise

    # Construim feature dataframe pentru monedă
    df: pd.DataFrame = build_feature_dataframe(coin_id=coin_id, days=days)

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

    if df_features.shape[0] < seq_length:
        raise ValueError(
            f"Not enough data for LSTM prediction. "
            f"Have {df_features.shape[0]} rows, need at least {seq_length}."
        )

    # MinMaxScaler – pentru predicție folosim un scaler nou pe seria curentă.
    # Pentru rezultate foarte precise, ideal am salva scaler-ul din training,
    # dar pentru prima versiune este suficient.
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features.values)

    # Luăm ultima fereastră de 'seq_length' timesteps
    last_window = scaled_data[-seq_length:, :]  # shape: (seq_length, num_features)
    X_input = np.expand_dims(last_window, axis=0)  # shape: (1, seq_length, num_features)

    # Predicția este în spațiul scalat
    y_scaled = model.predict(X_input)[0, 0]

    # Pentru a inversa scaling-ul, construim un vector "dummy"
    # cu num_features valori, punem y_scaled pe coloana de preț (index 0),
    # restul 0, și aplicăm inverse_transform.
    num_features = scaled_data.shape[1]
    dummy = np.zeros((1, num_features))
    dummy[0, 0] = y_scaled  # prețul prezis pe coloana "price"

    y_unscaled = scaler.inverse_transform(dummy)[0, 0]

    return float(y_unscaled)
