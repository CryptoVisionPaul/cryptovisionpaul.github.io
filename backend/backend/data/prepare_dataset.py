"""
prepare_dataset.py
------------------

Builds a feature dataset for CryptoVision based on:
- real historical prices from CoinGecko
- technical indicators (RSI, MACD, Bollinger Bands, ADX)

This dataset will later be used to train the LSTM model.
"""

from typing import List, Dict

import pandas as pd

from .fetch_historical import fetch_real_historical_data
from ..indicators.rsi import calculate_rsi
from ..indicators.macd import calculate_macd
from ..indicators.bollinger import calculate_bollinger
from ..indicators.adx import calculate_adx


def build_feature_dataframe(coin_id: str = "bitcoin", days: int = 90) -> pd.DataFrame:
    """
    Fetch real historical data and build a DataFrame with:
    - timestamp
    - price
    - RSI
    - MACD (value, signal, histogram)
    - Bollinger bands (middle, upper, lower)
    - ADX

    :param coin_id: CoinGecko coin id (e.g. 'bitcoin', 'ethereum')
    :param days: how many days of history to download
    :return: pandas DataFrame with features
    """
    raw: List[Dict] = fetch_real_historical_data(coin_id=coin_id, days=days)

    if not raw:
        raise ValueError("No data returned from fetch_real_historical_data")

    df = pd.DataFrame(raw)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert timestamp from ms to pandas datetime (optional but useful)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    prices = df["price"].tolist()

    # Calculate indicators
    rsi_value = []
    macd_value = []
    macd_signal = []
    macd_hist = []
    bb_middle = []
    bb_upper = []
    bb_lower = []
    adx_list = []

    for i in range(len(prices)):
        window_prices = prices[: i + 1]

        rsi = calculate_rsi(window_prices)
        macd, signal, hist = calculate_macd(window_prices)
        mb, ub, lb = calculate_bollinger(window_prices)

        # For now we approximate ADX using the same price list
        adx = calculate_adx(window_prices, window_prices, window_prices)

        rsi_value.append(rsi)
        macd_value.append(macd)
        macd_signal.append(signal)
        macd_hist.append(hist)
        bb_middle.append(mb)
        bb_upper.append(ub)
        bb_lower.append(lb)
        adx_list.append(adx)

    df["rsi"] = rsi_value
    df["macd"] = macd_value
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["bb_middle"] = bb_middle
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["adx"] = adx_list

    return df


if __name__ == "__main__":
    # Example manual usage:
    # When you run this file on a PC/server with Python, it will
    # create a CSV file with features for the chosen coin.
    coin = "bitcoin"
    days = 90

    print(f"Building feature dataset for {coin} (last {days} days)...")
    features_df = build_feature_dataframe(coin_id=coin, days=days)

    output_path = f"{coin}_features_{days}d.csv}"
    features_df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path} with shape {features_df.shape}")
