"""
rsi.py
------

Real implementation of the Relative Strength Index (RSI) indicator.

We use the classic 14-period RSI formula:
- RSI = 100 - (100 / (1 + RS))
- RS = average_gain / average_loss
"""

from typing import List
import numpy as np
import pandas as pd


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate the RSI for a list of closing prices.

    :param prices: list of price values (floats)
    :param period: RSI period (default 14)
    :return: last RSI value as float (0–100)
    """
    if len(prices) <= period:
        # Not enough data – return neutral RSI
        return 50.0

    series = pd.Series(prices, dtype="float64")

    # Price changes
    delta = series.diff()

    # Gains (up moves) and losses (down moves)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Average gains and losses (Wilder's smoothing approximation)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, np.nan)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # If the last RSI is NaN (not enough data), return neutral
    last_rsi = rsi.iloc[-1]
    if np.isnan(last_rsi):
        return 50.0

    return float(last_rsi)
