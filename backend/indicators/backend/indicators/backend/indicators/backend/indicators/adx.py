"""
adx.py
------

Real implementation of the Average Directional Index (ADX).

Default:
- period: 14
"""

from typing import List
import pandas as pd
import numpy as np


def calculate_adx(
    high_prices: List[float],
    low_prices: List[float],
    close_prices: List[float],
    period: int = 14
) -> float:
    """
    Calculate the ADX (trend strength) for given high, low, close price lists.

    :return: last ADX value (0–100)
    """
    length = len(close_prices)
    if length <= period + 1:
        return 25.0  # neutral/trending dummy

    high = pd.Series(high_prices, dtype="float64")
    low = pd.Series(low_prices, dtype="float64")
    close = pd.Series(close_prices, dtype="float64")

    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = low.diff().abs()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm)
    minus_dm = pd.Series(minus_dm)

    # Smoothed averages (Wilder)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).abs() ) * 100
    adx = dx.rolling(window=period).mean()

    last_adx = adx.iloc[-1]
    if np.isnan(last_adx):
        return 25.0

    return float(last_adx)
