"""
macd.py
-------

Real implementation of the MACD (Moving Average Convergence Divergence).

Default settings:
- fast EMA: 12 periods
- slow EMA: 26 periods
- signal line: 9-period EMA of MACD
"""

from typing import List
import pandas as pd


def calculate_macd(
    prices: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
):
    """
    Calculate MACD, signal line and histogram for a list of prices.

    :param prices: list of closing prices
    :param fast_period: fast EMA length (default 12)
    :param slow_period: slow EMA length (default 26)
    :param signal_period: signal EMA length (default 9)
    :return: tuple (macd_last, signal_last, hist_last)
    """
    if len(prices) < slow_period + signal_period:
        # Not enough data, return neutral values
        return 0.0, 0.0, 0.0

    series = pd.Series(prices, dtype="float64")

    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_last = float(macd_line.iloc[-1])
    signal_last = float(signal_line.iloc[-1])
    hist_last = float(histogram.iloc[-1])

    return macd_last, signal_last, hist_last
