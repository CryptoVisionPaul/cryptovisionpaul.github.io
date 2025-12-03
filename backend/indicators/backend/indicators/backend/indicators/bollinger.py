"""
bollinger.py
------------

Real implementation of Bollinger Bands.

Default settings:
- period: 20
- num_std: 2.0
"""

from typing import List, Tuple
import pandas as pd


def calculate_bollinger(
    prices: List[float],
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands for a list of prices.

    :param prices: list of closing prices
    :param period: moving average period (default 20)
    :param num_std: number of standard deviations (default 2.0)
    :return: tuple (middle_band, upper_band, lower_band) for the last value
    """
    if len(prices) < period:
        # Not enough data – return neutral bands
        return 0.0, 0.0, 0.0

    series = pd.Series(prices, dtype="float64")

    middle_band = series.rolling(window=period).mean()
    std_dev = series.rolling(window=period).std()

    upper_band = middle_band + num_std * std_dev
    lower_band = middle_band - num_std * std_dev

    mb_last = float(middle_band.iloc[-1])
    ub_last = float(upper_band.iloc[-1])
    lb_last = float(lower_band.iloc[-1])

    # Dacă din orice motiv e NaN (primele valori), întoarcem 0
    if any(map(pd.isna, [mb_last, ub_last, lb_last])):
        return 0.0, 0.0, 0.0

    return mb_last, ub_last, lb_last
