"""
fetch_historical.py
--------------------

This module will be responsible for downloading and preparing
historical price data for CryptoVision.

For now, it contains a simple placeholder function that we will
upgrade step by step.
"""

from typing import List, Dict


def fetch_sample_data() -> List[Dict]:
    """
    Temporary test function.
    Later this will call a real crypto API (for example CoinGecko)
    to download historical prices.

    Returns a small in-memory dataset we can use for tests.
    """
    data = [
        {"timestamp": "2025-01-01T00:00:00Z", "price": 40000.0},
        {"timestamp": "2025-01-02T00:00:00Z", "price": 41000.0},
        {"timestamp": "2025-01-03T00:00:00Z", "price": 39500.0},
    ]
    return data


if __name__ == "__main__":
    # Small debug print – in the future we will replace this
    # with real data download + save to file.
    sample = fetch_sample_data()
    for row in sample:
        print(row)
