"""
lstm_predict.py
---------------

This module will later load the trained LSTM model and generate predictions
for CryptoVision.
"""

from typing import List


def predict_next_price(prices: List[float]) -> float:
    """
    Temporary prediction function.
    Later it will:
    - load the saved LSTM model
    - prepare the input sequence
    - run the model and return a predicted price.

    For now, it simply returns the last price.
    """
    if not prices:
        raise ValueError("Price list is empty.")
    return float(prices[-1])


if __name__ == "__main__":
    # Simple manual test
    example = [100.0, 101.5, 102.3]
    print("Predicted price:", predict_next_price(example))
