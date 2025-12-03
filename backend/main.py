from typing import List, Dict

from fastapi import FastAPI

# Import from our own modules
from data.fetch_historical import fetch_sample_data
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.bollinger import calculate_bollinger
from indicators.adx import calculate_adx
from models.lstm_predict import predict_next_price

app = FastAPI(
    title="CryptoVision Backend",
    version="0.1.0",
    description="Backend API for data, indicators and AI-based predictions for CryptoVision."
)


@app.get("/health")
def health_check():
    """
    Simple endpoint to verify that the backend is alive.
    """
    return {
        "status": "ok",
        "message": "CryptoVision backend is running",
    }


@app.get("/demo/prediction")
def demo_prediction():
    """
    Demo endpoint that:
    - loads sample price data
    - calculates placeholder indicators
    - calls the placeholder prediction function

    This lets us test the full pipeline before we add real AI logic.
    """
    # 1. Get sample data
    data: List[Dict] = fetch_sample_data()
    prices: List[float] = [row["price"] for row in data]

    if not prices:
        return {"error": "No price data available"}

    last_price = float(prices[-1])

    # 2. Calculate indicators (placeholder values for now)
    rsi_value = calculate_rsi(prices)
    macd_value, macd_signal, macd_hist = calculate_macd(prices)
    bb_middle, bb_upper, bb_lower = calculate_bollinger(prices)
    adx_value = calculate_adx(prices, prices, prices)

    # 3. "Predict" next price using our dummy LSTM predictor
    predicted_price = predict_next_price(prices)

    return {
        "last_price": last_price,
        "predicted_price": predicted_price,
        "indicators": {
            "rsi": rsi_value,
            "macd": {
                "macd": macd_value,
                "signal": macd_signal,
                "histogram": macd_hist,
            },
            "bollinger": {
                "middle": bb_middle,
                "upper": bb_upper,
                "lower": bb_lower,
            },
            "adx": adx_value,
        },
        "data_points": len(prices),
    }
