from typing import List, Dict

from fastapi import FastAPI

# Import from our own modules (relative imports)
from .data.fetch_historical import fetch_real_historical_data, fetch_sample_data
from .indicators.rsi import calculate_rsi
from .indicators.macd import calculate_macd
from .indicators.bollinger import calculate_bollinger
from .indicators.adx import calculate_adx
from .models.lstm_predict import predict_next_price

app = FastAPI(
    title="CryptoVision Backend",
    version="0.2.0",
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
def demo_prediction(coin_id: str = "bitcoin", days: int = 30):
    """
    Demo endpoint that:
    - loads REAL historical data from CoinGecko (with fallback to sample data)
    - calculates REAL indicators (RSI, MACD, Bollinger, ADX)
    - calls the (for now simple) LSTM predictor

    :param coin_id: CoinGecko coin id (e.g. 'bitcoin', 'ethereum')
    :param days: how many days of history to load
    """
    # 1. Try to get REAL data from CoinGecko
    try:
        data: List[Dict] = fetch_real_historical_data(coin_id=coin_id, days=days)
        source = "coingecko"
    except Exception as e:
        # If something goes wrong, use sample data so the API still works
        data = fetch_sample_data()
        source = f"sample_fallback ({str(e)})"

    prices: List[float] = [row["price"] for row in data]

    if not prices:
        return {"error": "No price data available"}

    last_price = float(prices[-1])

    # 2. Calculate REAL indicators
    rsi_value = calculate_rsi(prices)
    macd_value, macd_signal, macd_hist = calculate_macd(prices)
    bb_middle, bb_upper, bb_lower = calculate_bollinger(prices)

    # For ADX we would ideally use high/low/close separate.
    # For now we approximate using the same price series for all three.
    adx_value = calculate_adx(prices, prices, prices)

    # 3. "Predict" next price using our (currently simple) LSTM predictor
    predicted_price = predict_next_price(prices)

    return {
        "coin_id": coin_id,
        "days": days,
        "data_source": source,
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
