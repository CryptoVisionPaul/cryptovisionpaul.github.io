import requests
from typing import List, Dict

COINGECKO_URL = "https://api.coingecko.com/api/v3"

def fetch_real_historical_data(coin_id: str = "bitcoin", days: int = 30) -> List[Dict]:
    """
    Fetch historical market data from CoinGecko.
    Returns closing prices for the last X days.
    
    Example coin_id:
    - 'bitcoin'
    - 'ethereum'
    - 'cardano'
    """
    url = f"{COINGECKO_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "hourly"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"CoinGecko API Error: {response.status_code}")
    
    data = response.json()
    
    prices = data.get("prices", [])
    result = []
    
    for timestamp, price in prices:
        result.append({
            "timestamp": timestamp,
            "price": float(price)
        })
    
    return result


def fetch_sample_data():
    """
    Temporary fallback used before real API integration.
    Returns 7 fake prices.
    """
    return [
        {"timestamp": 1, "price": 100},
        {"timestamp": 2, "price": 101},
        {"timestamp": 3, "price": 102},
        {"timestamp": 4, "price": 103},
        {"timestamp": 5, "price": 104},
        {"timestamp": 6, "price": 103},
        {"timestamp": 7, "price": 105},
    ]
