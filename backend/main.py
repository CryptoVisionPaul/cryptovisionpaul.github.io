from fastapi import FastAPI

app = FastAPI(
    title="CryptoVision Backend",
    version="0.1.0",
    description="Backend API for AI models, predictions, and market insights."
)

@app.get("/health")
def health_check():
    """
    Endpoint to verify that the backend is alive.
    """
    return {
        "status": "ok",
        "message": "CryptoVision backend is running",
    }


@app.get("/hello")
def hello(name: str = "Paul"):
    """
    Test endpoint to verify that API parameters work.
    Example: /hello?name=Paul
    """
    return {"message": f"Hello, {name}!"}
