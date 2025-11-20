from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_stock

app = FastAPI()

class StockFeatures(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjclose: float
    Return: float
    MA20: float
    MA50: float
    Volatility: float
    RSI: float

@app.post("/predict")
def predict(features: StockFeatures):
    result = predict_stock(features.dict())
    return {"prediction": result}
