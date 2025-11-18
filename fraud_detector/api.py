# fraud_detector/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from datetime import datetime
import json
import os

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions",
    version="1.0.0"
)

MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"
model = None
model_metrics = None

class Transaction(BaseModel):
    tx_id: str = Field(..., description="Unique transaction ID")
    user_id: str = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    is_night: int = Field(..., ge=0, le=1, description="Night transaction flag (0 or 1)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    tx_count_last_hour: int = Field(..., ge=0, description="Transaction count in last hour")
    device: str = Field(..., description="Device type (android/ios/web)")

    class Config:
        json_schema_extra = {
            "example": {
                "tx_id": "tx_12345",
                "user_id": "user_100",
                "merchant_id": "merchant_50",
                "amount": 125.50,
                "hour": 14,
                "is_night": 0,
                "weekday": 2,
                "tx_count_last_hour": 3,
                "device": "android"
            }
        }

class PredictionResponse(BaseModel):
    tx_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    timestamp: str

@app.on_event("startup")
def load_artifacts():
    """Load model and metrics on startup."""
    global model, model_metrics
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            model_metrics = json.load(f)
        print(f"✅ Loaded metrics from {METRICS_PATH}")
    else:
        model_metrics = {}

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "fraud-detection-api",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.get("/metrics")
def get_metrics():
    """Get model performance metrics."""
    if not model_metrics:
        raise HTTPException(status_code=404, detail="Metrics not available")
    return model_metrics

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """Predict fraud probability for a transaction."""
    try:
        # Prepare features
        X = pd.DataFrame([{
            "amount": transaction.amount,
            "hour": transaction.hour,
            "is_night": transaction.is_night,
            "weekday": transaction.weekday,
            "tx_count_last_hour": transaction.tx_count_last_hour
        }])
        
        # Predict
        fraud_prob = float(model.predict_proba(X)[0, 1])
        is_fraud = bool(fraud_prob > 0.5)
        
        # Determine risk level
        if fraud_prob < 0.3:
            risk_level = "low"
        elif fraud_prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return PredictionResponse(
            tx_id=transaction.tx_id,
            fraud_probability=round(fraud_prob, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)