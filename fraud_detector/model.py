# fraud_detector/model.py
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report
import joblib
import json
from datetime import datetime

FEATURES = ["amount", "hour", "is_night", "weekday", "tx_count_last_hour"]

def train_model(df, model_path="model.joblib", metrics_path="metrics.json"):
    """Train fraud detection model with time-based split."""
    print("Starting model training...")
    
    # Sort by timestamp (critical for time-based split)
    df = df.sort_values("timestamp")
    X = df[FEATURES]
    y = df["is_fraud"]
    
    # Time-based split: train on first 80%, test on last 20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training fraud rate: {y_train.mean():.2%}")
    print(f"Test fraud rate: {y_test.mean():.2%}")
    
    # Train XGBoost model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save model and metrics
    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("="*50)
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    
    return model, metrics

def load_model(path="model.joblib"):
    """Load trained model."""
    return joblib.load(path)

if __name__ == "__main__":
    # Train model
    df = pd.read_parquet("data/transactions.parquet")
    model, metrics = train_model(df)