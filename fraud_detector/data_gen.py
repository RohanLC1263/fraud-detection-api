# fraud_detector/data_gen.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

RNG = np.random.default_rng(42)

def generate_transactions(n=100000, fraud_rate=0.02, seed=42):
    """Generate synthetic transaction data with realistic fraud patterns."""
    np.random.seed(seed)
    random.seed(seed)
    
    users = [f"user_{i}" for i in range(1000)]
    merchants = [f"merchant_{i}" for i in range(200)]
    start = datetime(2024, 1, 1)

    rows = []
    for i in range(n):
        ts = start + timedelta(seconds=int(RNG.integers(0, 60*60*24*365)))
        user = RNG.choice(users)
        merchant = RNG.choice(merchants)
        amount = max(1.0, float(np.round(RNG.exponential(scale=50.0), 2)))
        
        # Extract temporal features
        hour = ts.hour
        is_night = int(hour < 6 or hour > 22)
        weekday = ts.weekday()
        
        # Velocity feature: simulate transaction bursts
        base_vel = RNG.poisson(1)
        if RNG.random() < 0.01:  # 1% of users have bursts
            base_vel += RNG.integers(5, 25)
        
        device = RNG.choice(["android", "ios", "web"])
        
        # Fraud labeling with realistic patterns
        fraud = 0
        
        # Pattern 1: High amount + high velocity
        if amount > 500 and base_vel > 10:
            fraud = 1
        
        # Pattern 2: Night transactions with high amounts
        if is_night and amount > 300 and RNG.random() < 0.3:
            fraud = 1
        
        # Pattern 3: Random fraud to reach target rate
        if RNG.random() < (fraud_rate - 0.005):
            fraud = 1

        rows.append({
            "tx_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "user_id": user,
            "merchant_id": merchant,
            "amount": amount,
            "hour": hour,
            "is_night": is_night,
            "weekday": weekday,
            "tx_count_last_hour": base_vel,
            "device": device,
            "is_fraud": fraud
        })
    
    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} transactions")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    return df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_transactions(100000)
    df.to_parquet("data/transactions.parquet", index=False)
    print(f"✅ Saved data/transactions.parquet")
    print(f"   Total transactions: {len(df)}")
    print(f"   Fraud transactions: {df['is_fraud'].sum()}")