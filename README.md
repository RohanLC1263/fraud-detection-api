# Fraud Detection API

Real-time fraud detection system for financial transactions using XGBoost and FastAPI.

## Features

- Real-time fraud prediction API
- Synthetic data generation with realistic fraud patterns
- Time-based model evaluation
- Docker containerization
- CI/CD with GitHub Actions
- Prometheus metrics (coming soon)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python fraud_detector/data_gen.py
```

### 3. Train Model
```bash
python fraud_detector/model.py
```

### 4. Run API
```bash
uvicorn fraud_detector.api:app --reload
```

### 5. Test API
Visit http://localhost:8000/docs for interactive API documentation.

## Docker

### Build
```bash
docker build -t fraud-detection-api .
```

### Run
```bash
docker run -p 8000:8000 fraud-detection-api
```

## Project Structure
```
fraud-detection-api/
├── fraud_detector/          # Main application code
│   ├── data_gen.py         # Synthetic data generation
│   ├── model.py            # Model training
│   └── api.py              # FastAPI application
├── tests/                   # Test suite
├── data/                    # Generated datasets
├── Dockerfile              # Container definition
└── requirements.txt        # Python dependencies
```

## Model Performance

Current metrics on holdout test set:
- Precision: TBD (target: ≥0.90)
- Recall: TBD (target: ≥0.80)
- ROC-AUC: TBD (target: ≥0.95)

## License

MIT