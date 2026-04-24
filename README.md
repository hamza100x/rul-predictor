# Predictive Maintenance — RUL Prediction

> Predicting **Remaining Useful Life (RUL)** of turbofan engines using NASA CMAPSS data.  
> Full ML pipeline: EDA → Feature Engineering → LSTM → MLflow → FastAPI → Docker → HuggingFace

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://kaggle.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

Industrial equipment failure is costly. By predicting RUL, maintenance can be scheduled proactively — reducing downtime and preventing catastrophic failures. This project applies ML to NASA's CMAPSS turbofan engine dataset to predict how many cycles remain before failure.

## Results

| Model | RMSE | MAE | NASA Score |
|-------|------|-----|------------|
| Random Forest (baseline) | 16.94 | 11.82 | 224,691 |
| LSTM (2-layer, 64 hidden) | 19.69 | 13.62 | 159,810 |

**Baseline Comparison**: The Random Forest baseline achieves 16.94 RMSE on the test set, with better generalization characteristics. The LSTM model at 19.69 RMSE shows broader predictions but has learned temporal patterns. The RF baseline is recommended for deployment on this dataset.

## Project Structure

```
rul-predictor/
├── notebooks/
│   ├── 01_eda.ipynb                      ← EDA and RUL distribution
│   ├── 02_feature_engineering.ipynb      ← Rolling features, scaling, sequences
│   ├── 03_baseline_rf.ipynb              ← Random Forest baseline training
│   └── 04_lstm.ipynb                     ← LSTM sequence model training
├── src/
│   ├── features.py                       ← CMAPSS loading, RUL labeling, feature creation
│   ├── train.py                          ← RF training CLI with metrics
│   └── predict.py                        ← Inference for RF model
├── api/
│   └── main.py                           ← FastAPI server with /predict, /metrics endpoints
├── data/
│   ├── raw/                              ← Original CMAPSS files (train/test/RUL)
│   └── processed/                        ← Feature tables, sequences, scaler artifacts
├── models/
│   ├── rf_baseline.pkl                   ← Trained RF model
│   ├── rf_metrics.json                   ← Test RMSE, MAE, NASA score
│   ├── rf_test_predictions.csv           ← Predictions on test set
│   ├── lstm_model.pth                    ← Trained LSTM weights (PyTorch)
│   └── lstm_metrics.json                 ← LSTM test metrics
├── reports/
│   └── figures/                          ← Generated plots (EDA, RF diagnostics, LSTM training)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quickstart

### Development
```bash
git clone https://github.com/hamza100x/rul-predictor.git
cd rul-predictor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Optional extras (MLflow and related tooling)
pip install -r requirements-optional.txt

# Run the pipeline
jupyter notebook notebooks/
# Execute in order: 01_eda → 02_feature_engineering → 03_baseline_rf → 04_lstm
```

### API Server
```bash
# Start FastAPI development server using the project venv interpreter
# Windows (PowerShell)
.venv\Scripts\python -m uvicorn api.main:app --reload --port 8000

# macOS/Linux
.venv/bin/python -m uvicorn api.main:app --reload --port 8000

# Optional: confirm you are using the project environment
# Windows: .venv\Scripts\python -m pip show fastapi uvicorn pydantic pyarrow
# macOS/Linux: .venv/bin/python -m pip show fastapi uvicorn pydantic pyarrow

# Visit http://localhost:8000/docs for interactive API docs
# GET  /health        - Health check and available models
# GET  /metrics       - Baseline RF metrics
# POST /predict       - Make predictions (JSON body with data_path)
```

### Docker
```bash
# Build and run in container
docker-compose up --build

# Predictions via container
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/processed/test_fd001_features.parquet"}'
```

### CLI Inference
```bash
# RF baseline predictions
python src/train.py \
  --train-data data/processed/train_fd001_features.parquet \
  --test-data data/processed/test_fd001_features.parquet \
  --output-dir models

# Predict on new data
python src/predict.py \
  --model models/rf_baseline.pkl \
  --data data/processed/test_fd001_features.parquet \
  --output predictions.csv
```

## Tech Stack

PyTorch · scikit-learn · MLflow · FastAPI · Docker · HuggingFace · Gradio

## Author

**Hamza Danish** — M.Sc. AI, BTU Cottbus-Senftenberg  
[GitHub](https://github.com/hamza100x) · [X](https://x.com/hamza100x)
