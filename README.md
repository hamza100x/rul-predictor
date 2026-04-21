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

| Model | RMSE | Score (NASA metric) |
|-------|------|---------------------|
| Random Forest (baseline) | ~22 | TBD |
| LSTM | ~15 | TBD |

## Project Structure

```
rul-predictor/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_rf.ipynb
│   └── 04_lstm.ipynb
├── src/
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── api/
│   └── main.py
├── mlflow/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/hamza100x/rul-predictor.git
cd rul-predictor
pip install -r requirements.txt
# Run notebooks 01 → 02 → 03 → 04
uvicorn api.main:app --reload
```

## Tech Stack

PyTorch · scikit-learn · MLflow · FastAPI · Docker · HuggingFace · Gradio

## Author

**Hamza Danish** — M.Sc. AI, BTU Cottbus-Senftenberg  
[GitHub](https://github.com/hamza100x) · [X](https://x.com/hamza100x)
