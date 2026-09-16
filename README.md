# 🛒 Reproducible ML Pipeline: Rossmann Store Sales Forecasting

A production-grade, end-to-end machine learning pipeline for **Rossmann Store Sales** forecasting that emphasizes reproducibility and version control. Built with **DVC** for data/model versioning, **XGBoost** for model training, **FastAPI** for model serving, and **Docker** for containerization.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Pipeline](#running-the-pipeline)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Experiment Tracking](#experiment-tracking)
- [Metrics and Results](#metrics-and-results)
- [Technology Stack](#technology-stack)

---

## 🎯 Overview

This project demonstrates a full **MLOps workflow** from data ingestion to deployment:

1. **Data Generation**: Synthetic Rossmann-style dataset (100K+ rows, 1115 stores)
2. **Data Version Control**: DVC tracks raw data, processed features, and model artifacts
3. **Reproducible Pipeline**: 4-stage DVC pipeline (`preprocess → feature_engineering → train → evaluate`)
4. **Model Serving**: FastAPI REST API with Pydantic validation
5. **Containerization**: Multi-stage Docker build for production deployment

---

## 🏗️ Architecture

### ML Pipeline DAG

```
┌──────────────────┐    ┌──────────────────┐
│  data/raw/       │    │  data/raw/       │
│  train.csv (DVC) │    │  store.csv (DVC) │
└────────┬─────────┘    └─────────┬────────┘
         │                        │
         └───────────┬────────────┘
                     │
              ┌──────▼──────┐
              │  PREPROCESS │  ← src/preprocess.py
              │             │    Merge, clean, fill NaN
              └──────┬──────┘
                     │
                     ▼
           ┌─────────────────┐
           │    FEATURE       │  ← src/feature_engineering.py
           │   ENGINEERING    │    Date, competition, promo features
           └────────┬────────┘    One-hot encoding, train/test split
                    │
            ┌───────┴───────┐
            │               │
       ┌────▼────┐    ┌─────▼──────┐
       │  TRAIN  │    │  test data │
       │         │    │            │
       └────┬────┘    └─────┬──────┘
            │               │
            └───────┬───────┘
                    │
             ┌──────▼──────┐
             │  EVALUATE   │  ← src/evaluate.py
             │             │    RMSE, MAE, R², RMSPE
             └──────┬──────┘
                    │
            ┌───────┴───────┐
            │               │
    ┌───────▼──────┐  ┌─────▼─────────┐
    │ metrics.json │  │ model.joblib  │
    │ (Git tracked)│  │ (DVC tracked) │
    └──────────────┘  └───────┬───────┘
                              │
                     ┌────────▼────────┐
                     │   FastAPI       │
                     │   /predict      │
                     │   /health       │
                     └─────────────────┘
```

### Data Flow

```
Raw CSV Data → DVC → Preprocess → Feature Engineering → Train (XGBoost) → Evaluate → API Serve
     │                    │               │                  │          │
     └── DVC tracked ─────┴── DVC out ────┴── DVC out ───────┴── DVC ──┘
```

---

## 📁 Project Structure

```
ml-sales-pipeline/
├── README.md                   # This file
├── submission.yml              # Automated evaluation commands
├── Dockerfile                  # Multi-stage Docker build
├── .gitignore                  # Git ignores
├── .dockerignore               # Docker build context excludes
├── dvc.yaml                    # DVC pipeline definition (4 stages)
├── dvc.lock                    # DVC lock file (reproducibility)
├── params.yaml                 # Hyperparameters & configuration
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── raw/                    # Raw data (DVC-tracked)
│   │   ├── train.csv           # 100K rows of sales data
│   │   ├── store.csv           # 1115 store metadata
│   │   ├── train.csv.dvc       # DVC pointer file
│   │   └── store.csv.dvc       # DVC pointer file
│   ├── processed/              # Cleaned data (DVC pipeline output)
│   │   └── train_cleaned.csv
│   └── features/               # Feature-engineered data (DVC pipeline output)
│       ├── train_features.csv
│       └── test_features.csv
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── preprocess.py           # Stage 1: Data cleaning
│   ├── feature_engineering.py  # Stage 2: Feature engineering
│   ├── train.py                # Stage 3: XGBoost training
│   └── evaluate.py             # Stage 4: Model evaluation
│
├── models/
│   ├── model.joblib            # Trained model (DVC-tracked)
│   └── feature_names.json      # Feature names for inference
│
├── metrics/
│   ├── metrics.json            # Evaluation metrics (Git-tracked)
│   └── feature_importance.json # Feature importances
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── schemas.py              # Pydantic request/response models
│   └── model_loader.py         # Model loading and feature prep
│
└── tests/
    ├── __init__.py
    └── test_api.py             # API endpoint tests
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Git
- Docker (for containerized deployment)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/the-sadanand/ml-sales-pipeline
cd ml-sales-pipeline

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure DVC remote (adjust path as needed)
dvc remote modify local_remote url /path/to/your/dvc-remote

# 5. Pull DVC-tracked data and models
dvc pull

# 6. (Optional) Reproduce the pipeline from scratch
dvc repro
```

---

## 🔄 Running the Pipeline

### Full Pipeline Reproduction

```bash
# Run the entire pipeline
dvc repro

# View the pipeline DAG
dvc dag

# Check pipeline status
dvc status
```

### Running Individual Stages

```bash
# Preprocess only
dvc repro preprocess

# Train only (will run dependencies if needed)
dvc repro train
```

### View Metrics

```bash
# Show current metrics
dvc metrics show

# Compare metrics across branches/commits
dvc metrics diff <branch-or-commit>
```

---

## 🌐 API Usage

### Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/model/info` | Model metadata |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Store": 1,
    "DayOfWeek": 5,
    "Open": 1,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 0,
    "StoreType": "c",
    "Assortment": "a",
    "CompetitionDistance": 1270.0,
    "Year": 2015,
    "Month": 7,
    "Day": 31
  }'
```

**Response:**
```json
{
  "store_id": 1,
  "predicted_sales": 7234.56,
  "model_version": "1.0"
}
```

### Example: Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"Store": 1, "DayOfWeek": 5, "Open": 1, "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0, "StoreType": "c", "Assortment": "a", "CompetitionDistance": 1270.0, "Year": 2015, "Month": 7, "Day": 31},
      {"Store": 2, "DayOfWeek": 3, "Open": 1, "Promo": 0, "StateHoliday": "0", "SchoolHoliday": 1, "StoreType": "a", "Assortment": "b", "CompetitionDistance": 500.0, "Year": 2015, "Month": 6, "Day": 15}
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t ml-sales-pipeline .

# Run the container
docker run -p 8000:8000 ml-sales-pipeline

# Run with environment variables
docker run -p 8000:8000 -e LOG_LEVEL=info ml-sales-pipeline
```

### Docker Compose (Optional)

```bash
docker-compose up --build
```

The Dockerfile uses a **multi-stage build**:
1. **Builder stage**: Compiles dependencies with build tools
2. **Runtime stage**: Minimal image with non-root user, health checks

---

## 🔬 Experiment Tracking

### Running Experiments with DVC

DVC makes it easy to track experiments by modifying hyperparameters:

```bash
# 1. Create a new experiment branch
git checkout -b experiment/more-trees

# 2. Modify hyperparameters in params.yaml
# Change n_estimators from 500 to 1000

# 3. Run the pipeline (only affected stages re-run)
dvc repro

# 4. View new metrics
dvc metrics show

# 5. Commit the experiment
git add .
git commit -m "experiment: increase n_estimators to 1000"

# 6. Compare with main branch
git checkout main
dvc metrics diff experiment/more-trees
```

### Key Parameters (params.yaml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.n_estimators` | 500 | Number of boosting rounds |
| `train.max_depth` | 6 | Maximum tree depth |
| `train.learning_rate` | 0.1 | Boosting learning rate |
| `train.subsample` | 0.8 | Subsample ratio of training data |
| `train.colsample_bytree` | 0.8 | Subsample ratio of features |
| `data.test_size` | 0.2 | Test split ratio |

---

## 📊 Metrics and Results

Current model evaluation metrics:

| Metric | Value |
|--------|-------|
| RMSE | 3775.42 |
| MAE | 2748.91 |
| R² | 0.077 |
| RMSPE | 0.778 |

> **Note**: These metrics are on synthetic data. Real Rossmann data would yield different results. The focus of this project is the **pipeline architecture and reproducibility**, not model performance.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Framework | XGBoost | Gradient boosted regression |
| Data Processing | Pandas, NumPy, Scikit-learn | Data manipulation & splitting |
| Pipeline | DVC | Data versioning & pipeline orchestration |
| API Framework | FastAPI | REST API for model serving |
| ASGI Server | Uvicorn | Production-grade async server |
| Validation | Pydantic | Request/response data validation |
| Containerization | Docker | Deployment packaging |
| Version Control | Git + DVC | Code and data versioning |

---

## 📜 License

This project is for educational purposes as part of a machine learning engineering coursework.
