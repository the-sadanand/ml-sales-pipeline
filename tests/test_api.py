import pytest
from fastapi.testclient import TestClient
import os

from api.main import app
from api.model_loader import load_model

client = TestClient(app)

@pytest.fixture(autouse=True)
def check_model_exists():
    model_path = os.path.join("models", "model.joblib")
    features_path = os.path.join("models", "feature_names.json")
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        pytest.skip(f"Model or features file not found, skipping tests. Model: {model_path}")
    else:
        load_model()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "timestamp" in data

def test_predict_single():
    payload = {
        "Store": 1,
        "DayOfWeek": 5,
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 1,
        "StoreType": "c",
        "Assortment": "a",
        "CompetitionDistance": 1270.0,
        "Year": 2015,
        "Month": 7,
        "Day": 31
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
    
    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == 1
    assert "predicted_sales" in data
    assert data["model_version"] == "1.0"

def test_predict_batch():
    payload = {
        "items": [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 1,
                "StoreType": "c",
                "Assortment": "a",
                "CompetitionDistance": 1270.0,
                "Year": 2015,
                "Month": 7,
                "Day": 31
            },
            {
                "Store": 2,
                "DayOfWeek": 5,
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 1,
                "StoreType": "a",
                "Assortment": "a",
                "CompetitionDistance": 570.0,
                "Year": 2015,
                "Month": 7,
                "Day": 31
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    if response.status_code == 503:
        pytest.skip("Model not loaded")
        
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["store_id"] == 1
    assert data["predictions"][1]["store_id"] == 2

def test_predict_invalid_input():
    payload = {
        "Store": 0, 
        "DayOfWeek": 8,
        "Open": 1,
        "Promo": 1,
        "StateHoliday": "d",
        "SchoolHoliday": 1,
        "StoreType": "x",
        "Assortment": "z",
        "CompetitionDistance": -10,
        "Year": 2010,
        "Month": 13,
        "Day": 32
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

def test_model_info():
    response = client.get("/model/info")
    if response.status_code == 503:
        pytest.skip("Model not loaded")
        
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "n_features" in data
    assert "feature_names" in data
