from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)
from api.model_loader import load_model, get_model, get_feature_names, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up API, loading models...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down API...")

app = FastAPI(
    title="Rossmann Sales Prediction API",
    description="API for serving XGBoost model predicting Rossmann Store Sales",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    model = get_model()
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    model = get_model()
    features = get_feature_names()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    return ModelInfoResponse(
        model_type=type(model).__name__,
        n_features=len(features),
        feature_names=features,
        metrics={"info": "Metrics not yet available in this version"}
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_single(request: PredictionRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    try:
        features_df = prepare_features(request)
        prediction = model.predict(features_df)[0]
        
        return PredictionResponse(
            store_id=request.Store,
            predicted_sales=float(prediction),
            model_version="1.0"
        )
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    try:
        predictions = []
        for item in request.items:
            features_df = prepare_features(item)
            pred = model.predict(features_df)[0]
            predictions.append(
                PredictionResponse(
                    store_id=item.Store,
                    predicted_sales=float(pred),
                    model_version="1.0"
                )
            )
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
