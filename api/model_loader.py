import joblib
import json
import logging
import os
import pandas as pd
from typing import Any, List, Optional
from datetime import date
from api.schemas import PredictionRequest

logger = logging.getLogger(__name__)

_model: Optional[Any] = None
_feature_names: Optional[List[str]] = None

MODEL_PATH = os.path.join("models", "model.joblib")
FEATURES_PATH = os.path.join("models", "feature_names.json")

def load_model() -> None:
    global _model, _feature_names
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")

        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, 'r') as f:
                _feature_names = json.load(f)
            logger.info("Feature names loaded successfully.")
        else:
            logger.warning(f"Feature names file not found at {FEATURES_PATH}")
            
    except Exception as e:
        logger.error(f"Error loading model or feature names: {e}")
        raise

def get_model() -> Any:
    return _model

def get_feature_names() -> List[str]:
    return _feature_names or []

def prepare_features(request: PredictionRequest) -> pd.DataFrame:
    # Single-row DataFrame
    df = pd.DataFrame([request.model_dump()])
    
    # StoreType one-hot
    store_types = ['a', 'b', 'c', 'd']
    for st in store_types:
        df[f'StoreType_{st}'] = (df['StoreType'].iloc[0] == st)
        df[f'StoreType_{st}'] = df[f'StoreType_{st}'].astype(int)
        
    # Assortment one-hot
    assortments = ['a', 'b', 'c']
    for a in assortments:
        df[f'Assortment_{a}'] = (df['Assortment'].iloc[0] == a)
        df[f'Assortment_{a}'] = df[f'Assortment_{a}'].astype(int)
        
    # Map StateHoliday
    state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}
    df['StateHoliday'] = df['StateHoliday'].map(state_holiday_map)
    
    # Derived features
    # WeekOfYear
    try:
        dt = date(request.Year, request.Month, request.Day)
        df['WeekOfYear'] = dt.isocalendar()[1]
    except Exception:
        df['WeekOfYear'] = 1
        
    df['CompetitionOpen'] = 0
    df['Promo2Open'] = 0
    df['IsPromo2Active'] = 0
    df['Promo2'] = 0
    
    # Align columns
    features = get_feature_names()
    if not features:
        return df
        
    for f in features:
        if f not in df.columns:
            df[f] = 0
            
    df = df[features]
    return df
