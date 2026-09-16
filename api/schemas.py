from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    Store: int = Field(..., ge=1, le=1115, description='Store ID')
    DayOfWeek: int = Field(..., ge=1, le=7)
    Open: int = Field(..., ge=0, le=1)
    Promo: int = Field(..., ge=0, le=1)
    StateHoliday: str = Field(default='0', description="'0','a','b','c'")
    SchoolHoliday: int = Field(..., ge=0, le=1)
    StoreType: str = Field(..., description="'a','b','c','d'")
    Assortment: str = Field(..., description="'a','b','c'")
    CompetitionDistance: float = Field(default=0.0, ge=0)
    Year: int = Field(..., ge=2013, le=2030)
    Month: int = Field(..., ge=1, le=12)
    Day: int = Field(..., ge=1, le=31)

    @field_validator('StateHoliday')
    @classmethod
    def validate_state_holiday(cls, v: str) -> str:
        v = v.lower()
        if v not in ['0', 'a', 'b', 'c']:
            raise ValueError("StateHoliday must be one of '0', 'a', 'b', 'c'")
        return v
        
    @field_validator('StoreType')
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ['a', 'b', 'c', 'd']:
            raise ValueError("StoreType must be one of 'a', 'b', 'c', 'd'")
        return v

    @field_validator('Assortment')
    @classmethod
    def validate_assortment(cls, v: str) -> str:
        v = v.lower()
        if v not in ['a', 'b', 'c']:
            raise ValueError("Assortment must be one of 'a', 'b', 'c'")
        return v
        
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

class PredictionResponse(BaseModel):
    store_id: int
    predicted_sales: float
    model_version: str = "1.0"

class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_names: List[str]
    metrics: Optional[Dict[str, Any]] = None
