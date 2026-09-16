import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    train_params = params.get("train", {})
    
    features_dir = Path(params.get("data", {}).get("features_dir", "data/features"))
    train_data_path = features_dir / "train_features.csv"
    
    logging.info(f"Loading training data from {train_data_path}")
    df_train = pd.read_csv(train_data_path)
    
    X_train = df_train.drop(columns=["Sales"])
    y_train = df_train["Sales"]
    
    model_params = {
        "n_estimators": train_params.get("n_estimators", 500),
        "max_depth": train_params.get("max_depth", 6),
        "learning_rate": train_params.get("learning_rate", 0.1),
        "subsample": train_params.get("subsample", 0.8),
        "colsample_bytree": train_params.get("colsample_bytree", 0.8),
        "random_state": train_params.get("random_state", 42),
        "objective": "reg:squarederror"
    }
    
    logging.info(f"Training XGBRegressor with params: {model_params}")
    model = XGBRegressor(**model_params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    feature_names = list(X_train.columns)
    feature_names_path = models_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=4)
    logging.info(f"Feature names saved to {feature_names_path}")


if __name__ == "__main__":
    main()
