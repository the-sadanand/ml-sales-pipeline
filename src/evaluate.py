import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    
    features_dir = Path(params.get("data", {}).get("features_dir", "data/features"))
    test_data_path = features_dir / "test_features.csv"
    
    metrics_dir_str = params.get("evaluate", {}).get("metrics_dir", "metrics")
    metrics_dir = Path(metrics_dir_str)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loading test data from {test_data_path}")
    df_test = pd.read_csv(test_data_path)
    
    X_test = df_test.drop(columns=["Sales"])
    y_test = df_test["Sales"]
    
    model_path = Path("models") / "model.joblib"
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logging.info("Generating predictions...")
    y_pred = model.predict(X_test)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    rmspe = float(calculate_rmspe(y_test.values, y_pred))
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmspe": rmspe
    }
    
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")
    
    logging.info(f"Metrics: {metrics}")
    
    # Feature importances
    feature_names = X_test.columns
    importances = model.feature_importances_
    
    feat_imp = []
    for name, imp in zip(feature_names, importances):
        feat_imp.append({"feature": name, "importance": float(imp)})
    
    # Sort by importance descending
    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)
    top_20 = feat_imp[:20]
    
    feature_importance_path = metrics_dir / "feature_importance.json"
    with open(feature_importance_path, "w") as f:
        json.dump(top_20, f, indent=4)
    logging.info(f"Top 20 feature importances saved to {feature_importance_path}")


if __name__ == "__main__":
    main()
