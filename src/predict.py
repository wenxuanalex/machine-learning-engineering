"""Batch inference script: loads the Production ChurnModel and scores a feature parquet.

Usage:
    python src/predict.py --input data/gold/feature_store.parquet --output data/gold/predictions.parquet

Output schema: customer_id, churn_probability, churn_prediction, predicted_at
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import mlflow.pyfunc
import pandas as pd

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "ChurnModel"
MODEL_STAGE = "Production"

NON_FEATURE_COLS = {
    "customer_id",
    "is_churn_label",
    "is_active_label",
    "scoring_date",
    "macro_lag_months",
}


def batch_predict(input_path: str, output_path: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_URI)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_STAGE}")

    df = pd.read_parquet(input_path)
    customer_ids = df["customer_id"]
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and str(df[c].dtype) != "datetime64[ms]"
    ]
    features = df[feature_cols]

    proba = model.predict(features)
    out = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": proba,
        "churn_prediction": (proba >= 0.5).astype(int),
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    })

    out.to_parquet(output_path, index=False)
    print(f"Predictions written to {output_path} ({len(out)} rows)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/gold/feature_store.parquet")
    parser.add_argument("--output", default="data/gold/predictions.parquet")
    args = parser.parse_args()
    batch_predict(args.input, args.output)
