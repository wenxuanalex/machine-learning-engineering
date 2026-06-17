"""Batch inference script: loads the Production ChurnModel and scores a feature parquet.

Usage:
    python src/predict.py --input data/gold/feature_store.parquet --output reports/predictions.parquet

Outputs:
  - <output>                          parquet: customer_id, churn_probability, churn_prediction, predicted_at
  - <output_dir>/churn_scores_YYYY-MM-DD.csv  CSV: customer_id, churn_probability, risk_tier

Risk tier thresholds (tune via RISK_HIGH_THRESHOLD / RISK_MED_THRESHOLD):
  High  churn_probability >= 0.65  — prioritise for retention outreach
  Med   churn_probability >= 0.35  — monitor, consider light-touch engagement
  Low   churn_probability <  0.35  — no immediate action required
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow.sklearn
import mlflow.tracking
import pandas as pd

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "ChurnModel"
MODEL_STAGE = "Production"

# Risk tier thresholds — edit these two constants to retune the alert boundaries.
RISK_HIGH_THRESHOLD = 0.65
RISK_MED_THRESHOLD = 0.35

NON_FEATURE_COLS = {
    "customer_id",
    "is_churn_label",
    "is_active_label",
    "scoring_date",
    "macro_lag_months",
}


def _load_optimal_threshold(client: mlflow.tracking.MlflowClient) -> float:
    """Retrieve the optimal_threshold param logged during training for the Production model."""
    prod_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE).version
    run_id = client.get_model_version(MODEL_NAME, prod_version).run_id
    run = client.get_run(run_id)
    return float(run.data.params.get("optimal_threshold", 0.5))


def _assign_risk_tier(proba: pd.Series) -> pd.Series:
    return pd.cut(
        proba,
        bins=[-float("inf"), RISK_MED_THRESHOLD, RISK_HIGH_THRESHOLD, float("inf")],
        labels=["Low", "Med", "High"],
    ).astype(str)


def batch_predict(input_path: str, output_path: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_STAGE}")
    threshold = _load_optimal_threshold(client)

    df = pd.read_parquet(input_path)
    customer_ids = df["customer_id"]
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and str(df[c].dtype) != "datetime64[ms]"
    ]
    features = df[feature_cols]

    proba = model.predict_proba(features)[:, 1]
    out = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": proba,
        "churn_prediction": (proba >= threshold).astype(int),
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    })

    out["risk_tier"] = _assign_risk_tier(out["churn_probability"])

    out.to_parquet(output_path, index=False)
    print(f"Predictions written to {output_path} ({len(out)} rows, threshold={threshold:.4f})")

    csv_name = f"churn_scores_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.csv"
    csv_path = Path(output_path).parent / csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out[["customer_id", "churn_probability", "risk_tier"]].to_csv(csv_path, index=False)
    print(f"Churn scores written to {csv_path}")
    print(out["risk_tier"].value_counts().to_string())

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/gold/feature_store.parquet")
    parser.add_argument("--output", default="data/gold/predictions.parquet")
    args = parser.parse_args()
    batch_predict(args.input, args.output)
