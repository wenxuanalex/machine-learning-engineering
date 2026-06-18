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

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import — required in headless/Docker environments
import matplotlib.pyplot as plt
import mlflow.sklearn
import mlflow.tracking
import numpy as np
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
    "auxiliary_lag_months",
    "auxiliary_snapshot_month",
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


def _log_shap_attribution(model, features: pd.DataFrame, output_dir: str) -> None:
    """Compute mean |SHAP| values and save a bar chart for feature attribution tracking.

    Called on every weekly inference run so attribution shifts are visible over time.
    Supports GradientBoosting, RandomForest (TreeExplainer) and LogisticRegression
    (LinearExplainer). Fails silently so it never blocks the inference pipeline.
    """
    try:
        import shap
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        clf = model.named_steps["clf"]
        preprocessor = model.named_steps["preprocessor"]

        sample = features.sample(min(200, len(features)), random_state=42)
        X_transformed = preprocessor.transform(sample)

        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except AttributeError:
            feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        if isinstance(clf, (GradientBoostingClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(clf)
            raw = explainer.shap_values(X_transformed)
            # RandomForest returns [neg_class, pos_class]; GBM returns 2-D array directly
            shap_vals = raw[1] if isinstance(raw, list) else raw
        elif isinstance(clf, LogisticRegression):
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_vals = explainer.shap_values(X_transformed)
        else:
            print(f"SHAP: unsupported classifier type {type(clf).__name__}, skipping.")
            return

        mean_abs = pd.Series(
            np.abs(shap_vals).mean(axis=0), index=feature_names
        ).sort_values(ascending=True).tail(20)

        fig, ax = plt.subplots(figsize=(8, max(4, len(mean_abs) * 0.35)))
        mean_abs.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Attribution — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        plt.tight_layout()

        out_path = Path(output_dir) / f"shap_attribution_{datetime.now(timezone.utc).strftime('%Y%m%d')}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"SHAP attribution saved: {out_path}")
        print(f"Top 5 features: {mean_abs.tail(5).index[::-1].tolist()}")
    except Exception as exc:
        print(f"SHAP attribution skipped: {exc}")


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

    _log_shap_attribution(model, features, str(Path(output_path).parent))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/gold/feature_store.parquet")
    parser.add_argument("--output", default="data/gold/predictions.parquet")
    args = parser.parse_args()
    batch_predict(args.input, args.output)
