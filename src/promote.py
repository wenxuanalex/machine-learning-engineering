"""Model promotion script: promotes ChurnModel from Staging to Production if it beats the incumbent.

Loads both Staging and Production versions of ChurnModel, evaluates each on the
held-out test set, and promotes Staging only if it improves AUC-ROC by at least
MIN_DELTA. The promotion decision and delta metrics are logged to MLflow.

Usage:
    python src/promote.py
"""

from __future__ import annotations

import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from sklearn.metrics import roc_auc_score

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "ChurnModel"
TARGET = "is_churn_label"
MIN_DELTA = 0.01

NON_FEATURE_COLS = {
    "customer_id",
    "is_churn_label",
    "is_active_label",
    "scoring_date",
    "macro_lag_months",
}


def _evaluate(model: mlflow.pyfunc.PyFuncModel, df: pd.DataFrame) -> float:
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and str(df[c].dtype) != "datetime64[ms]"
    ]
    return roc_auc_score(df[TARGET], model.predict(df[feature_cols]))


def promote_if_better(test_path: str = "data/gold/test_labeled.parquet") -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("churn_prediction")
    client = mlflow.tracking.MlflowClient()
    test_df = pd.read_parquet(test_path)

    def _get_alias_version(alias: str) -> str | None:
        try:
            return client.get_model_version_by_alias(MODEL_NAME, alias).version
        except mlflow.exceptions.MlflowException:
            return None

    staging_version = _get_alias_version("Staging")
    production_version = _get_alias_version("Production")

    if not staging_version:
        print("No Staging model found. Nothing to promote.")
        return

    staging_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Staging")
    staging_auc = _evaluate(staging_model, test_df)

    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"promotion_check_{ts}"):
        mlflow.log_metric("staging_auc_roc", staging_auc)
        mlflow.log_param("staging_version", staging_version)

        if production_version:
            prod_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Production")
            prod_auc = _evaluate(prod_model, test_df)
            delta = staging_auc - prod_auc

            mlflow.log_metric("production_auc_roc", prod_auc)
            mlflow.log_metric("auc_delta", delta)
            mlflow.log_param("production_version", production_version)

            print(
                f"Staging  AUC={staging_auc:.4f} (v{staging_version})\n"
                f"Production AUC={prod_auc:.4f} (v{production_version})\n"
                f"Delta: {delta:+.4f} (threshold={MIN_DELTA})"
            )

            if delta >= MIN_DELTA:
                client.set_registered_model_alias(MODEL_NAME, "Production", staging_version)
                mlflow.log_param("promoted", True)
                print(f"Promoted ChurnModel v{staging_version} to Production.")
            else:
                mlflow.log_param("promoted", False)
                print(f"Staging does not beat Production by {MIN_DELTA * 100:.0f}%. Promotion skipped.")
        else:
            # No incumbent — promote unconditionally
            client.set_registered_model_alias(MODEL_NAME, "Production", staging_version)
            mlflow.log_param("promoted", True)
            mlflow.log_param("reason", "no_incumbent")
            print(f"No existing Production model. Promoted ChurnModel v{staging_version} to Production.")


if __name__ == "__main__":
    promote_if_better()
