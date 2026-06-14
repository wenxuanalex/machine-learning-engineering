"""Feature distribution drift monitor using Evidently 0.7.x.

Compares the training set (reference) against the validation set (current)
to detect distribution shifts in model input features. Results are saved
as HTML and JSON to reports/ and key metrics are logged to MLflow.

Evidently 0.7.21 API (NOT 0.4.x):
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset

Usage:
    python src/monitor.py
    python src/monitor.py --reference data/gold/train_labeled.parquet \
                          --current   data/gold/val_labeled.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path("reports")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT = "churn_monitoring"

NON_FEATURE_COLS = {
    "customer_id", "is_churn_label", "is_active_label",
    "scoring_date", "macro_lag_months",
}

# Drift threshold: flag run if this share of features drifts
DRIFT_SHARE_THRESHOLD = 0.20


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and str(df[c].dtype) != "datetime64[ms]"
        and str(df[c].dtype) != "object"
    ]


def run_drift_report(
    reference_path: str = "data/gold/train_labeled.parquet",
    current_path: str = "data/gold/val_labeled.parquet",
) -> dict:
    from evidently import Dataset, DataDefinition, Report
    from evidently.presets import DataDriftPreset

    ref_df = pd.read_parquet(reference_path)
    cur_df = pd.read_parquet(current_path)

    feature_cols = _get_feature_cols(ref_df)
    print(f"  Checking drift on {len(feature_cols)} numeric features "
          f"({len(ref_df):,} reference rows | {len(cur_df):,} current rows)")

    ref_features = ref_df[feature_cols].copy()
    cur_features = cur_df[feature_cols].copy()

    dd = DataDefinition()
    ref_ds = Dataset.from_pandas(ref_features, data_definition=dd)
    cur_ds = Dataset.from_pandas(cur_features, data_definition=dd)

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    json_path = REPORTS_DIR / f"drift_report_{ts}.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    raw = json.loads(json_path.read_text(encoding="utf-8"))

    # Parse DriftedColumnsCount from the metrics list
    drifted_count = None
    drifted_share = None
    for metric in raw.get("metrics", []):
        name = metric.get("metric_name", "")
        if "DriftedColumnsCount" in name:
            val = metric.get("value", {})
            drifted_count = val.get("count")
            drifted_share = val.get("share")
            break

    result = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "reference_path": reference_path,
        "current_path": current_path,
        "n_features_checked": len(feature_cols),
        "drifted_columns_count": drifted_count,
        "drifted_columns_share": drifted_share,
        "drift_threshold": DRIFT_SHARE_THRESHOLD,
        "drift_detected": (drifted_share or 0) > DRIFT_SHARE_THRESHOLD,
        "html_report": str(html_path),
        "json_report": str(json_path),
    }
    return result


def log_to_mlflow(result: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="feature_drift"):
        mlflow.log_metric("drifted_columns_count", result["drifted_columns_count"] or 0)
        mlflow.log_metric("drifted_columns_share", result["drifted_columns_share"] or 0.0)
        mlflow.log_metric("drift_detected", int(result["drift_detected"]))
        mlflow.log_metric("n_features_checked", result["n_features_checked"])
        mlflow.log_artifact(result["html_report"])
        mlflow.log_artifact(result["json_report"])
        print(f"  MLflow run logged to experiment '{EXPERIMENT}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature distribution drift monitor")
    parser.add_argument("--reference", default="data/gold/train_labeled.parquet")
    parser.add_argument("--current", default="data/gold/val_labeled.parquet")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    print("Running feature drift report...")
    result = run_drift_report(args.reference, args.current)

    drift_symbol = "!!" if result["drift_detected"] else "OK"
    print(f"  [{drift_symbol}] Drifted features: {result['drifted_columns_count']} "
          f"/ {result['n_features_checked']} "
          f"({(result['drifted_columns_share'] or 0):.1%})")
    print(f"  HTML: {result['html_report']}")
    print(f"  JSON: {result['json_report']}")

    if not args.no_mlflow:
        try:
            log_to_mlflow(result)
        except Exception as exc:
            print(f"  MLflow logging skipped: {exc}")

    if result["drift_detected"]:
        print(f"\nWARNING: drift share {(result['drifted_columns_share'] or 0):.1%} "
              f"exceeds threshold {DRIFT_SHARE_THRESHOLD:.0%}. "
              "Consider retraining the model.")


if __name__ == "__main__":
    main()
