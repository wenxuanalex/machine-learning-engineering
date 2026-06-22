"""Weekly batch inference + Evidently drift monitoring DAG.

Runs every week. Two tasks in strict sequence:
  1. run_predict  — scores feature_store.parquet via src/predict.py,
                    writes data/gold/predictions.parquet
  2. run_monitor  — generates an Evidently HTML drift report via src/monitor.py,
                    writes reports/drift_report_YYYYMMDD.html

The upstream monthly data pipeline (churn_data_pipeline) is not modified.
This DAG assumes feature_store.parquet and train_labeled.parquet have already
been produced by the monthly pipeline before the weekly window fires.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def _ml_env() -> dict:
    return {"MLFLOW_TRACKING_URI": MLFLOW_URI, "PYTHONPATH": PROJECT_ROOT}


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="churn_weekly_inference",
    default_args=default_args,
    description="Weekly batch inference + Evidently drift report",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["inference", "monitoring", "churn"],
) as dag:

    run_predict = BashOperator(
        task_id="run_predict",
        bash_command=(
            "python src/predict.py "
            # Score the latest snapshot (SCORING_SNAPSHOT in utils/gold.py).
            "--input data/gold/feature_store_2011-08-31.parquet "
            "--output reports/predictions.parquet"
        ),
        env=_ml_env(),
        cwd=PROJECT_ROOT,
    )

    run_monitor = BashOperator(
        task_id="run_monitor",
        bash_command=(
            "python src/monitor.py "
            # Reference = training snapshot; current = later scoring snapshot, so
            # drift is a genuine earlier-vs-later comparison (see utils/gold.py).
            "--train data/gold/train_labeled.parquet "
            "--feature-store data/gold/feature_store_2011-08-31.parquet "
            "--predictions reports/predictions.parquet "
            "--output-dir reports"
        ),
        env=_ml_env(),
        cwd=PROJECT_ROOT,
    )

    run_predict >> run_monitor
