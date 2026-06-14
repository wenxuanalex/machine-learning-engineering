"""Airflow DAG: Bronze -> Silver -> Gold churn data pipeline.

Wraps the existing ``utils/`` functions (the same ones ``main.py`` calls) into a
scheduled, retryable DAG with three layered tasks that enforce
``ingest_bronze >> clean_silver >> build_gold`` execution order.

The task callables change the working directory to the project root so the
``utils`` functions resolve their relative ``data/...`` paths the same way they
do when ``main.py`` is run from the repo root. This is safe because the DAG runs
under the SequentialExecutor (one task at a time).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from utils.bronze import ingest
from utils.gold import build_gold_feature_store, build_gold_train_test_split
from utils.silver import (
    clean_customer_metadata,
    clean_macro_monthly,
    clean_transactions,
    date_dim,
)

# /opt/airflow/dags/pipeline_dag.py -> project root is the parent of dags/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (raw csv, bronze parquet) pairs — mirrors BRONZE_SOURCES in main.py
BRONZE_SOURCES = [
    ("data/data.csv", "data/bronze/transactions.parquet"),
    ("data/bronze_customer_metadata_synthetic.csv", "data/bronze/customer_metadata.parquet"),
    ("data/ancillary_20101201_to_20111231.csv", "data/bronze/ancillary.parquet"),
]


def ingest_bronze() -> dict:
    """Ingest every raw CSV source into the bronze layer."""
    os.chdir(PROJECT_ROOT)
    results = {}
    for src, dest in BRONZE_SOURCES:
        result = ingest(src, dest)
        results[dest] = result["rows"]
    return results


def clean_silver() -> dict:
    """Build all silver tables.

    Order matters: ``clean_customer_metadata`` validates against the silver
    transactions table, so transactions must be written first.
    """
    os.chdir(PROJECT_ROOT)
    tx = clean_transactions()
    crm = clean_customer_metadata()
    macro = clean_macro_monthly()
    cal = date_dim()
    return {
        "transactions_rows": tx["rows_out"],
        "customer_metadata_rows": crm["rows"],
        "macro_monthly_rows": macro["rows"],
        "date_dim_rows": cal["rows"],
    }


def build_gold() -> dict:
    """Build the gold feature store, then derive the train/val/test splits."""
    os.chdir(PROJECT_ROOT)
    fs = build_gold_feature_store()
    split = build_gold_train_test_split()
    return {
        "feature_store_rows": fs["rows"],
        "train_rows": split["train_rows"],
        "val_rows": split["val_rows"],
        "test_rows": split["test_rows"],
    }


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="churn_data_pipeline",
    default_args=default_args,
    description="Bronze -> Silver -> Gold data pipeline for churn prediction",
    schedule="@monthly",
    start_date=datetime(2011, 1, 1),
    catchup=False,
    tags=["data-pipeline", "churn"],
) as dag:
    t_bronze = PythonOperator(task_id="ingest_bronze", python_callable=ingest_bronze)
    t_silver = PythonOperator(task_id="clean_silver", python_callable=clean_silver)
    t_gold = PythonOperator(task_id="build_gold", python_callable=build_gold)

    t_bronze >> t_silver >> t_gold
