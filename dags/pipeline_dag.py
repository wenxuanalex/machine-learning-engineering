"""Airflow DAG: Bronze -> Silver -> Gold -> Train -> Promote (monthly CI/CD pipeline).

Full pipeline dependency graph:

  ingest_bronze
      ├─ silver_transactions ─ silver_customer_metadata ─┐
      ├─ silver_macro ────────────────────────────────────┼─ silver_quality_gate
      └─ silver_date_dim ─────────────────────────────────┘
                                                           │
                                              gold_feature_store
                                                           │
                                              gold_train_test_split
                                                           │
                                                      train_model
                                                           │
                                                     promote_model

Separate weekly DAG (churn_weekly_inference in weekly_inference_dag.py):
  run_predict -> run_monitor  (batch scoring + PSI drift summary)

The monthly pipeline retrains and gates promotion on AUC improvement.
The weekly pipeline scores the latest Production model and monitors for drift.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from utils.bronze import ingest
from utils.gold import (
    TRAINING_SNAPSHOT,
    build_gold_snapshots,
    build_gold_train_test_split,
    snapshot_path,
)
from utils.silver import (
    clean_customer_metadata,
    clean_macro_monthly,
    clean_transactions,
    date_dim,
)

log = logging.getLogger(__name__)

# /opt/airflow/dags/pipeline_dag.py -> project root is the parent of dags/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# (raw csv, bronze parquet) pairs — mirrors BRONZE_SOURCES in main.py
BRONZE_SOURCES = [
    ("data/data.csv", "data/bronze/transactions.parquet"),
    ("data/bronze_customer_metadata_synthetic.csv", "data/bronze/customer_metadata.parquet"),
    ("data/ancillary_20101201_to_20111231.csv", "data/bronze/ancillary.parquet"),
]

# Quality thresholds for the silver data-quality gate
SILVER_TX_MIN_ROWS = 300_000
SILVER_NULL_MAX_PCT = 5.0


# ---------------------------------------------------------------------------
# Failure callback
# ---------------------------------------------------------------------------

def _on_failure(context: dict) -> None:
    """Log task failure details to the Airflow task log."""
    dag_id = context["dag"].dag_id
    task_id = context["task_instance"].task_id
    execution_date = context["execution_date"]
    exception = context.get("exception", "unknown")
    log.error(
        "Task failed | dag=%s | task=%s | execution_date=%s | exception=%s",
        dag_id,
        task_id,
        execution_date,
        exception,
    )


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

def ingest_bronze() -> dict:
    """Ingest every raw CSV source into the bronze layer."""
    os.chdir(PROJECT_ROOT)
    results = {}
    for src, dest in BRONZE_SOURCES:
        result = ingest(src, dest)
        results[dest] = result["rows"]
    return results


# ---------------------------------------------------------------------------
# Silver (four separate callables so the DAG can express parallelism)
# ---------------------------------------------------------------------------

def silver_transactions() -> dict:
    os.chdir(PROJECT_ROOT)
    return clean_transactions()


def silver_customer_metadata() -> dict:
    # Depends on silver_transactions having written data/silver/transactions.parquet
    os.chdir(PROJECT_ROOT)
    return clean_customer_metadata()


def silver_macro() -> dict:
    os.chdir(PROJECT_ROOT)
    return clean_macro_monthly()


def silver_date_dim() -> dict:
    os.chdir(PROJECT_ROOT)
    return date_dim()


# ---------------------------------------------------------------------------
# Data-quality gate
# ---------------------------------------------------------------------------

def silver_quality_gate() -> None:
    """Validate silver layer before building gold.

    Checks:
    - Silver transactions row count >= SILVER_TX_MIN_ROWS
    - Null percentage in silver transactions <= SILVER_NULL_MAX_PCT
    - Silver customer_metadata, macro_monthly, and date_dim are non-empty
    """
    os.chdir(PROJECT_ROOT)

    tx = pd.read_parquet("data/silver/transactions.parquet")
    tx_rows = len(tx)
    if tx_rows < SILVER_TX_MIN_ROWS:
        raise AirflowFailException(
            f"Silver transactions has {tx_rows:,} rows — below minimum {SILVER_TX_MIN_ROWS:,}. "
            "Upstream cleaning may have dropped too many rows."
        )

    null_pct = tx.isnull().mean().mean() * 100
    if null_pct > SILVER_NULL_MAX_PCT:
        raise AirflowFailException(
            f"Silver transactions null rate is {null_pct:.2f}% — exceeds {SILVER_NULL_MAX_PCT}%."
        )

    for label, path in [
        ("customer_metadata", "data/silver/customer_metadata.parquet"),
        ("macro_monthly", "data/silver/macro_monthly.parquet"),
        ("date_dim", "data/silver/silver_date_dim.parquet"),
    ]:
        rows = len(pd.read_parquet(path))
        if rows == 0:
            raise AirflowFailException(f"Silver {label} is empty after cleaning.")
        log.info("Quality gate: silver/%s OK (%d rows)", label, rows)

    log.info(
        "Quality gate passed: silver/transactions %d rows, null_pct=%.2f%%",
        tx_rows,
        null_pct,
    )


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

def gold_feature_store() -> dict:
    # Build every point-in-time snapshot (training baseline + later scoring batch).
    os.chdir(PROJECT_ROOT)
    return build_gold_snapshots()


def gold_train_test_split() -> dict:
    # Model dev splits derive from the training snapshot only.
    os.chdir(PROJECT_ROOT)
    return build_gold_train_test_split(feature_store_path=snapshot_path(TRAINING_SNAPSHOT))


# ---------------------------------------------------------------------------
# ML — train and promote
# ---------------------------------------------------------------------------

def _ml_env() -> dict:
    """Environment passed to ML BashOperators: sets MLflow URI and PYTHONPATH."""
    return {"MLFLOW_TRACKING_URI": MLFLOW_URI, "PYTHONPATH": str(PROJECT_ROOT)}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "on_failure_callback": _on_failure,
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

    # Silver tasks — transactions and (macro, date_dim) run in parallel;
    # CRM is sequenced after transactions since it validates against silver tx.
    t_silver_tx = PythonOperator(task_id="silver_transactions", python_callable=silver_transactions)
    t_silver_crm = PythonOperator(task_id="silver_customer_metadata", python_callable=silver_customer_metadata)
    t_silver_macro = PythonOperator(task_id="silver_macro", python_callable=silver_macro)
    t_silver_date = PythonOperator(task_id="silver_date_dim", python_callable=silver_date_dim)

    t_quality_gate = PythonOperator(task_id="silver_quality_gate", python_callable=silver_quality_gate)

    # Gold tasks split into feature engineering and train/val/test splitting
    t_gold_features = PythonOperator(task_id="gold_feature_store", python_callable=gold_feature_store)
    t_gold_splits = PythonOperator(task_id="gold_train_test_split", python_callable=gold_train_test_split)

    # ML tasks — train then gate promotion on AUC improvement
    t_train = BashOperator(
        task_id="train_model",
        bash_command="python src/train.py",
        env=_ml_env(),
        cwd=str(PROJECT_ROOT),
    )
    t_promote = BashOperator(
        task_id="promote_model",
        bash_command="python src/promote.py",
        env=_ml_env(),
        cwd=str(PROJECT_ROOT),
    )

    # Dependency graph:
    #
    #                      ┌─ t_silver_tx ─┐
    #                      │               └─ t_silver_crm ─┐
    # t_bronze ────────────┤                                 ├─ t_quality_gate ─ t_gold_features ─ t_gold_splits ─ t_train ─ t_promote
    #                      ├─ t_silver_macro ───────────────┤
    #                      └─ t_silver_date ────────────────┘
    #
    t_bronze >> [t_silver_tx, t_silver_macro, t_silver_date]
    t_silver_tx >> t_silver_crm
    [t_silver_crm, t_silver_macro, t_silver_date] >> t_quality_gate
    t_quality_gate >> t_gold_features >> t_gold_splits >> t_train >> t_promote
