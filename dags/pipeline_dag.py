"""Airflow DAG: Bronze -> Silver -> Gold churn data pipeline.

Wraps the existing ``utils/`` functions (the same ones ``main.py`` calls) into a
scheduled, retryable DAG enforcing ``ingest_bronze >> clean_silver >> build_gold``
execution order.

Silver tasks are split into four independent operators so that macro and date-dim
cleaning run in parallel with transactions cleaning, while CRM cleaning (which
validates against the silver transactions table) is correctly sequenced after it.

A data-quality gate runs after all silver tasks and before the gold layer; it
raises ``AirflowFailException`` on row-count or null violations so the gold build
never silently consumes bad upstream data.

The task callables change the working directory to the project root so the
``utils`` functions resolve their relative ``data/...`` paths the same way they
do when ``main.py`` is run from the repo root. This is safe because the DAG runs
under the SequentialExecutor (one task at a time).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator

from utils.bronze import ingest
from utils.gold import build_gold_feature_store, build_gold_train_test_split
from utils.silver import (
    clean_customer_metadata,
    clean_macro_monthly,
    clean_transactions,
    date_dim,
)

log = logging.getLogger(__name__)

# /opt/airflow/dags/pipeline_dag.py -> project root is the parent of dags/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    os.chdir(PROJECT_ROOT)
    return build_gold_feature_store()


def gold_train_test_split() -> dict:
    os.chdir(PROJECT_ROOT)
    return build_gold_train_test_split()


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

    # Dependency graph:
    #
    #                      ┌─ t_silver_tx ─┐
    #                      │               └─ t_silver_crm ─┐
    # t_bronze ────────────┤                                 ├─ t_quality_gate ─ t_gold_features ─ t_gold_splits
    #                      ├─ t_silver_macro ───────────────┤
    #                      └─ t_silver_date ────────────────┘
    #
    t_bronze >> [t_silver_tx, t_silver_macro, t_silver_date]
    t_silver_tx >> t_silver_crm
    [t_silver_crm, t_silver_macro, t_silver_date] >> t_quality_gate
    t_quality_gate >> t_gold_features >> t_gold_splits
