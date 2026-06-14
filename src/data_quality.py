"""Validate Silver and Gold parquet files and emit a quality JSON report.

Checks performed:
  - Silver transactions: row count >= 300,000, no nulls in key columns
  - Silver customer_metadata, macro_monthly, silver_date_dim: non-empty, no critical nulls
  - Gold feature_store: no nulls in numeric feature cols, churn rate in expected range

Usage:
    python src/data_quality.py
    python src/data_quality.py --fail-fast   # exit 1 on first failed check
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPORTS_DIR = Path("reports")

# Minimum row thresholds
MIN_SILVER_TX_ROWS = 300_000
MIN_MACRO_ROWS = 10
MIN_DATE_DIM_ROWS = 100
MIN_CRM_ROWS = 1

# Key columns that must never be null in each table
SILVER_TX_NON_NULL = ["customer_id", "invoice_date", "revenue", "invoice_no"]
SILVER_CRM_NON_NULL = ["customer_id"]
MACRO_NON_NULL = ["year_month"]
GOLD_NON_NULL = ["customer_id", "is_churn_label", "recency", "frequency", "monetary"]

# Churn rate expected range for the 2010-2011 dataset
CHURN_RATE_MIN = 0.30
CHURN_RATE_MAX = 0.55


def _check(results: list[dict], name: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    results.append({"check": name, "status": status, "detail": detail})
    return passed


def validate_silver_transactions(path: str = "data/silver/transactions.parquet") -> list[dict]:
    results: list[dict] = []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        _check(results, "silver_tx.readable", False, str(exc))
        return results

    _check(results, "silver_tx.row_count",
           len(df) >= MIN_SILVER_TX_ROWS,
           f"{len(df):,} rows (min {MIN_SILVER_TX_ROWS:,})")

    for col in SILVER_TX_NON_NULL:
        null_ct = int(df[col].isna().sum()) if col in df.columns else -1
        exists = col in df.columns
        _check(results, f"silver_tx.no_nulls.{col}",
               exists and null_ct == 0,
               f"{null_ct} nulls" if exists else "column missing")

    neg_rev = int((df["revenue"] < 0).sum()) if "revenue" in df.columns else -1
    _check(results, "silver_tx.revenue_positive",
           "revenue" in df.columns and neg_rev == 0,
           f"{neg_rev} rows with negative revenue")

    return results


def validate_silver_customer_metadata(path: str = "data/silver/customer_metadata.parquet") -> list[dict]:
    results: list[dict] = []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        _check(results, "silver_crm.readable", False, str(exc))
        return results

    _check(results, "silver_crm.non_empty", len(df) >= MIN_CRM_ROWS, f"{len(df):,} rows")

    for col in SILVER_CRM_NON_NULL:
        null_ct = int(df[col].isna().sum()) if col in df.columns else -1
        exists = col in df.columns
        _check(results, f"silver_crm.no_nulls.{col}",
               exists and null_ct == 0,
               f"{null_ct} nulls" if exists else "column missing")

    dup_ct = int(df["customer_id"].duplicated().sum()) if "customer_id" in df.columns else -1
    _check(results, "silver_crm.unique_customer_id",
           "customer_id" in df.columns and dup_ct == 0,
           f"{dup_ct} duplicate customer_id rows")

    return results


def validate_silver_macro_monthly(path: str = "data/silver/macro_monthly.parquet") -> list[dict]:
    results: list[dict] = []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        _check(results, "silver_macro.readable", False, str(exc))
        return results

    _check(results, "silver_macro.row_count",
           len(df) >= MIN_MACRO_ROWS,
           f"{len(df)} rows (min {MIN_MACRO_ROWS})")

    for col in MACRO_NON_NULL:
        null_ct = int(df[col].isna().sum()) if col in df.columns else -1
        exists = col in df.columns
        _check(results, f"silver_macro.no_nulls.{col}",
               exists and null_ct == 0,
               f"{null_ct} nulls" if exists else "column missing")

    if "year_month" in df.columns:
        months = sorted(df["year_month"].astype(str).tolist())
        _check(results, "silver_macro.has_dec2010",
               "2010-12" in months,
               f"earliest month: {months[0] if months else 'none'}")
        _check(results, "silver_macro.has_dec2011",
               "2011-12" in months,
               f"latest month: {months[-1] if months else 'none'}")

    return results


def validate_silver_date_dim(path: str = "data/silver/silver_date_dim.parquet") -> list[dict]:
    results: list[dict] = []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        _check(results, "silver_date_dim.readable", False, str(exc))
        return results

    _check(results, "silver_date_dim.row_count",
           len(df) >= MIN_DATE_DIM_ROWS,
           f"{len(df)} rows")

    _check(results, "silver_date_dim.has_Date",
           "Date" in df.columns,
           "Date column present" if "Date" in df.columns else "Date column missing")

    return results


def validate_gold_feature_store(path: str = "data/gold/feature_store.parquet") -> list[dict]:
    results: list[dict] = []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        _check(results, "gold_features.readable", False, str(exc))
        return results

    _check(results, "gold_features.non_empty", len(df) > 0, f"{len(df):,} rows")

    for col in GOLD_NON_NULL:
        null_ct = int(df[col].isna().sum()) if col in df.columns else -1
        exists = col in df.columns
        _check(results, f"gold_features.no_nulls.{col}",
               exists and null_ct == 0,
               f"{null_ct} nulls" if exists else "column missing")

    if "is_churn_label" in df.columns:
        churn_rate = float(df["is_churn_label"].mean())
        _check(results, "gold_features.churn_rate_in_range",
               CHURN_RATE_MIN <= churn_rate <= CHURN_RATE_MAX,
               f"churn_rate={churn_rate:.3%} (expected {CHURN_RATE_MIN:.0%}–{CHURN_RATE_MAX:.0%})")

    total_nulls = int(df.isnull().sum().sum())
    _check(results, "gold_features.zero_total_nulls",
           total_nulls == 0,
           f"{total_nulls} total nulls across all columns")

    return results


def run_all(fail_fast: bool = False) -> dict:
    all_results: dict[str, list[dict]] = {}

    checks = [
        ("silver_transactions", validate_silver_transactions),
        ("silver_customer_metadata", validate_silver_customer_metadata),
        ("silver_macro_monthly", validate_silver_macro_monthly),
        ("silver_date_dim", validate_silver_date_dim),
        ("gold_feature_store", validate_gold_feature_store),
    ]

    any_failure = False
    for table, fn in checks:
        results = fn()
        all_results[table] = results
        for r in results:
            if r["status"] == "FAIL":
                any_failure = True
                print(f"  FAIL  [{table}] {r['check']}: {r['detail']}")
                if fail_fast:
                    break
            else:
                print(f"  PASS  [{table}] {r['check']}: {r['detail']}")
        if fail_fast and any_failure:
            break

    pass_ct = sum(r["status"] == "PASS" for v in all_results.values() for r in v)
    fail_ct = sum(r["status"] == "FAIL" for v in all_results.values() for r in v)

    report = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "summary": {"passed": pass_ct, "failed": fail_ct, "overall": "PASS" if fail_ct == 0 else "FAIL"},
        "tables": all_results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Data quality checks for Silver and Gold layers")
    parser.add_argument("--fail-fast", action="store_true", help="Exit 1 on first failed check")
    parser.add_argument("--output", default=None, help="Override output JSON path")
    args = parser.parse_args()

    print("Running data quality checks...")
    report = run_all(fail_fast=args.fail_fast)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output) if args.output else REPORTS_DIR / f"data_quality_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print(f"\n{'='*50}")
    print(f"Result: {summary['overall']}  ({summary['passed']} passed, {summary['failed']} failed)")
    print(f"Report saved to: {out_path}")

    if summary["overall"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
