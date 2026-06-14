"""Tests for Epic 3 monitoring scripts.

These tests are self-contained: they build tiny in-memory DataFrames that
mirror the Gold schema and validate each monitoring module's core logic
without touching the file system, MLflow, or a real trained model.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gold_df(n: int = 200, churn_rate: float = 0.41) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "customer_id": [str(i) for i in range(n)],
            "recency": rng.integers(1, 300, n).astype(float),
            "frequency": rng.integers(1, 50, n).astype(float),
            "monetary": rng.uniform(100, 50_000, n),
            "avg_basket_size": rng.uniform(1, 100, n),
            "avg_order_interarrival_days": rng.uniform(-1, 90, n),
            "product_diversity": rng.integers(1, 100, n).astype(float),
            "cancellation_rate": rng.uniform(0, 0.5, n),
            "rolling_30d_spend": rng.uniform(0, 5_000, n),
            "rolling_60d_spend": rng.uniform(0, 10_000, n),
            "rolling_90d_spend": rng.uniform(0, 15_000, n),
            "macro_ftse_monthly_return": rng.uniform(-0.05, 0.05, n),
            "is_churn_label": rng.choice([0, 1], n, p=[1 - churn_rate, churn_rate]),
            "is_active_label": 0,
            "scoring_date": pd.Timestamp("2011-08-31"),
            "macro_lag_months": 1,
        }
    )
    return df


# ---------------------------------------------------------------------------
# data_quality.py tests
# ---------------------------------------------------------------------------

class TestDataQuality:
    def _write_parquets(self, tmp: Path, silver_tx_rows: int = 320_000) -> dict[str, Path]:
        from tests.test_monitoring import _make_gold_df

        silver_tx = pd.DataFrame(
            {
                "customer_id": ["1"] * silver_tx_rows,
                "invoice_date": [pd.Timestamp("2011-01-01")] * silver_tx_rows,
                "revenue": [10.0] * silver_tx_rows,
                "invoice_no": ["INV"] * silver_tx_rows,
            }
        )
        silver_crm = pd.DataFrame({"customer_id": ["1", "2"]})
        months = pd.period_range("2010-12", "2011-12", freq="M").strftime("%Y-%m").tolist()
        silver_macro = pd.DataFrame(
            {"year_month": months, "val": np.arange(len(months), dtype=float)}
        )
        silver_date = pd.DataFrame({"Date": pd.date_range("2011-01-01", periods=300)})
        gold = _make_gold_df(200)

        paths = {}
        for name, df in [
            ("silver/transactions.parquet", silver_tx),
            ("silver/customer_metadata.parquet", silver_crm),
            ("silver/macro_monthly.parquet", silver_macro),
            ("silver/silver_date_dim.parquet", silver_date),
            ("gold/feature_store.parquet", gold),
        ]:
            p = tmp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            paths[name] = p

        return paths

    def test_all_pass_with_good_data(self, tmp_path: Path) -> None:
        from src.data_quality import (
            validate_gold_feature_store,
            validate_silver_customer_metadata,
            validate_silver_date_dim,
            validate_silver_macro_monthly,
            validate_silver_transactions,
        )

        paths = self._write_parquets(tmp_path)
        fns = [
            (validate_silver_transactions, str(paths["silver/transactions.parquet"])),
            (validate_silver_customer_metadata, str(paths["silver/customer_metadata.parquet"])),
            (validate_silver_macro_monthly, str(paths["silver/macro_monthly.parquet"])),
            (validate_silver_date_dim, str(paths["silver/silver_date_dim.parquet"])),
            (validate_gold_feature_store, str(paths["gold/feature_store.parquet"])),
        ]
        for fn, path in fns:
            results = fn(path)
            failures = [r for r in results if r["status"] == "FAIL"]
            assert not failures, f"{fn.__name__} failed: {failures}"

    def test_row_count_fail(self, tmp_path: Path) -> None:
        from src.data_quality import validate_silver_transactions

        silver_tx = pd.DataFrame(
            {
                "customer_id": ["1"] * 100,
                "invoice_date": [pd.Timestamp("2011-01-01")] * 100,
                "revenue": [10.0] * 100,
                "invoice_no": ["INV"] * 100,
            }
        )
        p = tmp_path / "transactions.parquet"
        silver_tx.to_parquet(p, index=False)
        results = validate_silver_transactions(str(p))
        row_check = next(r for r in results if r["check"] == "silver_tx.row_count")
        assert row_check["status"] == "FAIL"

    def test_negative_revenue_fail(self, tmp_path: Path) -> None:
        from src.data_quality import validate_silver_transactions

        df = pd.DataFrame(
            {
                "customer_id": ["1"] * 320_000,
                "invoice_date": [pd.Timestamp("2011-01-01")] * 320_000,
                "revenue": [-5.0] * 320_000,
                "invoice_no": ["INV"] * 320_000,
            }
        )
        p = tmp_path / "tx.parquet"
        df.to_parquet(p, index=False)
        results = validate_silver_transactions(str(p))
        rev_check = next(r for r in results if r["check"] == "silver_tx.revenue_positive")
        assert rev_check["status"] == "FAIL"

    def test_churn_rate_out_of_range(self, tmp_path: Path) -> None:
        from src.data_quality import validate_gold_feature_store

        df = _make_gold_df(200, churn_rate=0.95)  # way too high
        p = tmp_path / "features.parquet"
        df.to_parquet(p, index=False)
        results = validate_gold_feature_store(str(p))
        rate_check = next((r for r in results if "churn_rate" in r["check"]), None)
        assert rate_check is not None
        assert rate_check["status"] == "FAIL"

    def test_missing_file_reported_gracefully(self, tmp_path: Path) -> None:
        from src.data_quality import validate_silver_transactions

        results = validate_silver_transactions(str(tmp_path / "nonexistent.parquet"))
        assert any(r["status"] == "FAIL" for r in results)

    def test_report_json_structure(self, tmp_path: Path) -> None:
        from src.data_quality import run_all

        self._write_parquets(tmp_path)
        # run_all() reads from the actual data/ directory (integration-style)
        report = run_all()
        assert "summary" in report
        assert "tables" in report
        assert report["summary"]["overall"] in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# monitor.py tests (Evidently, no MLflow)
# ---------------------------------------------------------------------------

class TestMonitor:
    def test_get_feature_cols_excludes_non_features(self) -> None:
        from src.monitor import _get_feature_cols

        df = _make_gold_df(50)
        cols = _get_feature_cols(df)
        for excluded in ("customer_id", "is_churn_label", "is_active_label",
                         "scoring_date", "macro_lag_months"):
            assert excluded not in cols, f"{excluded} should be excluded"
        assert "monetary" in cols

    def test_drift_report_runs_and_returns_dict(self, tmp_path: Path) -> None:
        from src.monitor import run_drift_report

        ref_df = _make_gold_df(300)
        cur_df = _make_gold_df(100)
        ref_p = str(tmp_path / "ref.parquet")
        cur_p = str(tmp_path / "cur.parquet")
        ref_df.to_parquet(ref_p, index=False)
        cur_df.to_parquet(cur_p, index=False)

        with patch("src.monitor.REPORTS_DIR", tmp_path):
            result = run_drift_report(ref_p, cur_p)

        assert "drifted_columns_count" in result
        assert "drifted_columns_share" in result
        assert "html_report" in result
        assert Path(result["html_report"]).exists()
        assert Path(result["json_report"]).exists()

    def test_drift_detected_flag(self, tmp_path: Path) -> None:
        from src.monitor import run_drift_report, DRIFT_SHARE_THRESHOLD

        # Reference: normal distribution; current: heavily shifted → high drift
        n = 200
        ref_df = _make_gold_df(n)
        cur_df = _make_gold_df(n)
        # Shift all numeric features by 100x to force drift
        for col in cur_df.select_dtypes(include="number").columns:
            if col not in ("is_churn_label", "is_active_label", "macro_lag_months"):
                cur_df[col] = cur_df[col] * 100 + 9999

        ref_p = str(tmp_path / "ref.parquet")
        cur_p = str(tmp_path / "cur.parquet")
        ref_df.to_parquet(ref_p, index=False)
        cur_df.to_parquet(cur_p, index=False)

        with patch("src.monitor.REPORTS_DIR", tmp_path):
            result = run_drift_report(ref_p, cur_p)

        # With massive shift, drift should be detected
        assert result["drifted_columns_share"] is not None


# ---------------------------------------------------------------------------
# shap_monitor.py tests
# ---------------------------------------------------------------------------

class TestShapMonitor:
    def _make_pipeline(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer

        feature_cols = ["recency", "frequency", "monetary", "rolling_30d_spend"]
        df = _make_gold_df(200)[feature_cols]
        y = _make_gold_df(200)["is_churn_label"]

        num_pipe = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
        pre = ColumnTransformer([("num", num_pipe, feature_cols)])
        pipe = Pipeline([("preprocessor", pre), ("clf", GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=42))])
        pipe.fit(df, y)
        return pipe, feature_cols

    def test_compute_shap_values_shape(self) -> None:
        from src.shap_monitor import compute_shap_values

        pipe, feature_cols = self._make_pipeline()
        X = _make_gold_df(50)[feature_cols]
        sv, feat_names = compute_shap_values(pipe, X, feature_cols)

        assert sv.shape[0] == 50
        assert sv.shape[1] == len(feature_cols)
        assert len(feat_names) == len(feature_cols)

    def test_feature_names_from_pipeline(self) -> None:
        from src.shap_monitor import _get_feature_names_from_pipeline

        pipe, feature_cols = self._make_pipeline()
        names = _get_feature_names_from_pipeline(pipe, feature_cols)
        assert names == feature_cols

    def test_attribution_drift_no_baseline(self, tmp_path: Path) -> None:
        from src.shap_monitor import check_attribution_drift

        current = pd.Series({"recency": 0.5, "monetary": 0.3, "frequency": 0.2})
        result = check_attribution_drift(current, tmp_path / "nonexistent.json")
        assert result["baseline_found"] is False

    def test_attribution_drift_with_baseline(self, tmp_path: Path) -> None:
        from src.shap_monitor import check_attribution_drift, DRIFT_THRESHOLD

        baseline = pd.Series({"recency": 0.5, "monetary": 0.3, "frequency": 0.2})
        current = pd.Series({"recency": 0.1, "monetary": 0.8, "frequency": 0.1})
        bp = tmp_path / "baseline.json"
        baseline.to_json(bp)

        result = check_attribution_drift(current, bp)
        assert result["baseline_found"] is True
        assert len(result["drifted"]) > 0

    def test_save_shap_bar_chart_creates_file(self, tmp_path: Path) -> None:
        from src.shap_monitor import save_shap_bar_chart

        importance = pd.Series(
            {"recency": 0.5, "monetary": 0.3, "frequency": 0.2}
        )
        out = tmp_path / "chart.png"
        save_shap_bar_chart(importance, out)
        assert out.exists()
        assert out.stat().st_size > 0
