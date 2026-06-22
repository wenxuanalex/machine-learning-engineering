import pandas as pd
import pytest

from utils.gold import build_gold_feature_store


@pytest.fixture
def gold_df(tmp_path):
    dest = str(tmp_path / "feature_store.parquet")
    build_gold_feature_store(dest_parquet=dest)
    return pd.read_parquet(dest)


def test_gold_one_row_per_customer(gold_df):
    assert gold_df["customer_id"].is_unique


def test_gold_has_required_g1_features(gold_df):
    required = {
        "recency",
        "frequency",
        "monetary",
        "avg_basket_size",
        "avg_order_interarrival_days",
        "product_diversity",
        "cancellation_rate",
        "rolling_30d_spend",
        "rolling_60d_spend",
        "rolling_90d_spend",
    }
    assert required.issubset(set(gold_df.columns))


def test_gold_no_future_leakage_from_observation_window(gold_df):
    assert (gold_df["scoring_date"] == pd.Timestamp("2011-08-31")).all()


def test_gold_churn_label_is_binary(gold_df):
    assert set(gold_df["is_churn_label"].dropna().unique()).issubset({0, 1})
    assert set(gold_df["is_active_label"].dropna().unique()).issubset({0, 1})


def test_gold_has_lagged_macro_columns(gold_df):
    assert "macro_lag_months" in gold_df.columns
    assert "macro_ftse_monthly_return" in gold_df.columns
    # macro_lag_months=3 for GDP publication lag (quarterly reporting)
    # Per-column lags: GDP=3mo, CPI=1mo, market_returns=0mo
    assert (gold_df["macro_lag_months"] == 3).all()


def test_gold_has_lagged_auxiliary_metadata(gold_df):
    assert "auxiliary_lag_months" in gold_df.columns
    assert "auxiliary_snapshot_month" in gold_df.columns
    assert (gold_df["auxiliary_lag_months"] == 1).all()
    assert (gold_df["auxiliary_snapshot_month"] == "2011-07").all()
