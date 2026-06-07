import pandas as pd
import pytest

from utils.timestamps import (
    create_join_keys,
    standardize_ancillary_timestamps,
    #standardize_customer_metadata_timestamps,
    standardize_transactions_timestamps,
)


@pytest.fixture
def timestamped_transactions(tmp_path):
    dest = str(tmp_path / "transactions_timestamped.parquet")
    standardize_transactions_timestamps(
        src_parquet="data/silver/transactions.parquet",
        dest_parquet=dest,
    )
    return pd.read_parquet(dest)


@pytest.fixture
def timestamped_ancillary(tmp_path):
    dest = str(tmp_path / "ancillary_timestamped.parquet")
    standardize_ancillary_timestamps(
        src_parquet="data/bronze/ancillary.parquet",
        dest_parquet=dest,
    )
    return pd.read_parquet(dest)


def test_transactions_invoice_date_is_datetime(timestamped_transactions):
    assert pd.api.types.is_datetime64_any_dtype(timestamped_transactions["invoice_date"])


def test_transactions_has_temporal_features(timestamped_transactions):
    expected_features = {
        "invoice_year",
        "invoice_month",
        "invoice_day",
        "invoice_dayofweek",
        "invoice_quarter",
        "invoice_week",
        "invoice_date_only",
    }
    assert expected_features.issubset(set(timestamped_transactions.columns))


def test_transactions_temporal_features_are_numeric(timestamped_transactions):
    numeric_features = ["invoice_year", "invoice_month", "invoice_day", "invoice_dayofweek", 
                        "invoice_quarter", "invoice_week"]
    for feature in numeric_features:
        assert pd.api.types.is_numeric_dtype(timestamped_transactions[feature])


def test_ancillary_date_is_datetime(timestamped_ancillary):
    assert pd.api.types.is_datetime64_any_dtype(timestamped_ancillary["date"])


def test_ancillary_has_temporal_features(timestamped_ancillary):
    expected_features = {"year", "month", "day", "dayofweek", "quarter", "week", "date_only"}
    assert expected_features.issubset(set(timestamped_ancillary.columns))


def test_ancillary_original_date_column_removed(timestamped_ancillary):
    assert "Date" not in timestamped_ancillary.columns


def test_ancillary_is_holiday_renamed(timestamped_ancillary):
    assert "is_holiday" in timestamped_ancillary.columns
    assert "Is_Holiday" not in timestamped_ancillary.columns


def test_join_keys_summary_has_required_fields():
    result = create_join_keys(
        transactions_parquet="data/silver/transactions_timestamped.parquet",
        ancillary_parquet="data/silver/ancillary_timestamped.parquet",
    )
    expected_keys = {
        "transactions_date_range",
        "ancillary_date_range",
        "join_keys_available",
        "transactions_rows",
        "ancillary_rows",
    }
    assert expected_keys.issubset(set(result.keys()))
