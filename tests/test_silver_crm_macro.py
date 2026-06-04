import pandas as pd
import pytest

from utils.silver import clean_customer_metadata, clean_macro_monthly


@pytest.fixture
def crm_df(tmp_path):
    dest = str(tmp_path / "crm.parquet")
    clean_customer_metadata(dest_parquet=dest)
    return pd.read_parquet(dest)


@pytest.fixture
def macro_df(tmp_path):
    dest = str(tmp_path / "macro.parquet")
    clean_macro_monthly(dest_parquet=dest)
    return pd.read_parquet(dest)


def test_crm_one_row_per_customer(crm_df):
    assert crm_df["customer_id"].is_unique


def test_crm_numeric_columns_cast(crm_df):
    assert pd.api.types.is_numeric_dtype(crm_df["credit_limit_gbp"])
    assert pd.api.types.is_numeric_dtype(crm_df["payment_terms_days"])
    assert pd.api.types.is_numeric_dtype(crm_df["is_vip"])


def test_crm_has_validation_flag(crm_df):
    assert "in_transactions" in crm_df.columns


def test_macro_covers_full_obs_label_window(macro_df):
    months = set(macro_df["year_month"])
    assert "2010-12" in months
    assert "2011-12" in months
    assert len(macro_df) == 13


def test_macro_has_no_nulls(macro_df):
    assert macro_df.isnull().sum().sum() == 0


def test_macro_has_return_and_volatility_columns(macro_df):
    assert "ftse_monthly_return" in macro_df.columns
    assert "ftse_volatility" in macro_df.columns