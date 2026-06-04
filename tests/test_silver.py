import pandas as pd
import pytest

from utils.silver import NON_PRODUCT_STOCKCODES, clean_transactions


@pytest.fixture
def silver_df(tmp_path):
    dest = str(tmp_path / "silver.parquet")
    clean_transactions(
        src_parquet="data/bronze/transactions.parquet",
        dest_parquet=dest,
    )
    return pd.read_parquet(dest)


def test_silver_has_no_null_customer_id(silver_df):
    assert silver_df["customer_id"].notna().all()


def test_silver_has_no_cancellations(silver_df):
    assert not silver_df["invoice_no"].astype(str).str.startswith("C").any()


def test_silver_has_only_positive_quantity(silver_df):
    assert (silver_df["quantity"] > 0).all()


def test_silver_has_only_positive_unit_price(silver_df):
    assert (silver_df["unit_price"] > 0).all()


def test_silver_excludes_non_product_skus(silver_df):
    assert not silver_df["stock_code"].isin(NON_PRODUCT_STOCKCODES).any()


def test_silver_revenue_equals_quantity_times_price(silver_df):
    assert (silver_df["revenue"] == silver_df["quantity"] * silver_df["unit_price"]).all()


def test_silver_columns_are_snake_case(silver_df):
    expected = {
        "invoice_no", "stock_code", "description", "quantity",
        "invoice_date", "unit_price", "customer_id", "country", "revenue",
    }
    assert set(silver_df.columns) == expected


def test_silver_invoice_date_is_datetime(silver_df):
    assert pd.api.types.is_datetime64_any_dtype(silver_df["invoice_date"])