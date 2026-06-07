from pathlib import Path

import pandas as pd


def standardize_transactions_timestamps(
    src_parquet: str = "data/silver/transactions.parquet",
    dest_parquet: str = "data/silver/transactions_timestamped.parquet",
) -> dict:
    """Standardize and enrich timestamps in silver transactions."""
    df = pd.read_parquet(src_parquet)
    rows_in = len(df)

    # Ensure invoice_date is datetime
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

    # Extract temporal features
    df["invoice_year"] = df["invoice_date"].dt.year
    df["invoice_month"] = df["invoice_date"].dt.month
    df["invoice_day"] = df["invoice_date"].dt.day
    df["invoice_dayofweek"] = df["invoice_date"].dt.dayofweek
    df["invoice_quarter"] = df["invoice_date"].dt.quarter
    df["invoice_week"] = df["invoice_date"].dt.isocalendar().week

    # Create date-only column for joining with ancillary data
    df["invoice_date_only"] = df["invoice_date"].dt.date

    rows_out = len(df)

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)

    return {
        "source": src_parquet,
        "destination": dest_parquet,
        "rows_processed": rows_out,
        "rows_dropped": rows_in - rows_out,
        "temporal_features_added": [
            "invoice_year",
            "invoice_month",
            "invoice_day",
            "invoice_dayofweek",
            "invoice_quarter",
            "invoice_week",
            "invoice_date_only",
        ],
    }


def standardize_ancillary_timestamps(
    src_parquet: str = "data/bronze/ancillary.parquet",
    dest_parquet: str = "data/silver/ancillary_timestamped.parquet",
) -> dict:
    """Standardize and enrich timestamps in ancillary economic data."""
    df = pd.read_parquet(src_parquet)
    rows_in = len(df)

    # Convert Date to datetime
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Rename original Date column and drop it
    df = df.drop(columns=["Date"])

    # Extract temporal features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df["date"].dt.isocalendar().week
    df["date_only"] = df["date"].dt.date

    # Rename is_holiday to is_holiday_flag for clarity
    df = df.rename(columns={"Is_Holiday": "is_holiday"})

    rows_out = len(df)

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)

    return {
        "source": src_parquet,
        "destination": dest_parquet,
        "rows_processed": rows_out,
        "rows_dropped": rows_in - rows_out,
        "temporal_features_added": [
            "year",
            "month",
            "day",
            "dayofweek",
            "quarter",
            "week",
            "date_only",
        ],
    }


def standardize_customer_metadata_timestamps(
    src_parquet: str = "data/silver/customer_metadata.parquet",
    dest_parquet: str = "data/silver/customer_metadata_timestamped.parquet",
) -> dict:
    """Prepare customer metadata for joins (adds a reference timestamp)."""
    df = pd.read_parquet(src_parquet)
    rows_in = len(df)

    # Add a reference timestamp for the metadata snapshot
    # This assumes metadata is as of the last transaction date in the dataset
    df["metadata_snapshot_date"] = pd.Timestamp("2011-12-31").date()

    rows_out = len(df)

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)

    return {
        "source": src_parquet,
        "destination": dest_parquet,
        "rows_processed": rows_out,
        "rows_dropped": rows_in - rows_out,
        "temporal_features_added": ["metadata_snapshot_date"],
    }


def create_join_keys(
    transactions_parquet: str = "data/silver/transactions_timestamped.parquet",
    ancillary_parquet: str = "data/silver/ancillary_timestamped.parquet",
) -> dict:
    """Generate summary of join keys available between datasets."""
    tx_df = pd.read_parquet(transactions_parquet)
    anc_df = pd.read_parquet(ancillary_parquet)

    # Extract date ranges
    tx_date_range = {
        "min": tx_df["invoice_date_only"].min(),
        "max": tx_df["invoice_date_only"].max(),
    }
    anc_date_range = {
        "min": anc_df["date_only"].min(),
        "max": anc_df["date_only"].max(),
    }

    return {
        "transactions_date_range": {
            "min": str(tx_date_range["min"]),
            "max": str(tx_date_range["max"]),
        },
        "ancillary_date_range": {
            "min": str(anc_date_range["min"]),
            "max": str(anc_date_range["max"]),
        },
        "join_keys_available": {
            "by_date": "Use 'invoice_date_only' (transactions) with 'date_only' (ancillary)",
            "by_year": "Use 'invoice_year' with 'year'",
            "by_month": "Use 'invoice_month' with 'month'",
            "by_quarter": "Use 'invoice_quarter' with 'quarter'",
            "by_week": "Use 'invoice_week' with 'week'",
        },
        "transactions_rows": len(tx_df),
        "ancillary_rows": len(anc_df),
    }
