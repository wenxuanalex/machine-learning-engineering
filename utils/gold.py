from pathlib import Path

import numpy as np
import pandas as pd


def _as_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def _build_rfm_and_behavior(obs_tx: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    grouped = obs_tx.groupby("customer_id")

    rfm = grouped.agg(
        last_purchase_date=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
        product_diversity=("stock_code", "nunique"),
    ).reset_index()

    rfm["recency"] = (obs_end - rfm["last_purchase_date"]).dt.days.astype(int)
    rfm = rfm.drop(columns=["last_purchase_date"])

    invoice_level = (
        obs_tx.groupby(["customer_id", "invoice_no"], as_index=False)
        .agg(invoice_date=("invoice_date", "min"), basket_quantity=("quantity", "sum"))
    )

    basket = (
        invoice_level.groupby("customer_id", as_index=False)
        .agg(avg_basket_size=("basket_quantity", "mean"))
    )

    def mean_interarrival_days(frame: pd.DataFrame) -> float:
        dates = frame["invoice_date"].drop_duplicates().sort_values()
        if len(dates) <= 1:
            return np.nan
        return float(dates.diff().dropna().dt.days.mean())

    interarrival = (
        invoice_level.groupby("customer_id")
        .apply(mean_interarrival_days)
        .rename("avg_order_interarrival_days")
        .reset_index()
    )

    return rfm.merge(basket, on="customer_id", how="left").merge(
        interarrival, on="customer_id", how="left"
    )


def _build_rolling_spend(obs_tx: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    rolling = obs_tx[["customer_id"]].drop_duplicates().copy()

    for days in [30, 60, 90]:
        start = obs_end - pd.Timedelta(days=days - 1)
        window = obs_tx[(obs_tx["invoice_date"] >= start) & (obs_tx["invoice_date"] <= obs_end)]
        spend = (
            window.groupby("customer_id", as_index=False)["revenue"]
            .sum()
            .rename(columns={"revenue": f"rolling_{days}d_spend"})
        )
        rolling = rolling.merge(spend, on="customer_id", how="left")

    return rolling


def _build_cancellation_rate(
    bronze_tx: pd.DataFrame,
    obs_start: pd.Timestamp,
    obs_end: pd.Timestamp,
    modelable_customers: pd.Index,
) -> pd.DataFrame:
    raw = bronze_tx.copy()
    raw["CustomerID"] = pd.to_numeric(raw["CustomerID"], errors="coerce")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw = raw[raw["CustomerID"].notna()]
    raw["customer_id"] = raw["CustomerID"].astype("int64")

    raw_obs = raw[(raw["InvoiceDate"] >= obs_start) & (raw["InvoiceDate"] <= obs_end)]
    raw_obs = raw_obs[raw_obs["customer_id"].isin(modelable_customers)]

    invoices = raw_obs[["customer_id", "InvoiceNo"]].drop_duplicates()
    invoices["is_cancelled"] = invoices["InvoiceNo"].astype(str).str.startswith("C")

    rate = (
        invoices.groupby("customer_id", as_index=False)
        .agg(
            total_invoices=("InvoiceNo", "count"),
            cancelled_invoices=("is_cancelled", "sum"),
        )
        .assign(
            cancellation_rate=lambda d: np.where(
                d["total_invoices"] > 0,
                d["cancelled_invoices"] / d["total_invoices"],
                0.0,
            )
        )
    )

    return rate[["customer_id", "cancellation_rate"]]


def _build_churn_labels(
    silver_tx: pd.DataFrame,
    label_start: pd.Timestamp,
    label_end: pd.Timestamp,
    modelable_customers: pd.Index,
) -> pd.DataFrame:
    label_tx = silver_tx[
        (silver_tx["invoice_date"] >= label_start) & (silver_tx["invoice_date"] <= label_end)
    ]
    active = set(label_tx["customer_id"].unique())

    labels = pd.DataFrame({"customer_id": modelable_customers})
    labels["is_active_label"] = labels["customer_id"].isin(active).astype(int)
    labels["is_churn_label"] = (1 - labels["is_active_label"]).astype(int)
    return labels


def _macro_snapshot(
    macro_monthly: pd.DataFrame,
    obs_end: pd.Timestamp,
    macro_lag_months: int,
) -> dict:
    target_month = (obs_end.to_period("M") - macro_lag_months).strftime("%Y-%m")
    row = macro_monthly.loc[macro_monthly["year_month"] == target_month]
    if row.empty:
        raise ValueError(
            f"No macro row found for lagged month {target_month}. "
            "Check macro_lag_months and silver macro coverage."
        )

    record = row.iloc[0].to_dict()
    return {f"macro_{k}": v for k, v in record.items() if k != "year_month"}


def build_gold_feature_store(
    silver_tx_parquet: str = "data/silver/transactions.parquet",
    bronze_tx_parquet: str = "data/bronze/transactions.parquet",
    silver_crm_parquet: str = "data/silver/customer_metadata.parquet",
    silver_macro_parquet: str = "data/silver/macro_monthly.parquet",
    dest_parquet: str = "data/gold/feature_store.parquet",
    obs_start: str = "2010-12-01",
    obs_end: str = "2011-08-31",
    label_start: str = "2011-09-01",
    label_end: str = "2011-12-31",
    macro_lag_months: int = 1,
) -> dict:
    """Build one-row-per-customer gold feature store with churn labels."""
    obs_start_ts = _as_timestamp(obs_start)
    obs_end_ts = _as_timestamp(obs_end)
    label_start_ts = _as_timestamp(label_start)
    label_end_ts = _as_timestamp(label_end)

    silver_tx = pd.read_parquet(silver_tx_parquet).copy()
    silver_tx["invoice_date"] = pd.to_datetime(silver_tx["invoice_date"], errors="coerce")

    obs_tx = silver_tx[
        (silver_tx["invoice_date"] >= obs_start_ts) & (silver_tx["invoice_date"] <= obs_end_ts)
    ].copy()

    modelable_customers = pd.Index(sorted(obs_tx["customer_id"].dropna().unique()))
    customer_frame = pd.DataFrame({"customer_id": modelable_customers})

    rfm_behavior = _build_rfm_and_behavior(obs_tx=obs_tx, obs_end=obs_end_ts)
    rolling = _build_rolling_spend(obs_tx=obs_tx, obs_end=obs_end_ts)

    bronze_tx = pd.read_parquet(bronze_tx_parquet)
    cancellation = _build_cancellation_rate(
        bronze_tx=bronze_tx,
        obs_start=obs_start_ts,
        obs_end=obs_end_ts,
        modelable_customers=modelable_customers,
    )

    labels = _build_churn_labels(
        silver_tx=silver_tx,
        label_start=label_start_ts,
        label_end=label_end_ts,
        modelable_customers=modelable_customers,
    )

    crm = pd.read_parquet(silver_crm_parquet)
    macro = pd.read_parquet(silver_macro_parquet)
    macro_values = _macro_snapshot(macro, obs_end=obs_end_ts, macro_lag_months=macro_lag_months)

    feature_store = (
        customer_frame.merge(rfm_behavior, on="customer_id", how="left")
        .merge(rolling, on="customer_id", how="left")
        .merge(cancellation, on="customer_id", how="left")
        .merge(crm, on="customer_id", how="left")
        .merge(labels, on="customer_id", how="left")
    )

    feature_store["avg_order_interarrival_days"] = feature_store[
        "avg_order_interarrival_days"
    ].fillna(-1.0)

    for col in ["rolling_30d_spend", "rolling_60d_spend", "rolling_90d_spend", "cancellation_rate"]:
        feature_store[col] = feature_store[col].fillna(0.0)

    feature_store["scoring_date"] = obs_end_ts
    feature_store["macro_lag_months"] = macro_lag_months
    for key, value in macro_values.items():
        feature_store[key] = value

    # in_transactions is a silver diagnostic flag, not a model feature
    feature_store = feature_store.drop(columns=["in_transactions"], errors="ignore")

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(dest_parquet, index=False)

    feature_list = [
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
    ]

    return {
        "destination": dest_parquet,
        "rows": len(feature_store),
        "modelable_customers": len(modelable_customers),
        "observation_window": f"{obs_start_ts.date()} to {obs_end_ts.date()}",
        "label_window": f"{label_start_ts.date()} to {label_end_ts.date()}",
        "macro_snapshot_month": (obs_end_ts.to_period("M") - macro_lag_months).strftime("%Y-%m"),
        "feature_list": feature_list,
        "label_rate_churn": round(float(feature_store["is_churn_label"].mean()), 4),
    }


def _build_churn_label_and_split(
    features: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # is_churn_label is already built in G1 — reuse it rather than re-deriving
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        features,
        test_size=test_size,
        stratify=features["is_churn_label"],
        random_state=random_state,
    )

    # Adjust val_size to be a proportion of the remaining data
    val_size_adjusted = val_size / (1 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        stratify=train_val["is_churn_label"],
        random_state=random_state,
    )

    return train, val, test


def build_gold_train_test_split(
    feature_store_path: str = "data/gold/feature_store.parquet",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    train_path: str = "data/gold/train_labeled.parquet",
    val_path: str = "data/gold/val_labeled.parquet",
    test_path: str = "data/gold/test_labeled.parquet",
) -> dict:
    """Split the gold feature store into train, validation, and test sets."""
    features = pd.read_parquet(feature_store_path)

    train, val, test = _build_churn_label_and_split(
        features=features,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    test.to_parquet(test_path, index=False)

    churn_rate = features["is_churn_label"].mean()

    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "churn_rate": f"{churn_rate:.3%}",
    }