from pathlib import Path

import pandas as pd

# Pulled from the EDA notebook
NON_PRODUCT_STOCKCODES = {
    "POST",
    "DOT",
    "M",
    "BANK CHARGES",
    "AMAZONFEE",
    "S",
    "CRUK",
    "D",
    "PADS",
    "C2",
}


def clean_transactions(
    src_parquet: str = "data/bronze/transactions.parquet",
    dest_parquet: str = "data/silver/transactions.parquet",
) -> dict:
    """Clean bronze transactions into silver."""
    df = pd.read_parquet(src_parquet)
    rows_in = len(df)

    # Type casting
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Drop null CustomerID
    df = df[df["CustomerID"].notna()]

    # Drop cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Drop negative quantity
    df = df[df["Quantity"] > 0]

    # Drop negatice unit price
    df = df[df["UnitPrice"] > 0]

    # Filter
    df = df[~df["StockCode"].isin(NON_PRODUCT_STOCKCODES)]

    # Compute revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df = df.rename(columns={
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
        "Revenue": "revenue",
    })

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)

    rows_out = len(df)
    return {
        "source": src_parquet,
        "destination": dest_parquet,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_in - rows_out,
        "drop_rate_pct": round((rows_in - rows_out) / rows_in * 100, 2),
    }


# Customer metadata 
def clean_customer_metadata(
    src_parquet: str = "data/bronze/customer_metadata.parquet",
    dest_parquet: str = "data/silver/customer_metadata.parquet",
    silver_tx_parquet: str = "data/silver/transactions.parquet",
) -> dict:
    """Clean bronze CRM metadata into silver."""
    df = pd.read_parquet(src_parquet)

    # Type casting
    df["credit_limit_gbp"] = pd.to_numeric(df["credit_limit_gbp"], errors="coerce")
    df["payment_terms_days"] = pd.to_numeric(df["payment_terms_days"], errors="coerce").astype("Int64")
    df["years_in_business"] = pd.to_numeric(df["years_in_business"], errors="coerce").astype("Int64")
    df["is_vip"] = pd.to_numeric(df["is_vip"], errors="coerce").astype("Int64")

    # Validate against known customer IDs from silver transactions
    tx = pd.read_parquet(silver_tx_parquet)
    valid_ids = set(tx["customer_id"].unique())
    df["in_transactions"] = df["customer_id"].isin(valid_ids)
    n_unmatched = int((~df["in_transactions"]).sum())

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)

    return {
        "destination": dest_parquet,
        "rows": len(df),
        "unmatched_crm_customers": n_unmatched,
        "nulls": int(df.drop(columns=["in_transactions"]).isnull().sum().sum()),
    }


#Macro monthly panel
RETURN_COLS = ["EWU", "^FTAS", "^FTMC", "^FTSE"]
LEVEL_COLS = [
    "UK_CPI", "UK_Ind_Production", "UK_GDP",
    "UK_CPI_YoY", "UK_Ind_Prod_YoY", "UK_GDP_QoQ", "UK_GDP_YoY",
]
OBS_LABEL_MONTHS = pd.period_range("2010-12", "2011-12", freq="M")


def clean_macro_monthly(
    src_parquet: str = "data/bronze/ancillary.parquet",
    dest_parquet: str = "data/silver/macro_monthly.parquet",
) -> dict:
    """Aggregate daily macro/market data into a complete monthly panel."""
    df = pd.read_parquet(src_parquet)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in RETURN_COLS + LEVEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Is_Holiday"] = df["Is_Holiday"].map({"True": True, "False": False}).astype(bool)
    df["year_month"] = df["Date"].dt.to_period("M")

    out = pd.DataFrame(index=sorted(df["year_month"].unique()))
    for c in RETURN_COLS:
        clean = c.replace("^", "").lower()
        out[f"{clean}_monthly_return"] = (
            df.groupby("year_month")[c].apply(lambda r: (1 + r).prod() - 1)
        )
        out[f"{clean}_volatility"] = df.groupby("year_month")[c].std()
    for c in LEVEL_COLS:
        out[c.lower()] = df.groupby("year_month")[c].last()
    out["holidays_in_month"] = df.groupby("year_month")["Is_Holiday"].sum()
    out["trading_days"] = df.groupby("year_month")["Date"].count()

    months_available = len(out)

    # Extend back to Dec 2010; pre-April months inherit earliest snapshot
    out = out.reindex(OBS_LABEL_MONTHS).bfill().ffill()
    out = out.reset_index().rename(columns={"index": "year_month"})
    out["year_month"] = out["year_month"].astype(str)

    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest_parquet, index=False)

    return {
        "destination": dest_parquet,
        "rows": len(out),
        "months_available_raw": months_available,
        "nulls": int(out.isnull().sum().sum()),
    }

def date_dim(src_parquet: str = "data/bronze/ancillary.parquet", dest_parquet: str = "data/silver/silver_date_dim.parquet") -> dict:
    df = pd.read_parquet(src_parquet)
    date_dim = df[["Date", "Is_Holiday"]].copy() 
    date_dim['Date']=pd.to_datetime(date_dim['Date'])
    date_dim["Is_Holiday"] = date_dim["Is_Holiday"].map({"True": 1, "False": 0})
    date_dim = date_dim.rename(columns={"Is_Holiday":"is_bank_holiday"})
    
    date_dim["is_q4"] = date_dim["Date"].dt.quarter.eq(4).astype(int)
    date_dim["month"] = date_dim["Date"].dt.month
    date_dim["quarter"] = date_dim["Date"].dt.quarter
    date_dim["year_month"] = date_dim["Date"].dt.to_period("M").astype(str)
    
    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    date_dim.to_parquet(dest_parquet, index=False)
    return {
        "source": src_parquet,
        "destination": dest_parquet,
        "rows": len(date_dim),
        "schema": {col: str(dtype) for col, dtype in date_dim.dtypes.items()},
    }
    