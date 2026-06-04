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
    """Clean bronze transactions into silver.

    Applies all quality filters defined in the EDA + medallion design:
    - Type casting (Quantity, UnitPrice, InvoiceDate)
    - Drop nulls on CustomerID
    - Drop cancellations (InvoiceNo starts with 'C')
    - Drop non-positive quantity / unit price
    - Drop non-product SKUs
    - Compute revenue
    - Snake_case columns
    """
    df = pd.read_parquet(src_parquet)
    rows_in = len(df)

    #Type casting
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    #Drop null rows 
    df = df[df["CustomerID"].notna()]

    #Drop cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    #Drop negatve quantity
    df = df[df["Quantity"] > 0]

    #Drop negative unit price
    df = df[df["UnitPrice"] > 0]

    #Filter
    df = df[~df["StockCode"].isin(NON_PRODUCT_STOCKCODES)]

    #Compute revenue
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