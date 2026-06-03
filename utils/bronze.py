from pathlib import Path

import pandas as pd


def ingest(src_csv: str, dest_parquet: str) -> dict:
    """Read a CSV as raw strings into Parquet with no transformations."""
    df = pd.read_csv(src_csv, dtype=str, encoding="latin-1")
    Path(dest_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_parquet, index=False)
    return {
        "source": src_csv,
        "destination": dest_parquet,
        "rows": len(df),
        "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
