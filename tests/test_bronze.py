import pandas as pd
import pytest

from utils.bronze import ingest

SOURCES = [
    "data/data.csv",
    "data/bronze_customer_metadata_synthetic.csv",
    "data/ancillary_20101201_to_20111231.csv",
]


@pytest.mark.parametrize("src_csv", SOURCES)
def test_bronze_row_count_matches_source(src_csv, tmp_path):
    expected_rows = sum(1 for _ in open(src_csv, encoding="latin-1")) - 1  # subtract header
    dest = str(tmp_path / "out.parquet")
    result = ingest(src_csv, dest)
    assert result["rows"] == expected_rows
    assert len(pd.read_parquet(dest)) == expected_rows


@pytest.mark.parametrize("src_csv", SOURCES)
def test_bronze_schema_preserved_as_strings(src_csv, tmp_path):
    dest = str(tmp_path / "out.parquet")
    result = ingest(src_csv, dest)
    assert all(dtype == "object" for dtype in result["schema"].values())
