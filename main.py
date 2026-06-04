from utils.bronze import ingest
from utils.silver import clean_transactions

BRONZE_SOURCES = [
    ("data/data.csv",                               "data/bronze/transactions.parquet"),
    ("data/bronze_customer_metadata_synthetic.csv", "data/bronze/customer_metadata.parquet"),
    ("data/ancillary.csv",                          "data/bronze/ancillary.parquet"),
]

print("=== Bronze Ingestion ===")
for src, dest in BRONZE_SOURCES:
    result = ingest(src, dest)
    print(f"\n{result['source']} → {result['destination']}")
    print(f"  Rows  : {result['rows']:,}")

print("\n=== Silver Cleaning ===")
silver_result = clean_transactions()
print(f"\n{silver_result['source']} → {silver_result['destination']}")
print(f"  Rows in     : {silver_result['rows_in']:,}")
print(f"  Rows out    : {silver_result['rows_out']:,}")
print(f"  Rows dropped: {silver_result['rows_dropped']:,} ({silver_result['drop_rate_pct']}%)")

print("\nPipeline complete.")
