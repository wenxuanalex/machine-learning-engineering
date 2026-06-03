from utils.bronze import ingest

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
    print(f"  Schema: {result['schema']}")

print("\nBronze ingestion complete.")
