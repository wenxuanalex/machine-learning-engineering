from utils.bronze import ingest
from utils.silver import clean_transactions, clean_customer_metadata, clean_macro_monthly, date_dim

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
print("\n=== Silver: CRM metadata ===")
crm_result = clean_customer_metadata()
print(f"  Rows: {crm_result['rows']:,}  |  Unmatched CRM customers: {crm_result['unmatched_crm_customers']}  |  Nulls: {crm_result['nulls']}")

print("\n=== Silver: Macro monthly ===")
macro_result = clean_macro_monthly()
print(f"  Rows: {macro_result['rows']}  |  Months in raw panel: {macro_result['months_available_raw']}  |  Nulls: {macro_result['nulls']}")

print("\n=== Silver: Calendar Features ===")
cal_fe = date_dim()
print(f"\n{cal_fe['source']} → {cal_fe['destination']}")
print(f"  Rows: {cal_fe['rows']}  |  Schema: {cal_fe['schema']} ")

print("\nPipeline complete.")
