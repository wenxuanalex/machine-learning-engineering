import os
import warnings
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_bronze_financials
import utils.data_processing_bronze_attributes
import utils.data_processing_bronze_clickstream
import utils.data_processing_silver_table
import utils.data_processing_silver_financials
import utils.data_processing_silver_attributes
import utils.data_processing_silver_clickstream
import utils.data_processing_gold_table
import utils.data_processing_gold_features

warnings.filterwarnings("ignore")
if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

# ── Spark ─────────────────────────────────────────────────────────────────────
spark = pyspark.sql.SparkSession.builder \
    .appName("CreditRiskPipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ── Config ────────────────────────────────────────────────────────────────────
start_date_str = "2023-01-01"
end_date_str   = "2024-12-01"


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    dates, current = [], datetime(start_date.year, start_date.month, 1)
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        month = current.month % 12 + 1
        year  = current.year + (1 if current.month == 12 else 0)
        current = datetime(year, month, 1)
    return dates


dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(f"Processing {len(dates_str_lst)} monthly snapshots: "
      f"{dates_str_lst[0]} → {dates_str_lst[-1]}")

# ── Directory paths ───────────────────────────────────────────────────────────
bronze_lms_directory         = "datamart/bronze/lms/"
bronze_financials_directory  = "datamart/bronze/financials/"
bronze_attributes_directory  = "datamart/bronze/attributes/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"

silver_loan_daily_directory  = "datamart/silver/loan_daily/"
silver_financials_directory  = "datamart/silver/financials/"
silver_attributes_directory  = "datamart/silver/attributes/"
silver_clickstream_directory = "datamart/silver/clickstream/"

gold_label_store_directory   = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"

for directory in [
    bronze_lms_directory,         bronze_financials_directory,
    bronze_attributes_directory,  bronze_clickstream_directory,
    silver_loan_daily_directory,  silver_financials_directory,
    silver_attributes_directory,  silver_clickstream_directory,
    gold_label_store_directory,   gold_feature_store_directory,
]:
    os.makedirs(directory, exist_ok=True)


def _source_dates(csv_path):
    """Return sorted list of unique snapshot_date values present in a source CSV."""
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    return sorted([
        row["snapshot_date"]
        for row in df.select("snapshot_date").distinct().collect()
        if row["snapshot_date"]
    ])


# ── Bronze Layer ──────────────────────────────────────────────────────────────
print("\n=== Bronze Layer ===")

for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(
        date_str, bronze_lms_directory, spark)

financials_dates_list  = _source_dates("data/features_financials.csv")
attributes_dates_list  = _source_dates("data/features_attributes.csv")
clickstream_dates_list = _source_dates("data/feature_clickstream.csv")

for date_str in financials_dates_list:
    utils.data_processing_bronze_financials.process_bronze_financials(
        date_str, bronze_financials_directory, spark)

for date_str in attributes_dates_list:
    utils.data_processing_bronze_attributes.process_bronze_attributes(
        date_str, bronze_attributes_directory, spark)

for date_str in clickstream_dates_list:
    utils.data_processing_bronze_clickstream.process_bronze_clickstream(
        date_str, bronze_clickstream_directory, spark)

# ── Silver Layer ──────────────────────────────────────────────────────────────
print("\n=== Silver Layer ===")

for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(
        date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

for date_str in financials_dates_list:
    utils.data_processing_silver_financials.process_silver_financials(
        date_str, bronze_financials_directory, silver_financials_directory, spark)

for date_str in attributes_dates_list:
    utils.data_processing_silver_attributes.process_silver_attributes(
        date_str, bronze_attributes_directory, silver_attributes_directory, spark)

for date_str in clickstream_dates_list:
    utils.data_processing_silver_clickstream.process_silver_clickstream(
        date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)

# ── Gold Layer ────────────────────────────────────────────────────────────────
print("\n=== Gold Layer ===")

for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(
        date_str, silver_loan_daily_directory, gold_label_store_directory,
        spark, dpd=30, mob=6)

feature_store_df = utils.data_processing_gold_features.process_gold_features(
    spark,
    silver_financials_directory,
    silver_loan_daily_directory,
    silver_clickstream_directory,
    silver_attributes_directory,
    gold_feature_store_directory,
)

print("\nPipeline complete.")
print(f"  Label store  : {gold_label_store_directory}")
print(f"  Feature store: {gold_feature_store_directory}")
print(f"  Feature store rows   : {feature_store_df.count():,}")
print(f"  Feature store columns: {len(feature_store_df.columns)}")
