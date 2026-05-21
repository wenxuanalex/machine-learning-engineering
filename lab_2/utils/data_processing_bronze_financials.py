import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col


def process_bronze_financials(snapshot_date_str, bronze_financials_directory, spark):
    """
    Ingest raw financial features for a single snapshot date into the Bronze layer.

    Reads the full source CSV with inferSchema=False so that raw string values
    (including dirty entries like '52312.68_') are preserved exactly as they
    appear in the source — faithful to Bronze's contract of no transformation.
    Writes a per-date Parquet partition, avoiding the Pandas round-trip and
    preserving schema metadata for downstream Silver processing.

    Args:
        snapshot_date_str: Date string in format "YYYY-MM-DD".
        bronze_financials_directory: Path to the Bronze financials directory.
        spark: Active SparkSession.

    Returns:
        Spark DataFrame containing the raw Bronze partition for the given date.
    """
    # Parse and validate the snapshot date
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Connect to source — IRL this would be an API or database read
    csv_file_path = "data/features_financials.csv"

    # Load with inferSchema=False: Bronze preserves the raw source strings.
    # Type casting and cleaning are Silver's responsibility.
    df = (
        spark.read.csv(csv_file_path, header=True, inferSchema=False)
        .filter(col("snapshot_date") == snapshot_date_str)
    )

    row_count = df.count()
    print(f"{snapshot_date_str} row count: {row_count}")

    # Save as Parquet — schema-preserving and efficient for downstream reads
    partition_name = "bronze_financials_" + snapshot_date_str.replace("-", "_")
    filepath = os.path.join(bronze_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"Saved to: {filepath}")

    return df
