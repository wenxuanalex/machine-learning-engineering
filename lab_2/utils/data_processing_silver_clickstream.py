import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Transform raw clickstream features from Bronze to Silver layer.
    
    Steps:
    1. Load Bronze clickstream data for the specific snapshot_date.
    2. Drop records missing critical primary keys (Customer_ID, snapshot_date).
    3. Replace placeholder '-1' values with actual NULLs.
    4. Cast all feature columns (fe_1 to fe_20) to FloatType.
    5. Cast snapshot_date to DateType.
    6. Save to Silver layer.
    
    Args:
        snapshot_date_str: Date string in format "YYYY-MM-DD"
        bronze_clickstream_directory: Path to Bronze layer directory
        silver_clickstream_directory: Path to Silver layer directory
        spark: SparkSession object
    
    Returns:
        Spark DataFrame with cleaned clickstream data
    """
    
    # Handle both string and datetime.date inputs
    if isinstance(snapshot_date_str, str):
        formatted_date_str = snapshot_date_str
    else:
        # Assume it's a datetime.date object
        formatted_date_str = snapshot_date_str.strftime("%Y-%m-%d")

    # Connect to Bronze Parquet partition
    partition_name = "bronze_clickstream_" + formatted_date_str.replace('-', '_')
    filepath = os.path.join(bronze_clickstream_directory, partition_name)

    # Load data
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # ===== CLEANING & TYPE CASTING =====
    # 1. Drop records where primary keys are null
    df = df.dropna(subset=["Customer_ID", "snapshot_date"])
    
    # 2. Replace '-1' with null and cast features to FloatType
    feature_columns = [f"fe_{i}" for i in range(1, 21)]
    
    for col_name in feature_columns:
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) == "-1", None)
             .otherwise(F.col(col_name))
        ).withColumn(
            col_name,
            F.col(col_name).cast(FloatType())
        )
        
    # 3. Cast snapshot_date to DateType
    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

    # ===== SAVE TO SILVER LAYER =====
    # Save the cleaned data to the Silver layer as Parquet
    output_path = os.path.join(silver_clickstream_directory, f"snapshot_date={formatted_date_str}")
    df.write.mode("overwrite").parquet(output_path)
    
    print(f"Successfully processed and saved data for {formatted_date_str} to {output_path}")

    return df
