import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, DateType


def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Transform raw attributes features from Bronze to Silver layer.
    
    Steps:
    1. Load Bronze attributes data.
    2. Drop missing primary keys.
    3. Drop the 'Name' column as it is not a predictive feature.
    4. Clean 'Age': Strip trailing underscores, cast to Integer, nullify values outside [18, 99].
    5. Clean 'SSN': Ensure standard XXX-XX-XXXX format, nullify otherwise.
    6. Clean 'Occupation': Replace dummy string '_______' with NULL.
    7. Cast snapshot_date to DateType.
    8. Save to Silver layer.
    """
    if isinstance(snapshot_date_str, str):
        formatted_date_str = snapshot_date_str
    else:
        formatted_date_str = snapshot_date_str.strftime("%Y-%m-%d")

    partition_name = "bronze_attributes_" + formatted_date_str.replace('-', '_')
    filepath = os.path.join(bronze_attributes_directory, partition_name)

    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # 1. Drop rows with missing primary keys
    df = df.dropna(subset=["Customer_ID", "snapshot_date"])

    # 2. Drop the 'Name' column
    if "Name" in df.columns:
        df = df.drop("Name")

    # 3. Clean Age
    df = df.withColumn(
        "Age",
        F.regexp_replace(F.col("Age"), "_+$", "")  # Strip trailing underscores
    ).withColumn(
        "Age",
        F.col("Age").cast(IntegerType())
    ).withColumn(
        "Age",
        F.when((F.col("Age") < 18) | (F.col("Age") > 99), None).otherwise(F.col("Age"))
    )

    # 4. Clean SSN
    # Standard SSN format: 3 digits - 2 digits - 4 digits
    ssn_pattern = r"^\d{3}-\d{2}-\d{4}$"
    df = df.withColumn(
        "SSN",
        F.when(F.col("SSN").rlike(ssn_pattern), F.col("SSN")).otherwise(None)
    )

    # 5. Clean Occupation
    # Replace the specific dummy string or empty strings
    df = df.withColumn(
        "Occupation",
        F.when((F.col("Occupation") == "_______") | (F.trim(F.col("Occupation")) == ""), None).otherwise(F.col("Occupation"))
    )

    # 6. Cast snapshot_date
    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

    # Save to Silver layer
    output_path = os.path.join(silver_attributes_directory, f"snapshot_date={formatted_date_str}")
    df.write.mode("overwrite").parquet(output_path)
    
    print(f"Successfully processed and saved data for {formatted_date_str} to {output_path}")

    return df
