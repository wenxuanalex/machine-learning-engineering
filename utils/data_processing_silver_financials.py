import os
from datetime import datetime

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Transform raw financial features from Bronze to Silver layer.
    
    Steps:
    1. Load Bronze financial data
    2. Filter by snapshot_date
    3. Enforce schema and data types
    4. Clean and standardize values
    5. Handle missing/malformed data
    6. Save to Silver layer
    
    Args:
        snapshot_date_str: Date string in format "YYYY-MM-DD"
        bronze_financials_directory: Path to Bronze layer directory
        silver_financials_directory: Path to Silver layer directory
        spark: SparkSession object
    
    Returns:
        Spark DataFrame with cleaned financial data
    """
    
    # prepare arguments
    # Handle both string and datetime.date inputs
    if isinstance(snapshot_date_str, str):
        formatted_date_str = snapshot_date_str
    else:
        # Assume it's a datetime.date object
        formatted_date_str = snapshot_date_str.strftime("%Y-%m-%d")

    # Connect to Bronze Parquet partition written by data_processing_bronze_financials
    partition_name = "bronze_financials_" + formatted_date_str.replace('-', '_')
    filepath = os.path.join(bronze_financials_directory, partition_name)

    # Bronze already stores raw strings (inferSchema=False at ingestion),
    # so Parquet schema will be all StringType — same starting point as before.
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # ===== SCHEMA ENFORCEMENT & TYPE CASTING =====
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),  # Keep as string initially, requires parsing
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # ===== CLEAN NUMERIC COLUMNS =====
    # Remove trailing underscores and other artifacts, then cast
    numeric_columns = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", 
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]
    
    for col_name in numeric_columns:
        # Remove trailing underscores and whitespace, then cast
        df = df.withColumn(
            col_name,
            F.trim(F.regexp_replace(F.col(col_name), "_+$", ""))
        ).withColumn(
            col_name,
            F.when(F.col(col_name).isNull() | (F.col(col_name) == ""), None).otherwise(F.col(col_name))
        ).withColumn(
            col_name,
            F.col(col_name).cast(column_type_map[col_name])
        )

    # ===== CAST REMAINING COLUMNS =====
    for column, new_type in column_type_map.items():
        if column not in numeric_columns and column != "snapshot_date":
            df = df.withColumn(column, F.col(column).cast(new_type))
    
    # Cast snapshot_date
    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

    # ===== DATA QUALITY RULES =====
    # Values that are present but violate business logic are nulled out rather than
    # flagged. The original invalid value is preserved in Bronze for audit purposes.
    # Nulls flow through to the Gold layer where median imputation handles them
    # consistently with other missing values.
    #
    # Rules applied:
    #   - Annual_Income < 0          : impossible; income cannot be negative
    #   - Monthly_Balance < 0        : impossible; balance floor is zero
    #   - Credit_Utilization_Ratio > 100 : impossible; utilisation is bounded at 100%
    #   - Interest_Rate > 48         : exceeds 4% per month regulatory cap (48% per annum);
    #                                  source data contains erroneous values up to 5,789
    df = df.withColumn(
        "Annual_Income",
        F.when(F.col("Annual_Income") < 0, None).otherwise(F.col("Annual_Income"))
    ).withColumn(
        "Monthly_Balance",
        F.when(F.col("Monthly_Balance") < 0, None).otherwise(F.col("Monthly_Balance"))
    ).withColumn(
        "Credit_Utilization_Ratio",
        F.when(F.col("Credit_Utilization_Ratio") > 100, None).otherwise(F.col("Credit_Utilization_Ratio"))
    ).withColumn(
        "Interest_Rate",
        F.when(F.col("Interest_Rate") > 48, None).otherwise(F.col("Interest_Rate"))
    )

    # ===== SAVE TO SILVER LAYER =====
    # Save the cleaned data to the Silver layer as Parquet
    output_path = os.path.join(silver_financials_directory, f"snapshot_date={snapshot_date_str}")
    df.write.mode("overwrite").parquet(output_path)
    
    print(f"Successfully processed and saved data for {snapshot_date_str} to {output_path}")

    return df
