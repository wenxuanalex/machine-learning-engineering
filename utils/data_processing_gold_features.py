import pyspark.sql.functions as F
from pyspark.sql import Window

def _as_of_join(loans_df, feature_df, feature_name):
    """
    For each loan row, attach the most recent feature record for that customer
    where feature_date <= loan snapshot_date (as-of / last-known-value join).

    Used for financials and attributes, which are monthly samples (~500 rows/month)
    rather than full refreshes. An exact snapshot_date join produces a 0% match
    rate because a customer's credit bureau date rarely coincides with their loan
    MOB snapshot date.

    A fin_lag_days audit column is added so downstream consumers can assess
    feature staleness without re-deriving it.
    """
    feature_df = feature_df.withColumnRenamed("snapshot_date", "feature_date")

    joined = loans_df.join(
        feature_df,
        on=(loans_df["Customer_ID"] == feature_df["Customer_ID"]) &
           (feature_df["feature_date"] <= loans_df["snapshot_date"]),
        how="left"
    ).drop(feature_df["Customer_ID"])

    w = Window.partitionBy(
        loans_df["Customer_ID"], loans_df["snapshot_date"]
    ).orderBy(F.col("feature_date").desc())

    joined = (
        joined
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
        .withColumn(
            f"{feature_name}_lag_days",
            F.datediff(loans_df["snapshot_date"], F.col("feature_date"))
        )
        .drop("feature_date")
    )
    print(f"As-of join ({feature_name}): "
          f"{joined.filter(F.col(f'{feature_name}_lag_days').isNotNull()).count():,} "
          f"of {loans_df.count():,} rows matched.")
    return joined


def process_gold_features(spark, silver_financials_directory, silver_loan_daily_directory, silver_clickstream_directory, silver_attributes_directory, gold_feature_store_directory):
    """
    Creates a gold feature store by joining silver financials, clickstream, attributes, and loan data,
    and applying feature engineering.

    Join strategy:
      - Clickstream : exact PIT join on [Customer_ID, snapshot_date] — full refresh every month.
      - Financials  : as-of join — most recent record per customer on or before snapshot_date.
      - Attributes  : as-of join — same rationale; demographics change slowly.
    """
    # Load silver tables
    financials_df = spark.read.parquet(silver_financials_directory)
    loans_df = spark.read.parquet(silver_loan_daily_directory)
    clickstream_df = spark.read.parquet(silver_clickstream_directory)
    attributes_df = spark.read.parquet(silver_attributes_directory)

    # --- 1. Feature Engineering on Financials Data ---

    # A. Parse Credit_History_Age to months using native Spark SQL functions.
    # F.regexp_extract returns "" (empty string) on no-match, which casts to null —
    # the same behaviour as the previous Python UDF without any JVM→Python overhead.
    financials_df = financials_df.withColumn(
        "Credit_History_Age_months",
        (
            F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Year", 1).cast("int") * 12
            + F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Month", 1).cast("int")
        )
    )

    # B. Ordinal encode Payment_Behaviour
    # Unknown / null values default to 2 (Standard — middle tier) rather than
    # silently producing NULL, which would propagate as a missing feature.
    financials_df = financials_df.withColumn(
        "Payment_Behaviour_encoded",
        F.when(F.col("Payment_Behaviour") == "Poor",      F.lit(1))
         .when(F.col("Payment_Behaviour") == "Standard",  F.lit(2))
         .when(F.col("Payment_Behaviour") == "Good",      F.lit(3))
         .when(F.col("Payment_Behaviour") == "Excellent", F.lit(4))
         .otherwise(F.lit(2))
    )

    # C. One-Hot Encode Credit_Mix
    credit_mix_types = [row['Credit_Mix'] for row in financials_df.select("Credit_Mix").distinct().collect() if row['Credit_Mix']]
    for mix_type in credit_mix_types:
        col_name = f"credit_mix_{mix_type.lower().replace(' ', '_')}"
        financials_df = financials_df.withColumn(
            col_name,
            F.when(F.col("Credit_Mix") == mix_type, 1).otherwise(0)
        )

    # D. Multi-label One-Hot Encode Type_of_Loan
    # Clean up the Type_of_Loan column
    financials_df = financials_df.withColumn(
        "Type_of_Loan_cleaned",
        F.regexp_replace(F.col("Type_of_Loan"), " and ", ",")
    )
    # Explode into individual loan types
    loan_types_df = financials_df.withColumn(
        "loan_type_single", 
        F.explode(F.split(F.col("Type_of_Loan_cleaned"), ",\\s*"))
    )
    # Get unique loan types
    all_loan_types = [row['loan_type_single'] for row in loan_types_df.select("loan_type_single").distinct().collect() if row['loan_type_single']]
    
    # Create binary columns
    for loan_type in all_loan_types:
        col_name = f'has_loan_{loan_type.lower().replace(" ", "_").replace("-", "_")}'
        financials_df = financials_df.withColumn(
            col_name,
            F.when(F.col("Type_of_Loan").contains(loan_type), 1).otherwise(0)
        )

    # --- 2. Feature Engineering on Loans Data ---

    # Loan age in days at the time of the snapshot.
    # Stored in Gold so consumers don't recompute it; also used to derive OOT
    # cutoff dates in model training without relying on observation-window heuristics.
    loans_df = loans_df.withColumn(
        "loan_age_days",
        F.datediff(F.col("snapshot_date"), F.col("loan_start_date"))
    )

    # --- 3. Join All Features with Loans ---
    # Clickstream: exact PIT join — data exists for every customer every month.
    # Financials / Attributes: as-of join — monthly samples; exact date match
    # produces 0% hit rate at MOB=6, so we use the most recent available record.
    feature_store_df = loans_df \
        .join(clickstream_df, ["Customer_ID", "snapshot_date"], "left")

    feature_store_df = _as_of_join(feature_store_df, financials_df, "financials")
    feature_store_df = _as_of_join(feature_store_df, attributes_df, "attributes")

    # --- 4. Engineered Ratio Features ---
    # Computed after the join so all required source columns are available.
    # Adding here means every downstream model reads pre-computed features from
    # a single authoritative source rather than re-implementing the same logic.
    #
    # +1 denominators prevent division-by-zero for new customers (mob=0, Num_of_Loan=0, etc.)
    feature_store_df = (
        feature_store_df
        .withColumn("emi_to_income",
            F.col("Total_EMI_per_month") / (F.col("Monthly_Inhand_Salary") + 1))
        .withColumn("debt_to_income",
            F.col("Outstanding_Debt") / (F.col("Annual_Income") + 1))
        .withColumn("debt_per_loan",
            F.col("Outstanding_Debt") / (F.col("Num_of_Loan") + 1))
        .withColumn("util_per_card",
            F.col("Credit_Utilization_Ratio") / (F.col("Num_Credit_Card") + 1))
        .withColumn("delay_frequency",
            F.col("installments_missed") / (F.col("mob") + 1))
    )

    # --- 5. Drop intermediate columns ---
    original_categoricals = ["Credit_History_Age", "Payment_Behaviour", "Credit_Mix", "Type_of_Loan", "Type_of_Loan_cleaned"]
    final_df = feature_store_df.drop(*original_categoricals)

    # --- 6. Median imputation for numeric feature columns ---
    # Identifiers, leakage columns, and string columns are excluded —
    # they are either dropped before modelling or not numeric.
    NON_IMPUTE_COLS = {
        "Customer_ID", "loan_id", "snapshot_date", "loan_start_date",
        "dpd", "overdue_amt", "installments_missed", "first_missed_date",
        "balance", "paid_amt",
        "SSN", "Occupation", "Payment_of_Min_Amount",
        "financials_lag_days", "attributes_lag_days",
    }
    NUMERIC_SPARK_TYPES = {"double", "float", "int", "bigint", "long", "integer", "short"}

    numeric_feature_cols = [
        col_name for col_name, dtype in final_df.dtypes
        if col_name not in NON_IMPUTE_COLS and dtype in NUMERIC_SPARK_TYPES
    ]

    # approxQuantile returns [] for fully-null columns — skip those safely
    medians = {}
    for col_name in numeric_feature_cols:
        result = final_df.approxQuantile(col_name, [0.5], 0.001)
        if result:
            medians[col_name] = result[0]

    final_df = final_df.na.fill(medians)
    print(f"Median imputation applied to {len(medians)} numeric columns.")

    # --- 7. Save to Gold Layer ---
    filepath = f"{gold_feature_store_directory}/feature_store"
    final_df.write.mode("overwrite").parquet(filepath)

    print(f"Gold feature store saved to: {filepath}")
    print(f"Total columns: {len(final_df.columns)}")
    
    return final_df
