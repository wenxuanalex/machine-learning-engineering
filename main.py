import os

import pyspark

if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

spark = pyspark.sql.SparkSession.builder \
    .appName("Pipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
