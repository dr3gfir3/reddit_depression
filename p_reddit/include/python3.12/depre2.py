from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import os
from pyspark.sql.functions import explode, split, col
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

def create_schema():
    return StructType([
        StructField("ID", StringType(), True),
        StructField("WRITING", ArrayType(StructType([
            StructField("TITLE", StringType(), True),
            StructField("DATE", StringType(), True),
            StructField("TEXT", StringType(), True)
        ])), True)
    ])

def load_single_file(spark, schema, file_path):
    return spark.read.format("xml").option("rowTag", "INDIVIDUAL").schema(schema).load(file_path)

def process_file(spark, schema, file_path):
    df = load_single_file(spark, schema, file_path)
    df = df.withColumn("WRITING", explode(df.WRITING))
    df = df.select("ID", "WRITING.TITLE", "WRITING.DATE", "WRITING.TEXT")
    df = df.repartition(20)
    words_df = df.withColumn("word", explode(split(col("TEXT"), "\\s+")))
    word_count_df = words_df.groupBy("ID", "word").count()
    word_count_df.cache()
    return word_count_df

def load_and_process_files(spark, chunk):
    schema = create_schema()
    xml_files = [os.path.join(chunk, f) for f in os.listdir(chunk) if f.endswith(".xml")]
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        processed_dfs = list(executor.map(lambda f: process_file(spark, schema, f), xml_files))
    
    # Union all processed DataFrames
    final_df = reduce(lambda df1, df2: df1.union(df2), processed_dfs)
    final_df.cache()  # Cache the DataFrame for reuse
    return final_df

spark = SparkSession.builder \
    .appName("RedditDepressionAnalysis") \
    .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.14.0") \
    .config("spark.executor.memory", "12g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.autoBroadcastJoinThreshold", -1) \
    .getOrCreate()

path_base = "/home/marco/Desktop/reddit_depression/p_reddit/dataset"
chunks = [f"{path_base}/chunk{i}" for i in range(1, 11)]

# Process each chunk and collect results
result_dfs = [load_and_process_files(spark, chunk) for chunk in chunks]

# Union all results into a single DataFrame
final_df = reduce(lambda df1, df2: df1.union(df2), result_dfs)

# Write the final result to CSV
output_path = "/home/marco/Desktop/reddit_depression/p_reddit/output/final_result.csv"
final_df.write.csv(output_path, header=True)

# Show the final DataFrame
final_df.show()