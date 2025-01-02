from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import os
from pyspark.sql.functions import explode, split, col

def create_schema():
    return StructType([
        StructField("ID", StringType(), True),
        StructField("WRITING", ArrayType(StructType([
            StructField("TITLE", StringType(), True),
            StructField("DATE", StringType(), True),
            StructField("TEXT", StringType(), True)
        ])), True)
    ])

def load_data(spark, chunk):
    schema = create_schema()
    xml_files = [os.path.join(chunk, f) for f in os.listdir(chunk) if f.endswith(".xml")]
    
    # Read XML files in parallel and union them
    df_list = [spark.read.format("xml").option("rowTag", "INDIVIDUAL").schema(schema).load(f) for f in xml_files]
    df = df_list[0]
    for temp_df in df_list[1:]:
        df = df.union(temp_df)
    
    df = df.withColumn("WRITING", explode(df.WRITING))
    df = df.select("ID", "WRITING.TITLE", "WRITING.DATE", "WRITING.TEXT")
    df.cache()  # Cache the DataFrame for reuse
    df.show()
    return df

def word_count(df):
    df = df.repartition(20)
    words_df = df.withColumn("word", explode(split(col("TEXT"), "\\s+")))
    word_count_df = words_df.groupBy("ID", "word").count()
    word_count_df.show()
    return word_count_df

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
result_dfs = [word_count(load_data(spark, chunk)) for chunk in chunks]

# Union all results into a single DataFrame
final_df = result_dfs[0]
for df in result_dfs[1:]:
    final_df = final_df.union(df)

# Write the final result to CSV
final_df.show()
