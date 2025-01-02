import os
from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType
from pyspark.sql.functions import explode, split, col, collect_list, concat_ws, regexp_replace
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, Tokenizer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

# Initialize Spark session with provided configuration parameters
spark = SparkSession.builder \
    .appName("XML Reader") \
    .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.10.0") \
    .config("spark.executor.memory", "14g") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.autoBroadcastJoinThreshold", -1) \
    .getOrCreate()

# Define the schema
schema = StructType([
    StructField("ID", StringType(), True),
    StructField("TITLE", StringType(), True),
    StructField("DATE", DateType(), True),
    StructField("TEXT", StringType(), True)
])

# Directory containing XML files
input_dir = "/home/marco/Desktop/reddit_depression/p_reddit/dataset_revised/"

# List all XML files in the directory
xml_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.xml')]

# Initialize an empty DataFrame
df = spark.createDataFrame([], schema=schema)

# Read each XML file into DataFrame and union them
for xml_file in xml_files:
    temp_df = spark.read.format("xml").options(rowTag="WRITING").schema(schema).load(xml_file)
    df = df.union(temp_df) if df.head(1) else temp_df

#df.show()


def word_count(df):
    df = df.repartition(10)
    
    # Remove numbers, spaces, and punctuation from TEXT and TITLE columns
    df = df.withColumn("TEXT", regexp_replace(col("TEXT"), "[^a-zA-Z\\s]", ""))
    df = df.withColumn("TITLE", regexp_replace(col("TITLE"), "[^a-zA-Z\\s]", ""))
    
    text_words_df = df.withColumn("word", explode(split(col("TEXT"), "\\s+")))  
    title_words_df = df.withColumn("word", explode(split(col("TITLE"), "\\s+")))
    
    # Union the two DataFrames
    words_df = text_words_df.union(title_words_df)
    
    # Filter out empty words
    words_df = words_df.filter(words_df.word != "")
    
    word_count_df = words_df.groupBy("ID", "word").count()
    
    return word_count_df

# Perform word count
print(df.rdd.getNumPartitions())
result_df = word_count(df)

# Show the result
result_df.show()

# Write the result to a CSV file
output_path = "/home/marco/Desktop/reddit_depression/p_reddit/output/result"
result_df.write.csv(output_path, header=True, mode='overwrite')

data_labels = spark.read.csv("/home/marco/Desktop/reddit_depression/p_reddit/dataset/risk-golden-truth-test.txt", sep="\t", schema=StructType([
    StructField("ID", StringType(), True),
    StructField("label", IntegerType(), True)
]))
data = result_df.join(data_labels, on="ID")
data.show()

# Separare depressi e non depressi
depressi = data.filter(col("label") == 1)
non_depressi = data.filter(col("label") == 0)

# Campionamento stratificato per training e test set
train_depressi, test_depressi = depressi.randomSplit([0.8, 0.2], seed=42)
train_non_depressi, test_non_depressi = non_depressi.randomSplit([0.8, 0.2], seed=42)

# Combina i training e test set bilanciati
train = train_depressi.union(train_non_depressi)
test = test_depressi.union(test_non_depressi)

# Bilanciamento del training set se ci va di farlo oppure lasciamo stare
count_depressi = train_depressi.count()
count_non_depressi = train_non_depressi.count()
ratio = count_depressi / count_non_depressi

if ratio > 1:
    sampled_non_depressi = train_non_depressi.sample(withReplacement=True, fraction=ratio, seed=42)
    balanced_train = train_depressi.union(sampled_non_depressi)
else:
    sampled_depressi = train_depressi.sample(withReplacement=True, fraction=1/ratio, seed=42)
    balanced_train = train_non_depressi.union(sampled_depressi)

# Tokenize the text data
tokenizer = Tokenizer(inputCol="word", outputCol="words")

# Feature Extraction
hashing_tf = HashingTF(inputCol="words", outputCol="raw_features") #converte il vettore di termini un uno numerico basato sullòa frequenza dei termini
idf = IDF(inputCol="raw_features", outputCol="features") #diamo più peso ai termini pù frequenti e meno peso agli altri
label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")

# Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="indexed_label")

# Creazione della pipeline
pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf, label_indexer, lr])

# Allenamento del modello
model = pipeline.fit(balanced_train)

# Predizioni sul test set
predictions = model.transform(test)

# Valutazione del modello
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexed_label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

# Salvare il modello
model.write().overwrite().save("depression_model")

# Stop the Spark session
spark.stop()