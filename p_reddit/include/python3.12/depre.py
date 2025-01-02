from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import os

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
    rdd_list = [spark.read.format("xml").option("rowTag", "INDIVIDUAL").schema(schema).load(f).rdd for f in xml_files]
    rdd = rdd_list[0]
    for temp_rdd in rdd_list[1:]:
        rdd = rdd.union(temp_rdd)
    
    rdd = rdd.flatMap(lambda row: [(row.ID, writing.TITLE, writing.DATE, writing.TEXT) for writing in row.WRITING])
    rdd.cache()  # Cache the RDD for reuse
    rdd.foreach(print)
    return rdd

def word_count(rdd):
    rdd = rdd.repartition(20)
    words_rdd = rdd.flatMap(lambda row: [(row[0], word) for word in row[3].split()])
    word_count_rdd = words_rdd.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a, b: a + b)
    word_count_rdd.foreach(print)
    return word_count_rdd

# Initialize Spark session
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
result_rdds = [word_count(load_data(spark, chunk)) for chunk in chunks]

# Union all results into a single RDD
final_rdd = result_rdds[0]
for rdd in result_rdds[1:]:
    final_rdd = final_rdd.union(rdd)

# Convert final RDD to DataFrame and write to CSV
final_df = final_rdd.map(lambda x: (x[0][0], x[0][1], x[1])).toDF(["ID", "word", "count"])
final_df.show()
final_df.write.csv("/home/marco/Desktop/reddit_depression/p_reddit/result/output.csv")




# Simulazione della funzione MapReduce che restituisce i dati
# (id_persona, termini). Sarebbero i file che fuoriescono dalla mapreduce che dovevi veder tu
data_terms = spark.createDataFrame([
    ("1", ["sad", "tired"]),
    ("2", ["happy", "excited"]),
    ("3", ["lonely", "hopeless"]),
    ("4", ["content", "joyful"]),
    ("5", ["anxious", "worried"]),
    ("6", ["energetic", "productive"])
], ["id_persona", "termini"])

# Simulazione del file con le label (id_persona, label). Sarebbero i dati che dobbiamo leggere dal file dei depressi e non
data_labels = spark.createDataFrame([
    ("1", 1),
    ("2", 0),
    ("3", 1),
    ("4", 0),
    ("5", 1),
    ("6", 0)
], ["id_persona", "label"])

# Join dei dati su id_persona. Qui uniamo in base all'id sia i termini che se è depresso o meno
data = data_terms.join(data_labels, on="id_persona")

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

# Feature Extraction
hashing_tf = HashingTF(inputCol="termini", outputCol="raw_features") #converte il vettore di termini un uno numerico basato sullòa frequenza dei termini
idf = IDF(inputCol="raw_features", outputCol="features") #diamo più peso ai termini pù frequenti e meno peso agli altri
label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")

# Random Forest Classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="indexed_label", numTrees=100, maxDepth=10, seed=42)

# Creazione della pipeline
pipeline = Pipeline(stages=[hashing_tf, idf, label_indexer, rf])

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
model.save("depression_model")