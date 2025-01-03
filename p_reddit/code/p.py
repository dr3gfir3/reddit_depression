import os
from pandas import DataFrame
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import explode, split, col, collect_list, concat_ws, regexp_replace, sum, date_format
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StringIndexer, Tokenizer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

class SparkXMLProcessor:
   
    def __init__(self, input_dir):
        self.spark = SparkSession.builder \
            .appName("SparkXMLProcessor") \
            .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.14.0") \
            .config("spark.executor.memory", "14g") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.autoBroadcastJoinThreshold", -1) \
            .getOrCreate()
        
        self.schema = StructType([
            StructField("ID", StringType(), True),
            StructField("TITLE", StringType(), True),
            StructField("DATE", TimestampType(), True),
            StructField("TEXT", StringType(), True)
        ])
        
        self.input_dir = input_dir
        self.df = self._load_data()

    def _load_data(self):
        xml_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith('.xml')]
        df = self.spark.createDataFrame([], schema=self.schema)
        
        for xml_file in xml_files:
            temp_df = self.spark.read.format("xml").options(rowTag="WRITING", timestampFormat="yyyy-MM-dd HH:mm:ss").schema(self.schema).load(xml_file)
            df = df.union(temp_df) if df.head(1) else temp_df
        
        return df

    def word_count(self):
        df:DataFrame = self.df.repartition(10)
        df = df.withColumn("TEXT", regexp_replace(col("TEXT"), "[^a-zA-Z\\s]", ""))
        df = df.withColumn("TITLE", regexp_replace(col("TITLE"), "[^a-zA-Z\\s]", ""))
        text_words_df = df.withColumn("word", explode(split(col("TEXT"), "\\s+")))  
        title_words_df = df.withColumn("word", explode(split(col("TITLE"), "\\s+")))
        words_df = text_words_df.union(title_words_df)
        words_df = words_df.filter(words_df.word != "")
        word_count_df = words_df.groupBy("ID", "word").count()
        word_count_df = word_count_df.filter(word_count_df['count'] >= 5)
        return word_count_df

    def model_creation_and_training(self, result_df):
        output_path = "/home/marco/Desktop/reddit_depression/p_reddit/output/result"
        result_df.write.csv(output_path, header=True, mode='overwrite')

        data_labels = self.spark.read.csv("/home/marco/Desktop/reddit_depression/p_reddit/dataset/risk-golden-truth-test.txt", sep="\t", schema=StructType([
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
    #    train = train_depressi.union(train_non_depressi)
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

    def rank_most_used_words_depressi(self):
        # Get the word count dataframe
        word_count_df = self.df
        
        # Read the labels
        data_labels = self.spark.read.csv("/home/marco/Desktop/reddit_depression/p_reddit/dataset/risk-golden-truth-test.txt", sep="\t", schema=StructType([
            StructField("ID", StringType(), True),
            StructField("label", IntegerType(), True)
        ]))
        
        # Join word count with labels
        data = word_count_df.join(data_labels, on="ID")
        
        # Filter for depressi
        depressi_words = data.filter(col("label") == 1)
        
        # Group by word and count occurrences
        word_rank_df = depressi_words.groupBy("word").agg(sum("count").alias("total_count"))
        
        # Order by total count in descending order
        word_rank_df = word_rank_df.orderBy(col("total_count").desc())
        
        return word_rank_df

    def analyze_post_frequency(self):
        # Load the labels
        data_labels = self.spark.read.csv("/home/marco/Desktop/reddit_depression/p_reddit/dataset/risk-golden-truth-test.txt", sep="\t", schema=StructType([
            StructField("ID", StringType(), True),
            StructField("label", IntegerType(), True)
        ]))
        
        # Join the labels with the main DataFrame
        data = self.df.join(data_labels, on="ID")
        
        # Filter to include only posts with label set to 1
        depressed_posts = data.filter(col("label") == 1)
        
        # Group by DATE and count the number of posts per day
        post_frequency_df = depressed_posts.groupBy(date_format(col("DATE"), "yyyy-MM-dd").alias("date")).count()
        post_frequency_df = post_frequency_df.orderBy("date")
        
        return post_frequency_df


# Example usage:
processor = SparkXMLProcessor("/home/marco/Desktop/reddit_depression/p_reddit/dataset_revised/")
result_df = processor.word_count()

processor.model_creation_and_training(processor, result_df)
# Show the result
result_df.show()



# Stop the Spark session
processor.spark.stop()