from pyspark.sql import SparkSession
import p_reddit.code.p
class SparkSessionManager:
    _spark = None

    @classmethod
    def get_spark_session(cls):
        if cls._spark is None:
            cls._spark =p_reddit.code.p.SparkXMLProcessor("/home/marco/Desktop/reddit_depression/p_reddit/dataset_revised/")
            cls._perform_initial_operations()
        return cls._spark

    @classmethod
    def _perform_initial_operations(cls):
         cls._spark._load_data()
         cls._spark.word_count()
         cls._spark.model_creation_and_training()
       

# Initialize Spark session when the server starts
spark_session = SparkSessionManager.get_spark_session()