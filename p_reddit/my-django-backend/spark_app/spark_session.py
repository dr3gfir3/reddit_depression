from pyspark.sql import SparkSession
import p_reddit.code.p
class SparkSessionManager:
    _spark = None
    MODEL_NAME="depression_model"

    @classmethod
    def get_spark_session(cls):
        if cls._spark is None:
            cls._spark =p_reddit.code.p.SparkXMLProcessor("/home/marco/Desktop/reddit_depression/p_reddit/dataset_revised/")
            cls._perform_initial_operations()
        return cls._spark

    @classmethod
    def _perform_initial_operations(cls):
         
         result=cls._spark.word_count(cls._spark.df)
         cls._spark.model_creation_and_training(result)
    @classmethod
    def word_count(cls):
        cls._spark.word_count()

    @classmethod
    def load_model(cls):
        # Load the model from the specified path
        cls._model = cls._spark.load_model(cls.MODEL_NAME)

    @classmethod
    def evaluate_text(cls, text):
        # Ensure the model is loaded
        if not hasattr(cls, '_model'):
            cls.load_model()
        # Create a DataFrame from the input text
        input_df = cls._spark.createDataFrame([(text,)], ["text"])
        # Transform the input text using the model's pipeline
        transformed_df = cls._model.transform(input_df)
        # Select the prediction and probability from the transformed DataFrame
        prediction_row = transformed_df.select("prediction", "probability").collect()[0]
        prediction = prediction_row["prediction"]
        probability = prediction_row["probability"]
        reliability = max(probability)
        return prediction, reliability
    

# Initialize Spark session when the server starts
spark_session = SparkSessionManager.get_spark_session()