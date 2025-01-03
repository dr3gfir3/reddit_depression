from django.http import JsonResponse
from django.views import View
from spark_app.spark_session import create_spark_session
from spark_app.word_count import get_word_count

class WordCountView(View):
    def get(self, request):
        # Create Spark session
        spark = create_spark_session()
        
        # Retrieve word count information
        word_count = get_word_count(spark)
        
        # Return the word count as a JSON response
        return JsonResponse({'word_count': word_count})