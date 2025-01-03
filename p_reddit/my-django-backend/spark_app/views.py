from django.http import JsonResponse
from .spark_session import SparkSession

def word_count(request):
    result = SparkSession.word_count()
    return JsonResponse({'word_count': result})

def evaluate_text(request):
    text = request.GET.get('text', '')
    result = SparkSession.evaluate_text(text)
    return JsonResponse({'evaluation': result})

def depressed_words(request):
    result = SparkSession.depressed_words()
    return JsonResponse({'depressed_words': result})

def post_frequency(request):
    result = SparkSession.post_frequency()
    return JsonResponse({'post_frequency': result})