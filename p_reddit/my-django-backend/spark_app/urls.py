from django.urls import path
from . import views

urlpatterns = [
    path('word_count/', views.word_count, name='word_count'),
    path('evaluate_text/', views.evaluate_text, name='evaluate_text'),
    path('depressed_words/', views.depressed_words, name='depressed_words'),
    path('post_frequency/', views.post_frequency, name='post_frequency'),
]