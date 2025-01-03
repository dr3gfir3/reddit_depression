from django.urls import path
from .views import WordCountView

urlpatterns = [
    path('word-count/', WordCountView.as_view(), name='word_count'),
]