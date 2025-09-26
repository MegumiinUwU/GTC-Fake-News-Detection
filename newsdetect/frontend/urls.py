from django.urls import path
from .views import chat_view, predict_view


urlpatterns = [
    path('', chat_view, name='chat'),
    path('predict/', predict_view, name='predict'),
]


