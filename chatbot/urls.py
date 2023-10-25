from django.urls import path
from . import views

urlpatterns = [
    path('', views.transcribe_audio, name='transcribe_audio'),
]
