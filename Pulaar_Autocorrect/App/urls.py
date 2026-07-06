from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('correct/', views.correct, name='correct'),
    path('health/', views.health, name='health'),
]
