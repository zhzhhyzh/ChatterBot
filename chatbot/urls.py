from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_home, name='home'),
    path('get/', views.get_response, name='get'),
]