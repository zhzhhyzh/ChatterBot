from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_home, name='home'),
    path('generate_response/', views.generate_response_view, name='get'),
    path('get_choices/', views.get_choices_view, name='get'),
    path('get_recommendations/', views.get_recommendations_view, name='get'),
]