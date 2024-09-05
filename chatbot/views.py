from django.shortcuts import render

# Create your views here.
from .chatbot import generate_response

def chatbot_home(request):
    return render(request, 'index.html')

from django.http import HttpResponse

def get_response(request):
    user_input = request.GET.get('msg')
    context = request.GET.get('context')
    response = generate_response(user_input, context)
    return HttpResponse(str(response))