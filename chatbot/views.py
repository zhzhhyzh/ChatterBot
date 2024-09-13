from django.shortcuts import render

# Create your views here.
from .chatbot import generate_response, get_choices, recommend

def chatbot_home(request):
    return render(request, 'index.html')

from django.http import HttpResponse, JsonResponse
def generate_response_view(request):
    user_input = request.GET.get('msg')
    context = request.GET.get('context')
    response = generate_response(user_input, context)
    return HttpResponse(str(response))

def get_choices_view(request):
    context = request.GET["context"]
    response = get_choices(context)
    return JsonResponse(response, safe=False)
def get_recommendations_view(request):
    ram = request.GET.get("ram")
    storage = request.GET.get("storage")
    max_price = request.GET.get("max_price")
    response = recommend(ram,storage,max_price)
    return JsonResponse(response, safe=False)