from rest_framework.decorators import api_view

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.http import JsonResponse
import os
from .ml_models.ImageCNN import classify_image  # assuming your CNN is here
from .ml_models.OCR import ocrText
from .ml_models.sentiment import detect_face_sentiment
from django.core.files.storage import default_storage
from rest_framework.decorators import parser_classes
from rest_framework.parsers import MultiPartParser

import logging 
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

api_key = os.getenv("GOOGLE_API_KEY")

# Configure your API key
genai.configure(api_key=api_key)

models = genai.list_models()

for m in models:
    print(m.name)

# Create a function to talk to Gemini
def explain_image_content(cnn_result, ocr_result, face_sentiment_result):
    prompt = f"""
    An image has been analyzed with the following results:

    - Detected object (CNN): {cnn_result}
    - Extracted Text (OCR): {ocr_result.get('text')}
    - Emotion from text: {ocr_result.get('emotions')}
    - Facial emotion detected: {face_sentiment_result}

    Based on these details, describe what is happening in the image.
    Decide whether it is good or bad content for the user. Explain why and say whether they should consume it.
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)  # ← FIXED (removed list)
    return response.text

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def image_classification_view(request): 
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        path = default_storage.save(image.name, image)

        resultcnn = classify_image(path)
        resultocr = ocrText(path) or {}  # prevent None crash
        resultsentimentFace = detect_face_sentiment(path)

        gemini_summary = explain_image_content(
            resultcnn,
            resultocr,
            resultsentimentFace
        )
       

        return JsonResponse({ 
            'gemini_summary': gemini_summary
        })

    return JsonResponse({'error': 'No image uploaded'}, status=400)


def home(request):
    return HttpResponse("Server is running...")