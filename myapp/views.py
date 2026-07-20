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

# Create a function to talk to Gemini with automatic model fallback
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

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

    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            logging.info(f"Successfully used model: {model_name}")
            return response.text
        except Exception as e:
            logging.warning(f"Model {model_name} failed: {e}")
            continue

    return "All AI models are currently at their daily limit. Please try again tomorrow."

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def image_classification_view(request): 
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        resultcnn = classify_image(image)
        image.seek(0)
        resultocr = ocrText(image) or {}
        image.seek(0)  # prevent None crash
        resultsentimentFace = detect_face_sentiment(image)

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