import pytesseract
from PIL import Image
from transformers import pipeline

# Set path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the emotion classifier once (you can also move this outside the function if needed)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

from PIL import Image

def ocrText(image_file):
    # Convert uploaded file → Pillow image
    image = Image.open(image_file)

    # Extract text using OCR
    text = pytesseract.image_to_string(image)

    # Run emotion classifier on extracted text
    results = emotion_classifier(text)[0]
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return {
        'text': text,
        'emotions': results
    }

