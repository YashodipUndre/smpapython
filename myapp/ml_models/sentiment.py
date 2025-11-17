from deepface import DeepFace
import cv2

import numpy as np
from PIL import Image

def detect_face_sentiment(image_file):
    # Convert uploaded file to Pillow image
    pil_img = Image.open(image_file).convert('RGB')
    
    # Convert Pillow image to numpy array
    img_np = np.array(pil_img)

    # Convert RGB (PIL) → BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # DeepFace analysis directly on numpy array
    analysis = DeepFace.analyze(
        img_path=img_bgr,
        actions=['emotion'],
        enforce_detection=False
    )

    return analysis[0]['dominant_emotion']
