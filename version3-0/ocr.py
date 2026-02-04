# ocr.py

import easyocr
import re
import cv2

reader = easyocr.Reader(['en'], verbose=False)

def read_ic_text(image):
    results = reader.readtext(image)

    texts = []

    for box, text, conf in results:
        text = text.strip()
        text = text.replace("IC ", "")

        if conf < 0.4:
            continue

        if len(text) < 3:
            continue

        if not any(c.isdigit() for c in text):
            continue
        
        if not re.match(r'^[A-Za-z0-9\-]+$', text):
            continue
        
        texts.append(text)

    return texts

def preprocess_for_ocr(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

