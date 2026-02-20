# ocr.py

import cv2
import re
from paddleocr import PaddleOCR

# initialize once
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def preprocess_for_ocr(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray


def read_full_image_text(image):
    """
    OCR on entire image.
    """

    processed = preprocess_for_ocr(image)
    result = ocr.ocr(processed, cls=True)

    texts = []

    if result is None:
        return texts

    for line in result:
        for word_info in line:
            text = word_info[1][0]
            conf = word_info[1][1]

            text = text.strip()

            if conf < 0.4:
                continue

            if len(text) < 2:
                continue

            texts.append(text)

    return texts

def filter_ic_candidates(texts):
    ic_candidates = []

    for text in texts:
        text = text.replace(" ", "").upper()

        # reject reference designators
        if re.match(r'^[RCUJ]\d+$', text):
            continue

        # allow only alphanumeric and dash
        if not re.match(r'^[A-Z0-9\-]+$', text):
            continue

        # must contain letters and digits
        if not any(c.isdigit() for c in text):
            continue

        if not any(c.isalpha() for c in text):
            continue

        # length constraint
        if len(text) < 3 or len(text) > 20:
            continue

        ic_candidates.append(text)

    return ic_candidates


def read_ic_text_from_image(image):
    """
    Full pipeline:
    OCR → filter → IC candidates
    """

    texts = read_full_image_text(image)
    ic_names = filter_ic_candidates(texts)

    return ic_names


def read_region_text(image, bbox):
    """
    OCR on a specific region (for later refinement).
    bbox: dict with x, y, w, h
    """

    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]

    crop = image[y:y+h, x:x+w]

    if crop.size == 0:
        return []

    processed = preprocess_for_ocr(crop)
    result = ocr.ocr(processed, cls=True)

    texts = []

    if result is None:
        return texts

    for line in result:
        for word_info in line:
            text = word_info[1][0]
            conf = word_info[1][1]

            text = text.strip()

            if conf < 0.4:
                continue

            if len(text) < 2:
                continue

            texts.append(text)

    return texts

def extract_reference_counts(texts):
    counts = {"R": 0, "C": 0, "U": 0, "J": 0}

    for t in texts:
        t = t.strip().upper()

        if len(t) < 2:
            continue

        prefix = t[0]

        if prefix in counts and t[1:].isdigit():
            counts[prefix] += 1

    return counts