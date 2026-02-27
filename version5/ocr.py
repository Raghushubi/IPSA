# ocr.py

import cv2
import re
from paddleocr import PaddleOCR

# initialize once
ocr = PaddleOCR(use_angle_cls=True, lang='en')


def preprocess_for_ocr(img):
    if img is None:
        return None

    try:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray
    except Exception:
        return None


def safe_text_clean(text):
    """
    Remove problematic characters (prevents encoding issues).
    """
    try:
        return text.encode("ascii", "ignore").decode().strip()
    except Exception:
        return ""


def read_full_image_text(image):
    """
    OCR on entire image.
    Always returns a list.
    """

    texts = []

    if image is None:
        return texts

    processed = preprocess_for_ocr(image)
    if processed is None:
        return texts

    try:
        result = ocr.ocr(processed, cls=True)
    except Exception:
        return texts

    if result is None:
        return texts

    for line in result or []:
        if not line:
            continue

        for word_info in line:
            try:
                text = word_info[1][0]
                conf = word_info[1][1]
            except Exception:
                continue

            text = safe_text_clean(text)

            if not text:
                continue

            if conf < 0.4:
                continue

            if len(text) < 2:
                continue

            texts.append(text)

    return texts


def filter_ic_candidates(texts):
    ic_candidates = []

    if not texts:
        return ic_candidates

    for text in texts:
        if not isinstance(text, str):
            continue

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

    return ic_names or []


def read_region_text(image, bbox):
    """
    OCR on a specific region
    bbox: dict with x, y, w, h
    Always returns a list.
    """

    texts = []

    if image is None or bbox is None:
        return texts

    try:
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
    except Exception:
        return texts

    if w <= 0 or h <= 0:
        return texts

    crop = image[y:y + h, x:x + w]

    if crop is None or crop.size == 0:
        return texts

    processed = preprocess_for_ocr(crop)
    if processed is None:
        return texts

    try:
        result = ocr.ocr(processed, cls=True)
    except Exception:
        return texts

    if result is None:
        return texts

    for line in result or []:
        if not line:
            continue

        for word_info in line:
            try:
                text = word_info[1][0]
                conf = word_info[1][1]
            except Exception:
                continue

            text = safe_text_clean(text)

            if not text:
                continue

            if conf < 0.4:
                continue

            if len(text) < 2:
                continue

            texts.append(text)

    return texts


def extract_reference_counts(texts):
    counts = {"R": 0, "C": 0, "U": 0, "J": 0}

    if not texts:
        return counts

    for t in texts:
        if not isinstance(t, str):
            continue

        t = safe_text_clean(t).upper()

        if len(t) < 2:
            continue

        prefix = t[0]

        if prefix in counts and t[1:].isdigit():
            counts[prefix] += 1

    return counts