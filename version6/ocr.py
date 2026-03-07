# ocr.py

import cv2
import re
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import tempfile
import os

# initialize once
ocr = ocr_predictor(pretrained=True)


def preprocess_for_ocr(img):
    if img is None:
        return None

    try:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
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


def _doctr_extract_words(image):

    texts = []

    try:
        # create temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name

        # save the image
        cv2.imwrite(temp_path, image)

        # let doctr read from file path
        doc = DocumentFile.from_images(temp_path)
        result = ocr(doc)
        data = result.export()

        os.remove(temp_path)

    except Exception:
        return texts

    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):

                    text = safe_text_clean(word.get("value", ""))

                    if not text:
                        continue

                    if len(text) < 2:
                        continue

                    texts.append(text)

    return texts

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
        texts = _doctr_extract_words(processed)
    except Exception:
        return []

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
        texts = _doctr_extract_words(processed)
    except Exception:
        return []

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

if __name__ == "__main__":

    image_path = "pcbclear2.jpg"

    print("\n--- LOADING IMAGE ---")
    img = cv2.imread(image_path)

    if img is None:
        print("Failed to load image:", image_path)
        exit()

    print("\n--- RUNNING OCR ---")
    texts = read_full_image_text(img)

    print("\n--- OCR RAW TEXT (first 50) ---")
    print(texts[:50])

    print("\n--- OCR REFERENCE COUNTS ---")
    counts = extract_reference_counts(texts)
    print(counts)

    print("\n--- OCR IC CANDIDATES ---")
    ic_names = filter_ic_candidates(texts)
    print(ic_names[:20])

    print("\n--- DONE ---")