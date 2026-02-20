# cv_pipeline.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

def run_cv(image_path: str) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    object_type, mask, cropped = detect_main_object(image)
    
    if object_type != "PCB":
        return {
            "object_type": object_type,
            "visualization": image,
            "components": []
        }
    
    preprocessed = preprocess_pcb(cropped, mask)
    candidates = detect_component_candidates(preprocessed, mask)
    merged = merge_boxes(candidates)
    filtered = filter_components(merged, mask)
    
    components = extract_features(filtered, preprocessed, cropped)
    components = heuristic_classification(components)
    components = mark_ic_candidates(components)
    visualization = visualize_components(cropped, components)
    
    return {
        "object_type": object_type,
        "visualization": visualization,
        "components": components
    }


def detect_main_object(image: np.ndarray) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "UNKNOWN", None, None
    
    largest = max(contours, key=cv2.contourArea)
    
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    is_convex = cv2.isContourConvex(approx)
    vertex_count = len(approx)
    
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest)
    area_ratio = contour_area / image_area
    
    if not is_convex or vertex_count < 4 or area_ratio < 0.1:
        return "UNKNOWN", None, None
    
    hull = cv2.convexHull(approx)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0
    
    if solidity > 0.85:
        object_type = "PCB"
    else:
        return "UNSUPPORTED", None, None
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(largest)
    cropped = image[y:y+h, x:x+w].copy()
    mask_cropped = mask[y:y+h, x:x+w].copy()
    
    return object_type, mask_cropped, cropped


def preprocess_pcb(pcb_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(pcb_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    
    normalized = np.zeros_like(gray, dtype=np.float32)
    mask_bool = mask > 0
    
    normalized[mask_bool] = gray[mask_bool].astype(np.float32) - blurred[mask_bool].astype(np.float32)
    
    if normalized[mask_bool].max() > normalized[mask_bool].min():
        min_val = normalized[mask_bool].min()
        max_val = normalized[mask_bool].max()
        normalized[mask_bool] = 255 * (normalized[mask_bool] - min_val) / (max_val - min_val)
    
    return normalized.astype(np.uint8)


def detect_component_candidates(preprocessed: np.ndarray, mask: np.ndarray):
    edges = cv2.Canny(preprocessed, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w < 5 or h < 5:
            continue
        
        boxes.append((x, y, w, h))
    
    return boxes


def merge_boxes(boxes):
    def boxes_close(b1, b2, thresh=20):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        cx1 = x1 + w1/2
        cy1 = y1 + h1/2
        cx2 = x2 + w2/2
        cy2 = y2 + h2/2
        return np.hypot(cx1-cx2, cy1-cy2) < thresh

    def boxes_overlap(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (
            x1+w1 < x2 or x2+w2 < x1 or
            y1+h1 < y2 or y2+h2 < y1
        )

    groups = []
    used = [False]*len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        group = [boxes[i]]
        used[i] = True

        for j in range(i+1, len(boxes)):
            if used[j]:
                continue
            if boxes_close(boxes[i], boxes[j]) or boxes_overlap(boxes[i], boxes[j]):
                group.append(boxes[j])
                used[j] = True

        groups.append(group)

    merged = []
    for group in groups:
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        xe = [b[0]+b[2] for b in group]
        ye = [b[1]+b[3] for b in group]
        merged.append((min(xs), min(ys), max(xe)-min(xs), max(ye)-min(ys)))

    return merged


def filter_components(boxes: List[Tuple[int, int, int, int]], mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    
    pcb_area = np.sum(mask > 0)
    areas = [w * h for x, y, w, h in boxes]
    median_area = np.median(areas)
    
    filtered = []
    for box in boxes:
        x, y, w, h = box
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        if area < 0.2 * median_area or area > 5 * median_area:
            continue
        
        if area > 0.5 * pcb_area:
            continue
        
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            continue
        
        filtered.append(box)
    
    return filtered


def extract_features(
    boxes: List[Tuple[int, int, int, int]],
    preprocessed: np.ndarray,
    pcb_image: np.ndarray
) -> List[Dict]:
    
    h_img, w_img = preprocessed.shape[:2]
    total_area = h_img * w_img
    
    components = []
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        area = w * h
        
        region_gray = preprocessed[y:y+h, x:x+w]
        region_color = pcb_image[y:y+h, x:x+w]
        
        if region_gray.size == 0:
            continue
        
        edges = cv2.Canny(region_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        
        intensity_std = float(np.std(region_gray))
        color_std = float(np.std(region_color))
        
        _, thresh = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fill_ratio = np.sum(thresh > 0) / (w * h)
        
        if area < 0.0005 * total_area:
            size = "tiny"
        elif area < 0.005 * total_area:
            size = "small"
        elif area < 0.02 * total_area:
            size = "medium"
        else:
            size = "large"
        
        component = {
            "id": int(i),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "area": int(area),
            "normalized_area": float(area / total_area),
            "aspect_ratio": float(w / h if h > 0 else 0),
            "centroid": {
                "x": float((x + w/2) / w_img),
                "y": float((y + h/2) / h_img)
            },
            "mean_intensity": float(np.mean(region_gray)),
            "intensity_std": intensity_std,
            "color_std": color_std,
            "edge_density": edge_density,
            "fill_ratio": fill_ratio,
            "size": size,
            "type": None,
            "confidence": None,
            "ocr_text": None
        }
        
        components.append(component)
    
    return components

def mark_ic_candidates(components):
    if not components:
        return components

    # sort by area descending
    sorted_comps = sorted(components, key=lambda x: x["area"], reverse=True)

    # top 5% or at least 3 components
    k = max(3, int(0.05 * len(components)))

    for i, comp in enumerate(sorted_comps):
        if i < k:
            comp["type"] = "IC"
            comp["confidence"] = 0.6

    return components

def heuristic_classification(components: List[Dict]) -> List[Dict]:
    for comp in components:
        ar = comp["aspect_ratio"]
        size = comp["size"]
        edges = comp["edge_density"]
        
        if size == "large" and 0.7 < ar < 2:
            comp["type"] = "IC"
            comp["confidence"] = 0.6
        
        elif size in ["small", "medium"] and ar > 2.5:
            comp["type"] = "resistor"
            comp["confidence"] = 0.5
        
        elif size in ["small", "medium"] and ar < 1.5 and edges < 0.3:
            comp["type"] = "capacitor"
            comp["confidence"] = 0.4
        
        else:
            comp["type"] = "unknown"
            comp["confidence"] = 0.2
    
    return components


def visualize_components(pcb_image: np.ndarray, components: List[Dict]) -> np.ndarray:
    vis = pcb_image.copy()
    
    for comp in components:
        bbox = comp["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        label = comp["type"]
        
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return vis

if __name__ == "__main__":
    res = run_cv("pcbclear2.jpg")
    print(res)
