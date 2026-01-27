import cv2
import numpy as np
import json

image = cv2.imread("pcbimagetrial.jfif")

print("Detecting main object...")

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# crude PCB check using green color
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

green_pixels = np.count_nonzero(green_mask)
total_pixels = img_gray.shape[0] * img_gray.shape[1]
green_ratio = green_pixels / total_pixels

if green_ratio > 0.3:
    object_type = "PCB"
else:
    object_type = "PCB"  # defaulting for now

print("Object:", object_type)

# isolate PCB using largest contour

ret, img_thresh = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV)
img_contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

maxarea = 0
index = 0

for i in range(len(img_contours)):
    area = cv2.contourArea(img_contours[i])
    if area > maxarea:
        maxarea = area
        index = i

pcb_mask = np.zeros(img_gray.shape, dtype=np.uint8)
cv2.drawContours(pcb_mask, img_contours, index, 255, -1)

cropped_pcb = cv2.bitwise_and(image, image, mask=pcb_mask)

pcb_gray = cv2.cvtColor(cropped_pcb, cv2.COLOR_BGR2GRAY)
pcb_gray_clean = pcb_gray.copy()
pcb_gray_clean[pcb_mask == 0] = 0

# lighting normalization
blur = cv2.GaussianBlur(pcb_gray_clean, (31, 31), 0)
normalized = cv2.subtract(pcb_gray_clean, blur)
pcb_phase1 = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

pcb_h, pcb_w = pcb_phase1.shape
pcb_area = pcb_h * pcb_w

edges = cv2.Canny(pcb_phase1, 50, 150)

component_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in component_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h > 50:
        boxes.append((x, y, w, h))

def boxes_close(b1, b2, thresh=20):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    cx1 = x1 + w1/2
    cy1 = y1 + h1/2
    cx2 = x2 + w2/2
    cy2 = y2 + h2/2
    return np.hypot(cx1-cx2, cy1-cy2) < thresh

def boxes_overlap(b1, b2):
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
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

merged_boxes = []

for group in groups:
    xs = [b[0] for b in group]
    ys = [b[1] for b in group]
    xe = [b[0]+b[2] for b in group]
    ye = [b[1]+b[3] for b in group]

    merged_boxes.append((min(xs), min(ys), max(xe)-min(xs), max(ye)-min(ys)))

def classify_component(w, h, area):
    ar = w/h if h>0 else 0

    if 1000 < area < 10000 and 0.5 < ar < 2.5:
        return "IC_CHIP"
    elif 100 < area < 2000 and 0.7 < ar < 1.4:
        return "CAPACITOR"
    elif 50 < area < 1000 and (ar > 2.5 or ar < 0.4):
        return "RESISTOR"
    elif area > 2000 and (ar > 3 or ar < 0.33):
        return "CONNECTOR"
    elif area < 100:
        return "SMALL_COMPONENT"
    else:
        return "MISC"

components_data = []

for idx,(x,y,w,h) in enumerate(merged_boxes):
    area = w*h
    ar = w/h if h>0 else 0

    region_gray = pcb_phase1[y:y+h, x:x+w]
    region_edges = edges[y:y+h, x:x+w]

    mean_intensity = float(np.mean(region_gray))
    edge_density = np.count_nonzero(region_edges)/area if area>0 else 0

    comp = {
        "id": idx,
        "category": classify_component(w,h,area),
        "bbox": {"x":int(x),"y":int(y),"w":int(w),"h":int(h)},
        "area": int(area),
        "norm_area": float(area/pcb_area),
        "aspect_ratio": float(ar),
        "centroid": {
            "x": float((x+w/2)/pcb_w),
            "y": float((y+h/2)/pcb_h)
        },
        "mean_intensity": mean_intensity,
        "edge_density": edge_density
    }

    components_data.append(comp)

category_counts = {}
for c in components_data:
    k = c["category"]
    category_counts[k] = category_counts.get(k,0)+1

output = {
    "object_type": object_type,
    "total_components": len(components_data),
    "component_counts": category_counts,
    "components": components_data
}

with open("detection_results.json","w") as f:
    json.dump(output,f,indent=2)

print("Saved detection_results.json")

vis = cv2.cvtColor(pcb_phase1, cv2.COLOR_GRAY2BGR)

colors = {
    "IC_CHIP":(255,0,0),
    "CAPACITOR":(0,255,0),
    "RESISTOR":(0,0,255),
    "CONNECTOR":(255,255,0),
    "SMALL_COMPONENT":(255,0,255),
    "MISC":(128,128,128)
}

for c in components_data:
    x=c["bbox"]["x"]
    y=c["bbox"]["y"]
    w=c["bbox"]["w"]
    h=c["bbox"]["h"]

    cv2.rectangle(vis,(x,y),(x+w,y+h),colors.get(c["category"],(255,255,255)),2)
    cv2.putText(vis,c["category"][:3],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

cv2.imshow("Detected Components",vis)
cv2.waitKey(0)
cv2.destroyAllWindows()