# Version 6 – Observations Compared to Version 5

Version 6 introduces two main changes to the system:

* Replacement of **PaddleOCR** with **DocTR (python-doctr)** for OCR
* Replacement of paragraph-style reasoning output with a **structured BOM estimation table**

The computer vision pipeline and the agent reasoning architecture remain unchanged. Because of this, the behavior of the system can be compared directly with the observations reported for Version 5.

---

## 1. OCR module change

The OCR subsystem now uses **DocTR** instead of PaddleOCR. The main motivation for this change was **environment compatibility**. PaddleOCR required Python 3.10 and introduced multiple dependency constraints, while DocTR works reliably in newer environments such as **Python 3.12**.

For clean PCB images, the DocTR-based OCR module extracts a larger number of readable text tokens such as:

* Reference labels (R, C, U, J)
* Possible IC markings

Example observation from the clean PCB image:

R: 33
C: 34
U: 20
J: 6

This provides a slightly stronger textual signal for the reasoning stage compared to Version 5.

However, OCR behavior remains highly dependent on image quality. For dense PCB images with small or unclear text, the OCR output still contains mostly noisy or unusable tokens.

**Observation:** OCR extraction improves on clean images but still degrades significantly on dense boards.

---

## 2. Structured BOM output

Version 6 replaces the previous paragraph-style output with a **structured BOM estimation table**.

Example:

| Component | Count | Unit Cost   | Estimated Total |
| --------- | ----- | ----------- | --------------- |
| Resistor  | 35    | 0.5–5 INR   | 17.5–175 INR    |
| Capacitor | 39    | 1–50 INR    | 39–1950 INR     |
| IC        | 17    | 20–500+ INR | 340–8500+ INR   |

This format improves readability and makes the system easier to demonstrate or integrate with user interfaces such as the Streamlit dashboard.

---

## 3. CV signals still dominate reasoning

Even with improved OCR extraction, the reasoning stage still relies mainly on **CV-derived component counts**.

For example in the clean PCB experiment:

* CV IC count ≈ 17
* OCR IC candidates ≈ 21

The final reasoning result aligns more closely with the CV counts. OCR is mainly treated as supporting evidence rather than the primary signal.

---

## 4. Perception limitations remain unchanged

The main limitations described in the Version 5 analysis remain the same in Version 6.

Dense PCB images still produce a large number of detected regions because the CV module detects **contour-based regions rather than actual electronic components**.

Example from dense PCB experiments:

* Total detected regions ≈ 850
* Unknown regions ≈ 808

This confirms that segmentation artifacts from traces and textures still inflate region counts.

---

## 5. Sensitivity to resolution remains similar

Resolution scaling experiments continue to show the same pattern observed in Version 5:

| Scale    | Behavior                       |
| -------- | ------------------------------ |
| 1.0      | Stable detection               |
| 0.75–0.5 | Rapid loss of small components |
| 0.25     | Sparse detections              |
| 0.1      | Detection collapse             |

At the lowest resolution, the system fails to detect any meaningful regions and the agent returns an undefined PCB estimate.

---

## 6. Blur / distance behavior remains non‑linear

Distance simulation experiments continue to show unstable behavior where component counts may increase or decrease depending on blur.

This occurs because blur redistributes pixel intensity gradients, creating additional artificial edges that are interpreted as contours.

As a result, estimated component counts and BOM ranges can vary widely under degraded visual conditions.

---

## Summary of Version 6 changes

| Aspect                 | Version 5            | Version 6                      |
| ---------------------- | -------------------- | ------------------------------ |
| OCR Engine             | PaddleOCR            | DocTR                          |
| Python compatibility   | Python 3.10 required | Works on Python 3.12           |
| Output format          | Narrative reasoning  | Structured BOM table           |
| OCR coverage           | Limited on clean PCB | Slightly improved on clean PCB |
| CV perception behavior | Unstable             | Unchanged                      |

---

## Key takeaway

Version 6 mainly improves **OCR portability and output presentation**, but the fundamental system behavior remains consistent with the observations reported for Version 5.

The experiments continue to show that the **computer vision perception stage is the primary bottleneck**, and reasoning results are strongly dependent on the quality and stability of the detected visual structures.
