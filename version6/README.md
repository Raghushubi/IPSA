Project Update – OCR Module

The OCR subsystem is currently being updated to improve environment compatibility and portability. Previous versions of this project used PaddleOCR, which required a Python 3.10 environment and introduced several dependency constraints.

In this version, PaddleOCR has been replaced with DocTR (python-doctr), a PyTorch-based OCR framework that works reliably on newer Python versions (e.g., Python 3.12). This allows the project to run in more environments without requiring specific Python setups.

Current OCR changes

* PaddleOCR removed due to Python version compatibility issues.
* OCR replaced with DocTR for text detection and recognition.
* Existing filtering logic for extracting IC part numbers remains unchanged.
