# Agent-Based PCB Analysis using Computer Vision and OCR

**Project (IPSA): Intelligent PCB Should Cost Estimator with Agentic AI Vision Analysis**

## Overview
This project presents an agent-based system for analyzing printed circuit board (PCB) images using classical computer vision (CV), optical character recognition (OCR), and large language model (LLM)-based reasoning.

The problem is challenging because real-world PCB images contain small, densely packed components, variable lighting, and limited readable text, making traditional detection and recognition unreliable.

The goal is to estimate PCB complexity and approximate the bill of materials (BOM) from a single image. The system combines geometric feature extraction, text detection, and structured reasoning to produce high-level insights. The system treats PCB analysis as a perception + reasoning problem, where CV and OCR extract signals and an agent-based LLM performs structured inference.

An experimental pipeline is also included to evaluate how the system behaves under different input conditions such as resolution and simulated camera distance.

---

## Repository Structure

The repository contains multiple versions of the system showing its evolution. The stable implementation described in the report is in `version5/`.
`version6/` contains ongoing improvements, particularly to the OCR subsystem.

```
root/
│
├── version1/
├── version2/
├── version3-0/
├── version3-1/
├── version4/
├── version5/           
├── version6/ (ongoing...)
│
├── PCB_Analysis_Report.pdf  # Detailed report of experiments and system design (from version5)      
```

---

## Version Evolution

### Version 1 — Basic CV + LLM Pipeline

* Detects PCB region and segments it from the background
* Identifies components using OpenCV-based contour detection
* Groups nearby regions and applies simple shape-based rules
* Sends extracted component counts to an LLM for:

  * PCB type inference
  * Approximate BOM estimation
* Includes a minimal Streamlit interface

This version establishes the basic pipeline but relies heavily on heuristic assumptions.

---

### Version 2 — Structured CV + Agent + LLM

* Improved component detection using:

  * Area filtering
  * Aspect ratio filtering
  * Bounding box merging
* Outputs **geometric features only** instead of guessed labels:

  * Bounding boxes
  * Area
  * Aspect ratio
  * Centroid
  * Intensity
* Introduces an **agent layer** to control the pipeline
* Integrates a **local LLM (Ollama / Qwen)**
* Updates Streamlit UI to run the full pipeline interactively

This version separates perception (CV) from reasoning (LLM).

---

### Version 3.0 — OCR Experiments

* Adds OCR using EasyOCR
* Applies OCR only to likely IC regions (based on size/shape)
* Adds simple preprocessing for better text detection
* Introduces size-based categorization (tiny / medium / large)

Observations:

* OCR works for small labels like "R1", "C18", "101"
* Fails for IC names due to small and unclear text
* High-resolution images increase noise in detection

OCR output is not yet used for reasoning due to unreliability.

---

### Version 3.1 — Agent Refactor

* Converts the pipeline into a fully **agent-based structure**
* LLM dynamically calls tools instead of following a fixed pipeline

This introduces iterative reasoning.

---

### Version 4 — Modular Agent System

Major architectural improvements:

* **Tool-based design**:

  * `run_cv`
  * `get_component_stats`
  * `get_ic_info`

* **Agent loop**:

  * Decide tool
  * Execute tool
  * Observe output
  * Repeat

* **State (memory)** to store intermediate results

* **Caching** to avoid repeated CV computation

* **Control logic** to prevent invalid tool usage

* **Reflection step** for answer correction

Simplifies data passed to LLM using summarized statistics.

Focus is on system design, not accuracy.

---

### Version 5 

#### Key Improvements

**1. Richer CV Output**

* Extracts geometric features
* Adds heuristic-based component type estimation
* Categorizes components by size

**2. Improved OCR (PaddleOCR)**

* Replaces EasyOCR with PaddleOCR
* Runs OCR on the full image
* Extracts:

  * Reference designators (R, C, U, J)
  * Possible IC text

**3. Multi-Modal Reasoning**

* Agent uses both:

  * CV features
  * OCR signals
* Reduces reliance on a single modality

**4. Robust Error Handling**

* Handles failures in CV, OCR, and agent modules
* Ensures structured outputs even in error cases
* Enables graceful degradation

**5. Experimental Pipeline**

* Simulates:

  * Resolution changes
  * Camera distance (blur)
* Stores results automatically for analysis

---

### Version 6 (in progress)
Current ongoing version

OCR changes as of now:

* PaddleOCR removed due to Python version compatibility issues.
* OCR replaced with DocTR (python-doctr), which works on newer Python versions.
* Existing filtering logic for extracting IC part numbers remains unchanged.

---

## Version 5 Project Structure

```
version5/
│
├── app.py                     # Streamlit UI
├── main.py                    # Main pipeline entry point
├── agent.py                   # Agent logic
├── agent_tools.py             # Tool definitions
├── cv_pipeline.py             # Computer vision pipeline
├── ocr.py                     # OCR module (PaddleOCR)
├── llm_pipeline.py            # LLM interaction
├── experiment_camera_spec.py  # Experiment pipeline
│
├── experiment_results_clean/  # Generated outputs for Image 1 (pcbclear2.jpg)
├── experiment_results_dense/  # Generated outputs for Image 2 (pcbimagetrial4k.png)
├── outputV5.pdf               # Example system output for sample PCB images
├── README.md
└── requirements.txt
```

---

## Setup Instructions

1. Clone the repository

```
git clone https://github.com/Raghushubi/IPSA.git
cd IPSA/version5
```

2. Create a virtual environment (Use Python 3.10 for compatibility with PaddleOCR)

```
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Ensure Ollama is installed and the required model is available:

```
ollama pull qwen2.5:7b-instruct
ollama serve
```

---

## Running the System (Version 5)

### Run Main Pipeline (Uses the default sample image pcbclear2.jpg)

```
python main.py
```

### Run Streamlit UI

```
streamlit run app.py
```

### Run Experimental Evaluation

```
python experiment_camera_spec.py
```

Results will be stored in the `experiment_results/` directory.

---

## Experimental Results

Experiments are conducted on two PCB types:

* **Clean PCB** (`pcbclear2.jpg`)

  * Clear components and readable text

* **Dense PCB** (`pcbimagetrial4k.png`)

  * High component density
  * Limited text visibility

Evaluation includes:

* Resolution scaling (1.0 → 0.1)
* Blur-based distance simulation

Results are stored separately for reproducibility.

---

## Limitations

* CV uses contour-based detection, not true component recognition
* Performance degrades on dense or low-quality images
* OCR accuracy depends heavily on text clarity
* Heuristic classification is approximate
* LLM outputs are heuristic and not grounded in real component pricing data

---

## Future Work

* Learning-based detection (e.g., YOLO)
* Improved OCR preprocessing and filtering
* Structured validation of LLM outputs
* Integration of real pricing data
* Uncertainty estimation in reasoning

---

## Report

A detailed explanation of the system, experiments, and observations is available in:

**`PCB_Analysis_Report.pdf` (root directory)**

