Version 1 implements a PCB analysis pipeline.
The system first detects the main PCB region, separates it from the background, then identifies sub-components using OpenCV-based processing. Nearby regions are grouped, components are categorized using shape-based prior rules, and structured JSON output is generated.
The extracted component counts are then sent to an LLM for high-level PCB type inference and approximate BOM cost estimation.
A minimal Streamlit GUI is included for image upload and viewing detection and analysis results.