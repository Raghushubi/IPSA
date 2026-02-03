Version 2 reorganizes the project into a CV + Agent + LLM pipeline.

The PCB is first isolated and validated, then components are detected using OpenCV with better filtering (area filtering, aspect ratio filtering, and box merging). Instead of guessing component types, the CV now only outputs geometric data like bounding boxes, area, aspect ratio, centroid, and intensity.

An agent layer was added to control the full flow (CV → LLM) inside the same app. Local LLM inference is integrated using Ollama (Qwen), so everything runs together without subprocess calls or intermediate JSON files. The Streamlit UI is updated to run detection and LLM reasoning in one flow.
