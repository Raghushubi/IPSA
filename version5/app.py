# app.py

import streamlit as st
import tempfile
import os
import time
from agent import run_agent


def safe_run_agent(path):
    try:
        result = run_agent(path)
        if isinstance(result, dict):
            return result
        return {"status": "error", "reason": "invalid_agent_output"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def main():
    st.title("PCB Inspection & BOM Estimation System")
    st.write("Upload a PCB image for analysis")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "jfif"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            try:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            except Exception:
                st.error("Failed to save uploaded file")
                return

        try:
            st.subheader("Uploaded Image")
            st.image(tmp_path, use_column_width=True)

            st.subheader("Analysis")

            start_time = time.time()

            with st.spinner("Running agent..."):
                result = safe_run_agent(tmp_path)

            end_time = time.time()
            st.write(f"Time taken: {end_time - start_time:.2f} seconds")

            if not isinstance(result, dict):
                st.error("Invalid agent response")
                return

            status = result.get("status")

            if status == "success":
                llm_result = result.get("result") or {}
                steps_used = result.get("steps_used", "unknown")

                st.success(f"Agent completed in {steps_used} step(s)")

                st.subheader("BOM Estimation Result")
                st.json(llm_result)

                logs = result.get("logs")
                if logs:
                    st.subheader("Agent Reasoning Steps")
                    for log in logs:
                        st.text(log)

                ic_info = result.get("ic_info")
                if ic_info:
                    st.subheader("OCR Insights")
                    st.json(ic_info)

                visualization = result.get("visualization")
                if visualization is not None:
                    st.subheader("Detected Components")
                    st.image(visualization, use_column_width=True)

            elif status == "llm_error":
                st.warning("LLM returned invalid response")
                st.json(result)

            elif status == "max_steps_exceeded":
                st.warning("Agent reached maximum reasoning steps")
                st.json(result)

            else:
                st.error(f"Agent failed with status: {status}")
                st.json(result)

        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()