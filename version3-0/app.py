# app.py

import streamlit as st
import cv2
import tempfile
import os
from agent import run_agent

def main():
    st.title("PCB Inspection & BOM Estimation System")
    st.write("Upload a PCB image for analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "jfif"]
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            st.subheader("Analysis")
            with st.spinner("Running pipeline"):
                result = run_agent(tmp_path)
            
            cv_result = result["cv"]
            llm_result = result["llm"]
            status = result["status"]
            
            object_type = cv_result["object_type"]
            visualization = cv_result["visualization"]
            components = cv_result["components"]
            
            st.write(f"Detected Object Type: {object_type}")
            
            if object_type == "PCB":
                st.image(
                    cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB),
                    caption="Detected Components",
                    use_container_width=True
                )
                st.write(f"Components Detected: {len(components)}")
                
                with st.expander("Component Data (JSON)"):
                    st.json(components)
                
                if llm_result is not None:
                    st.subheader("LLM Analysis")
                    st.json(llm_result)
                else:
                    st.warning(f"LLM skipped: {status}")
            else:
                st.warning(f"Object type '{object_type}' not supported.")
                st.image(
                    cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()