import streamlit as st
from PIL import Image
import subprocess
import json
import os

st.title("PCB Component Detection")

uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg","png","jpeg","jfif"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded PCB")

    # Saving image so detector can read it
    with open("pcbimagetrial.jfif", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Run Detection"):

        st.write("Running component detection...")

        subprocess.run(["python", "component_detector.py"])

        if os.path.exists("detection_results.json"):
            with open("detection_results.json") as f:
                data = json.load(f)

            st.success("Detection complete")

            st.json(data)

        else:
            st.error("detection_results.json not found")

    if st.button("Run LLM Analysis"):

        st.write("Running LLM...")

        subprocess.run(["python", "llm_analyzer.py"])

        if os.path.exists("bom_analysis.json"):
            with open("bom_analysis.json") as f:
                analysis = json.load(f)

            st.success("LLM complete")
            st.json(analysis)

        else:
            st.error("bom_analysis.json not found")
