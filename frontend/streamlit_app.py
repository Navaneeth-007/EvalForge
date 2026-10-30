"""Optional alternate client. The primary UI is served by FastAPI at /."""

import os
import requests
import streamlit as st

st.set_page_config(page_title="EvalForge", layout="wide")
st.title("EvalForge · Text-to-SQL")
api = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
try:
    response = requests.get(f"{api}/examples", timeout=10)
    response.raise_for_status()
    data = response.json()
except requests.RequestException:
    st.error("API unavailable. Start the API, or set API_BASE to your deployment URL.")
    st.stop()

if data["mode"] == "walkthrough":
    st.info("Authored examples only. No model inference or completed training is claimed.")
else:
    st.info("Using the configured adapter. Inspect generated SQL before use.")
example = data["examples"][0]
question = st.selectbox("Question", example["sample_questions"])
schema = st.text_area("Schema", example["schema"], disabled=data["mode"] == "walkthrough")
if st.button("Show SQL"):
    try:
        response = requests.post(
            f"{api}/generate-sql", json={"question": question, "schema": schema}, timeout=60
        )
        response.raise_for_status()
        result = response.json()
        st.code(result["sql"], language="sql")
        st.caption(f"Source: {result['model']} · {result['latency_ms']} ms")
    except requests.RequestException:
        st.error("Request failed. Check API availability and its supported mode.")
st.link_button("Project repository", "https://github.com/Navaneeth-007/EvalForge")
