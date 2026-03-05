import streamlit as st
import tempfile
import ollama

from model_inference import predict_xray
from medical_reasoning import generate_medical_summary

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Pneumonia Detection System")
st.title("🩺 Pneumonia Detection System (AI Explanation)")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Chest X-ray", width=300)

    with st.spinner("🔍 Analyzing X-ray..."):
        result = predict_xray(image_path)

    # Generate structured medical features
    summary = generate_medical_summary(result)

    st.subheader("📊 Model Output (Features)")
    st.text(summary)

    # ------------------ LLM Explanation ------------------
    with st.spinner("🧠 Generating AI medical explanation..."):
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a medical assistant. "
                        "Explain the following pneumonia detection result "
                        "in simple clinical language:\n\n"
                        f"{summary}"
                    )
                }
            ]
        )

    st.subheader("🧠 AI Medical Explanation")
    st.markdown(response["message"]["content"])
