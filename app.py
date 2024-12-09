import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image

# Custom CSS for modern UI
def local_css():
    st.markdown("""
    <style>
    .title {
        background-image: linear-gradient(to right, #4c6ff4, #7f3df3);
        color: white;
        text-align: center;
        font-weight: bold;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .subtitle {
        color: #34495e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stFile {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .normal {
        background-color: #2ecc71;
        color: white;
    }
    .cancer {
        background-color: #e74c3c;
        color: white;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model with enhanced error handling
@st.cache_resource
def load_model_from_file(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Advanced image preprocessing
def preprocess_image(image):
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize with anti-aliasing
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)

    # Normalize pixel values
    image = image / 255.0

    # Expand dimensions to match batch size
    image = np.expand_dims(image, axis=0)

    return image

# Predict using the model with confidence interpretation
def predict_image(model, image):
    prediction = model.predict(image)
    confidence = prediction[0][0]
    return confidence

def main():
    # Set page config
    st.set_page_config(
        page_title="Lung Cancer Detector",
        page_icon="🫁",
        layout="centered"
    )

    # Apply custom CSS
    local_css()

    # Title and Subtitle
    st.markdown('<h1 class="title">Lung Cancer Detection System 🫁</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">AI-Powered Chest X-ray Analysis</h3>', unsafe_allow_html=True)

    # Model Loading
    model_path = r"model/lung_cancer_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please check the file path.")
        return
    
    model = load_model_from_file(model_path)

    # File Uploader with Enhanced UI
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for lung cancer detection",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Read and display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess and predict
        processed_image = preprocess_image(image_cv)
        prediction = predict_image(model, processed_image)

        # Prediction Display with Detailed Interpretation
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        if prediction > 0.5:
            st.markdown(f'''
            <div class="prediction-box cancer">
                ⚠️ Potential Lung Cancer Detected
                <br>Confidence: {prediction*100:.2f}%
            </div>
            ''', unsafe_allow_html=True)
            st.warning("🚨 Please consult a medical professional for further evaluation.")
            st.info("Recommendations:")
            st.info("- Schedule a follow-up chest X-ray or CT scan to confirm the diagnosis")
            st.info("- Consider discussing treatment options with an oncologist if cancer is confirmed")
        else:
            st.markdown(f'''
            <div class="prediction-box normal">
                ✅ No Cancer Detected
                <br>Confidence: {(1-prediction)*100:.2f}%
            </div>
            ''', unsafe_allow_html=True)
            st.info("Regular health check-ups are recommended to monitor your lung health.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Additional Information
    # st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.sidebar.markdown("""
    ## About This App 🏥
    - **Purpose**: Early Lung Cancer Detection
    - **Technology**: Deep Learning AI
    - **Model**: Trained on Chest X-ray Datasets
    - **Disclaimer**: Not a Substitute for Professional Medical Advice
    """)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

