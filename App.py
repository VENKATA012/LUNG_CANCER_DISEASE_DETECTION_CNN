import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow import keras
from PIL import Image
import pandas as pd




# Set page configuration
st.set_page_config(page_title="🩺 Lung Cancer Detection using CNN", layout="wide")

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="lung_cancer_classifier_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
model = load_tflite_model()

# Define class labels
class_labels = ['Normal', 'Adenocarcinoma', 'Squamous Cell Carcinoma']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (256, 256))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Simulated cloud storage predictions (Replace this with actual cloud storage retrieval)
stored_predictions = [
    {"image_name": "scan_01.jpg", "predicted_class": "Adenocarcinoma", "timestamp": "2025-03-07 10:15:30"},
    {"image_name": "scan_02.jpg", "predicted_class": "Squamous Cell Carcinoma", "timestamp": "2025-03-07 10:17:45"},
    {"image_name": "scan_03.jpg", "predicted_class": "Normal", "timestamp": "2025-03-07 10:20:10"},
]

# Sidebar Navigation
st.sidebar.image("logo.png.webp", width=120)
st.sidebar.title("🔍 Lung Cancer Detection")
page = st.sidebar.radio("Go to", ["🏠 Home", "📖 About", "🔬 Lung Detection", "📊 History", "❓ Help"])

# Home Page
if page == "🏠 Home":
    st.image("digital-lung-anatomy-visualization-ai-generated-detailed-visualization-human-lungs-illuminated-vibrant-blue-orange-326767653.webp", use_column_width=True)
    st.title("🏠 Welcome to Lung Cancer Detection System")
    st.write("Detect lung diseases using AI-powered deep learning models.")
    
    if st.button("🔬 Start Detection"):
        st.session_state.page = "🔬 Lung Detection"

# About Page
elif page == "📖 About":
    st.title("📖 About")
    st.write("This AI-based application detects lung cancer by analyzing lung histopathology images.")
    st.write("This web app uses a Convolutional Neural Network (CNN) to detect lung cancer from histopathological images.")
    st.markdown("""
    - *Normal*: Healthy lung tissue.
    - *Adenocarcinoma*: A type of non-small cell lung cancer.
    - *Squamous Cell Carcinoma*: Another form of non-small cell lung cancer.
    """)

# Lung Detection Page
elif page == "🔬 Lung Detection":
    st.title("🔬 Lung Cancer Detection")
    st.write("Upload a lung histopathology image to analyze.")

    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        predicted_label = class_labels[predicted_class]

        st.markdown(f"### 🔍 Prediction: *{predicted_label}*")

        st.write("### 📊 Class Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"✔ *{label}*: {prediction[0][i] * 100:.2f}%")

        st.warning("⚠️ This is an AI-based prediction. Consult a medical professional for an accurate diagnosis.")

# History Page
elif page == "📊 History":
    st.title("📊 Prediction History")

    if stored_predictions:
        df = pd.DataFrame(stored_predictions)
        df = df.rename(columns={"image_name": "🖼 Image", "predicted_class": "🔬 Predicted Class", "timestamp": "⏳ Timestamp"})

        st.dataframe(df, use_container_width=True)
    else:
        st.write("No past predictions found.")

# Help Page
elif page == "❓ Help":
    st.title("❓ Help & Support")
    st.write("""
    1️⃣ Navigate to *Lung Disease Detection*.  
    2️⃣ Upload a histopathology image.  
    3️⃣ AI will analyze and predict the disease category.  
    4️⃣ View the results and confidence scores.  
    5️⃣ Consult a medical professional for confirmation.  
    """)

st.sidebar.write("© 2025 Lung Cancer Detection System")
