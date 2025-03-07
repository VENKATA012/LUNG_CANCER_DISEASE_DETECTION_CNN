import streamlit as st
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import sqlite3
import pandas as pd
import datetime

# Set page configuration
st.set_page_config(page_title="🩺 Lung Cancer Detection using CNN", layout="wide")

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="lung_cancer_classifier_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ['Normal', 'Adenocarcinoma', 'Squamous Cell Carcinoma']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((256, 256))  # Resize using PIL
    img = np.array(img).astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# **🔹 Database Setup for History**
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        predicted_class TEXT,
        confidence REAL,
        timestamp TEXT
    )
""")
conn.commit()

# Function to save prediction to the database
def save_prediction(image_name, predicted_class, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history (image_name, predicted_class, confidence, timestamp) VALUES (?, ?, ?, ?)",
                   (image_name, predicted_class, confidence, timestamp))
    conn.commit()

# Fetch prediction history
def fetch_prediction_history():
    try:
        cursor.execute("SELECT image_name, predicted_class, confidence, timestamp FROM history ORDER BY timestamp DESC")
        rows = cursor.fetchall()

        # Convert to Pandas DataFrame
        df = pd.DataFrame(rows, columns=["🖼 Image", "🔬 Predicted Class", "📊 Confidence (%)", "⏳ Timestamp"])
        
        # Ensure text columns are properly decoded
        df["🖼 Image"] = df["🖼 Image"].astype(str)
        df["🔬 Predicted Class"] = df["🔬 Predicted Class"].astype(str)
        df["⏳ Timestamp"] = df["⏳ Timestamp"].astype(str)
        
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading prediction history: {e}")
        return pd.DataFrame()

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
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        processed_image = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class_index = np.argmax(prediction)
        predicted_label = class_labels[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index]) * 100  # Convert to float

        st.markdown(f"### 🔍 Prediction: *{predicted_label}*")
        st.write("### 📊 Class Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"✔ *{label}*: {prediction[0][i] * 100:.2f}%")

        st.warning("⚠️ This is an AI-based prediction. Consult a medical professional for an accurate diagnosis.")

        # **Save prediction to database**
        save_prediction(uploaded_file.name, predicted_label, confidence)

# History Page
elif page == "📊 History":
    st.title("📊 Prediction History")

    try:
        df = fetch_prediction_history()

        if not df.empty:
            # Convert confidence values safely
            def safe_convert(value):
                try:
                    return float(value) if isinstance(value, (int, float, str)) else "Invalid"
                except Exception:
                    return "Error"

            df["📊 Confidence (%)"] = df["📊 Confidence (%)"].apply(safe_convert)

            # Display the cleaned table
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No past predictions found.")

    except Exception as e:
        st.error(f"⚠️ Error loading prediction history: {e}")

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
