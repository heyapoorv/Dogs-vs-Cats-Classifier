import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import os
os.environ["STREAMLIT_SERVER_PORT"] = os.environ.get("PORT", "8501")
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Dogs vs Cats AI",
    page_icon="🐶",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
}
.badge {
    background-color: #0f172a;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
    display: inline-block;
}
.footer {
    text-align: center;
    opacity: 0.7;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/model_v1.keras")

model = load_model()

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🐶 Dogs vs Cats AI Classifier 🐱</h1>
    <p>Deep Learning powered image classification</p>
    <span class="badge">Accuracy: ~78% • CNN v1</span>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# -----------------------------
# Sample Image Loader
# -----------------------------
st.subheader("Try a sample image")

sample_col1, sample_col2 = st.columns(2)

sample_image = None

with sample_col1:
    if st.button("Load Sample Cat 🐱"):
        sample_image = Image.open("assets/untitled.jpg")

with sample_col2:
    if st.button("Load Sample Dog 🐶"):
        sample_image = Image.open("assets/sample_dog.webp")

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("Upload your own image")

uploaded_file = st.file_uploader(
    "Drag & drop or click to upload",
    type=["jpg", "jpeg", "png"]
)

# Decide image source
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif sample_image:
    image = sample_image
else:
    image = None

# -----------------------------
# Prediction Logic
# -----------------------------
def predict(image):
    img = image.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "Dog 🐶", float(pred)
    else:
        return "Cat 🐱", float(1 - pred)

# -----------------------------
# Display Prediction
# -----------------------------
if image:
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Input Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            label, confidence = predict(image)

        st.subheader("Prediction")
        st.success(label)

        st.subheader("Confidence")
        st.progress(confidence)
        st.write(f"**{confidence:.2%} confidence**")

# -----------------------------
# About the Model
# -----------------------------
st.divider()

with st.expander("ℹ️ About the model"):
    st.markdown("""
**Model Architecture**
- Custom CNN built using TensorFlow
- 4 Convolutional blocks + BatchNorm + Dropout
- Binary classification (Cat vs Dog)

**Training**
- Dataset: Microsoft Cats vs Dogs
- Image size: 256×256
- Optimizer: Adam
- Loss: Binary Crossentropy

**Performance**
- Accuracy: ~78%
- Model version: v1 (baseline)
- Next upgrade: Transfer Learning (MobileNetV2)
""")

# -----------------------------
# Confusion Matrix
# -----------------------------
if os.path.exists("assets/confusion_matrix.png"):
    st.subheader("Model Evaluation")
    st.image(
        "assets/confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

# -----------------------------
# Footer
# -----------------------------
st.write("")
st.markdown(
    "<div class='footer'>Built with TensorFlow • Streamlit • Render</div>",
    unsafe_allow_html=True
)
