import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Page settings
st.set_page_config(page_title="Astronomical Object Classifier", page_icon="🔭", layout="centered")

# Load model
model = joblib.load("astro_classifier.pkl")
label_map = {0: "GALAXY", 1: "QSO (Quasar)", 2: "STAR"}
image_map = {
    "GALAXY": "assets/galaxy.jpg",
    "QSO (Quasar)": "assets/quasar.jpg",
    "STAR": "assets/star.jpg"
}

# Title
st.markdown("<h1 style='text-align: center; color: #00aaff;'>🔭 Astronomical Object Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict stars, galaxies, and quasars using SDSS photometric data</p>", unsafe_allow_html=True)
st.markdown("---")

# Input fields
st.subheader("📥 Enter Object Features")
col1, col2 = st.columns(2)

with col1:
    u = st.number_input("**u-band (ultraviolet)**", 0.0, 30.0, 18.0, step=0.01)
    g = st.number_input("**g-band (green)**", 0.0, 30.0, 17.5, step=0.01)
    r = st.number_input("**r-band (red)**", 0.0, 30.0, 17.2, step=0.01)

with col2:
    i = st.number_input("**i-band (infrared)**", 0.0, 30.0, 17.0, step=0.01)
    z = st.number_input("**z-band (deep IR)**", 0.0, 30.0, 16.8, step=0.01)
    redshift = st.number_input("**Redshift (z)**", 0.0, 10.0, 0.1, step=0.01)

st.markdown("---")

# Predict
if st.button("🚀 Classify"):
    features = np.array([[u, g, r, i, z, redshift]])
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    label = label_map.get(prediction, "Unknown")
    st.success(f"🎯 The object is classified as: **{label}**")

    # Show image
    img_path = image_map.get(label, None)
    if img_path:
        st.image(Image.open(img_path), caption=label, use_container_width=True)

    # Show confidence
    st.markdown("### 🔢 Prediction Probabilities:")
    for idx, class_label in label_map.items():
        st.markdown(f"- **{class_label}**: `{probs[idx]*100:.2f}%`")

    # Simple 2D Visualization (fake projection)
    st.markdown("### 🧭 Feature Projection (r vs redshift)")
    fig, ax = plt.subplots()
    ax.scatter(r, redshift, c='gold', s=100, edgecolors='black')
    ax.set_xlabel("r-band (Red)")
    ax.set_ylabel("Redshift (z)")
    ax.set_title("Object Projection")
    st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit | Data from SDSS</p>", unsafe_allow_html=True)
