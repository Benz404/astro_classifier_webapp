# 🔭 Astronomical Object Classifier

This web app classifies celestial objects (Stars, Galaxies, Quasars) based on Sloan Digital Sky Survey (SDSS) data using machine learning.

![Galaxy](assets/galaxy.jpg)

## 🚀 Try the App

🔗 [Click here to open the live app](https://your-username.streamlit.app)  
*(Deployed using Streamlit Cloud)*

---

## 📊 Project Overview

This project uses a trained Random Forest classifier to predict the type of astronomical object using 6 input features:
- u, g, r, i, z (photometric magnitudes)
- Redshift (z)

The model was trained using real SDSS data and achieves high accuracy in classifying:
- 🌟 Stars
- 🌌 Galaxies
- ✨ Quasars (QSOs)

---

## 🖥️ Features

- 📥 Input photometric data and redshift
- 🔮 Predict object class with confidence scores
- 📸 Shows sample images of predicted objects
- 📊 Visualizes object position in 2D (r vs redshift)
- 💡 Clean, responsive UI with modern layout

---

## 📁 Folder Structure