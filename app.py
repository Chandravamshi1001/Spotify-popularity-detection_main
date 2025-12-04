# app.py
import streamlit as st
import numpy as np
import joblib
import os
import os
import gdown  # safer for Google Drive links

models = {
    "rf_model.joblib": "https://drive.google.com/uc?id=1Va6lxlHMO-coFtIjdmaSM9zHFdMQ1NWZ",
    "scaler.joblib":   "https://drive.google.com/uc?id=1p9ASCYCzka2SJO1ET1214Wl9u5QClA48"
}

os.makedirs("models", exist_ok=True)

for filename, url in models.items():
    path = os.path.join("models", filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        gdown.download(url, path, quiet=False)
    else:
        print(f"{filename} already exists.")



MODEL_PATH = "models/rf_model.joblib"
SCALER_PATH = "models/scaler.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler not found. Run training first: `python src/train.py`")
    st.stop()


st.set_page_config(page_title="Spotify Popularity Predictor", layout="centered")

st.title("🎧 Spotify Popularity Predictor")
st.markdown("Enter the song features and get prediction whether the song is likely to be popular.")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler not found. Run training first: `python src/train.py`")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Input sliders
danceability = st.slider("Danceability (0-1)", 0.0, 1.0, 0.5, 0.01)
energy = st.slider("Energy (0-1)", 0.0, 1.0, 0.5, 0.01)
key = st.slider("Key (0-11)", 0, 11, 5)
time_signature = st.slider("Time Signature", 3, 7, 4) 
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0, 0.1)
mode = st.selectbox("Mode (0 = minor, 1 = major)", [0,1], index=1)
speechiness = st.slider("Speechiness (0-1)", 0.0, 1.0, 0.05, 0.01)
acousticness = st.slider("Acousticness (0-1)", 0.0, 1.0, 0.1, 0.01)
instrumentalness = st.number_input("Instrumentalness (0-1)", 0.0, 1.0, 0.0, 0.001)
liveness = st.slider("Liveness (0-1)", 0.0, 1.0, 0.1, 0.01)
valence = st.slider("Valence (0-1)", 0.0, 1.0, 0.5, 0.01)
tempo = st.number_input("Tempo (BPM)", 0.0, 300.0, 120.0, 0.1)
duration_ms = st.number_input("Duration (ms)", 10000, 600000, 210000, 1000)

features = np.array([[
    danceability, energy, key, time_signature, loudness, mode, speechiness,
    acousticness, instrumentalness, liveness, valence, tempo, duration_ms
]])

features_scaled = scaler.transform(features)

if st.button("Predict Popularity"):
    prob = model.predict_proba(features_scaled)[0,1]
    pred = model.predict(features_scaled)[0]
    st.write("---")
    if pred == 1:
        st.success(f"🔥 Predicted: POPULAR (probability = {prob:.2f})")
    else:
        st.info(f"Predicted: Not Popular (probability = {prob:.2f})")
