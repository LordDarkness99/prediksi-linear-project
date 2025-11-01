# src/app.py
import streamlit as st
import numpy as np
from src.model import load_model, predict
from src.data_utils import load_scaler

st.title("Prediksi Gaji Berdasarkan Pengalaman & Pendidikan")

model_path = "models/manual_model.joblib"
if st.button("Muat Model Default"):
    model = load_model(model_path)
    st.success("Model dimuat.")

# input
pengalaman = st.number_input("Pengalaman (tahun)", min_value=0, step=1, value=3)
pendidikan = st.selectbox("Pendidikan", options=[1,2,3,4], format_func=lambda x: {1:'SMA',2:'D3',3:'S1',4:'S2'}[x])

if st.button("Prediksi Gaji"):
    model = load_model(model_path)
    scaler = model["scaler"]
    w = model["w"]
    x = np.array([[pengalaman, pendidikan]]).astype(float)
    x_scaled = scaler.transform(x)
    x_input = np.hstack([np.ones((1,1)), x_scaled])
    pred = predict(x_input, w)[0]
    st.write(f"Perkiraan gaji: Rp {pred:,.0f}")
