import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os

# -----------------------------
# CONFIGURASI APLIKASI
# -----------------------------
st.set_page_config(
    page_title="Prediksi Gaji Berdasarkan Pengalaman",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Prediksi Gaji Berdasarkan Pengalaman Kerja")
st.write("""
Aplikasi ini menggunakan **Regresi Linier Berganda** untuk memprediksi **gaji (dalam Rupiah)** 
berdasarkan **tahun pengalaman kerja** dan **tingkat pendidikan**.
""")

# -----------------------------
# PILIH MODEL
# -----------------------------
BASE_DIR = Path(os.path.dirname(__file__)).resolve().parent
models_path = BASE_DIR / "models"

model_choice = st.selectbox(
    "Pilih model yang ingin digunakan:",
    ("manual_model.joblib", "sklearn_model.joblib")
)

model_path = models_path / model_choice
model = joblib.load(model_path)

# Penjelasan model
with st.expander("🧩 Penjelasan Jenis Model"):
    st.markdown("""
    **1. manual_model.joblib**  
    Model hasil pelatihan manual yang disimpan sebagai *dictionary* berisi:
    - `w`: bobot hasil training (parameter regresi)
    - `scaler`: objek normalisasi fitur  
    Digunakan untuk pembelajaran konsep dasar regresi linier.

    **2. sklearn_model.joblib**  
    Model yang dilatih menggunakan **Scikit-Learn LinearRegression()**  
    Mempunyai fungsi `.predict()` untuk memprediksi langsung tanpa preprocessing tambahan.
    """)

# -----------------------------
# INPUT DATA MANUAL
# -----------------------------
st.subheader("Masukkan Data Pengalaman")

years_exp = st.number_input("Tahun pengalaman kerja:", min_value=0.0, step=0.5, value=3.0)
education_lvl = st.number_input("Tingkat pendidikan (1–4):", min_value=1, max_value=4, step=1, value=3)

# Penjelasan level pendidikan
with st.expander("ℹ️ Penjelasan Tingkat Pendidikan (1–4)"):
    st.markdown("""
    Nilai **1–4** merepresentasikan **tingkat pendidikan formal** secara berurutan.  
    Semakin tinggi nilai, semakin tinggi tingkat pendidikan seseorang.

    | Nilai | Tingkat Pendidikan | Keterangan |
    |:------:|:------------------|:------------|
    | 1 | SMA / SMK | Pendidikan menengah atas. |
    | 2 | Diploma (D1–D3) | Pendidikan vokasi atau keahlian khusus. |
    | 3 | Sarjana (S1) | Pendidikan tinggi akademik atau profesional. |
    | 4 | Pascasarjana (S2/S3) | Pendidikan lanjutan atau riset profesional. |

    Model ini mengasumsikan bahwa **pendidikan berpengaruh positif terhadap gaji**,  
    sehingga semakin tinggi angka (pendidikan), maka prediksi gaji juga meningkat.
    """)

# -----------------------------
# PREDIKSI SATU DATA
# -----------------------------
if st.button("🔮 Prediksi Gaji"):
    model_data = model

    # Jika model manual (dictionary berisi w dan scaler)
    if isinstance(model_data, dict) and "w" in model_data:
        scaler = model_data["scaler"]
        X_input = np.array([[years_exp, education_lvl]])
        X_scaled = scaler.transform(X_input)
        X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        w = model_data["w"]
        prediction = np.dot(X_scaled, w)

    # Jika model sklearn (dictionary berisi sklearn_model dan scaler)
    elif isinstance(model_data, dict) and "sklearn_model" in model_data:
        scaler = model_data["scaler"]
        X_input = np.array([[years_exp, education_lvl]])
        X_scaled = scaler.transform(X_input)
        sklearn_model = model_data["sklearn_model"]
        prediction = sklearn_model.predict(X_scaled)

    # Jika model sklearn langsung (tanpa dict)
    else:
        prediction = model_data.predict([[years_exp, education_lvl]])

    st.success(f"💰 Perkiraan gaji: Rp {prediction[0]:,.0f}")

# -----------------------------
# PENJELASAN LOGIKA MODEL
# -----------------------------
st.markdown("""
**📘 Bagaimana Model Ini Bekerja:**

Model ini menggunakan **Regresi Linier Berganda**, yang memprediksi gaji berdasarkan:
- `experience` → tahun pengalaman kerja  
- `education` → tingkat pendidikan (1–4)

Rumus sederhananya adalah:
> `salary = b₀ + b₁ * experience + b₂ * education`

Artinya, setiap tambahan **1 tahun pengalaman** atau **1 level pendidikan**  
akan meningkatkan estimasi gaji berdasarkan bobot hasil pelatihan.
""")

# -----------------------------
# UPLOAD FILE UNTUK BATCH PREDIKSI
# -----------------------------
st.subheader("📂 Atau Unggah File CSV untuk Prediksi Massal")

uploaded_file = st.file_uploader("Upload file CSV dengan kolom 'experience' dan 'education'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if {'experience', 'education'}.issubset(df.columns):
        model_data = model
        X_input = df[['experience', 'education']]

        if isinstance(model_data, dict):
            scaler = model_data["scaler"]
            X_scaled = scaler.transform(X_input)
            X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
            w = model_data["w"]
            df['predicted_salary'] = np.dot(X_scaled, w)
        else:
            df['predicted_salary'] = model.predict(X_input)

        st.dataframe(df)

        fig, ax = plt.subplots()
        ax.scatter(df['experience'], df['predicted_salary'], color='red', label='Prediksi')
        ax.set_xlabel("Pengalaman (tahun)")
        ax.set_ylabel("Gaji (Rp)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("❌ File harus memiliki kolom 'experience' dan 'education'!")

# -----------------------------
# TEMPLATE CSV
# -----------------------------
st.subheader("📄 Contoh Template CSV yang Dapat Digunakan")

template_df = pd.DataFrame({
    "experience": [1, 2, 2, 3, 5, 7, 8, 4, 6, 3],
    "education": [1, 1, 2, 3, 3, 4, 4, 3, 4, 2],
    "salary": [4000000, 5000000, 6000000, 8000000, 10000000,
               15000000, 17000000, 9000000, 12000000, 7000000]
})

st.dataframe(template_df)

csv = template_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Template CSV",
    data=csv,
    file_name="sample_salary_template.csv",
    mime="text/csv"
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh **LordDarkness99** — Menggunakan Streamlit & Scikit-Learn")
