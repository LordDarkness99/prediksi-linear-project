# 💼 Prediksi Gaji Berdasarkan Pengalaman Kerja

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi **gaji (Rp)** berdasarkan **tahun pengalaman kerja** dan **tingkat pendidikan** menggunakan model **Regresi Linier Berganda**. Aplikasi ini juga dilengkapi fitur edukatif dan visualisasi prediksi agar mudah dipahami oleh pengguna maupun mahasiswa yang sedang belajar *Machine Learning* dasar.

## 📘 Deskripsi Proyek

Aplikasi ini menggunakan dua pendekatan model untuk memprediksi gaji berdasarkan pengalaman kerja dan tingkat pendidikan. Tujuan utamanya adalah untuk memahami bagaimana model regresi linier berganda dengan fungsi:

\[
f(x_1, x_2) = b_0 + b_1 \times x_1 + b_2 \times x_2
\]

dapat digunakan untuk melakukan prediksi gaji, di mana:
- \(x_1\) = pengalaman kerja (tahun)
- \(x_2\) = tingkat pendidikan (skala 1-4)
- \(b_0\) = intercept (gaji dasar)
- \(b_1, b_2\) = koefisien untuk masing-masing fitur

## ⚙️ Fitur Utama

- Menggunakan **Streamlit** untuk antarmuka web interaktif
- Menggunakan **Scikit-Learn** dan **NumPy** untuk komputasi machine learning
- Menggunakan **Matplotlib** dan **Pandas** untuk visualisasi dan manipulasi data
- Dukungan **dua model**: manual calculation dan Scikit-Learn
- Prediksi **tunggal** dan **massal** (via CSV upload)
- Visualisasi hasil prediksi yang informatif
- Satuan disesuaikan dengan konteks Indonesia:
  - Pengalaman kerja → Tahun
  - Tingkat pendidikan → Skala 1-4
  - Gaji → Rupiah (Rp)

## 📊 Dataset dan Fitur

### Tingkat Pendidikan
| Kode | Tingkat Pendidikan |
|------|-------------------|
| 1    | SMA/Sederajat |
| 2    | D3/Diploma |
| 3    | S1/Sarjana |
| 4    | S2/Magister atau lebih |

### Contoh Data Training
| Pengalaman (Tahun) | Pendidikan | Gaji (Rp) |
|-------------------|------------|-----------|
| 1                 | 1          | 4,000,000 |
| 3                 | 3          | 8,000,000 |
| 5                 | 4          | 10,000,000 |
| 7                 | 4          | 15,000,000 |

## 🧠 Algoritma yang Digunakan

1. **Regresi Linier Berganda**
   \[
   y = b_0 + b_1x_1 + b_2x_2
   \]
   Di mana:
   - \(y\) = gaji prediksi
   - \(x_1\) = pengalaman kerja
   - \(x_2\) = tingkat pendidikan

2. **Fungsi Biaya (Cost Function)**
   Menggunakan *Mean Squared Error (MSE)*:
   \[
   J(b_0, b_1, b_2) = \frac{1}{2m} \sum_{i=1}^{m}(f(x_1^{(i)}, x_2^{(i)}) - y^{(i)})^2
   \]

3. **Normalisasi Fitur**
   Standardisasi fitur untuk konvergensi gradient descent yang lebih baik

## 🚀 Cara Menjalankan

### Menjalankan apk
```bash
git clone https://github.com/LordDarkness99/prediksi-linear-project.git
cd prediksi-linear-project

pip install -r requirements.txt

streamlit run webapp/main_app.py
