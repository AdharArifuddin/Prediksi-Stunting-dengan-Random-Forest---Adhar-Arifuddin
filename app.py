import streamlit as st
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Nama file model yang akan disimpan atau dimuat
model_filename = "stunting_model.pkl"

# Jika model belum ada, buat model baru
if not os.path.exists(model_filename):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = np.random.rand(100, 4) * 100  # Simulasi data latih
    y_train = np.random.randint(0, 2, 100)  # Simulasi label
    rf.fit(X_train, y_train)
    with open(model_filename, "wb") as file:
        pickle.dump(rf, file)

# Memuat model yang telah disimpan
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Judul aplikasi
st.title("Prediksi Stunting dengan Random Forest - Adhar Arifuddin")
st.write("""
Aplikasi ini membantu memprediksi risiko stunting pada anak berdasarkan data umur, berat badan, tinggi badan, dan jenis kelamin.
Gunakan sidebar untuk memasukkan data anak, dan aplikasi akan memberikan prediksi apakah anak tersebut berisiko stunting atau tidak.
""")

# Sidebar untuk input pengguna
st.sidebar.header("Masukkan Data Anak")
umur = st.sidebar.number_input("Umur (bulan)", min_value=0, max_value=60, value=24)
berat_badan = st.sidebar.number_input("Berat Badan (kg)", min_value=2.0, max_value=30.0, value=10.0)
tinggi_badan = st.sidebar.number_input("Tinggi Badan (cm)", min_value=40.0, max_value=120.0, value=75.0)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

# Konversi jenis kelamin menjadi angka
jenis_kelamin_num = 1 if jenis_kelamin == "Laki-laki" else 0

# Data input
input_features = np.array([[umur, berat_badan, tinggi_badan, jenis_kelamin_num]])

# Menampilkan informasi tentang input yang dimasukkan
st.write(f"**Data yang dimasukkan**:")
st.write(f"Umur: {umur} bulan")
st.write(f"Berat Badan: {berat_badan} kg")
st.write(f"Tinggi Badan: {tinggi_badan} cm")
st.write(f"Jenis Kelamin: {jenis_kelamin}")

# Tombol prediksi
if st.button("Prediksi Stunting"):
    prediction = model.predict(input_features)
    result = "Anak berisiko stunting! ⚠️" if prediction[0] == 1 else "Anak tidak berisiko stunting ✅"
    st.success(result)

    # Menampilkan kemungkinan prediksi
    probability = model.predict_proba(input_features)[0][prediction[0]]
    st.write(f"Probabilitas: {probability*100:.2f}%")

# Footer aplikasi
st.write("""
Aplikasi ini dikembangkan oleh Adhar Arifuddin untuk membantu analisis risiko stunting pada anak. Data yang dimasukkan akan digunakan untuk memprediksi apakah anak tersebut berisiko stunting berdasarkan model machine learning.
""")
