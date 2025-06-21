import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os

def load_model_params(model_path):
    with open(model_path, 'r') as file:
        model_data = json.load(file)
    w = np.array(model_data['weights'])
    b = float(model_data['bias'])
    return w, b

def predict_manual_svm(X, w, b):
    scores = np.dot(X, w) + b
    return np.where(scores >= 0, 1, 0)

def encode_user_input(user_input, categorical_columns, encoders):
    encoded_data = []
    for col in categorical_columns:
        col_data = np.array([[user_input[col]]])
        try:
            enc_col = encoders[col].transform(col_data).flatten()
            encoded_data.append(enc_col)
        except ValueError as e:
            st.error(f"Kesalahan encoding untuk kolom {col}: {e}. Pastikan input sesuai dengan kategori pelatihan.")
            return None
    
    encoded_data = np.concatenate(encoded_data)
    
    numeric_data = np.array([
        user_input['age'],
        user_input['hypertension'],
        user_input['heart_disease'],
        user_input['avg_glucose_level'],
        user_input.get('bmi', 0.0) 
    ])
    
    final_data = np.concatenate([encoded_data, numeric_data])
    
    return final_data

def load_encoders(categorical_columns):
    encoders = {}
    for col in categorical_columns:
        encoder_path = f"manual_svm_model_output/encoder_{col}.joblib"
        if os.path.exists(encoder_path):
            encoders[col] = joblib.load(encoder_path)
        else:
            st.error(f"File encoder untuk {col} tidak ditemukan di {encoder_path}")
            return None
    return encoders

def main():
    st.title("Prediksi Kemungkinan Stroke")
    st.write("Masukkan informasi berikut untuk memprediksi kemungkinan stroke:")

    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    categories = {
        'gender': ['Male', 'Female'],
        'ever_married': ['Yes', 'No'],
        'work_type': ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
        'Residence_type': ['Urban', 'Rural'],
        'smoking_status': ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
    }

    encoders = load_encoders(categorical_columns)
    if encoders is None:
        st.error("Gagal memuat encoder. Pastikan file encoder tersedia di folder 'manual_svm_model_output'.")
        return

    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Jenis Kelamin", options=categories['gender'])
            age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=30.0, step=0.1)
            hypertension = st.selectbox("Hipertensi", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            heart_disease = st.selectbox("Penyakit Jantung", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
            ever_married = st.selectbox("Pernah Menikah", options=categories['ever_married'])
        
        with col2:
            work_type = st.selectbox("Jenis Pekerjaan", options=categories['work_type'])
            residence_type = st.selectbox("Tipe Tempat Tinggal", options=categories['Residence_type'])
            avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", min_value=0.0, value=100.0, step=0.1)
            bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=0.0, value=25.0, step=0.1)
            smoking_status = st.selectbox("Status Merokok", options=categories['smoking_status'])
        
        submitted = st.form_submit_button("Prediksi")

        if submitted:
            user_input = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }

            encoded_input = encode_user_input(user_input, categorical_columns, encoders)
            if encoded_input is None:
                return
            
            model_path = 'manual_svm_model_output/manual_svm_params.json'
            try:
                w, b = load_model_params(model_path)
            except FileNotFoundError:
                st.error(f"File model tidak ditemukan di {model_path}")
                return
            
            try:
                prediction = predict_manual_svm(encoded_input.reshape(1, -1), w, b)
                
                st.subheader("Hasil Prediksi")
                if prediction[0] == 1:
                    st.error("Peringatan: Anda memiliki risiko tinggi untuk terkena stroke!")
                else:
                    st.success("Anda memiliki risiko rendah untuk terkena stroke.")
            except ValueError as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

if __name__ == "__main__":
    main()