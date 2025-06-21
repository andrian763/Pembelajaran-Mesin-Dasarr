# 🧠 Pembelajaran Mesin Dasar: Prediksi Risiko Stroke dengan SVM Manual

## 🔖 Deskripsi

Repositori ini berisi proyek pembelajaran mesin dasar yang mengimplementasikan algoritma **Support Vector Machine (SVM)** secara **manual tanpa library siap pakai seperti `sklearn.svm`**, untuk memprediksi risiko stroke pada pasien berdasarkan data kesehatan. Model ini dilengkapi dengan antarmuka pengguna berbasis **Streamlit**, memungkinkan prediksi risiko stroke secara **interaktif**.

## 📊 Dataset

Dataset diambil dari [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), berisi 5110 data pasien dan atribut medis seperti:

* Jenis kelamin
* Usia
* Riwayat hipertensi dan penyakit jantung
* Status pernikahan
* Jenis pekerjaan dan tempat tinggal
* Kadar glukosa
* BMI
* Status merokok
* Diagnosa stroke (0 = tidak, 1 = stroke)

## 📅 Tahapan Proyek

### 1. Eksplorasi dan Pra-pemrosesan

* Penghapusan duplikasi
* Penanganan nilai hilang dan outlier
* One-Hot Encoding untuk fitur kategorikal
* Normalisasi fitur numerik

### 2. Penyeimbangan Data

* Menggunakan **SMOTE** untuk oversampling data stroke
* Digabungkan dengan **random undersampling** untuk menjaga proporsi data

### 3. Implementasi SVM Manual

* SVM manual dengan **kernel polinomial**
* Menggunakan metode **gradient descent** untuk optimasi
* Fungsi hinge loss dan margin maksimal

### 4. Evaluasi Model

* Metode evaluasi: **Akurasi, Precision, Recall**
* Hasil training vs testing dievaluasi untuk mendeteksi overfitting

### 5. Antarmuka Pengguna (Streamlit)

* Input data pasien seperti usia, glukosa, BMI, riwayat medis
* Output berupa prediksi risiko stroke
* Visualisasi hasil interaktif

## 🎓 Tujuan Pembelajaran

* Memahami cara kerja algoritma SVM dari nol
* Melatih keterampilan preprocessing data dan penanganan imbalance
* Membuat aplikasi interaktif untuk prediksi medis

## 🚀 Cara Menjalankan

1. Clone repository:

```bash
git clone https://github.com/username/pembelajaran-mesin-dasar.git
cd pembelajaran-mesin-dasar
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

## 🔗 Referensi

* [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
* Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning
* [SMOTE Technique - imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

> Dibuat oleh: Nando Septian Prisandy, Aufatir Diaul Haq, Andrian Simanjuntak
> Kelas: 2023C, Universitas Negeri Surabaya
