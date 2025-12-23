"""
================================================================================
UAS BENGKEL KODING - DEPLOYMENT
Aplikasi Prediksi Customer Churn
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import os

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Prediksi Customer Churn",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL DAN PREPROCESSOR
# ============================================================================
@st.cache_resource
def load_resource():
    """Load model dan preprocessor yang disimpan dalam satu file pkl"""
    model_path = 'model_churn_terbaik.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"File '{model_path}' tidak ditemukan!")
        return None

    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Mengambil model dan preprocessor dari dictionary
        return data['model'], data['preprocessor']
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Inisialisasi model dan preprocessor
resource = load_resource()
if resource:
    model, preprocessor = resource
else:
    model, preprocessor = None, None

# ============================================================================
# HEADER DAN SIDEBAR
# ============================================================================
st.title("Prediksi Customer Churn Telco")
st.markdown("---")

with st.sidebar:
    st.header("Informasi Aplikasi")
    st.info("""
    Aplikasi Prediksi Customer Churn
    
    Aplikasi ini memprediksi apakah pelanggan akan:
    - Tetap Berlangganan (No Churn)
    - Berhenti Berlangganan (Churn)
    
    Dataset: Telco Customer Churn
    Model: Machine Learning dengan SMOTE dan Preprocessing
    """)
    
    st.markdown("---")
    st.markdown("**Identitas Mahasiswa:**")
    st.text("Nama  : Angelica Widyastuti Kolo")
    st.text("NIM   : A11.2021.13212")
    st.text("Kelas : DS01")
    
    st.markdown("---")
    st.markdown("**UAS BENGKEL KODING**")
    st.markdown("Data Science 2025")

# ============================================================================
# TABS UNTUK NAVIGASI
# ============================================================================
tab1, tab2, tab3 = st.tabs(["Prediksi", "Panduan", "Tentang Model"])

# ============================================================================
# TAB 1: FORM PREDIKSI
# ============================================================================
with tab1:
    st.header("Form Prediksi Customer Churn")
    st.markdown("Silakan isi data pelanggan di bawah ini:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Demografis")
        gender_display = st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"])
        gender = "Male" if gender_display == "Laki-laki" else "Female"
        
        senior_citizen_display = st.selectbox("Status Lansia", options=["Tidak", "Iya"])
        senior_citizen = 1 if senior_citizen_display == "Iya" else 0
        
        partner_display = st.selectbox("Memiliki Pasangan", options=["Tidak", "Iya"])
        partner = "Yes" if partner_display == "Iya" else "No"
        
        dependents_display = st.selectbox("Memiliki Tanggungan", options=["Tidak", "Iya"])
        dependents = "Yes" if dependents_display == "Iya" else "No"
        
        st.markdown("---")
        st.subheader("Layanan Telepon")
        phone_service_display = st.selectbox("Layanan Telepon", options=["Tidak", "Iya"])
        phone_service = "Yes" if phone_service_display == "Iya" else "No"
        
        multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"])
    
    with col2:
        st.subheader("Layanan Internet")
        internet_service = st.selectbox("Jenis Internet", options=["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Keamanan Online", options=["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Backup Online", options=["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Proteksi Perangkat", options=["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])
    
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("Informasi Akun")
        tenure = st.slider("Lama Berlangganan (bulan)", 0, 72, 12)
        contract = st.selectbox("Jenis Kontrak", options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Tagihan Tanpa Kertas", options=["No", "Yes"])
    
    with col4:
        st.subheader("Informasi Pembayaran")
        payment_method = st.selectbox("Metode Pembayaran", options=[
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col5:
        st.subheader("Biaya Langganan")
        monthly_charges = st.number_input("Biaya Bulanan (USD)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Biaya (USD)", min_value=0.0, value=tenure * monthly_charges)
    
    st.markdown("---")
    if st.button("PREDIKSI CHURN", use_container_width=True, type="primary"):
        if model is not None and preprocessor is not None:
            # Buat DataFrame input
            input_df = pd.DataFrame([{
                'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
                'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
                'MultipleLines': multiple_lines, 'InternetService': internet_service,
                'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
                'DeviceProtection': device_protection, 'TechSupport': tech_support,
                'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
                'Contract': contract, 'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }])
            
            try:
                # Transformasi data menggunakan preprocessor yang dimuat
                input_preprocessed = preprocessor.transform(input_df)
                
                # Prediksi
                prediction = model.predict(input_preprocessed)[0]
                proba = model.predict_proba(input_preprocessed)[0]
                
                st.header("Hasil Prediksi")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction == 1:
                        st.error("PELANGGAN BERPOTENSI CHURN")
                        st.write("Rekomendasi: Berikan penawaran khusus atau diskon retensi.")
                    else:
                        st.success("PELANGGAN TIDAK CHURN")
                        st.write("Rekomendasi: Pertahankan kualitas layanan dan program loyalitas.")
                
                with res_col2:
                    st.write(f"Probabilitas Churn: {proba[1]*100:.2f}%")
                    st.progress(float(proba[1]))
            except Exception as e:
                st.error(f"Error saat pemrosesan data: {e}")
        else:
            st.error("Model atau Preprocessor tidak tersedia.")

# ============================================================================
# TAB 2 & 3 (PANDUAN & TENTANG) - Versi Ringkas Tanpa Emoji
# ============================================================================
with tab2:
    st.header("Panduan Penggunaan")
    st.write("1. Masukkan data profil pelanggan pada form prediksi.")
    st.write("2. Klik tombol Prediksi Churn.")
    st.write("3. Hasil akan muncul berupa status potensi churn dan persentase probabilitas.")

with tab3:
    st.header("Tentang Model")
    st.write("Model ini dikembangkan menggunakan dataset Telco Customer Churn.")
    st.write("Algoritma: Logistic Regression / Random Forest / Voting Classifier.")
    st.write("Teknik: SMOTE digunakan untuk menangani ketidakseimbangan data.")

st.markdown("---")
st.markdown("<div style='text-align: center'>2025 UAS Bengkel Koding - Data Science</div>", unsafe_allow_html=True)