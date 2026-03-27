import streamlit as st
import requests
import os 

# Konfigurasi Halaman
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="🏦",
    layout="centered"
)

# Endpoint FastAPI Anda
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.title("🏦 Sistem Credit Scoring")
st.markdown("Masukkan data nasabah di bawah ini untuk mengevaluasi kelayakan pengajuan kredit berdasarkan standar NPL OJK (<5%).")
st.divider()

# Membuat Form Input
with st.form("form_pengajuan"):
    st.subheader("Data Finansial Nasabah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        person_income = st.number_input("Pendapatan Tahunan ($)", min_value=1000.0, value=50000.0, step=1000.0)
        loan_amnt = st.number_input("Jumlah Pinjaman ($)", min_value=100.0, value=10000.0, step=500.0)
        loan_int_rate = st.number_input("Suku Bunga (%)", min_value=1.0, value=10.5, step=0.1)
        
    with col2:
        person_home_ownership = st.selectbox(
            "Status Tempat Tinggal", 
            options=["RENT", "OWN", "MORTGAGE", "OTHER"]
        )
        loan_intent = st.selectbox(
            "Tujuan Pinjaman", 
            options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        
    submit_button = st.form_submit_button("Evaluasi Pengajuan", type="primary", use_container_width=True)

# Logika ketika tombol ditekan
if submit_button:
    # 1. Siapkan payload JSON sesuai schemas.py di FastAPI
    payload = {
        "person_income": person_income,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent
    }
    
    with st.spinner("Menghitung skor kredit..."):
        try:
            # 2. Tembak API
            response = requests.post(API_URL, json=payload)
            
            # Jika API membalas dengan sukses (status code 200)
            if response.status_code == 200:
                hasil = response.json()
                
                st.divider()
                st.subheader("📊 Hasil Evaluasi")
                
                # Tampilkan metrik skor utama
                col_skor, col_status = st.columns(2)
                col_skor.metric("Total Skor Kredit", hasil["total_skor"])
                
                status = hasil["status_pengajuan"]
                if status == "DITERIMA":
                    col_status.success(f"**{status}**")
                    st.success(hasil["pesan"])
                else:
                    col_status.error(f"**{status}**")
                    st.error(hasil["pesan"])
                    
                # Fitur tambahan: Menampilkan rincian poin jika teller ingin tahu
                with st.expander("Lihat Rincian Poin Model"):
                    st.json(hasil["rincian_poin"])
                    
            else:
                st.error(f"Error dari API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Gagal terhubung ke API. Pastikan server FastAPI sudah berjalan di terminal lain!")