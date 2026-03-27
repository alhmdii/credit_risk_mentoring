from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_serangan_tipe_data_string_di_angka():
    """
    Skenario: Teller mengetik huruf di kolom angka.
    """
    payload_kacau = {
        "person_income": "sepuluh ribu", #Serangan Tipe Data
        "loan_amnt": 10000.0,
        "loan_int_rate": 10.5,
        "person_home_ownership": "MORTGAGE",
        "loan_intent": "PERSONAL"
    }
    response = client.post("/predict", json=payload_kacau)
    
    assert response.status_code == 422
    assert "person_income" in response.text

def test_serangan_angka_nol_dan_negatif():
    """
    Skenario: Mencegah ZeroDivisionError di backend dengan memastikan
    Pydantic menolak gaji 0 atau negatif.
    """
    payload_negatif = {
        "person_income": 0,       #Serangan Gaji Nol (Bisa bikin aplikasi mati dibagi nol)
        "loan_amnt": -5000.0,     #Serangan Pinjaman Negatif
        "loan_int_rate": 10.5,
        "person_home_ownership": "RENT",
        "loan_intent": "MEDICAL"
    }
    response = client.post("/predict", json=payload_negatif)

    assert response.status_code == 422

def test_serangan_payload_kosong():
    """
    Skenario: Aplikasi UI mengalami blank dan mengirimkan dictionary kosong ({}).
    """
    response = client.post("/predict", json={})
    
    assert response.status_code == 422

def test_payload_injeksi_ekstra():
    """
    Skenario: Ada tambahan field yang tidak dikenali di-inject ke dalam JSON payload.
    """
    payload_injeksi = {
        "person_income": 50000.0,
        "loan_amnt": 10000.0,
        "loan_int_rate": 10.0,
        "person_home_ownership": "OWN",
        "loan_intent": "EDUCATION",
        "role": "admin_bypass",                #Data Injeksi
        "cb_person_default_on_file": "N"       #Data Injeksi (Mencoba bypass SLIK)
    }
    response = client.post("/predict", json=payload_injeksi)

    assert response.status_code == 200