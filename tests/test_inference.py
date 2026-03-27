import pytest 
import math 
from api.inference import process_prediction

def test_skenario_nasabah_alien():
    """
    Skenario: Teller memasukkan kategori yang tidak ada di dunia nyata.
    """
    data_aneh = {
        "person_income": 50000.0,
        "loan_amnt": 10000.0,
        "loan_int_rate": 10.0,
        "person_home_ownership": "GUA_BATU", 
        "loan_intent": "BELI_PLANET"         
    }
    hasil = process_prediction(data_aneh)
    
    assert hasil["status_pengajuan"] == "DITOLAK"
    assert "Faktor utama penyebab penolakan" in hasil["pesan"]

def test_skenario_rasio_gila():
    """
    Skenario: Nasabah gaji $1, tapi pinjam $1,000,000.
    """
    data_gila = {
        "person_income": 1.0, 
        "loan_amnt": 1000000.0,
        "loan_int_rate": 20.0,
        "person_home_ownership": "RENT",
        "loan_intent": "VENTURE"
    }
    hasil = process_prediction(data_gila)
    
    assert hasil["status_pengajuan"] == "DITOLAK"
    assert hasil["total_skor"] < 555

def test_skenario_boundary_binning():
    """
    Skenario: Nilai pas berada di perbatasan binning (Boundary Value Analysis).
    Misal di config: bin_person_income ada 35000.0 dan 60000.0. 
    """
    data_boundary = {
        "person_income": 60000.0, #Pas di perbatasan binning
        "loan_amnt": 12500.0,     #Pas di perbatasan binning
        "loan_int_rate": 9.5,     #Pas di perbatasan binning
        "person_home_ownership": "OWN",
        "loan_intent": "PERSONAL"
    }
    hasil = process_prediction(data_boundary)
    
    assert isinstance(hasil["total_skor"], int)
    assert "status_pengajuan" in hasil