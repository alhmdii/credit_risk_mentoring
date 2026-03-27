import pandas as pd
from src.utils import load_config, deserialize_data

config = load_config()
scorecard_dict = deserialize_data(config["path_scorecard_dict"])

max_poin_fitur = {}
for fitur, mapping in scorecard_dict.items():
    max_poin_fitur[fitur] = max(mapping.values())

def mock_slik_ojk() -> dict:
    """
    Anggap saja ini seolah olah nembak API OJK/Internal
    """
    return {
        "cb_person_default_on_file": "N",
        "loan_grade": "B"
    }

def process_prediction(data_teller: dict) -> dict:

    data_eksternal = mock_slik_ojk()

    loan_percent_income = data_teller["loan_amnt"] / data_teller["person_income"]

    data_lengkap = {
        "person_home_ownership": data_teller["person_home_ownership"],
        "loan_intent": data_teller["loan_intent"],
        "cb_person_default_on_file": data_eksternal["cb_person_default_on_file"],
        "person_income": data_teller["person_income"],
        "loan_int_rate": data_teller["loan_int_rate"],
        "loan_percent_income": loan_percent_income
    }

    df = pd.DataFrame([data_lengkap])

    cat_mapping = config.get("categorical_grouping", {})
    for col, mapping in cat_mapping.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    
    num_bins = {
        "person_income": config.get("bin_person_income"),
        "loan_int_rate": config.get("bin_loan_int_rate"),
        "loan_percent_income": config.get("bin_loan_percent_income")
    }

    for col, bins in num_bins.items():
        if col in df.columns and bins is not None:
            parsed_bins = [float(b) for b in bins]
            df[col] = pd.cut(df[col], bins=parsed_bins).astype(str)
    
    total_skor = 0 
    rincian_poin = {}
    rincian_kekurangan = {}

    for col in df.columns:
        if col in scorecard_dict:
            kategori_nasabah = df[col].iloc[0]
            poin_didapat = int(scorecard_dict[col].get(kategori_nasabah, 0))
            poin_maksimal = int(max_poin_fitur.get(col, 0))
            
            total_skor += poin_didapat
            rincian_poin[col] = poin_didapat
            
            rincian_kekurangan[col] = poin_maksimal - poin_didapat
            
    CUT_OFF_SCORE = 555 #Mengikuti batas NPL CAP OJK DI 5%
    
    if total_skor >= CUT_OFF_SCORE:
        status = "DITERIMA"
        pesan = "Aman untuk dicairkan. Skor memenuhi standar kepatuhan OJK (Estimasi NPL < 5%)."
    else:
        status = "DITOLAK"
        kekurangan_total = CUT_OFF_SCORE - total_skor
        
        fitur_bermasalah = sorted(rincian_kekurangan.items(), key=lambda x: x[1], reverse=True)
        
        alasan_penolakan = []
        for fitur, gap in fitur_bermasalah[:3]:
            if gap > 0:
                alasan_penolakan.append(f"{fitur} (-{gap} poin dari skor ideal)")
                
        teks_alasan = ", ".join(alasan_penolakan)
        pesan = f"DITOLAK: Skor kurang {kekurangan_total} poin dari batas minimum {CUT_OFF_SCORE}. Faktor utama penyebab penolakan: {teks_alasan}."
        
    return {
        "status_pengajuan": status,
        "total_skor": total_skor,
        "rincian_poin": rincian_poin,
        "pesan": pesan
    }