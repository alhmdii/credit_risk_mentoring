from pydantic import BaseModel, Field
from typing import Dict

class PengajuanKredit(BaseModel):
    person_income: float = Field(..., gt=0, description="Pendapatan tahunan nasabah")
    loan_amnt: float = Field(..., gt=0, description="Jumlah pinjaman yang diajukan")
    loan_int_rate: float = Field(..., gt=0, description="Suku bunga pinjaman (%)")
    person_home_ownership: str = Field(..., description="Status kepemilikan tempat tinggal (RENT, OWN, MORTGAGE, OTHER)")
    loan_intent: str = Field(..., description="Tujuan pinjaman")

class PrediksiResponse(BaseModel):
    status_pengajuan: str
    total_skor: int
    rincian_poin: Dict[str, int]
    pesan: str