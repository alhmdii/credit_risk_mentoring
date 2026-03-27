import uvicorn
from fastapi import FastAPI, HTTPException
import logging

from api.schemas import PengajuanKredit, PrediksiResponse
from api.inference import process_prediction

from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Credit Scorecard",
    description="Endpoint untuk evaluasi risiko kredit",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Server API aktif. Buka http://127.0.0.1:8000/docs untuk Swagger UI."}

@app.post("/predict", response_model=PrediksiResponse)
def predict_score(pengajuan: PengajuanKredit):
    try:
        data_input = pengajuan.model_dump()
        hasil = process_prediction(data_input)
        
        return hasil
        
    except Exception as e:
        logger.error(f"Gagal memproses prediksi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)