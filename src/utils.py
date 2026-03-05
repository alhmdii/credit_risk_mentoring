import os 
import yaml 
import joblib 
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Any, Dict

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Mengembalikan path ke root folder project"""
    return Path(__file__).resolve().parent.parent

def load_config(filename: str = "config.yaml") -> Dict[str, Any]:
    """
    Memuat file config

    Args:
        filename (str, optional): nama file config. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: file config
    """
    #Mengembalikan path ke root folder project
    root = get_project_root()

    #Membaca path folder config
    path_config = root / "config" / filename

    #Memuat file config dengan yaml safe load
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)

    return config

def load_data(fname: str) -> pd.DataFrame:
    """
    Memuat dataset dari file csv ke dalam pandas dataframe.

    Args:
        fname (str): lokasi file (path) .csv 

    Returns: 
        pd.DataFrame: Dataset yang berisi data dari file CSV tersebut
    """
    #Membaca dari root folder project
    root = get_project_root()

    #Setelah root, baru masuk ke file
    filepath  = root / fname

    if not filepath.exists():
        #Rekam jejak error sebelum program dimatikan
        logger.error(f"   -> GAGAL: File tidak ditemukan pada path {filepath}")
        raise FileNotFoundError(f"File tidak ditemukan di {filepath}")

    #Catat proses yang sedang berjalan
    logger.info(f"   -> Mulai memuat data dari: {filepath}")
    
    #Memanggil pandas utk membaca file
    data = pd.read_csv(filepath)
    
    # Catat keberhasilan dan bentuk datanya
    logger.info(f"   -> Sukses memuat data. Dimensi dataset: {data.shape}")
    
    return data

def serialize_data(data: Any, path: str) -> None:
    """
    Menyimpan python object ke dalam file serealisasi 
    Fungsi ini menggunakan library joblib untuk mengubah objek seperti
    DataFrame atau Model menjadi file .pkl agar bisa digunakan kembali
    tanpa harus memproses ulang dari awal.

    Args:
        data (Any): Objek yang ingin disimpan, bisa berupa DataFrame, list, atau model
        path (str): Lokasi dan nama file tujuan
    """
    #Membaca dari root folder project
    root = get_project_root()

    #Setelah root, baru masuk ke file
    filepath  = root / path

    #Buat direktori jika belum ada 
    os.makedirs(filepath.parent, exist_ok=True)
    
    joblib.dump(data, filepath)

    logger.info(f"   -> Sukses menyimpan data ke: {filepath}")

def deserialize_data(path: str) -> Any:
    """
    Memuat kembali objek yang telah disimpan dari hasil serealisasi
    Fungsi ini membaca file .pkl yang ada di folder dan mengembalikannya
    ke dalam variabel python

    Args:
        path (str): Lokasi file .pkl yang ingin dimuat

    Returns:
        Any: Objek asli yang sebelumnya disimpan 
    """
    #Membaca dari root folder project
    root = get_project_root()

    #Setelah root, baru masuk ke file
    filepath  = root / path

    if not filepath.exists():
        logger.error(f"File tidak ditemukan: {filepath}")
        raise FileNotFoundError(f"File .pkl tidak ditemukan di {filepath}")
    
    return joblib.load(filepath)