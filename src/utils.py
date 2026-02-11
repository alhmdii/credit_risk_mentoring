import pandas as pd
import joblib
import yaml
import os
from pathlib import Path
from typing import Tuple, Any, Dict
from sklearn.model_selection import train_test_split

def get_project_root() -> Path:
    """Mengembalikan path ke root folder project"""
    return Path(__file__).resolve().parent.parent

def load_config(filename: str = "config.yaml") -> Dict[str, Any]:

    root = get_project_root()
    path_config = root / "config" / filename

    with open(path_config, "r") as file:
        config = yaml.safe_load(file)

    config["path_raw_dir"] = str(root / config["directories"]["raw"])
    config["path_interim_dir"] = str(root / config["directories"]["interim"])
    config["path_processed_dir"] = str(root / config["directories"]["processed"])
    config["path_raw_data"] = str(root / config["directories"]["raw"] / config["files"]["dataset_name"])

    interim_files = config["files"]["interim"]
    for key, filename in interim_files.items():
        full_path = Path(config["path_interim_dir"]) / filename
        config[f"path_{key}"] = str(full_path)    
    return config

def load_data(fname: str) -> pd.DataFrame:
    """
    Memuat dataset dari file csv ke dalam pandas dataframe.

    Args:
        fname (str): lokasi file (path) .csv 

    Returns: 
        pd.DataFrame: DataFrame yang berisi data dari file CSV tersebut
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File tidak ditemukan di {fname}")

    data = pd.read_csv(fname)
    print(f"Data Shape: {data.shape}")

    return data

def split_input_output(data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Memisahkan dataset menjadi fitur (X) dan target (y)

    Fungsi ini menghapus kolom target dari dataframe utama untuk membuat
    kumpulan fitur, dan mengekstrak kolom target tersebut secara terpisah 

    Args:
        data (pd.DataFrame): Dataframe asli (raw) yang berisi fitur dan target
        target_col (str): Nama kolom yang akan dijadikan label/target

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X (pd.DataFrame): fitur (semua kolom kecuali target)
            - y (pd.Series): target (hanya kolom target)
    """

    X = data.drop(target_col, axis=1)
    y = data[target_col]

    print(f"Original data shape: {data.shape}")
    print(f"X data shape       : {X.shape}")
    print(f"y data shape       : {y.shape}")

    return X, y

def split_train_test(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Membagi data menjadi training set dan testing set
    Fungsi ini menggunakan train_test_split dari sklearn untuk memastikan 
    konsistensi random state

    Args:
        X (pd.DataFrame): Fitur
        y (pd.Series): Target
        test_size (float): Proporsi data untuk test set (0.0 - 1.0)
        random_state (int): Seed untuk reproduksibilitas hasil

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            Urutannya adalah: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"X train shape: {X_train.shape}")
    print(f"X test shape : {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y test shape : {y_test.shape}")

    return X_train, X_test, y_train, y_test

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

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    joblib.dump(data, path)
    print(f"Sukses menyimpan data ke: {path}")

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
    if not os.path.exists(path):
        raise FileNotFoundError(f"File .pkl tidak ditemuka di {path}")

    data = joblib.load(path)
    return data