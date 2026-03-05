import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple, Any, Dict

from src.utils import load_config, load_data, serialize_data

logger = logging.getLogger(__name__)

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

    logger.info("   -> -----------------------------------")
    logger.info(f"   -> Original data shape : {data.shape}")
    logger.info("   -> -----------------------------------")
    logger.info(f"   -> X data shape        : {X.shape}")
    logger.info(f"   -> y data shape        : {y.shape}")

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

    return X_train, X_test, y_train, y_test

def main():
    logger.info("=== MEMULAI PIPELINE DATA PREPARATION ===")

    # 1. Memuat file config
    logger.info("1. Memuat file konfigurasi (config.yaml)")
    config = load_config()

    # 2. Memuat Raw Dataset
    logger.info("2. Memuat Dataset Mentah (Load Raw data)")
    data = load_data(config["path_raw_data"])

    # 3. Memisahkan Input & Output
    logger.info("3. Memisahkan Kolom Fitur dan Target (Split Input Output)")
    X, y = split_input_output(data=data, target_col=config["target_col"])

    # 4. Membagi Data (Train, Valid, Test)
    logger.info("4. Membagi Data Menjadi Train, Valid, dan Test Set")
    #Memanggil random state dari file config
    RANDOM_STATE = config["random_state"]
    #Split pertama: Pisahkan Train (misal 80%) dan Sisanya (20%)
    X_train, X_not_train, y_train, y_not_train = split_train_test(
        X=X,
        y=y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    #Split kedua: Bagi 'Sisanya' menjadi Validasi (50%) dan Test (50%) -> Masing-masing 10% dari total
    X_valid, X_test, y_valid, y_test = split_train_test(
        X=X_not_train,
        y=y_not_train,
        test_size=0.5,
        random_state=RANDOM_STATE
    )

    logger.info("   -> -----------------------------------")
    logger.info(f"   -> X_train shape : {X_train.shape}")
    logger.info(f"   -> y_train shape : {y_train.shape}")
    logger.info("   -> -----------------------------------")
    logger.info(f"   -> X_valid shape : {X_valid.shape}")
    logger.info(f"   -> y_valid shape : {y_valid.shape}")
    logger.info("   -> -----------------------------------")
    logger.info(f"   -> X_test shape : {X_test.shape}")
    logger.info(f"   -> y_test shape : {y_test.shape}")

    # 5. Serialization (Menyimpan seluruh data)
    logger.info("5. Menyimpan Dataset Hasil Pembagian Split Train, Valid, Test (Serialization)")
    
    path_X_train, path_y_train = config["path_train_set"]
    path_X_valid, path_y_valid = config["path_valid_set"]
    path_X_test, path_y_test = config["path_test_set"]

    # Eksekusi penyimpanan
    serialize_data(X_train, path_X_train)
    serialize_data(y_train, path_y_train)
    
    serialize_data(X_valid, path_X_valid)
    serialize_data(y_valid, path_y_valid)
    
    serialize_data(X_test, path_X_test)
    serialize_data(y_test, path_y_test)

    logger.info("=== PIPELINE DATA PREPARATION SELESAI ===")

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        main()
    except Exception as e:
        logger.critical(f"PIPELINE GAGAL ERROR: {str(e)}")
        raise e