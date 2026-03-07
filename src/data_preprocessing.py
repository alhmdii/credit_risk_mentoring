import pandas as pd
import numpy as np 
import math
import yaml
import logging
from typing import Tuple, Any, Dict, List
from src.utils import load_config, load_data, serialize_data, deserialize_data, get_project_root

logger = logging.getLogger(__name__)

def drop_duplicate_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Mengecek data duplikat, dan menghilangkannya

    Args:
        X (pd.DataFrame): Dataset fitur input (pd.DataFrame)
        y (pd.Series): Kolom fitur output target (pd.Series)

    Raises:
        TypeError: Tipe data X tidak sesuai
        TypeError: Tipe data y tidak sesuai

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - X (pd.DataFrame)      : Dataset fitur input yang sudah dihilangkan kolom duplikatnya
            - y (pd.Series)         : Dataset kolom fitur output/target yang sudah dihilangkan kolom duplikatnnya
    """
    logger.info("   Memulai fungsi drop_duplicate_data.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fungsi drop_duplicate_data: Parameter (X) harus bertipe pd.DataFrame.")
    elif not isinstance(y, pd.Series):
        raise TypeError("Fungsi drop_duplicate_data: Parameter (y) harus bertipe pd.Series.")
    else:
        logger.info("    -> Fungsi drop_duplicate_data: parameter telah divalidasi.")
    
    #Jangan ubah data asli
    X = X.copy()
    y = y.copy()

    logger.info(f"    -> Fungsi drop_duplicate_data: shape dataset sebelum dropping duplicate adalah ->        {X.shape}.")

    #Pengecekan data duplikat
    X_duplicate = X[X.duplicated()]
    logger.info(f"    -> Fungsi drop_duplicate_data: shape dari data yang duplicate adalah ->                  {X_duplicate.shape}.")

    X_clean = (X.shape[0] - X_duplicate.shape[0], X.shape[1])
    logger.info(f"    -> Fungsi drop_duplicate_data: shape dataset setelah drop duplicate seharusnya adalah -> {X_clean}.")

    #Drop data duplikat
    X.drop_duplicates(inplace=True)
    y = y.loc[X.index]

    logger.info(f"    -> Fungsi drop_duplicate_data: shape dataset setelah dropping duplicate adalah ->        {X.shape}.")
    logger.info("   Fungsi drop_duplicate_data selesai.")

    return X, y

def filter_domain_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pengecekan kolom/fitur umur dan lama bekerja, jika dianggap outlier (umur > 100) / (lama bekerja > 60 / lama bekerja > umur)
    maka dijadikan np.NaN

    Args:
        data (pd.DataFrame): Dataset dengan fitur input yang ingin dicek outliernya

    Raises:
        TypeError: Tipe data tidak sesuai

    Returns:
        pd.DataFrame: Dataset yang telah diubah outliernya menjadi np.NaN
    """

    logger.info("   Memulai fungsi filter_domain_outliers.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi filter_domain_outliers: parameter data harus bertipe DataFrame.")
    else: 
        logger.info("    -> Fungsi filter_domain_outliers: parameter telah divalidasi.")
    
    logger.info("    -> Fungsi filter_domain_outliers: memulai pengecekan logika umur.")

    #Jangan ubah data asli
    data = data.copy()

    #Jika umur lebih dari 100 dianggap outlier
    age_outliers = data["person_age"] > 100
    logger.info(f"    -> Ditemukan {age_outliers.sum()} data dengan person_age > 100. Mengubah ke NaN.")

    #Mengubah umur yang outlier menjadi NaN
    data.loc[age_outliers, "person_age"] = np.nan

    #Jika lama bekerja diatas 60 tahun atau lama bekerja lebih dari umurnya dianggap outlier
    emp_length_outliers = (data["person_emp_length"] > 60) | (data["person_emp_length"] >= data["person_age"])
    logger.info(f"    -> Ditemukan {emp_length_outliers.sum()} data dengan person_emp_length yang mustahil. Mengubah ke NaN.")

    #Mengubah lama bekerja yang outlier menjadi NaN
    data.loc[emp_length_outliers, "person_emp_length"] = np.nan

    logger.info("   Fungsi filter_domain_outliers: pembersihan selesai.\n")

    return data

def fit_median_imputation(data: pd.DataFrame, subset_data: List[str]) -> Dict[str, float]:
    """
    Fungsi (fit) dengan mencari nilai median untuk mengisi kolom yang mempunyai np.NaN

    Args:
        data (pd.DataFrame): Dataset fitur input (pd.DataFrame)
        subset_data (List[str]): Kolom yang ingin dicari mediannya (list)

    Raises:
        TypeError: Tipe data tidak sesuai
        TypeError: Tipe data tidak sesuai 

    Returns:
        Dict[str, float]: Dict yang berisi nilai median untuk imputasi
    """

    logger.info("   Memulai fungsi fit_median_imputation.")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi fit_median_imputation: parameter data harus bertipe DataFrame.")
    elif not isinstance(subset_data, list):
        raise TypeError("Fungsi fit_median_imputation: parameter subset_data harus bertipe list.")
    else:
        logger.info("    -> Fungsi fit_median_imputation: parameter telah divalidasi.")
    
    #Dict kosong untuk menyimpan nilai median pada fitur
    imputation_data = {}

    #Pengecekan fitur lama bekerja dan umur
    if "person_emp_length" in subset_data and "person_age" in data.columns:

        #Menghilangkan np.NaN pada kolom umur dan lama bekerja 
        valid_data = data.dropna(subset=["person_age", "person_emp_length"])

        #Mencari median umur awal bekerja dari data yang valid (tidak NaN)
        #Mengecek dengan mengurangkan umur dengan lama bekerja
        starting_age = valid_data["person_age"] - valid_data["person_emp_length"]

        #Mencari median umur awal bekerja dari data yang tidak np.NaN
        median_starting_age = starting_age.median()

        #Mengisi dict kosong dengan key = median awal bekerja dan valuenya
        imputation_data["median_starting_age"] = median_starting_age
        logger.info(f"    -> [Domain Rule] Median usia mulai bekerja didapatkan: {median_starting_age} tahun.")
    
    #Pencarian median utk fitur selain lama bekerja 
    for col in subset_data:
        if col != "person_emp_length":
            imputation_data[col] = data[col].median()
    
    logger.info(f"    -> Fungsi fit_median_imputation: proses fitting telah selesai, berikut hasilnya {imputation_data}.")
    logger.info("   Fungsi fit_median_imputation: pencarian median selesai.\n")

    return imputation_data

def transform_median_imputation(data: pd.DataFrame, imputation_data: Dict[str, float]) -> pd.DataFrame:

    logger.info("   Memulai fungsi transform_median_imputation.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi transform_median_imputation: parameter data harus bertipe DataFrame.")
    elif not isinstance(imputation_data, dict):
        raise TypeError("Fungsi transform_median_imputation: parameter imputation_data harus bertipe dict.")
    else:
        logger.info("    -> Fungsi transform_median_imputation: parameter telah divalidasi.")
    
    #Jangan ubah data asli
    data = data.copy()

    #Mengecek kolom di imputation data selain median starting age
    cols_to_check = [col for col in imputation_data.keys() if col != "median_starting_age"]
    #Jika ada median starting age, masukan ke cols to check
    if "median_starting_age" in imputation_data:
        cols_to_check.append("person_emp_length")
    
    logger.info("    -> Fungsi transform_median_imputation: informasi count na SEBELUM dilakukan imputasi:")
    na_sebelum = data[cols_to_check].isna().sum().to_dict()
    for col, count in na_sebelum.items():
        logger.info(f"       - {col}: {count} NaN")


    standard_impute_dict = {k:v for k, v in imputation_data.items() if k != "median_starting_age"}

    if standard_impute_dict:
        data.fillna(standard_impute_dict, inplace=True)

    if "median_starting_age" in imputation_data and "person_emp_length" in data.columns:
        missing_emp_idx = data["person_emp_length"].isna()

        if missing_emp_idx.sum() > 0:
            calculated_emp = data.loc[missing_emp_idx, "person_age"] - imputation_data["median_starting_age"]
            calculated_emp = calculated_emp.apply(lambda x: max(0, x))
            data.loc[missing_emp_idx, "person_emp_length"] = calculated_emp
        
    logger.info("    -> Fungsi transform_median_imputation: informasi count na SESUDAH dilakukan imputasi:")
    na_sesudah = data[cols_to_check].isna().sum().to_dict()
    for col, count in na_sesudah.items():
        logger.info(f"       - {col}: {count} NaN")
    
    logger.info("   Fungsi transform_median_imputation selesai.")

    return data

def float_convert(data: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:

    logger.info("   Memulai fungsi float_convert.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi float_convert: parameter data harus bertipe DataFrame.")
    elif not isinstance(num_cols, list):
        raise TypeError("Fungsi float_convert: parameter num_cols harus bertipe list.")
    else:
        logger.info("    -> Fungsi float_convert: parameter telah divalidasi.")
    
    data = data.copy()
    valid_cols = [col for col in num_cols if col in data.columns]

    logger.info("    -> Fungsi float_convert: tipe data SEBELUM konversi:")
    dtypes_sebelum = data[valid_cols].dtypes.astype(str).to_dict()
    for col, dtype in dtypes_sebelum.items():
        logger.info(f"       - {col}: {dtype}")

    for col in valid_cols:
        data[col] = data[col].astype("float64")

    logger.info("    -> Fungsi float_convert: tipe data SESUDAH konversi:")
    dtypes_sesudah = data[valid_cols].dtypes.astype(str).to_dict()
    for col, dtype in dtypes_sesudah.items():
        logger.info(f"       - {col}: {dtype}")

    logger.info("   Fungsi float_convert selesai.")
    return data

def fit_mode_imputation(data: pd.DataFrame, cat_cols: List[str]) -> Dict[str, str]:

    logger.info("   Memulai fungsi fit_mode_imputation.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi fit_mode_imputation: parameter data harus bertipe DataFrame.")
    elif not isinstance(cat_cols, list): # Typo cat_col -> cat_cols telah diperbaiki di sini
        raise TypeError("Fungsi fit_mode_imputation: parameter cat_cols harus bertipe list.")
    else:
        logger.info("    -> Fungsi fit_mode_imputation: parameter telah divalidasi.")

    imputation_data = {}
    for col in cat_cols:
        if col in data.columns:
            imputation_data[col] = data[col].mode()[0]
            
    logger.info(f"    -> Fungsi fit_mode_imputation: proses fitting selesai, hasil:")
    logger.info(f"     {imputation_data}")
    logger.info("   Fungsi fit_mode_imputation selesai.")
    return imputation_data

def transform_mode_imputation(data: pd.DataFrame, imputation_data: Dict[str, str]) -> pd.DataFrame:

    logger.info("   Memulai fungsi transform_mode_imputation.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi transform_mode_imputation: parameter data harus bertipe DataFrame.")
    elif not isinstance(imputation_data, dict):
        raise TypeError("Fungsi transform_mode_imputation: parameter imputation_data harus bertipe dict.")
    else:
        logger.info("    -> Fungsi transform_mode_imputation: parameter telah divalidasi.")
        
    data = data.copy()
    valid_cols = list(imputation_data.keys())

    logger.info("    -> Fungsi transform_mode_imputation: count na SEBELUM imputasi:")
    na_sebelum = data[valid_cols].isna().sum().to_dict()
    for col, count in na_sebelum.items():
        logger.info(f"       - {col}: {count} NaN")

    data.fillna(imputation_data, inplace=True)
    
    logger.info("    -> Fungsi transform_mode_imputation: count na SESUDAH imputasi:")
    na_sesudah = data[valid_cols].isna().sum().to_dict()
    for col, count in na_sesudah.items():
        logger.info(f"       - {col}: {count} NaN")

    logger.info("   Fungsi transform_mode_imputation selesai.")
    return data

def object_convert(data: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:

    logger.info("   Memulai fungsi object_convert.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi object_convert: parameter data harus bertipe DataFrame.")
    elif not isinstance(cat_cols, list):
        raise TypeError("Fungsi object_convert: parameter cat_cols harus bertipe list.")
    else:
        logger.info("    -> Fungsi object_convert: parameter telah divalidasi.")

    logger.info("    -> Fungsi object_convert: memulai konversi")
    data = data.copy()

    valid_cols = [col for col in cat_cols if col in data.columns]

    logger.info("    -> Fungsi object_convert: tipe data SEBELUM konversi:")
    dtypes_sebelum = data[valid_cols].dtypes.astype(str).to_dict()
    for col, dtype in dtypes_sebelum.items():
        logger.info(f"       - {col}: {dtype}")
    
    for col in valid_cols:
        data[col] = data[col].astype("object")
    
    logger.info("    -> Fungsi object_convert: tipe data SESUDAH konversi:")
    dtypes_sesudah = data[valid_cols].dtypes.astype(str).to_dict()
    for col, dtype in dtypes_sesudah.items():
        logger.info(f"       - {col}: {dtype}")
    
    logger.info("   Fungsi object_convert selesai.")
    return data

def main():
    logger.info("=== MEMULAI PIPELINE DATA PREPROCESSING ===")

    # 1. Memuat file konfigurasi
    logger.info("1. Memuat file konfigurasi (config.yaml)")
    config = load_config()

    # 2. Memuat data hasil split (Deserialize)
    logger.info("2. Memuat data hasil split (Train, Valid, Test)")
    X_train = deserialize_data(config["path_train_set"][0])
    y_train = deserialize_data(config["path_train_set"][1])
    
    X_valid = deserialize_data(config["path_valid_set"][0])
    y_valid = deserialize_data(config["path_valid_set"][1])
    
    X_test = deserialize_data(config["path_test_set"][0])
    y_test = deserialize_data(config["path_test_set"][1])

    # 3. Drop Duplicates (HANYA UNTUK TRAIN SET)
    logger.info("3. Melakukan Drop Duplicates (Hanya pada Data Training)")
    X_train, y_train = drop_duplicate_data(X_train, y_train)

    # 4. Filter Domain Outliers (Ubah nilai mustahil ke NaN)
    logger.info("4. Memfilter Outlier berdasarkan Logika Domain")
    X_train = filter_domain_outliers(X_train)
    X_valid = filter_domain_outliers(X_valid)
    X_test = filter_domain_outliers(X_test)

    # 5. Imputasi Numerikal (Median & Domain Rule)
    logger.info("5. Melakukan Imputasi Data Numerikal")
    # Fit HANYA di X_train, lalu simpan ke .pkl
    imputer_median_dict = fit_median_imputation(X_train, config["columns_num"])
    serialize_data(imputer_median_dict, config["path_imputer_median"])

    # Transform ke semua set data
    X_train = transform_median_imputation(X_train, imputer_median_dict)
    X_valid = transform_median_imputation(X_valid, imputer_median_dict)
    X_test = transform_median_imputation(X_test, imputer_median_dict)

    # 6. Imputasi Kategorikal (Modus)
    logger.info("6. Melakukan Imputasi Data Kategorikal")
    # Fit HANYA di X_train, lalu simpan ke .pkl
    imputer_mode_dict = fit_mode_imputation(X_train, config["columns_cat"])
    serialize_data(imputer_mode_dict, config["path_imputer_mode"])

    # Transform ke semua set data
    X_train = transform_mode_imputation(X_train, imputer_mode_dict)
    X_valid = transform_mode_imputation(X_valid, imputer_mode_dict)
    X_test = transform_mode_imputation(X_test, imputer_mode_dict)

    # 7. Penyeragaman Tipe Data (Casting)
    logger.info("7. Melakukan Penyeragaman Tipe Data (Casting)")
    # Numerikal menjadi float64
    X_train = float_convert(X_train, config["columns_num"])
    X_valid = float_convert(X_valid, config["columns_num"])
    X_test = float_convert(X_test, config["columns_num"])

    # Kategorikal menjadi object
    X_train = object_convert(X_train, config["columns_cat"])
    X_valid = object_convert(X_valid, config["columns_cat"])
    X_test = object_convert(X_test, config["columns_cat"])

    # 8. Menyimpan Data Bersih (Serialization)
    logger.info("8. Menyimpan Dataset yang Telah Dibersihkan (Serialization)")
    serialize_data(X_train, config["path_train_clean"][0])
    serialize_data(y_train, config["path_train_clean"][1])

    serialize_data(X_valid, config["path_valid_clean"][0])
    serialize_data(y_valid, config["path_valid_clean"][1])

    serialize_data(X_test, config["path_test_clean"][0])
    serialize_data(y_test, config["path_test_clean"][1])

    logger.info("=== PIPELINE DATA PREPROCESSING SELESAI ===")


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
