import pandas as pd
import numpy as np
import logging
from typing import Tuple, Any, Dict, List
from src.utils import load_config, load_data, serialize_data, deserialize_data, get_project_root

logger = logging.getLogger(__name__)


def data_binning(X: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Mengubah fitur numerikal dengan binning menjadi kategori, dan mengubah kategori yang populasinya < 0.5% ke kategori terdekat

    Args:
        X (pd.DataFrame): Dataset dengan fitur numerikal dan kategorikal yang ingin diubah
        config (Dict[str, Any]): File config yang menuimpan angka batas binning untuk kolom numerik

    Raises:
        TypeError: Tipe data tidak sesuai
        TypeError: Tipe data tidak sesuai

    Returns:
        pd.DataFrame: Dataset yang fitur numerikalnya sudah dibinning, dan tidak ada lagi kategori yang populasinya < 0.5%
    """
    logger.info("   Memulai fungsi data_binning.")
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fungsi data_binning: Parameter (X) harus bertipe pd.DataFrame.")
    if not isinstance(config, dict):
        raise TypeError("Fungsi data_binning: Parameter (config) harus bertipe dictionary.")
    else:
        logger.info("    -> Fungsi data_binning: parameter telah divalidasi.")
    
    #Jangan ubah data asli 
    df_binned = X.copy()
    #Mengambil grouping kategorikal dari file config
    cat_mapping = config.get("categorical_grouping", {})
    
    #Mengambil nilai bining fitur numerikal dari file config 
    num_bins = {
        "person_age": config.get("bin_person_age"),
        "person_income": config.get("bin_person_income"),
        "person_emp_length": config.get("bin_person_emp_length"),
        "loan_amnt": config.get("bin_loan_amnt"),
        "loan_int_rate": config.get("bin_loan_int_rate"),
        "loan_percent_income": config.get("bin_loan_percent_income"),
        "cb_person_cred_hist_length": config.get("bin_cb_person_cred_hist_length")
    }
    
    #Mengubah fitur kategori yang populasinya kurang dari 0.5% dan menyatukannya
    for col, mapping in cat_mapping.items():
        if col in df_binned.columns:
            df_binned[col] = df_binned[col].replace(mapping)
            logger.info(f"    -> Berhasil melakukan mapping kategori pada kolom: {col}")
            
    #Mengubah fitur numerikal ke kategorikal dengan binning
    for col, bins in num_bins.items():
        if col in df_binned.columns and bins is not None:
            df_binned[col] = pd.cut(df_binned[col], bins=bins).astype(str)
            logger.info(f"    -> Berhasil melakukan binning numerikal pada kolom: {col}")
            
    logger.info("   -> Fungsi data_binning selesai dieksekusi.")

    return df_binned


def fit_woe_mappings(X_train_binned: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Mencari nilai Weight of Evidence pada setiap kategori fitur HANYA dari data train

    Args:
        X_train_binned (pd.DataFrame): Parameter input data train yang ingin dicari WoEnya
        y_train (pd.Series): Kolom target 0=NoNDefault dan 1=Default

    Raises:
        TypeError: Tipe data tidak sesuai
        TypeError: Tipe data tidak sesuai

    Returns:
        Tuple[Dict[str, Any], pd.DataFrame]: Dictionary yang berisi nilai WoE utk setiap kategori fitur
    """
    
    logger.info("   Memulai fungsi fit_woe_mappings.")
    
    if not isinstance(X_train_binned, pd.DataFrame):
        raise TypeError("Fungsi fit_woe_mappings: Parameter (X_train_binned) harus bertipe pd.DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("Fungsi fit_woe_mappings: Parameter (y_train) harus bertipe pd.Series.")
    else:
        logger.info("    -> Fungsi fit_woe_mappings: parameter telah divalidasi.")

    #Dict kosong untuk menyimpan nilai WoE setiap kategori fitur
    woe_mappings = {}
    #List kosong untuk menyimpan
    iv_summary = []
    
    #Membuat df baru
    df_temp = X_train_binned.copy()
    df_temp["target"] = y_train.values
    
    #kolom target = 0 itu tidak gagal bayar / non-default
    total_cakep = (df_temp["target"] == 0).sum()
    #kolom target = 1 itu gagal bayar / default
    total_galbay = (df_temp["target"] == 1).sum()
    eps = 0.00001 #Melindungi dari error log(0)
    
    #Menyimpan nama fitur dari parameter X_train
    features = [col for col in X_train_binned.columns]
    
    for col in features:
        grouped = df_temp.groupby(col, observed=False).agg(
            cakep=("target", lambda x: (x == 0).sum()),
            galbay=("target", lambda x: (x == 1).sum())
        )
        
        grouped["persentase_cakep"] = (grouped["cakep"] + eps) / total_cakep
        grouped["persentase_galbay"] = (grouped["galbay"] + eps) / total_galbay
        
        # Rumus WoE
        grouped["woe"] = np.log(grouped["persentase_cakep"] / grouped["persentase_galbay"])
        grouped["kontribusi_iv"] = (grouped["persentase_cakep"] - grouped["persentase_galbay"]) * grouped["woe"]
        
        total_iv = grouped["kontribusi_iv"].sum()
        w_dict = grouped["woe"].to_dict()
        
        woe_mappings[col] = w_dict

        iv_summary.append({"Nama Fitur": col, "Total IV": total_iv})
        
        logger.info(f"    -> Kolom {col}: Kalkulasi WoE selesai. Total IV = {total_iv:.4f}")
        
    logger.info("   -> Fungsi fit_woe_mappings selesai dieksekusi.")

    return woe_mappings, pd.DataFrame(iv_summary)


def data_binned_to_woe(X_binned: pd.DataFrame, woe_mappings: Dict[str, Any]) -> pd.DataFrame:
    """
    Mengubah kategori pada tiap fitur menjadi angka WoE

    Args:
        X_binned (pd.DataFrame): Dataset yang sudah dibinning
        woe_mappings (Dict[str, Any]): Dict yang berisi angka WoE utk tiap kategori fitur

    Raises:
        TypeError: Tipe data tidak sesuai
        TypeError: Tipe data tidak sesuai

    Returns:
        pd.DataFrame: Dataset yang kategori fiturnya sudah diubah menjadi angka WoE
    """
    logger.info("   Memulai fungsi data_binned_to_woe.")
    
    if not isinstance(X_binned, pd.DataFrame):
        raise TypeError("Fungsi data_binned_to_woe: Parameter (X_binned) harus bertipe pd.DataFrame.")
    if not isinstance(woe_mappings, dict):
        raise TypeError("Fungsi data_binned_to_woe: Parameter (woe_mappings) harus bertipe dictionary.")
    else:
        logger.info("    -> Fungsi data_binned_to_woe: parameter telah divalidasi.")
    
    #Membuat dataframe baru 
    X_woe = pd.DataFrame()
    
    for col in X_binned.columns:
        if col in woe_mappings:
            #Menggunakan .map dan diamankan dengan .fillna(0)
            X_woe[col] = X_binned[col].map(woe_mappings[col]).fillna(0)
            logger.info(f"    -> Berhasil mapping WoE pada kolom: {col}")
        else:
            X_woe[col] = X_binned[col]
            
    logger.info("   -> Fungsi data_binned_to_woe selesai dieksekusi.")

    return X_woe


def main():
    logger.info("=== MEMULAI PIPELINE DATA BINNING & WoE ===")

    #Memuat file config
    config = load_config()
    
    #Deserealize dataset 
    logger.info("1. Memuat Dataset yang Telah Dibersihkan (Clean Data)")
    #Mengambil lokasi file .pkl dataset dari file config
    path_X_train_clean, path_y_train = config["path_train_clean"]
    path_X_valid_clean, path_y_valid = config["path_valid_clean"]
    path_X_test_clean, path_y_test = config["path_test_clean"]
    
    X_train_clean = deserialize_data(path_X_train_clean)
    y_train = deserialize_data(path_y_train)
    
    X_valid_clean = deserialize_data(path_X_valid_clean)
    y_valid = deserialize_data(path_y_valid)
    
    X_test_clean = deserialize_data(path_X_test_clean)
    y_test = deserialize_data(path_y_test)
    
    #Binning kolom numerikal
    logger.info("2. Melakukan Proses Binning (Coarse Classing)")
    X_train_binned = data_binning(X_train_clean, config)
    X_valid_binned = data_binning(X_valid_clean, config)
    X_test_binned = data_binning(X_test_clean, config)
    
    #Mencari nilai Weight of Evidence
    logger.info("3. Mempelajari dan Mengkalkulasi WoE dari Data Latih")
    #Hanya cari/fit dari data Train
    woe_mappings, df_iv_summary = fit_woe_mappings(X_train_binned, y_train)
    
    #Serialisasi nilai Weight of Evidence
    logger.info("4. Menyimpan Mapping WoE (Serialization)")
    serialize_data(woe_mappings, config["path_woe_mappings"])
    
    #Mengubah kolom numerikal yang sudah dibinning menjadi nilai WoE
    logger.info("5. Mentransformasi Dataset menjadi Angka WoE")

    X_train_woe = data_binned_to_woe(X_train_binned, woe_mappings)
    X_valid_woe = data_binned_to_woe(X_valid_binned, woe_mappings)
    X_test_woe = data_binned_to_woe(X_test_binned, woe_mappings)
    
    #Serealisasi dataset yang sudah diubah ke nilai WoE
    logger.info("6. Menyimpan Dataset WoE Final (Serialization)")
    
    serialize_data(X_train_woe, config["path_train_woe"][0])
    serialize_data(y_train, config["path_train_woe"][1])
    
    serialize_data(X_valid_woe, config["path_valid_woe"][0])
    serialize_data(y_valid, config["path_valid_woe"][1])
    
    serialize_data(X_test_woe, config["path_test_woe"][0])
    serialize_data(y_test, config["path_test_woe"][1])
    
    logger.info("=== PIPELINE DATA BINNING & WoE SELESAI ===")

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