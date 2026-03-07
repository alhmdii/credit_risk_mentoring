import pandas as pd
import numpy as np
import logging
from typing import Tuple, Any, Dict, List
from src.utils import load_config, load_data, serialize_data, deserialize_data, get_project_root

logger = logging.getLogger(__name__)


def data_binning(X: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Melakukan proses coarse classing (binning) pada data numerik 
    dan penggabungan kategori pada data kategorikal.
    """
    logger.info("   Memulai fungsi data_binning.")
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fungsi data_binning: Parameter (X) harus bertipe pd.DataFrame.")
    if not isinstance(config, dict):
        raise TypeError("Fungsi data_binning: Parameter (config) harus bertipe dictionary.")
    else:
        logger.info("    -> Fungsi data_binning: parameter telah divalidasi.")
        
    df_binned = X.copy()
    cat_mapping = config.get("categorical_grouping", {})
    
    # Ambil batas binning dari config
    num_bins = {
        "person_age": config.get("bin_person_age"),
        "person_income": config.get("bin_person_income"),
        "person_emp_length": config.get("bin_person_emp_length"),
        "loan_amnt": config.get("bin_loan_amnt"),
        "loan_int_rate": config.get("bin_loan_int_rate"),
        "loan_percent_income": config.get("bin_loan_percent_income"),
        "cb_person_cred_hist_length": config.get("bin_cb_person_cred_hist_length")
    }
    
    # Eksekusi mapping kategorikal
    for col, mapping in cat_mapping.items():
        if col in df_binned.columns:
            df_binned[col] = df_binned[col].replace(mapping)
            logger.info(f"    -> Berhasil melakukan mapping kategori pada kolom: {col}")
            
    # Eksekusi binning numerikal
    for col, bins in num_bins.items():
        if col in df_binned.columns and bins is not None:
            # astype(str) agar aman dari error tipe data kategori bawaan pandas
            df_binned[col] = pd.cut(df_binned[col], bins=bins).astype(str)
            logger.info(f"    -> Berhasil melakukan binning numerikal pada kolom: {col}")
            
    logger.info("   -> Fungsi data_binning selesai dieksekusi.")
    return df_binned


def fit_woe_mappings(X_train_binned: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Mempelajari dan menghitung nilai WoE serta IV murni dari data latih (Train).
    """
    logger.info("   Memulai fungsi fit_woe_mappings.")
    
    if not isinstance(X_train_binned, pd.DataFrame):
        raise TypeError("Fungsi fit_woe_mappings: Parameter (X_train_binned) harus bertipe pd.DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("Fungsi fit_woe_mappings: Parameter (y_train) harus bertipe pd.Series.")
    else:
        logger.info("    -> Fungsi fit_woe_mappings: parameter telah divalidasi.")
        
    woe_mappings = {}
    iv_summary = []
    
    # Gabungkan sementara untuk kebutuhan groupby
    df_temp = X_train_binned.copy()
    df_temp['target'] = y_train.values
    
    total_cakep = (df_temp['target'] == 0).sum()
    total_galbay = (df_temp['target'] == 1).sum()
    eps = 0.00001 # Pelindung dari error log(0)
    
    features = [col for col in X_train_binned.columns]
    
    for col in features:
        grouped = df_temp.groupby(col, observed=False).agg(
            cakep=('target', lambda x: (x == 0).sum()),
            galbay=('target', lambda x: (x == 1).sum())
        )
        
        grouped['persentase_cakep'] = (grouped['cakep'] + eps) / total_cakep
        grouped['persentase_galbay'] = (grouped['galbay'] + eps) / total_galbay
        
        # Rumus WoE
        grouped['woe'] = np.log(grouped['persentase_cakep'] / grouped['persentase_galbay'])
        grouped['kontribusi_iv'] = (grouped['persentase_cakep'] - grouped['persentase_galbay']) * grouped['woe']
        
        total_iv = grouped['kontribusi_iv'].sum()
        w_dict = grouped['woe'].to_dict()
        
        woe_mappings[col] = w_dict
        iv_summary.append({"Nama Fitur": col, "Total IV": total_iv})
        
        logger.info(f"    -> Kolom {col}: Kalkulasi WoE selesai. Total IV = {total_iv:.4f}")
        
    logger.info("   -> Fungsi fit_woe_mappings selesai dieksekusi.")
    return woe_mappings, pd.DataFrame(iv_summary)


def data_binned_to_woe(X_binned: pd.DataFrame, woe_mappings: Dict[str, Any]) -> pd.DataFrame:
    """
    Mengubah nilai kategori/string interval menjadi angka WoE.
    Dilengkapi pengaman (fillna(0)) untuk unseen data.
    """
    logger.info("   Memulai fungsi data_binned_to_woe.")
    
    if not isinstance(X_binned, pd.DataFrame):
        raise TypeError("Fungsi data_binned_to_woe: Parameter (X_binned) harus bertipe pd.DataFrame.")
    if not isinstance(woe_mappings, dict):
        raise TypeError("Fungsi data_binned_to_woe: Parameter (woe_mappings) harus bertipe dictionary.")
    else:
        logger.info("    -> Fungsi data_binned_to_woe: parameter telah divalidasi.")
        
    X_woe = pd.DataFrame()
    
    for col in X_binned.columns:
        if col in woe_mappings:
            # Menggunakan .map dan diamankan dengan .fillna(0)
            X_woe[col] = X_binned[col].map(woe_mappings[col]).fillna(0)
            logger.info(f"    -> Berhasil mapping WoE pada kolom: {col}")
        else:
            X_woe[col] = X_binned[col]
            
    logger.info("   -> Fungsi data_binned_to_woe selesai dieksekusi.")
    return X_woe


def main():
    logger.info("=== MEMULAI PIPELINE DATA BINNING & WoE ===")
    config = load_config()
    
    logger.info("1. Memuat Dataset yang Telah Dibersihkan (Clean Data)")
    path_X_train_clean, path_y_train = config["path_train_clean"]
    path_X_valid_clean, path_y_valid = config["path_valid_clean"]
    path_X_test_clean, path_y_test = config["path_test_clean"]
    
    X_train_clean = deserialize_data(path_X_train_clean)
    y_train = deserialize_data(path_y_train)
    
    X_valid_clean = deserialize_data(path_X_valid_clean)
    y_valid = deserialize_data(path_y_valid)
    
    X_test_clean = deserialize_data(path_X_test_clean)
    y_test = deserialize_data(path_y_test)
    
    logger.info("2. Melakukan Proses Binning (Coarse Classing)")
    X_train_binned = data_binning(X_train_clean, config)
    X_valid_binned = data_binning(X_valid_clean, config)
    X_test_binned = data_binning(X_test_clean, config)
    
    logger.info("3. Mempelajari dan Mengkalkulasi WoE dari Data Latih")
    woe_mappings, df_iv_summary = fit_woe_mappings(X_train_binned, y_train)
    
    logger.info("4. Menyimpan Mapping WoE (Serialization)")
    serialize_data(woe_mappings, config["path_woe_mappings"])
    
    logger.info("5. Mentransformasi Dataset menjadi Angka WoE")
    X_train_woe = data_binned_to_woe(X_train_binned, woe_mappings)
    X_valid_woe = data_binned_to_woe(X_valid_binned, woe_mappings)
    X_test_woe = data_binned_to_woe(X_test_binned, woe_mappings)
    
    logger.info("6. Menyimpan Dataset WoE Final (Serialization)")
    serialize_data(X_train_woe, config["path_test_woe"][0].replace("test", "train"))
    serialize_data(y_train, config["path_test_woe"][1].replace("test", "train"))
    
    serialize_data(X_valid_woe, config["path_test_woe"][0].replace("test", "valid"))
    serialize_data(y_valid, config["path_test_woe"][1].replace("test", "valid"))
    
    serialize_data(X_test_woe, config["path_test_woe"][0])
    serialize_data(y_test, config["path_test_woe"][1])
    
    logger.info("=== PIPELINE DATA BINNING & WoE SELESAI DENGAN SUKSES ===")


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