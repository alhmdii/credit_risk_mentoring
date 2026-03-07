import pandas as pd
import logging
from typing import Dict, Any
from sklearn.preprocessing import OneHotEncoder
from src.utils import load_config, deserialize_data, serialize_data

# Inisialisasi logger level modul
logger = logging.getLogger(__name__)


def fit_ohe(X_train: pd.DataFrame, config: Dict[str, Any]) -> OneHotEncoder:
    """
    Melakukan fitting OneHotEncoder langsung pada DataFrame X_train
    untuk semua kolom kategorikal secara serentak.
    """
    logger.info("   Memulai fungsi fit_ohe.")
    
    # Validasi parameter
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("Fungsi fit_ohe: Parameter (X_train) harus bertipe pd.DataFrame.")
    if not isinstance(config, dict):
        raise TypeError("Fungsi fit_ohe: Parameter (config) harus bertipe dictionary.")
    else:
        logger.info("    -> Fungsi fit_ohe: parameter telah divalidasi.")
        
    # Ambil daftar kolom kategorikal dari config
    cat_cols = config.get("columns_cat", [])
    
    # Inisialisasi OHE (sparse_output=False agar jadi array 2D biasa, handle_unknown='ignore' agar kebal unseen data)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fitting model
    logger.info("    -> Melakukan fitting OneHotEncoder pada kolom kategorikal.")
    ohe.fit(X_train[cat_cols])
    
    # Mencetak kategori yang berhasil dipelajari
    for col, categories in zip(cat_cols, ohe.categories_):
        logger.info(f"    -> Kategori dipelajari untuk '{col}': {categories.tolist()}")
        
    logger.info("   -> Fungsi fit_ohe selesai dieksekusi.")
    return ohe


def transform_ohe(X: pd.DataFrame, ohe: OneHotEncoder, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Melakukan transformasi One-Hot Encoding pada data kategorikal
    dan menggabungkannya kembali dengan sisa data numerikal.
    """
    logger.info("   Memulai fungsi transform_ohe.")
    
    # Validasi parameter
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fungsi transform_ohe: Parameter (X) harus bertipe pd.DataFrame.")
    if not isinstance(ohe, OneHotEncoder):
        raise TypeError("Fungsi transform_ohe: Parameter (ohe) harus bertipe OneHotEncoder.")
    if not isinstance(config, dict):
        raise TypeError("Fungsi transform_ohe: Parameter (config) harus bertipe dictionary.")
    else:
        logger.info("    -> Fungsi transform_ohe: parameter telah divalidasi.")
        
    cat_cols = config.get("columns_cat", [])
    
    logger.info("    -> Melakukan transformasi matriks One-Hot Encoding.")
    ohe_features = ohe.transform(X[cat_cols])
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    
    # Membungkus array menjadi DataFrame dengan index asli (Anti-Bocor)
    df_ohe = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=X.index)
    
    # Membuang kolom kategorikal asli
    X_dropped = X.drop(columns=cat_cols)
    
    logger.info("    -> Menggabungkan hasil OHE dengan kolom numerikal.")
    X_final = pd.concat([X_dropped, df_ohe], axis=1)
    
    logger.info(f"    -> Transformasi berhasil. Total fitur sekarang menjadi: {X_final.shape[1]} kolom.")
    logger.info("   -> Fungsi transform_ohe selesai dieksekusi.")
    
    return X_final


def main():
    logger.info("=== MEMULAI PIPELINE DATA ENCODING (OHE) ===")
    config = load_config()
    
    logger.info("   1. Memuat Dataset yang Telah Dibersihkan (Clean Data)")
    path_X_train_clean, path_y_train = config["path_train_clean"]
    path_X_valid_clean, path_y_valid = config["path_valid_clean"]
    path_X_test_clean, path_y_test = config["path_test_clean"]
    
    X_train_clean = deserialize_data(path_X_train_clean)
    y_train = deserialize_data(path_y_train)
    
    X_valid_clean = deserialize_data(path_X_valid_clean)
    y_valid = deserialize_data(path_y_valid)
    
    X_test_clean = deserialize_data(path_X_test_clean)
    y_test = deserialize_data(path_y_test)
    
    logger.info("   2. Mempelajari dan Memfitting OneHotEncoder dari Data Latih")
    ohe_model = fit_ohe(X_train_clean, config)
    
    logger.info("   3. Menyimpan Model OHE (Serialization)")
    serialize_data(ohe_model, config.get("path_ohe_model", "models/ohe_model.pkl"))
    
    logger.info("   4. Mentransformasi Dataset menjadi OHE")
    X_train_ohe = transform_ohe(X_train_clean, ohe_model, config)
    X_valid_ohe = transform_ohe(X_valid_clean, ohe_model, config)
    X_test_ohe = transform_ohe(X_test_clean, ohe_model, config)
    
    logger.info("   5. Menyimpan Dataset OHE Final ke folder modeling_ready/")
    # Default path diarahkan langsung ke data/modeling_ready/
    path_test_ohe = config.get("path_test_ohe", [
        "data/modeling_ready/X_test_ohe.pkl", 
        "data/modeling_ready/y_test_ohe.pkl"
    ])
    
    serialize_data(X_train_ohe, path_test_ohe[0].replace("test", "train"))
    serialize_data(y_train, path_test_ohe[1].replace("test", "train"))
    
    serialize_data(X_valid_ohe, path_test_ohe[0].replace("test", "valid"))
    serialize_data(y_valid, path_test_ohe[1].replace("test", "valid"))
    
    serialize_data(X_test_ohe, path_test_ohe[0])
    serialize_data(y_test, path_test_ohe[1])
    
    logger.info("=== PIPELINE DATA ENCODING (OHE) SELESAI DENGAN SUKSES ===")


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