
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, List

COLUMNAS_EXCLUIR = [
    "timestamp", "target_subida", "target_bajada", "target_direccion",
    "retorno_futuro_4h", "retorno_futuro_12h", "retorno_futuro_24h",
    "max_drawdown_24h", "max_subida_24h", "regimen_etiqueta",
    "fear_greed_zona", "open", "high", "low", "close", "volume", "close_btc",
]
TARGET_PRINCIPAL = "target_subida"

def cargar_dataset(simbolo, dir_features="data/features"):
    ruta = os.path.join(dir_features, f"{simbolo}_features.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró {ruta}")
    df = pd.read_csv(ruta, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert TARGET_PRINCIPAL in df.columns
    assert len(df) > 1000
    return df

def obtener_columnas_features(df):
    excluir = set(COLUMNAS_EXCLUIR)
    cols_texto = set(df.select_dtypes(include=["object"]).columns)
    return [c for c in df.columns if c not in excluir and c not in cols_texto]

def dividir_temporal(df, pct_train=0.70, pct_val=0.15):
    n = len(df)
    idx_val  = int(n * pct_train)
    idx_test = int(n * (pct_train + pct_val))
    train = df.iloc[:idx_val].copy()
    val   = df.iloc[idx_val:idx_test].copy()
    test  = df.iloc[idx_test:].copy()
    print(f"    Train: {len(train):>7,} filas  ({train['timestamp'].iloc[0].date()} → {train['timestamp'].iloc[-1].date()})")
    print(f"    Val:   {len(val):>7,} filas  ({val['timestamp'].iloc[0].date()} → {val['timestamp'].iloc[-1].date()})")
    print(f"    Test:  {len(test):>7,} filas  ({test['timestamp'].iloc[0].date()} → {test['timestamp'].iloc[-1].date()})")
    return train, val, test

def normalizar(train, val, test, features, dir_modelos="models"):
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train[features].values)
    X_val   = scaler.transform(val[features].values)
    X_test  = scaler.transform(test[features].values)
    for X in [X_train, X_val, X_test]:
        np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
    os.makedirs(dir_modelos, exist_ok=True)
    joblib.dump(scaler, os.path.join(dir_modelos, "scaler.pkl"))
    print(f"    ✅ Scaler guardado en {dir_modelos}/scaler.pkl")
    return X_train, X_val, X_test

def crear_secuencias(X, y, longitud=48):
    Xs, ys = [], []
    for i in range(longitud, len(X)):
        Xs.append(X[i - longitud:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def preparar_datos(simbolo, dir_features="data/features", dir_modelos="models",
                   longitud_lstm=48, target_col=TARGET_PRINCIPAL):  # ← CAMBIO 1: nuevo parámetro
    print(f"\n  📂 Cargando dataset de {simbolo}...")
    df = cargar_dataset(simbolo, dir_features)
    features = obtener_columnas_features(df)
    print(f"  📊 Features encontradas: {len(features)}")
    print(f"\n  ✂️  Dividiendo temporalmente...")
    train, val, test = dividir_temporal(df)

    y_train = train[target_col].values  # ← CAMBIO 2: usa target_col
    y_val   = val[target_col].values    # ← CAMBIO 2
    y_test  = test[target_col].values   # ← CAMBIO 2

    print(f"\n  ⚖️  Balance de clases ({target_col}):")  # ← CAMBIO 3: muestra qué target se usa
    print(f"    Train — positivo: {y_train.mean()*100:.1f}%  negativo: {(1-y_train.mean())*100:.1f}%")
    print(f"    Val   — positivo: {y_val.mean()*100:.1f}%")
    print(f"\n  📐 Normalizando features...")
    X_train, X_val, X_test = normalizar(train, val, test, features, dir_modelos)
    print(f"\n  🔗 Creando secuencias para LSTM (ventana={longitud_lstm}h)...")
    X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, longitud_lstm)
    X_val_seq,   y_val_seq   = crear_secuencias(X_val,   y_val,   longitud_lstm)
    X_test_seq,  y_test_seq  = crear_secuencias(X_test,  y_test,  longitud_lstm)
    print(f"    XGBoost — Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"    LSTM    — Train: {X_train_seq.shape}  Val: {X_val_seq.shape}")
    joblib.dump(features, os.path.join(dir_modelos, "features.pkl"))
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "X_train_seq": X_train_seq, "y_train_seq": y_train_seq,
        "X_val_seq":   X_val_seq,   "y_val_seq":   y_val_seq,
        "X_test_seq":  X_test_seq,  "y_test_seq":  y_test_seq,
        "features": features, "df_test": test, "df_val": val,
    }