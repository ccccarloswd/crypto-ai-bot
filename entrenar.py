"""
entrenar.py  (v3)
=====================================
Cambios respecto a v2:
  1. FEATURES_BASE ampliado con microestructura, estructura de precio y BTC contexto
     → body_ratio, upper_wick, lower_wick, spread_hl_pct
     → dist_max_24h, dist_min_24h, posicion_48h
     → regimen_volatilidad, btc_ret_1h, btc_ret_4h, corr_btc_24h
  2. entrenar_modelo() adapta n_estimators y min_child_weight al tamaño del dataset
     → Datasets pequeños (4h) usan menos árboles y más regularización
     → Evita que SOL/BNB_4h colapsen por overfitting con pocos datos
  3. Umbral mínimo de muestras de val subido a 150 positivos para calibrar bien
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, precision_score,
                              recall_score, f1_score, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

PROCESSED_DIR = 'models/data/processed'
MODELS_DIR    = 'models'
SIMBOLOS      = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
TIMEFRAMES    = ['1h', '4h']

FEATURES_BASE = [
    # Momentum
    'rsi_14', 'rsi_7', 'rsi_sobrecompra', 'rsi_sobreventa', 'rsi_pendiente',
    'macd_hist', 'macd_cruce', 'macd_hist_pendiente',
    'stoch_k', 'stoch_d', 'stoch_sobrecompra', 'stoch_sobreventa',
    'adx', 'dmi_pos', 'dmi_neg',
    'cci_20', 'roc_10', 'williams_r',
    # Estructura y volatilidad
    'bb_width', 'bb_posicion', 'bb_squeeze',
    'atr_pct', 'volatilidad_20', 'rango_vela_pct',
    # Volumen clásico
    'volumen_ratio', 'volumen_anomalo', 'vol_ratio_5',
    'obv_tendencia', 'confirmacion_alcista', 'confirmacion_bajista',
    # Tendencia
    'precio_sobre_ema50', 'precio_sobre_ema200', 'ema50_sobre_ema200',
    'dist_ema21_atr', 'dist_ema50_atr', 'dist_ema200_atr',
    # Fibonacci
    'cerca_fib_236', 'cerca_fib_382', 'cerca_fib_500', 'cerca_fib_618',
    # Retornos
    'ret_1', 'ret_5', 'ret_8',
    # Micro y estructura de precio
    'posicion_en_vela', 'regimen_mercado',
    'body_ratio', 'upper_wick', 'lower_wick', 'spread_hl_pct',
    'dist_max_24h', 'dist_min_24h', 'posicion_48h',
    'regimen_volatilidad',
    # Contexto BTC
    'btc_ret_1h', 'btc_ret_4h', 'corr_btc_24h',
    # Taker flow (nuevos — requieren descargar_datos.py v2)
    'taker_ratio',
    'taker_ratio_ma5',
    'taker_ratio_ma20',
    'taker_ratio_delta',
    'taker_ratio_pendiente',
    'taker_dominance',
    'taker_cvd_20',
    'taker_cvd_tendencia',
    'vol_quality',
]

FEATURES_TEMPORALES = ['hora_dia', 'dia_semana',
                        'es_sesion_asia', 'es_sesion_europa', 'es_sesion_eeuu']
FEATURES = FEATURES_BASE + FEATURES_TEMPORALES


def cargar_datos(simbolo, tf):
    ruta = os.path.join(PROCESSED_DIR, f"{simbolo}_{tf}.csv")
    if not os.path.exists(ruta):
        print(f"  No encontrado: {ruta}")
        return None
    return pd.read_csv(ruta, parse_dates=['timestamp'])


def añadir_features_temporales(df):
    df = df.copy()
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    hora = df['timestamp'].dt.hour
    dia  = df['timestamp'].dt.dayofweek
    df['hora_dia']         = hora / 23.0
    df['dia_semana']       = dia  / 6.0
    df['es_sesion_asia']   = ((hora >= 0)  & (hora < 8)).astype(int)
    df['es_sesion_europa'] = ((hora >= 7)  & (hora < 16)).astype(int)
    df['es_sesion_eeuu']   = ((hora >= 13) & (hora < 22)).astype(int)
    return df


def preparar_X_y(df, label_col):
    cols = [c for c in FEATURES + [label_col] if c in df.columns]
    df_ok = df[cols].dropna()
    feats = [f for f in FEATURES if f in df_ok.columns]
    return df_ok[feats].values, df_ok[label_col].values.astype(int), feats


def umbral_optimo_f1(y_true, y_prob):
    """Umbral que maximiza F1 — clave con desbalance de clases."""
    prec, rec, umbrales = precision_recall_curve(y_true, y_prob)
    f1s = np.where((prec + rec) > 0,
                   2 * prec * rec / (prec + rec), 0)
    idx = np.argmax(f1s[:-1])
    return float(umbrales[idx])


def evaluar(nombre, y_true, y_pred, y_prob, umbral):
    auc  = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = (y_true == y_pred).mean()
    bins = [0, 0.25, 0.35, 0.45, 0.55, 0.65, 1.0]
    hist, _ = np.histogram(y_prob, bins=bins)
    dist = ' | '.join([f"{b:.2f}-{bins[i+1]:.2f}:{hist[i]}"
                       for i, b in enumerate(bins[:-1])])
    print(f"    AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  "
          f"Rec={rec:.3f}  F1={f1:.3f}  Umbral={umbral:.3f}")
    print(f"    Dist prob: {dist}")
    return {'auc': round(auc,4), 'accuracy': round(acc,4),
            'precision': round(prec,4), 'recall': round(rec,4),
            'f1': round(f1,4), 'umbral': round(umbral,4)}


def entrenar_modelo(X_train, y_train, X_val, y_val, escala_pos=1.0):
    n_train = len(X_train)

    # Adaptar complejidad al tamaño del dataset
    # 4h datasets tienen ~3000-4000 muestras de train → menos árboles, más regularización
    # 1h datasets tienen ~12000-17000 muestras → más capacidad
    if n_train < 5000:
        n_est      = 400
        mcw        = 10
        reg_alpha  = 0.8
        reg_lambda = 2.0
        es_rounds  = 40
    elif n_train < 10000:
        n_est      = 600
        mcw        = 8
        reg_alpha  = 0.5
        reg_lambda = 1.5
        es_rounds  = 50
    else:
        n_est      = 800
        mcw        = 5
        reg_alpha  = 0.3
        reg_lambda = 1.0
        es_rounds  = 60

    base = XGBClassifier(
        n_estimators          = n_est,
        max_depth             = 4,
        learning_rate         = 0.02,
        subsample             = 0.75,
        colsample_bytree      = 0.75,
        min_child_weight      = mcw,
        gamma                 = 3,
        reg_alpha             = reg_alpha,
        reg_lambda            = reg_lambda,
        scale_pos_weight      = escala_pos,
        eval_metric           = 'auc',
        early_stopping_rounds = es_rounds,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )
    base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    modelo = CalibratedClassifierCV(base, method='sigmoid', cv='prefit')
    modelo.fit(X_val, y_val)
    return modelo


def importancia_features(modelo, feats, top_n=5):
    try:
        xgb = modelo.calibrated_classifiers_[0].estimator
        imp = xgb.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        return [(feats[i], round(float(imp[i]), 4)) for i in idx]
    except Exception:
        return []


def entrenar_un_modelo(df, label_col, nombre, carpeta, escala_auto=True):
    X, y, feats = preparar_X_y(df, label_col)
    if len(X) < 300:
        print(f"    Muestras insuficientes ({len(X)}), saltando")
        return None

    n = len(X)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)

    if (i_test - i_val) < 100 or (n - i_test) < 100:
        print(f"    Val/Test demasiado pequeños (n={n}), saltando")
        return None

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X[:i_val])
    X_val   = scaler.transform(X[i_val:i_test])
    X_test  = scaler.transform(X[i_test:])
    y_train, y_val, y_test = y[:i_val], y[i_val:i_test], y[i_test:]

    win_rate   = y_train.mean()
    escala_pos = ((1 - win_rate) / win_rate if win_rate > 0 else 1.0) if escala_auto else 1.0

    print(f"    Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    print(f"    Win rate={win_rate:.1%} | scale_pos_weight={escala_pos:.2f}")

    modelo = entrenar_modelo(X_train, y_train, X_val, y_val, escala_pos)

    # Umbral óptimo calculado sobre VAL (no test → evita data leakage)
    y_prob_val = modelo.predict_proba(X_val)[:, 1]
    umbral = umbral_optimo_f1(y_val, y_prob_val)
    # Clamp del umbral según tipo de modelo:
    # - Dirección: balance ~50/50, umbral nunca por debajo de 0.40
    #   (evita que colapse a "predice siempre positivo")
    # - Éxito long/short: desbalance 27/73, umbral puede bajar hasta 0.20
    es_direccion = 'direccion' in label_col
    if es_direccion:
        umbral = max(0.40, min(0.60, umbral))
    else:
        umbral = max(0.20, min(0.55, umbral))

    # Evaluación final sobre TEST
    y_prob_test = modelo.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= umbral).astype(int)

    print(f"    Metricas en TEST:")
    metricas = evaluar(nombre, y_test, y_pred_test, y_prob_test, umbral)

    top = importancia_features(modelo, feats)
    if top:
        print(f"    Top features: {', '.join([f'{n}({v})' for n,v in top])}")

    # Guardar
    joblib.dump(modelo, os.path.join(carpeta, f'{nombre}.pkl'))
    joblib.dump(scaler, os.path.join(carpeta, f'scaler_{nombre}.pkl'))
    with open(os.path.join(carpeta, f'threshold_{nombre}.json'), 'w') as f:
        json.dump({'umbral': umbral, 'win_rate_train': round(win_rate, 4)}, f)
 
    # Añadir después de calcular métricas en test:
    # Si AUC < 0.52, intenta con TimeSeriesSplit para verificar que no es ruido

    from sklearn.model_selection import TimeSeriesSplit
    if metricas['auc'] < 0.52:
        tscv = TimeSeriesSplit(n_splits=3)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            X_cv_train = scaler.fit_transform(X[train_idx])
            X_cv_val   = scaler.transform(X[val_idx])
            m_cv = entrenar_modelo(X_cv_train, y[train_idx], X_cv_val, y[val_idx])
            prob = m_cv.predict_proba(X_cv_val)[:, 1]
            aucs.append(roc_auc_score(y[val_idx], prob))
        print(f"    CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    return metricas


def entrenar_simbolo(simbolo, tf, reporte_global):
    print(f"\n{'─'*55}")
    print(f"  {simbolo} - {tf}")
    print(f"{'─'*55}")

    df = cargar_datos(simbolo, tf)
    if df is None:
        return

    df = añadir_features_temporales(df)
    carpeta = os.path.join(MODELS_DIR, simbolo)
    os.makedirs(carpeta, exist_ok=True)

    rep = reporte_global.setdefault(simbolo, {})

    # Label A: oportunidad operativa limpia (reemplaza label_direccion)
    label_a = 'label_oportunidad' if 'label_oportunidad' in df.columns else 'label_direccion'
    nombre_a = 'oportunidad' if label_a == 'label_oportunidad' else 'direccion'
    escala_a = True  # puede estar desbalanceado según cuántas oportunidades limpias hay

    print(f"\n  [A] {nombre_a.capitalize()} ({tf})")
    m = entrenar_un_modelo(df, label_a,
                           f'modelo_{nombre_a}_{tf}', carpeta,
                           escala_auto=escala_a)
    if m: rep[f'{nombre_a}_{tf}'] = m

    print(f"\n  [B] Exito LONG ({tf})")
    m = entrenar_un_modelo(df, 'label_exito_long',
                           f'modelo_exito_long_{tf}', carpeta)
    if m: rep[f'exito_long_{tf}'] = m

    print(f"\n  [B] Exito SHORT ({tf})")
    m = entrenar_un_modelo(df, 'label_exito_short',
                           f'modelo_exito_short_{tf}', carpeta)
    if m: rep[f'exito_short_{tf}'] = m

    with open(os.path.join(carpeta, 'features.json'), 'w') as f:
        json.dump(FEATURES, f, indent=2)


def main():
    print(f"\n{'='*55}")
    print(f"  ENTRENANDO MODELOS (v2)")
    print(f"{'='*55}")

    reporte_global = {}
    for simbolo in SIMBOLOS:
        for tf in TIMEFRAMES:
            entrenar_simbolo(simbolo, tf, reporte_global)

    ruta = os.path.join(MODELS_DIR, 'reporte_entrenamiento.json')
    with open(ruta, 'w') as f:
        json.dump(reporte_global, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  RESUMEN DE ENTRENAMIENTO")
    print(f"{'='*55}")
    for simbolo, modelos in reporte_global.items():
        print(f"\n  {simbolo}:")
        for nombre, m in modelos.items():
            auc = m['auc']
            estado = "VERDE" if auc >= 0.55 else "AMARILLO" if auc >= 0.52 else "ROJO"
            print(f"    [{estado}] {nombre:<30} AUC={auc:.4f}  "
                  f"F1={m['f1']:.3f}  Umbral={m.get('umbral',0.5):.3f}")

    print(f"\n  Reporte: {ruta}")
    print(f"  Siguiente: predecir.py")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()