
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_score, recall_score, f1_score)
from typing import Dict, Tuple, List

def _calcular_scale_pos_weight(y):
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)

def _hiperparametros_defecto():
    return {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.04,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 7,
        "gamma": 0.1, "reg_alpha": 0.2, "reg_lambda": 1.5,
    }

def optimizar_hiperparametros(X_train, y_train, X_val, y_val, n_trials=50):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("    Optuna no instalado. Usando hiperparametros por defecto.")
        return _hiperparametros_defecto()

    def objetivo(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
            "objective":        "binary:logistic",
            "scale_pos_weight": _calcular_scale_pos_weight(y_train),
            "eval_metric":      "logloss",
            "random_state":     42, "n_jobs": -1,
        }
        modelo = XGBClassifier(**params)
        modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        probs = modelo.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, probs)

    print(f"    Optimizando hiperparametros ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objetivo, n_trials=n_trials, show_progress_bar=False)
    print(f"    Mejor AUC en validacion: {study.best_value:.4f}")
    return study.best_params

def walk_forward_validation(X, y, n_splits=6, min_train_size=0.5):
    n = len(X)
    min_train = int(n * min_train_size)
    bloque = (n - min_train) // n_splits
    metricas_splits = []
    print(f"    Walk-forward con {n_splits} splits, bloque: {bloque:,} muestras")
    for i in range(n_splits):
        idx_fin_train = min_train + i * bloque
        idx_fin_val   = min(idx_fin_train + bloque, n)
        if idx_fin_val <= idx_fin_train:
            break
        X_tr, y_tr = X[:idx_fin_train], y[:idx_fin_train]
        X_vl, y_vl = X[idx_fin_train:idx_fin_val], y[idx_fin_train:idx_fin_val]
        modelo = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective="binary:logistic",
            scale_pos_weight=_calcular_scale_pos_weight(y_tr),
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        modelo.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        probs = modelo.predict_proba(X_vl)[:, 1]
        preds = (probs >= 0.55).astype(int)
        auc       = roc_auc_score(y_vl, probs) if len(np.unique(y_vl)) > 1 else 0.5
        precision = precision_score(y_vl, preds, zero_division=0)
        recall    = recall_score(y_vl, preds, zero_division=0)
        f1        = f1_score(y_vl, preds, zero_division=0)
        metricas_splits.append({
            "split": i+1, "n_train": len(X_tr), "n_val": len(X_vl),
            "auc": auc, "precision": precision, "recall": recall, "f1": f1,
        })
        print(f"    Split {i+1}/{n_splits} — AUC: {auc:.3f}  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    df_metricas = pd.DataFrame(metricas_splits)
    print(f"\n    Promedio walk-forward:")
    print(f"    AUC:       {df_metricas['auc'].mean():.3f} +/- {df_metricas['auc'].std():.3f}")
    print(f"    Precision: {df_metricas['precision'].mean():.3f} +/- {df_metricas['precision'].std():.3f}")
    print(f"    F1:        {df_metricas['f1'].mean():.3f} +/- {df_metricas['f1'].std():.3f}")
    return {"metricas": df_metricas}

def entrenar_xgboost(X_train, y_train, X_val, y_val, features,
                     dir_modelos="models", optimizar=True, n_trials=50):
    if optimizar:
        params = optimizar_hiperparametros(X_train, y_train, X_val, y_val, n_trials)
    else:
        params = _hiperparametros_defecto()
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    print(f"\n    Entrenando modelo final con {len(X_full):,} muestras...")
    params_finales = {
        **params,
        "objective": "binary:logistic",
        "scale_pos_weight": _calcular_scale_pos_weight(y_full),
        "eval_metric": "logloss",
        "random_state": 42, "n_jobs": -1,
    }
    modelo = XGBClassifier(**params_finales)
    modelo.fit(X_full, y_full, verbose=False)
    print("    Modelo entrenado correctamente")
    importancias = pd.DataFrame({
        "feature": features,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=False)
    print(f"\n    Top 15 features mas importantes:")
    for _, row in importancias.head(15).iterrows():
        barra = chr(9608) * int(row["importancia"] * 200)
        print(f"    {row['feature']:<35} {barra} {row['importancia']:.4f}")
    os.makedirs(dir_modelos, exist_ok=True)
    joblib.dump(modelo, os.path.join(dir_modelos, "xgboost.pkl"))
    importancias.to_csv(os.path.join(dir_modelos, "feature_importances.csv"), index=False)
    print(f"\n    Modelo XGBoost guardado en {dir_modelos}/xgboost.pkl")
    return modelo, {"importancias": importancias, "params": params_finales}

def evaluar_xgboost(modelo, X_test, y_test, umbral=0.55):
    probs = modelo.predict_proba(X_test)[:, 1]
    preds = (probs >= umbral).astype(int)
    auc       = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)
    print(f"\n    Evaluacion en TEST (datos nunca vistos)")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, preds, target_names=['No sube', 'Sube'])}")
    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1,
            "probs": probs, "preds": preds}
