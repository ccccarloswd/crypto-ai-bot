
import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score

class EnsembleTrading:
    def __init__(self, peso_xgb=0.6, peso_lstm=0.4):
        self.peso_xgb  = peso_xgb
        self.peso_lstm = peso_lstm
        self.umbral    = 0.55

    def calibrar_pesos(self, probs_xgb, probs_lstm, y_true):
        if len(np.unique(y_true)) < 2:
            return
        auc_xgb  = roc_auc_score(y_true, probs_xgb)
        auc_lstm = roc_auc_score(y_true, probs_lstm)
        total = auc_xgb + auc_lstm
        if total > 0:
            self.peso_xgb  = auc_xgb  / total
            self.peso_lstm = auc_lstm / total
        print(f"    Pesos calibrados — XGBoost: {self.peso_xgb:.3f}  LSTM: {self.peso_lstm:.3f}")
        print(f"    AUC individual  — XGBoost: {auc_xgb:.4f}  LSTM: {auc_lstm:.4f}")

    def predecir_proba(self, probs_xgb, probs_lstm=None):
        if probs_lstm is None or np.all(probs_lstm == 0.5):
            return probs_xgb
        return self.peso_xgb * probs_xgb + self.peso_lstm * probs_lstm

    def guardar(self, dir_modelos="models"):
        config = {"peso_xgb": self.peso_xgb, "peso_lstm": self.peso_lstm, "umbral": self.umbral}
        joblib.dump(config, os.path.join(dir_modelos, "ensemble_config.pkl"))
        print(f"    ✅ Configuracion del ensemble guardada")

    def cargar(self, dir_modelos="models"):
        ruta = os.path.join(dir_modelos, "ensemble_config.pkl")
        if os.path.exists(ruta):
            config = joblib.load(ruta)
            self.peso_xgb  = config["peso_xgb"]
            self.peso_lstm = config["peso_lstm"]
            self.umbral    = config["umbral"]

def evaluar_ensemble(probs_xgb, probs_lstm, probs_ensemble, y_true, umbral=0.55):
    from sklearn.metrics import precision_score, recall_score, f1_score
    def metricas(probs, nombre):
        if len(np.unique(y_true)) < 2:
            return {}
        auc  = roc_auc_score(y_true, probs)
        pred = (probs >= umbral).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        f1   = f1_score(y_true, pred, zero_division=0)
        señales = pred.sum()
        print(f"    {nombre:<15} AUC: {auc:.4f}  P: {prec:.3f}  R: {rec:.3f}  F1: {f1:.3f}  Señales: {señales:,} ({señales/len(pred)*100:.1f}%)")
        return {"auc": auc, "precision": prec, "recall": rec, "f1": f1}
    print(f"\n    Comparativa de modelos en TEST")
    m_xgb      = metricas(probs_xgb, "XGBoost")
    m_lstm     = metricas(probs_lstm if probs_lstm is not None else np.full(len(y_true), 0.5), "LSTM")
    m_ensemble = metricas(probs_ensemble, "Ensemble")
    return {"xgboost": m_xgb, "lstm": m_lstm, "ensemble": m_ensemble}

def umbral_por_regimen(regimen, umbral_base=0.55):
    ajustes = {2: -0.03, 1: -0.01, 0: 0.00, -1: 0.03, -2: 0.06}
    ajuste  = ajustes.get(regimen, 0.0)
    return float(np.clip(umbral_base + ajuste, 0.50, 0.75))
