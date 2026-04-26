"""
=============================================================
  CRYPTO AI BOT — Paper Trading v15
=============================================================
  Cambios respecto a v14:
  ────────────────────────
  IA REACTIVADA con arquitectura de predecir.py (nueva IA entrenada)

  La nueva IA NO reemplaza la lógica técnica existente.
  Se usa para DOS cosas concretas:

  1. FILTRO DE ENTRADA (¿tiene sentido abrir la operación?)
     ─────────────────────────────────────────────────────
     Antes de abrir cualquier posición, se llama a generar_señal()
     de predecir.py. Si la IA dice FLAT o la dirección contraria
     con confianza suficiente, la operación se cancela aunque el
     score técnico sea bueno.

     Reglas de filtro:
       · IA dice FLAT y score técnico < 5  → cancela (señal débil sin apoyo IA)
       · IA dice dirección CONTRARIA con confianza ≥ UMBRAL_IA_VETO → cancela
       · IA apoya la dirección → entrada normal, posiblemente mejorada

  2. AJUSTE DE SL/TP (¿cómo de agresivo ser?)
     ──────────────────────────────────────────
     La fuerza del modelo 4h (prob_long_4h o prob_short_4h) modifica
     el ratio TP:SL antes de buscarlo en los niveles de mercado.

     · fuerza_4h ≥ 0.65  → tp_ratio × 1.20 (TP más ambicioso, 4H confirma fuerte)
     · fuerza_4h ≥ 0.55  → tp_ratio × 1.10 (TP algo más amplio)
     · fuerza_4h < 0.45  → sl_mult  × 1.10 (SL más ajustado, señal débil en 4H)
     · IA en contra 4h   → tp_ratio × 0.85 (TP más conservador, salir antes)

  El resto de la lógica (score técnico, divergencias, patrones chartistas,
  gestión de posiciones, trailing, etc.) es idéntico a v14.

=============================================================
"""

import os, json, time, urllib.request, numpy as np, pandas as pd
import joblib, warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

VERSION = 'v14'

# ──────────────────────────────────────────────
#  CONFIGURACIÓN
# ──────────────────────────────────────────────
CRIPTOS  = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
VELAS_N  = 300

CAPITAL_INICIAL   = 1000.0
COMISION_PCT      = 0.001
SLIPPAGE_PCT      = 0.0005
FUNDING_RATE_HORA = 0.0003

MAX_MARGEN_TOTAL_PCT = 0.80

MAX_HORAS_BASE_LONG  = 20
MAX_HORAS_BASE_SHORT = 16
MAX_HORAS_ATR_REF    = 0.012
MAX_HORAS_MIN        = 6
MAX_HORAS_MAX        = 48

TRAILING_ATR_ACTIVACION = 1.5
TRAILING_ATR_DISTANCIA  = 1.2

SOLO_XGBOOST     = ['BTC_USDT', 'BNB_USDT']
CAP_MULT_FUNDING = 8.0

ADAPTIVE_VENTANA  = 20
ADAPTIVE_WR_BAJO  = 0.48
ADAPTIVE_WR_ALTO  = 0.65
ADAPTIVE_FACTOR_C = 0.50
ADAPTIVE_FACTOR_A = 1.20
ADAPTIVE_DURACION = 15

TRADE_CONFIG = {
    'nivel1': {'leverage': 4,  'margen': 0.08, 'sl_mult': 2.0, 'tp_ratio': 1.8, 'rr_min': 1.6},
    'nivel2': {'leverage': 12, 'margen': 0.14, 'sl_mult': 1.8, 'tp_ratio': 2.2, 'rr_min': 1.8},
    'nivel3': {'leverage': 20, 'margen': 0.18, 'sl_mult': 1.6, 'tp_ratio': 2.6, 'rr_min': 2.0},
    'nivel4': {'leverage': 30, 'margen': 0.22, 'sl_mult': 1.4, 'tp_ratio': 3.0, 'rr_min': 2.2},
}

RR_MINIMO          = 1.6
SL_ATR_MIN_MULT    = 0.8
SL_ATR_MAX_MULT    = 5.0
TP_ATR_MAX_MULT    = 12.0
BUFFER_NIVEL_PCT   = 0.004

PROB_ELITE_LONG    = 0.74
PROB_PREMIUM_LONG  = 0.35
PROB_ELITE_SHORT   = 0.70
PROB_PREMIUM_SHORT = 0.35

IA_AMP_SUBE_UN_NIVEL   = 0.65
IA_AMP_SIN_CAMBIO_INF  = 0.45
IA_AMP_BAJA_UN_NIVEL   = 0.30

IA_CT_FUERTE   = 0.60
IA_CT_DEBIL    = 0.50
IA_CT_CONTRA   = 0.40

# ── Parámetros de integración de la nueva IA ──────────────────────────
# Rutas de modelos (relativas al directorio de ejecución)
IA_MODELS_DIR    = 'models'
IA_PROCESSED_DIR = 'models/data/processed'

# Umbral de confianza mínima del modelo 1h para que el filtro IA actúe.
# Si confianza < UMBRAL_IA_ACTIVA la IA no filtra (pasa igual que v14).
UMBRAL_IA_ACTIVA = 0.50   # la IA tiene opinión si confianza ≥ 50%

# Si la IA dice la dirección CONTRARIA con esta confianza → veto duro de entrada.
UMBRAL_IA_VETO   = 0.60   # prob contraria ≥ 60% → no se abre

# Si score técnico es bueno pero IA dice FLAT:
# solo se bloquea si el score técnico es menor que este umbral.
SCORE_MINIMO_SIN_IA = 5.0  # score < 5 + IA FLAT → no entra

# Umbrales para el ajuste de TP/SL según fuerza 4h de la IA
IA_FUERZA_4H_FUERTE    = 0.65  # fuerza 4h ≥ este valor → TP más agresivo (+20%)
IA_FUERZA_4H_MODERADA  = 0.55  # fuerza 4h ≥ este valor → TP algo más amplio (+10%)
IA_FUERZA_4H_DEBIL     = 0.45  # fuerza 4h < este valor → SL más ajustado (+10%)

def score_minimo(prob: float, direccion: str) -> int:
    return 3  # consistente con v14

DIR_MODELOS   = 'models'
ESTADO_FILE   = f'paper_trading/{VERSION}/estado.json'
LOG_FILE      = f'paper_trading/{VERSION}/operaciones.csv'
METRICAS_FILE = f'paper_trading/{VERSION}/metricas.json'

KRAKEN_MAP = {
    'BTC/USDT': 'XBTUSD', 'ETH/USDT': 'ETHUSD',
    'BNB/USDT': 'BNBUSD', 'SOL/USDT': 'SOLUSD',
}

# Mapeo símbolo Kraken → símbolo Binance (para predecir.py)
BINANCE_MAP = {
    'BTC/USDT': 'BTCUSDT', 'ETH/USDT': 'ETHUSDT',
    'BNB/USDT': 'BNBUSDT', 'SOL/USDT': 'SOLUSDT',
}


# ──────────────────────────────────────────────
#  INTEGRACIÓN CON predecir.py (nueva IA)
# ──────────────────────────────────────────────

def _cargar_modelo_ia(simbolo_binance: str, nombre: str):
    """
    Carga modelo + scaler + umbral de la nueva IA entrenada.
    nombre: 'exito_long_1h', 'exito_short_1h', 'exito_long_4h', 'exito_short_4h'
    Devuelve (modelo, scaler, umbral) o (None, None, None).
    """
    carpeta = os.path.join(IA_MODELS_DIR, simbolo_binance)
    ruta_m  = os.path.join(carpeta, f'modelo_{nombre}.pkl')
    ruta_s  = os.path.join(carpeta, f'scaler_modelo_{nombre}.pkl')
    ruta_t  = os.path.join(carpeta, f'threshold_modelo_{nombre}.json')

    if not os.path.exists(ruta_m):
        return None, None, None

    modelo = joblib.load(ruta_m)
    scaler = joblib.load(ruta_s) if os.path.exists(ruta_s) else None
    umbral = 0.5
    if os.path.exists(ruta_t):
        with open(ruta_t) as f:
            umbral = json.load(f).get('umbral', 0.5)
    return modelo, scaler, umbral


def _cargar_features_ia(simbolo_binance: str) -> list:
    """Lee la lista de features con la que se entrenó el modelo IA."""
    FEATURES_BASE = [
        'rsi_14', 'rsi_7', 'rsi_sobrecompra', 'rsi_sobreventa', 'rsi_pendiente',
        'macd_hist', 'macd_cruce', 'macd_hist_pendiente',
        'stoch_k', 'stoch_d', 'stoch_sobrecompra', 'stoch_sobreventa',
        'adx', 'dmi_pos', 'dmi_neg',
        'cci_20', 'roc_10', 'williams_r',
        'bb_width', 'bb_posicion', 'bb_squeeze',
        'atr_pct', 'volatilidad_20', 'rango_vela_pct',
        'volumen_ratio', 'volumen_anomalo', 'vol_ratio_5',
        'obv_tendencia', 'confirmacion_alcista', 'confirmacion_bajista',
        'precio_sobre_ema50', 'precio_sobre_ema200', 'ema50_sobre_ema200',
        'dist_ema21_atr', 'dist_ema50_atr', 'dist_ema200_atr',
        'cerca_fib_236', 'cerca_fib_382', 'cerca_fib_500', 'cerca_fib_618',
        'ret_1', 'ret_5', 'ret_8',
        'posicion_en_vela', 'regimen_mercado',
        'body_ratio', 'upper_wick', 'lower_wick', 'spread_hl_pct',
        'dist_max_24h', 'dist_min_24h', 'posicion_48h',
        'regimen_volatilidad',
        'btc_ret_1h', 'btc_ret_4h', 'corr_btc_24h',
        'taker_ratio', 'taker_ratio_ma5', 'taker_ratio_ma20',
        'taker_ratio_delta', 'taker_ratio_pendiente',
        'taker_dominance', 'taker_cvd_20', 'taker_cvd_tendencia', 'vol_quality',
    ]
    FEATURES_TEMPORALES = [
        'hora_dia', 'dia_semana',
        'es_sesion_asia', 'es_sesion_europa', 'es_sesion_eeuu'
    ]
    ruta = os.path.join(IA_MODELS_DIR, simbolo_binance, 'features.json')
    if os.path.exists(ruta):
        with open(ruta) as f:
            return json.load(f)
    return FEATURES_BASE + FEATURES_TEMPORALES


def _añadir_features_temporales_ia(df: pd.DataFrame) -> pd.DataFrame:
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


def _cargar_ultima_fila_procesada(simbolo_binance: str, tf: str) -> Optional[pd.Series]:
    """
    Carga el CSV procesado (generado por preparar_datos.py) y devuelve
    la última fila con features completas.
    Devuelve None si no existe o no hay datos válidos.
    """
    ruta = os.path.join(IA_PROCESSED_DIR, f"{simbolo_binance}_{tf}.csv")
    if not os.path.exists(ruta):
        return None
    try:
        df = pd.read_csv(ruta, parse_dates=['timestamp'])
        df = _añadir_features_temporales_ia(df)
        return df.dropna().iloc[-1] if len(df) > 0 else None
    except Exception:
        return None


def _predecir_prob_ia(modelo, scaler, fila: pd.Series, features: list) -> Optional[float]:
    """Aplica scaler y modelo a una fila, devuelve prob de clase 1."""
    feats_ok = [f for f in features if f in fila.index]
    if not feats_ok:
        return None
    x = fila[feats_ok].values.reshape(1, -1).astype(float)
    if np.any(np.isnan(x)):
        return None
    if scaler is not None:
        x = scaler.transform(x)
    return float(modelo.predict_proba(x)[0][1])


def obtener_señal_ia(simbolo_kraken: str) -> Dict:
    """
    Genera la señal de la nueva IA para un símbolo dado.
    Usa los modelos entrenados con predecir.py y los CSVs de processed/.

    Devuelve dict con:
      direccion:     'LONG' | 'SHORT' | 'FLAT'
      confianza:     prob del modelo 1h ganador (0-1)
      fuerza:        prob del modelo 4h en la dirección ganadora (0-1)
      prob_long_1h:  float | None
      prob_short_1h: float | None
      prob_long_4h:  float | None
      prob_short_4h: float | None
      disponible:    bool (False si no hay modelos o datos)
      razon:         str explicativa
    """
    simbolo_binance = BINANCE_MAP.get(simbolo_kraken)
    resultado = {
        'direccion': 'FLAT', 'confianza': 0.0, 'fuerza': 0.5,
        'prob_long_1h': None, 'prob_short_1h': None,
        'prob_long_4h': None, 'prob_short_4h': None,
        'disponible': False, 'razon': 'no inicializado',
    }

    if not simbolo_binance:
        resultado['razon'] = f'símbolo no mapeado: {simbolo_kraken}'
        return resultado

    features = _cargar_features_ia(simbolo_binance)

    # ── Cargar modelos 1h ──
    m_long_1h,  s_long_1h,  u_long_1h  = _cargar_modelo_ia(simbolo_binance, 'exito_long_1h')
    m_short_1h, s_short_1h, u_short_1h = _cargar_modelo_ia(simbolo_binance, 'exito_short_1h')

    if m_long_1h is None or m_short_1h is None:
        resultado['razon'] = 'modelos 1h no encontrados — ejecuta entrenar.py primero'
        return resultado

    # ── Cargar datos 1h procesados ──
    fila_1h = _cargar_ultima_fila_procesada(simbolo_binance, '1h')
    if fila_1h is None:
        resultado['razon'] = 'sin datos procesados 1h — ejecuta preparar_datos.py primero'
        return resultado

    # ── Probabilidades 1h ──
    prob_long_1h  = _predecir_prob_ia(m_long_1h,  s_long_1h,  fila_1h, features)
    prob_short_1h = _predecir_prob_ia(m_short_1h, s_short_1h, fila_1h, features)

    if prob_long_1h is None or prob_short_1h is None:
        resultado['razon'] = 'NaN en features 1h — revisar datos procesados'
        return resultado

    resultado['prob_long_1h']  = round(prob_long_1h,  3)
    resultado['prob_short_1h'] = round(prob_short_1h, 3)
    resultado['disponible']    = True

    # ── Probabilidades 4h (para ajuste de SL/TP) ──
    m_long_4h,  s_long_4h,  _  = _cargar_modelo_ia(simbolo_binance, 'exito_long_4h')
    m_short_4h, s_short_4h, _  = _cargar_modelo_ia(simbolo_binance, 'exito_short_4h')
    fila_4h = _cargar_ultima_fila_procesada(simbolo_binance, '4h')

    prob_long_4h  = None
    prob_short_4h = None
    if fila_4h is not None and m_long_4h is not None and m_short_4h is not None:
        prob_long_4h  = _predecir_prob_ia(m_long_4h,  s_long_4h,  fila_4h, features)
        prob_short_4h = _predecir_prob_ia(m_short_4h, s_short_4h, fila_4h, features)
        resultado['prob_long_4h']  = round(prob_long_4h,  3) if prob_long_4h  is not None else None
        resultado['prob_short_4h'] = round(prob_short_4h, 3) if prob_short_4h is not None else None

    # ── Decisión de dirección ──
    # La IA tiene opinión clara si la confianza supera UMBRAL_IA_ACTIVA.
    # Si ambas son altas → ambiguo → FLAT.
    hay_long  = prob_long_1h  >= UMBRAL_IA_ACTIVA
    hay_short = prob_short_1h >= UMBRAL_IA_ACTIVA

    if hay_long and hay_short:
        resultado['razon'] = (f'IA ambigua — ambas altas '
                              f'(long={prob_long_1h:.2f}, short={prob_short_1h:.2f})')
        return resultado

    if hay_long:
        fuerza_4h = prob_long_4h  if prob_long_4h  is not None else 0.5
        resultado.update({
            'direccion': 'LONG',
            'confianza': round(prob_long_1h,  3),
            'fuerza':    round(fuerza_4h,     3),
            'razon':     (f'LONG — 1h={prob_long_1h:.2f} | '
                          f'4h_fuerza={fuerza_4h:.2f}'),
        })
    elif hay_short:
        fuerza_4h = prob_short_4h if prob_short_4h is not None else 0.5
        resultado.update({
            'direccion': 'SHORT',
            'confianza': round(prob_short_1h, 3),
            'fuerza':    round(fuerza_4h,     3),
            'razon':     (f'SHORT — 1h={prob_short_1h:.2f} | '
                          f'4h_fuerza={fuerza_4h:.2f}'),
        })
    else:
        resultado['razon'] = (f'confianza insuficiente '
                              f'(long={prob_long_1h:.2f}, short={prob_short_1h:.2f}, '
                              f'mínimo={UMBRAL_IA_ACTIVA})')

    return resultado


def filtrar_entrada_con_ia(señal_ia: Dict, direccion: str, score_adj: float) -> Tuple[bool, str]:
    """
    Decide si la operación puede abrirse según la señal de la IA.
    Devuelve (permitir: bool, motivo: str).

    Lógica:
      · Si IA no disponible → siempre permitir (igual que v14)
      · Si IA dice la misma dirección → siempre permitir
      · Si IA dice FLAT:
          · score técnico ≥ SCORE_MINIMO_SIN_IA → permitir (el técnico es suficiente)
          · score técnico <  SCORE_MINIMO_SIN_IA → bloquear (señal débil + IA neutral)
      · Si IA dice dirección CONTRARIA con confianza ≥ UMBRAL_IA_VETO → bloquear
      · Si IA dice dirección CONTRARIA con confianza < UMBRAL_IA_VETO → permitir
        (la contradicción es leve, el técnico manda)
    """
    if not señal_ia.get('disponible', False):
        return True, 'IA no disponible — entrada libre'

    dir_ia     = señal_ia['direccion']          # 'LONG' | 'SHORT' | 'FLAT'
    confianza  = señal_ia.get('confianza', 0.0)

    # La IA concuerda
    if dir_ia == direccion.upper():
        return True, f'IA confirma {dir_ia} (conf={confianza:.2f})'

    # La IA está en FLAT
    if dir_ia == 'FLAT':
        if score_adj >= SCORE_MINIMO_SIN_IA:
            return True, f'IA FLAT pero score técnico={score_adj:.1f} ≥ {SCORE_MINIMO_SIN_IA} — OK'
        else:
            return False, (f'IA FLAT y score técnico bajo ({score_adj:.1f} < {SCORE_MINIMO_SIN_IA}) '
                           f'— señal insuficiente sin apoyo IA')

    # La IA dice lo contrario
    if confianza >= UMBRAL_IA_VETO:
        return False, (f'IA dice {dir_ia} con conf={confianza:.2f} ≥ {UMBRAL_IA_VETO} '
                       f'— VETO de entrada {direccion.upper()}')

    # Contradicción leve → el técnico manda
    return True, (f'IA contradice levemente ({dir_ia} conf={confianza:.2f} < {UMBRAL_IA_VETO}) '
                  f'— técnico prevalece')


def ajustar_cfg_por_ia(cfg: dict, señal_ia: Dict, direccion: str) -> Tuple[dict, str]:
    """
    Modifica una copia del cfg de nivel para ajustar SL/TP según la fuerza 4h de la IA.
    Devuelve (cfg_ajustado, nota_ajuste).

    No modifica leverage ni margen — solo tp_ratio y sl_mult.
    """
    if not señal_ia.get('disponible', False):
        return cfg, 'IA no disponible — SL/TP sin ajuste'

    cfg_aj = dict(cfg)  # copia para no mutar el original

    # Obtener la fuerza 4h en la dirección de la operación
    if direccion.lower() == 'long':
        fuerza_4h = señal_ia.get('prob_long_4h')
        fuerza_contra = señal_ia.get('prob_short_4h')
    else:
        fuerza_4h = señal_ia.get('prob_short_4h')
        fuerza_contra = señal_ia.get('prob_long_4h')

    # Si no hay datos 4h, sin ajuste
    if fuerza_4h is None:
        return cfg_aj, 'sin datos IA 4h — SL/TP sin ajuste'

    nota = ''
    if fuerza_4h >= IA_FUERZA_4H_FUERTE:
        # 4H apoya fuerte → TP más agresivo (podemos aguantar más)
        cfg_aj['tp_ratio'] = round(cfg_aj['tp_ratio'] * 1.20, 2)
        nota = f'IA 4h fuerte ({fuerza_4h:.2f}) → tp_ratio ×1.20 ({cfg_aj["tp_ratio"]:.2f})'
    elif fuerza_4h >= IA_FUERZA_4H_MODERADA:
        # 4H apoya moderado → TP algo más amplio
        cfg_aj['tp_ratio'] = round(cfg_aj['tp_ratio'] * 1.10, 2)
        nota = f'IA 4h moderada ({fuerza_4h:.2f}) → tp_ratio ×1.10 ({cfg_aj["tp_ratio"]:.2f})'
    elif fuerza_4h < IA_FUERZA_4H_DEBIL:
        # 4H débil para esta dirección → SL más ajustado (salir antes si falla)
        cfg_aj['sl_mult'] = round(cfg_aj['sl_mult'] * 1.10, 2)
        nota = f'IA 4h débil ({fuerza_4h:.2f}) → sl_mult ×1.10 ({cfg_aj["sl_mult"]:.2f})'
    else:
        nota = f'IA 4h neutra ({fuerza_4h:.2f}) — SL/TP sin ajuste'

    # Si encima la dirección contraria tiene fuerza alta en 4h → TP conservador
    if fuerza_contra is not None and fuerza_contra >= IA_FUERZA_4H_FUERTE:
        cfg_aj['tp_ratio'] = round(cfg_aj['tp_ratio'] * 0.85, 2)
        nota += f' | contra 4h fuerte ({fuerza_contra:.2f}) → tp_ratio ×0.85 ({cfg_aj["tp_ratio"]:.2f})'

    return cfg_aj, nota


# ──────────────────────────────────────────────
#  TELEGRAM
# ──────────────────────────────────────────────
def telegram(msg: str):
    token   = os.environ.get('TELEGRAM_TOKEN_V15', '') or os.environ.get('TELEGRAM_TOKEN_V14', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_V15', '') or os.environ.get('TELEGRAM_CHAT_ID_V14', '')
    if not token or not chat_id:
        print(f"[TG-{VERSION}] {msg[:120]}")
        return
    try:
        chat_id_clean = chat_id.strip().strip('"').strip("'")
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({'chat_id': chat_id_clean, 'text': msg,
                           'parse_mode': 'HTML'}).encode()
        req  = urllib.request.Request(url, data=data,
                                       headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  ⚠️  TG: {e}")
        print(f"  [MSG] {msg[:200]}")


# ──────────────────────────────────────────────
#  ESTADO
# ──────────────────────────────────────────────
def cargar_estado() -> Dict:
    if os.path.exists(ESTADO_FILE):
        with open(ESTADO_FILE) as f:
            return json.load(f)
    return {
        'capital': CAPITAL_INICIAL, 'capital_max': CAPITAL_INICIAL,
        'posiciones': [], 'bot_activo': True,
        'en_pausa': False, 'fin_pausa_hora': None,
        'n_ops': 0, 'n_wins': 0, 'n_loses': 0,
        'pnl_total': 0.0, 'funding_total': 0.0,
        'ops_hoy': [], 'ultima_ejecucion': None,
        'primera_ejecucion': True,
        'historial_pnl': [],
        'adaptive_factor': 1.0,
        'adaptive_trades_left': 0,
    }

def guardar_estado(estado: Dict):
    os.makedirs(os.path.dirname(ESTADO_FILE), exist_ok=True)
    estado['ultima_ejecucion'] = datetime.now(timezone.utc).isoformat()
    with open(ESTADO_FILE, 'w') as f:
        json.dump(estado, f, indent=2, default=str)

def registrar_op(op: Dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    df = pd.DataFrame([op])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def guardar_metricas(estado: Dict) -> Dict:
    capital = estado['capital']
    rent    = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    n       = estado['n_ops']
    wr      = estado['n_wins'] / max(n, 1) * 100
    m = {
        'version': VERSION,
        'capital_actual': round(capital, 2),
        'capital_inicial': CAPITAL_INICIAL,
        'rentabilidad_pct': round(rent, 2),
        'n_operaciones': n,
        'win_rate_pct': round(wr, 1),
        'pnl_total': round(estado['pnl_total'], 2),
        'funding_total': round(estado['funding_total'], 2),
        'posiciones_abiertas': len(estado['posiciones']),
        'adaptive_factor': estado.get('adaptive_factor', 1.0),
        'bot_activo': estado['bot_activo'],
        'ultima_actualizacion': datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(METRICAS_FILE), exist_ok=True)
    with open(METRICAS_FILE, 'w') as f:
        json.dump(m, f, indent=2)
    return m


# ──────────────────────────────────────────────
#  DATOS DE MERCADO
# ──────────────────────────────────────────────
def obtener_velas(simbolo: str, intervalo: int = 60,
                  limite: int = 300) -> Optional[pd.DataFrame]:
    par   = KRAKEN_MAP.get(simbolo, simbolo.replace('/', ''))
    desde = int(time.time()) - (limite * intervalo * 60)
    url   = (f"https://api.kraken.com/0/public/OHLC"
             f"?pair={par}&interval={intervalo}&since={desde}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if data.get('error'):
            return None
        result = data.get('result', {})
        clave  = [k for k in result if k != 'last']
        if not clave:
            return None
        velas = result[clave[0]]
        if len(velas) < 50:
            return None
        df = pd.DataFrame(velas, columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['timestamp'] = pd.to_datetime(df['time'].astype(int), unit='s')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)
    except Exception as e:
        print(f"  ❌ Kraken {simbolo} {intervalo}m: {e}")
        return None


# ──────────────────────────────────────────────
#  INDICADORES
# ──────────────────────────────────────────────
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta
    df = df.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [9, 21, 50, 100, 200]: df[f'ema_{p}'] = ta.ema(c, length=p)
    for p in [20, 50, 200]:         df[f'sma_{p}'] = ta.sma(c, length=p)

    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df['macd']        = macd_df.iloc[:, 0]
        df['macd_signal'] = macd_df.iloc[:, 1] if macd_df.shape[1] > 1 else np.nan
        df['macd_hist']   = macd_df.iloc[:, 2] if macd_df.shape[1] > 2 else np.nan
    else:
        df['macd'] = df['macd_signal'] = df['macd_hist'] = np.nan

    df['macd_cruce'] = np.where(
        (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
        np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0))

    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None and not adx_df.empty:
        df['adx']     = adx_df.iloc[:, 0]
        df['dmi_pos'] = adx_df.iloc[:, 1] if adx_df.shape[1] > 1 else np.nan
        df['dmi_neg'] = adx_df.iloc[:, 2] if adx_df.shape[1] > 2 else np.nan
    else:
        df['adx'] = df['dmi_pos'] = df['dmi_neg'] = np.nan

    df['golden_cross']        = ((df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1))).astype(int)
    df['death_cross']         = ((df['ema_50'] < df['ema_200']) & (df['ema_50'].shift(1) >= df['ema_200'].shift(1))).astype(int)
    df['precio_sobre_ema50']  = (c > df['ema_50']).astype(int)
    df['precio_sobre_ema200'] = (c > df['ema_200']).astype(int)
    df['ema50_sobre_ema200']  = (df['ema_50'] > df['ema_200']).astype(int)

    df['rsi_14'] = ta.rsi(c, length=14)
    df['rsi_7']  = ta.rsi(c, length=7)
    df['rsi_sobrecompra']  = (df['rsi_14'] >= 70).astype(int)
    df['rsi_sobreventa']   = (df['rsi_14'] <= 30).astype(int)
    df['rsi_zona_neutral'] = ((df['rsi_14'] > 40) & (df['rsi_14'] < 60)).astype(int)
    ps = (c > c.shift(5)).astype(int)
    rs = (df['rsi_14'] > df['rsi_14'].shift(5)).astype(int)
    df['divergencia_bajista_rsi'] = ((ps == 1) & (rs == 0)).astype(int)
    df['divergencia_alcista_rsi'] = ((ps == 0) & (rs == 1)).astype(int)

    st = ta.stoch(h, l, c, k=14, d=3)
    if st is not None and not st.empty:
        df['stoch_k'] = st.iloc[:, 0]
        df['stoch_d'] = st.iloc[:, 1] if st.shape[1] > 1 else np.nan
        df['stoch_sobrecompra'] = (df['stoch_k'] >= 80).astype(int)
        df['stoch_sobreventa']  = (df['stoch_k'] <= 20).astype(int)
    else:
        df['stoch_k'] = df['stoch_d'] = np.nan
        df['stoch_sobrecompra'] = df['stoch_sobreventa'] = 0

    df['cci_20']          = ta.cci(h, l, c, length=20)
    df['roc_10']          = ta.roc(c, length=10)
    df['roc_20']          = ta.roc(c, length=20)
    df['williams_r']      = ta.willr(h, l, c, length=14)
    df['momentum_10']     = c - c.shift(10)
    df['momentum_pct_10'] = c.pct_change(periods=10) * 100

    bb = ta.bbands(c, length=20, std=2)
    if bb is not None and not bb.empty:
        cols = bb.columns.tolist()
        uc = next((x for x in cols if 'U' in x or 'upper' in x.lower()), None)
        mc = next((x for x in cols if 'M' in x or 'mid' in x.lower()), None)
        lc = next((x for x in cols if 'L' in x or 'lower' in x.lower()), None)
        df['bb_upper'] = bb[uc] if uc else np.nan
        df['bb_mid']   = bb[mc] if mc else np.nan
        df['bb_lower'] = bb[lc] if lc else np.nan
        rbb = df['bb_upper'] - df['bb_lower']
        df['bb_width']    = rbb / df['bb_mid'].replace(0, np.nan)
        df['bb_posicion'] = np.where(rbb > 0, (c - df['bb_lower']) / rbb, 0.5)
        df['bb_squeeze']  = (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.7).astype(int)
    else:
        for x in ['bb_upper', 'bb_mid', 'bb_lower', 'bb_width', 'bb_posicion', 'bb_squeeze']:
            df[x] = np.nan

    df['atr_14']  = ta.atr(h, l, c, length=14)
    df['atr_pct'] = df['atr_14'] / c * 100
    kc = ta.kc(h, l, c, length=20)
    if kc is not None and not kc.empty:
        df['kc_upper'] = kc.iloc[:, 0]
        df['kc_mid']   = kc.iloc[:, 1] if kc.shape[1] > 1 else np.nan
        df['kc_lower'] = kc.iloc[:, 2] if kc.shape[1] > 2 else np.nan
    else:
        df['kc_upper'] = df['kc_mid'] = df['kc_lower'] = np.nan

    ret = c.pct_change()
    df['volatilidad_20'] = ret.rolling(20).std() * np.sqrt(24 * 365) * 100
    df['volatilidad_50'] = ret.rolling(50).std() * np.sqrt(24 * 365) * 100
    df['rango_vela']     = h - l
    df['rango_vela_pct'] = (h - l) / c * 100

    df['obv']           = ta.obv(c, v)
    df['obv_ema']       = ta.ema(df['obv'], length=20)
    df['obv_tendencia'] = (df['obv'] > df['obv_ema']).astype(int)
    try:
        dt = df.set_index('timestamp')
        vr = ta.vwap(dt['high'], dt['low'], dt['close'], dt['volume'])
        df['vwap'] = vr.values if vr is not None else np.nan
    except:
        df['vwap'] = np.nan
    df['precio_sobre_vwap'] = (c > df['vwap']).fillna(False).astype(int)
    df['mfi_14']          = ta.mfi(h, l, c, v, length=14)
    df['mfi_sobrecompra'] = (df['mfi_14'] >= 80).astype(int)
    df['mfi_sobreventa']  = (df['mfi_14'] <= 20).astype(int)

    vm = v.rolling(20).mean()
    vs = v.rolling(20).std()
    df['volumen_ratio']    = v / vm
    df['volumen_anomalo']  = (v > vm + 2 * vs).astype(int)
    df['volumen_muy_bajo'] = (v < vm * 0.3).astype(int)
    va = (c > df['open']).astype(int)
    df['confirmacion_alcista'] = ((va == 1) & (df['volumen_anomalo'] == 1)).astype(int)
    df['confirmacion_bajista'] = ((va == 0) & (df['volumen_anomalo'] == 1)).astype(int)

    df['es_resistencia'] = (h == h.rolling(20, center=True).max()).astype(int)
    df['es_soporte']     = (l == l.rolling(20, center=True).min()).astype(int)
    ur = h.where(df['es_resistencia'] == 1).ffill()
    us = l.where(df['es_soporte'] == 1).ffill()
    df['dist_resistencia_pct'] = (ur - c) / c * 100
    df['dist_soporte_pct']     = (c - us) / c * 100

    mx = h.rolling(50).max()
    mn = l.rolling(50).min()
    rg = mx - mn
    df['fib_236'] = mx - rg * 0.236
    df['fib_382'] = mx - rg * 0.382
    df['fib_500'] = mx - rg * 0.500
    df['fib_618'] = mx - rg * 0.618
    for nv in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
        df[f'cerca_{nv}'] = (abs(c - df[nv]) / c < 0.005).astype(int)

    df['ema200_pendiente'] = df['ema_200'].diff(50) / df['ema_200'].shift(50) * 100
    se200    = (c > df['ema_200']).astype(int)
    se50     = (c > df['ema_50']).astype(int)
    e5_200   = (df['ema_50'] > df['ema_200']).astype(int)
    conds    = [
        (se200 == 1) & (se50 == 1) & (e5_200 == 1) & (df['ema200_pendiente'] > 0.5),
        (se200 == 1) & (e5_200 == 1),
        (se200 == 0) & (e5_200 == 0) & (df['ema200_pendiente'] < -0.5),
        (se200 == 0) & (e5_200 == 0),
    ]
    df['regimen_mercado'] = np.select(conds, [2, 1, -2, -1], default=0)
    return df


def contexto_diario(simbolo: str) -> Dict:
    df1d = obtener_velas(simbolo, intervalo=1440, limite=120)
    if df1d is None or len(df1d) < 30:
        return {'disponible': False}
    try:
        import pandas_ta as ta
        c1d = df1d['close']
        ema50_1d  = ta.ema(c1d, length=50)
        ema200_1d = ta.ema(c1d, length=200)
        rsi1d     = ta.rsi(c1d, length=14)
        macd1d_df = ta.macd(c1d, fast=12, slow=26, signal=9)
        macd1d_h  = (macd1d_df.iloc[:, 2]
                     if (macd1d_df is not None and not macd1d_df.empty
                         and macd1d_df.shape[1] > 2)
                     else pd.Series([0.0]))
        precio1d   = float(c1d.iloc[-1])
        ema50_val  = float(ema50_1d.iloc[-1])  if ema50_1d  is not None else precio1d
        ema200_val = float(ema200_1d.iloc[-1]) if ema200_1d is not None else precio1d
        rsi1d_val  = float(rsi1d.iloc[-1])     if not rsi1d.empty else 50.0
        mh1d       = float(macd1d_h.iloc[-1])  if not macd1d_h.empty else 0.0
        mh1d_prev  = float(macd1d_h.iloc[-2])  if len(macd1d_h) > 1 else 0.0
        if precio1d > ema50_val > ema200_val:
            tendencia1d = 'alcista'
        elif precio1d < ema50_val < ema200_val:
            tendencia1d = 'bajista'
        else:
            tendencia1d = 'lateral'
        if mh1d > 0 and mh1d > mh1d_prev:     momentum1d = 'alcista_acelerando'
        elif mh1d > 0 and mh1d < mh1d_prev:   momentum1d = 'alcista_desacelerando'
        elif mh1d < 0 and mh1d < mh1d_prev:   momentum1d = 'bajista_acelerando'
        elif mh1d < 0 and mh1d > mh1d_prev:   momentum1d = 'bajista_desacelerando'
        else:                                   momentum1d = 'neutral'
        return {'disponible': True, 'tendencia': tendencia1d, 'momentum': momentum1d,
                'rsi': rsi1d_val, 'precio': precio1d, 'ema50': ema50_val,
                'ema200': ema200_val, 'df': df1d}
    except Exception as e:
        print(f"  ⚠️  1D contexto error: {e}")
        return {'disponible': False}


def contexto_4h(simbolo: str) -> Dict:
    df4 = obtener_velas(simbolo, intervalo=240, limite=100)
    if df4 is None or len(df4) < 30:
        return {'disponible': False}
    try:
        import pandas_ta as ta
        c4 = df4['close']
        h4 = df4['high']
        l4 = df4['low']
        rsi4     = ta.rsi(c4, length=14)
        ema50_4  = ta.ema(c4, length=50)
        ema200_4 = ta.ema(c4, length=200)
        macd4_df = ta.macd(c4, fast=12, slow=26, signal=9)
        macd4_hist = macd4_df.iloc[:, 2] if (macd4_df is not None and not macd4_df.empty and macd4_df.shape[1] > 2) else pd.Series([0])
        ultima4    = df4.iloc[-1]
        rsi4_val   = float(rsi4.iloc[-1]) if not rsi4.empty else 50
        ema50_val  = float(ema50_4.iloc[-1]) if ema50_4 is not None else float(c4.iloc[-1])
        ema200_val = float(ema200_4.iloc[-1]) if ema200_4 is not None else float(c4.iloc[-1])
        precio4    = float(ultima4['close'])
        mh4        = float(macd4_hist.iloc[-1]) if not macd4_hist.empty else 0
        mh4_prev   = float(macd4_hist.iloc[-2]) if len(macd4_hist) > 1 else 0
        if precio4 > ema50_val > ema200_val:    tendencia4 = 'alcista'
        elif precio4 < ema50_val < ema200_val:  tendencia4 = 'bajista'
        else:                                    tendencia4 = 'lateral'
        if mh4 > 0 and mh4 > mh4_prev:         momentum4 = 'alcista_acelerando'
        elif mh4 > 0 and mh4 < mh4_prev:        momentum4 = 'alcista_desacelerando'
        elif mh4 < 0 and mh4 < mh4_prev:        momentum4 = 'bajista_acelerando'
        elif mh4 < 0 and mh4 > mh4_prev:        momentum4 = 'bajista_desacelerando'
        else:                                    momentum4 = 'neutral'
        return {'disponible': True, 'tendencia': tendencia4, 'momentum': momentum4,
                'rsi': rsi4_val, 'precio': precio4, 'ema50': ema50_val,
                'ema200': ema200_val, 'df': df4}
    except Exception as e:
        print(f"  ⚠️  4H contexto error: {e}")
        return {'disponible': False}


# ──────────────────────────────────────────────
#  DETECCIÓN DE PATRONES CHARTISTAS
# ──────────────────────────────────────────────
def detectar_patrones(df: pd.DataFrame, df4h: pd.DataFrame = None) -> Dict:
    resultado = {
        'patrones_short': [], 'patrones_long': [],
        'score_short': 0, 'score_long': 0, 'niveles_clave': [],
    }
    if df is None or len(df) < 50:
        return resultado

    c   = df['close'].values.astype(float)
    h   = df['high'].values.astype(float)
    l   = df['low'].values.astype(float)
    n   = len(c)
    precio_actual = c[-1]

    def detectar_zonas_sr(highs_arr, lows_arr, closes_arr, banda_pct=0.003):
        picos, valles = [], []
        for i in range(2, len(highs_arr) - 2):
            if (highs_arr[i] > highs_arr[i-1] and highs_arr[i] > highs_arr[i-2]
                    and highs_arr[i] > highs_arr[i+1] and highs_arr[i] > highs_arr[i+2]):
                if closes_arr[i+1] < closes_arr[i]:
                    picos.append(highs_arr[i])
            if (lows_arr[i] < lows_arr[i-1] and lows_arr[i] < lows_arr[i-2]
                    and lows_arr[i] < lows_arr[i+1] and lows_arr[i] < lows_arr[i+2]):
                if closes_arr[i+1] > closes_arr[i]:
                    valles.append(lows_arr[i])
        zonas_res, zonas_sop = [], []
        for precio_pico in sorted(picos):
            agrupado = False
            for zona in zonas_res:
                if abs(precio_pico - zona['nivel']) / zona['nivel'] < banda_pct:
                    zona['nivel'] = (zona['nivel'] * zona['toques'] + precio_pico) / (zona['toques'] + 1)
                    zona['toques'] += 1; agrupado = True; break
            if not agrupado:
                zonas_res.append({'nivel': precio_pico, 'toques': 1, 'tipo': 'resistencia'})
        for precio_valle in sorted(valles):
            agrupado = False
            for zona in zonas_sop:
                if abs(precio_valle - zona['nivel']) / zona['nivel'] < banda_pct:
                    zona['nivel'] = (zona['nivel'] * zona['toques'] + precio_valle) / (zona['toques'] + 1)
                    zona['toques'] += 1; agrupado = True; break
            if not agrupado:
                zonas_sop.append({'nivel': precio_valle, 'toques': 1, 'tipo': 'soporte'})
        return [z for z in zonas_res if z['toques'] >= 2], [z for z in zonas_sop if z['toques'] >= 2]

    ventana_1h = min(60, n)
    zonas_res_1h, zonas_sop_1h = detectar_zonas_sr(h[-ventana_1h:], l[-ventana_1h:], c[-ventana_1h:])
    zonas_res_4h, zonas_sop_4h = [], []
    if df4h is not None and len(df4h) >= 30:
        h4 = df4h['high'].values.astype(float)
        l4 = df4h['low'].values.astype(float)
        c4 = df4h['close'].values.astype(float)
        zonas_res_4h, zonas_sop_4h = detectar_zonas_sr(h4, l4, c4, banda_pct=0.004)

    def zona_confirmada_4h(nivel_1h, zonas_4h, banda=0.006):
        return any(abs(nivel_1h - z4['nivel']) / z4['nivel'] < banda for z4 in zonas_4h)

    for zona in sorted(zonas_res_1h, key=lambda z: -z['toques']):
        nivel = zona['nivel']; toques = zona['toques']
        dist = (nivel - precio_actual) / precio_actual
        if -0.005 < dist < 0.015:
            confirma_4h = zona_confirmada_4h(nivel, zonas_res_4h)
            score_zona  = 1.0 + (0.5 if toques >= 3 else 0) + (1.0 if confirma_4h else 0)
            label = f"Resistencia ${nivel:,.2f} ({toques} toques{'  ✅4H' if confirma_4h else ''})"
            resultado['niveles_clave'].append(('resistencia', nivel))
            resultado['patrones_short'].append(label); resultado['score_short'] += score_zona; break

    for zona in sorted(zonas_sop_1h, key=lambda z: -z['toques']):
        nivel = zona['nivel']; toques = zona['toques']
        dist = (precio_actual - nivel) / precio_actual
        if -0.005 < dist < 0.015:
            confirma_4h = zona_confirmada_4h(nivel, zonas_sop_4h)
            score_zona  = 1.0 + (0.5 if toques >= 3 else 0) + (1.0 if confirma_4h else 0)
            label = f"Soporte ${nivel:,.2f} ({toques} toques{'  ✅4H' if confirma_4h else ''})"
            resultado['niveles_clave'].append(('soporte', nivel))
            resultado['patrones_long'].append(label); resultado['score_long'] += score_zona; break

    # Doble techo
    ventana_dt = min(30, n)
    highs_dt   = h[-ventana_dt:]; lows_dt = l[-ventana_dt:]
    idx_picos  = [i for i in range(2, len(highs_dt) - 2)
                  if highs_dt[i] > highs_dt[i-1] and highs_dt[i] > highs_dt[i-2]
                  and highs_dt[i] > highs_dt[i+1] and highs_dt[i] > highs_dt[i+2]]
    if len(idx_picos) >= 2:
        p1, p2 = highs_dt[idx_picos[-2]], highs_dt[idx_picos[-1]]
        sep = idx_picos[-1] - idx_picos[-2]
        if abs(p1 - p2) / p1 < 0.015 and sep >= 5:
            tramo = lows_dt[idx_picos[-2]:idx_picos[-1]]
            nk = float(np.min(tramo)) if len(tramo) > 0 else min(p1, p2) * 0.98
            resultado['niveles_clave'].extend([('doble_techo', (p1+p2)/2), ('neckline_dt', nk)])
            nk_perf = precio_actual <= nk * 1.005
            nk_cerca = precio_actual <= nk * 1.020
            inicio_p = max(0, n - ventana_dt - 15); fin_p = max(0, n - ventana_dt + idx_picos[-2])
            tend_alc = False
            if fin_p > inicio_p + 5:
                tp = c[inicio_p:fin_p]
                try:
                    cf = np.polyfit(np.arange(len(tp)), tp, 1)
                    tend_alc = cf[0] > 0 and (tp[-1]-tp[0])/tp[0] > 0.02
                except: pass
            if tend_alc and nk_cerca:
                score_dt = 2.5 if nk_perf else 2.0
                resultado['patrones_short'].append(
                    f"Doble techo ${(p1+p2)/2:,.2f} | Neckline ${nk:,.2f} {'⚡PERFORADA' if nk_perf else '⚠️cerca'}")
                resultado['score_short'] += score_dt

    # Doble suelo
    lows_ds = l[-ventana_dt:]; highs_ds = h[-ventana_dt:]
    idx_valles = [i for i in range(2, len(lows_ds) - 2)
                  if lows_ds[i] < lows_ds[i-1] and lows_ds[i] < lows_ds[i-2]
                  and lows_ds[i] < lows_ds[i+1] and lows_ds[i] < lows_ds[i+2]]
    if len(idx_valles) >= 2:
        v1, v2 = lows_ds[idx_valles[-2]], lows_ds[idx_valles[-1]]
        sep = idx_valles[-1] - idx_valles[-2]
        if abs(v1 - v2) / v1 < 0.015 and sep >= 5:
            tramo = highs_ds[idx_valles[-2]:idx_valles[-1]]
            nk = float(np.max(tramo)) if len(tramo) > 0 else max(v1, v2) * 1.02
            resultado['niveles_clave'].extend([('doble_suelo', (v1+v2)/2), ('neckline_ds', nk)])
            nk_perf = precio_actual >= nk * 0.995
            nk_cerca = precio_actual >= nk * 0.980
            inicio_p = max(0, n - ventana_dt - 15); fin_p = max(0, n - ventana_dt + idx_valles[-2])
            tend_baj = False
            if fin_p > inicio_p + 5:
                tp = c[inicio_p:fin_p]
                try:
                    cf = np.polyfit(np.arange(len(tp)), tp, 1)
                    tend_baj = cf[0] < 0 and (tp[0]-tp[-1])/tp[0] > 0.02
                except: pass
            if tend_baj and nk_cerca:
                score_ds = 2.5 if nk_perf else 2.0
                resultado['patrones_long'].append(
                    f"Doble suelo ${(v1+v2)/2:,.2f} | Neckline ${nk:,.2f} {'⚡PERFORADA' if nk_perf else '⚠️cerca'}")
                resultado['score_long'] += score_ds

    # Canal
    ventana_canal = min(25, n)
    highs_canal = h[-ventana_canal:]; lows_canal = l[-ventana_canal:]
    x_canal = np.arange(ventana_canal)
    pend_h_norm = pend_l_norm = None
    try:
        coef_h = np.polyfit(x_canal, highs_canal, 1)
        coef_l = np.polyfit(x_canal, lows_canal, 1)
        pend_h_norm = coef_h[0] / precio_actual
        pend_l_norm = coef_l[0] / precio_actual
        techo_canal = np.polyval(coef_h, ventana_canal - 1)
        suelo_canal = np.polyval(coef_l, ventana_canal - 1)
        ancho_canal = (techo_canal - suelo_canal) / precio_actual
        toques_techo = sum(1 for i in x_canal if abs(highs_canal[i] - np.polyval(coef_h, i)) / np.polyval(coef_h, i) < 0.008)
        toques_suelo = sum(1 for i in x_canal if abs(lows_canal[i]  - np.polyval(coef_l, i)) / np.polyval(coef_l, i) < 0.008)
        canal_valido = (ancho_canal > 0.01 and abs(pend_h_norm - pend_l_norm) < 0.0015
                        and toques_techo >= 2 and toques_suelo >= 2)
        if canal_valido:
            dist_techo = (techo_canal - precio_actual) / precio_actual
            dist_suelo = (precio_actual - suelo_canal) / precio_actual
            info = f"({toques_techo}t techo / {toques_suelo}t suelo)"
            if pend_h_norm > 0.001 and pend_l_norm > 0.001:
                resultado['niveles_clave'].extend([('techo_canal_alcista', techo_canal), ('suelo_canal_alcista', suelo_canal)])
                if dist_techo < 0.015: resultado['patrones_short'].append(f"Canal alcista: precio en techo ~${techo_canal:,.2f} {info}"); resultado['score_short'] += 1.5
                if dist_suelo < 0.015: resultado['patrones_long'].append(f"Canal alcista: precio en suelo ~${suelo_canal:,.2f} {info}"); resultado['score_long'] += 1.0
            elif pend_h_norm < -0.001 and pend_l_norm < -0.001:
                resultado['niveles_clave'].extend([('techo_canal_bajista', techo_canal), ('suelo_canal_bajista', suelo_canal)])
                if dist_techo < 0.015: resultado['patrones_short'].append(f"Canal bajista: precio en techo ~${techo_canal:,.2f} {info}"); resultado['score_short'] += 1.5
                if dist_suelo < 0.015: resultado['patrones_long'].append(f"Canal bajista: precio en suelo ~${suelo_canal:,.2f} {info}"); resultado['score_long'] += 1.0
    except: pass

    # Cuña
    ventana_cuna = min(20, n)
    highs_cuna = h[-ventana_cuna:]; lows_cuna = l[-ventana_cuna:]
    x_cuna = np.arange(ventana_cuna)
    try:
        coef_hc = np.polyfit(x_cuna, highs_cuna, 1)
        coef_lc = np.polyfit(x_cuna, lows_cuna, 1)
        ph = coef_hc[0] / precio_actual; pl = coef_lc[0] / precio_actual
        techo_cuna = np.polyval(coef_hc, ventana_cuna - 1)
        suelo_cuna = np.polyval(coef_lc, ventana_cuna - 1)
        ancho_cuna = (techo_cuna - suelo_cuna) / precio_actual
        toques_t = sum(1 for i in x_cuna if abs(highs_cuna[i] - np.polyval(coef_hc, i)) / np.polyval(coef_hc, i) < 0.008)
        toques_s = sum(1 for i in x_cuna if abs(lows_cuna[i]  - np.polyval(coef_lc, i)) / np.polyval(coef_lc, i) < 0.008)
        conv = abs(ph - pl)
        if 0.0008 < conv < 0.006 and ancho_cuna > 0.007 and toques_t >= 2 and toques_s >= 2:
            info_c = f"({toques_t}t/{toques_s}t)"
            if ph > 0 and pl > 0 and ph < pl:
                resultado['patrones_short'].append(f"Cuña alcista convergente {info_c} (ruptura bajista probable)")
                resultado['score_short'] += 1.5
            elif ph < 0 and pl < 0 and pl > ph:
                resultado['patrones_long'].append(f"Cuña bajista convergente {info_c} (ruptura alcista probable)")
                resultado['score_long'] += 1.5
    except: pass

    # Bandera
    if n >= 25:
        try:
            mejor_mov = 0.0; mejor_ini = -1; mejor_fin = -1
            v_imp = 7; v_con = 8; min_ini = max(0, n - 30)
            for ini in range(min_ini, n - v_imp - v_con):
                fin = ini + v_imp
                if fin + v_con > n: break
                rm = (np.max(h[ini:fin]) - np.min(l[ini:fin])) / np.min(l[ini:fin])
                if rm > mejor_mov: mejor_mov = rm; mejor_ini = ini; mejor_fin = fin
            if mejor_mov > 0.025 and mejor_fin > 0:
                ci = mejor_fin; cf = min(n, ci + v_con)
                imp_c = c[mejor_ini:mejor_fin]; con_c = c[ci:cf]
                con_h = h[ci:cf]; con_l = l[ci:cf]
                if len(con_c) >= 3:
                    rc = (np.max(con_h) - np.min(con_l)) / np.min(con_l)
                    slope = (float(con_c[-1]) - float(con_c[0])) / float(con_c[0])
                    if rc < mejor_mov * 0.40:
                        if imp_c[-1] > imp_c[0] and slope < -0.003:
                            resultado['patrones_long'].append(f"Bandera alcista (impulso +{mejor_mov*100:.1f}%, consol {slope*100:+.1f}%)")
                            resultado['score_long'] += 1.5
                        elif imp_c[-1] < imp_c[0] and slope > 0.003:
                            resultado['patrones_short'].append(f"Bandera bajista (impulso -{mejor_mov*100:.1f}%, consol {slope*100:+.1f}%)")
                            resultado['score_short'] += 1.5
        except: pass

    # Confirmación 4H
    if df4h is not None and len(df4h) >= 30:
        try:
            import pandas_ta as ta
            c4 = df4h['close']
            ema50_4 = ta.ema(c4, length=50); ema200_4 = ta.ema(c4, length=200)
            precio4 = float(c4.iloc[-1]); e50 = float(ema50_4.iloc[-1]); e200 = float(ema200_4.iloc[-1])
            t4 = 'alcista' if precio4 > e50 > e200 else ('bajista' if precio4 < e50 < e200 else 'lateral')
            if t4 == 'bajista' and resultado['score_short'] > 0:
                resultado['score_short'] *= 1.3; resultado['patrones_short'].append("4H confirma dirección bajista del patrón")
            elif t4 == 'alcista' and resultado['score_long'] > 0:
                resultado['score_long'] *= 1.3; resultado['patrones_long'].append("4H confirma dirección alcista del patrón")
        except: pass

    resultado['score_short'] = round(resultado['score_short'], 1)
    resultado['score_long']  = round(resultado['score_long'],  1)
    return resultado


# ──────────────────────────────────────────────
#  DIVERGENCIAS
# ──────────────────────────────────────────────
def detectar_divergencias(df: pd.DataFrame, ventana: int = 40) -> Dict:
    resultado = {
        'div_bajista_rsi': False, 'div_bajista_rsi_fuerza': 0.0,
        'div_alcista_rsi': False, 'div_alcista_rsi_fuerza': 0.0,
        'div_bajista_macd': False, 'div_bajista_macd_fuerza': 0.0,
        'div_alcista_macd': False, 'div_alcista_macd_fuerza': 0.0,
        'div_bajista_oculta': False, 'div_alcista_oculta': False,
        'convergencia_bajista': False, 'convergencia_alcista': False,
        'resumen': [],
    }
    n = min(ventana, len(df))
    if n < 10: return resultado
    sub      = df.iloc[-n:].reset_index(drop=True)
    highs    = sub['high'].values.astype(float)
    lows     = sub['low'].values.astype(float)
    rsi_vals = sub['rsi_14'].values.astype(float) if 'rsi_14' in sub else None
    macd_vals = sub['macd_hist'].values.astype(float) if 'macd_hist' in sub else None

    def picos_locales(arr, radio=2):
        return [i for i in range(radio, len(arr)-radio)
                if all(arr[i] > arr[i-j] for j in range(1,radio+1))
                and all(arr[i] > arr[i+j] for j in range(1,radio+1))]

    def valles_locales(arr, radio=2):
        return [i for i in range(radio, len(arr)-radio)
                if all(arr[i] < arr[i-j] for j in range(1,radio+1))
                and all(arr[i] < arr[i+j] for j in range(1,radio+1))]

    def fuerza_div(v1, v2, escala):
        return float(np.clip(abs(v1-v2)/escala, 0, 1)) if escala != 0 else 0.0

    picos_precio  = picos_locales(highs)
    valles_precio = valles_locales(lows)

    if rsi_vals is not None and not np.all(np.isnan(rsi_vals)):
        pr = np.nan_to_num(rsi_vals, nan=50)
        picos_rsi = picos_locales(pr); valles_rsi = valles_locales(pr)
        if len(picos_precio) >= 2 and len(picos_rsi) >= 2:
            pp1, pp2 = picos_precio[-2], picos_precio[-1]
            rp_cerca = [r for r in picos_rsi if abs(r-pp2) <= 3]
            rp_prev  = [r for r in picos_rsi if abs(r-pp1) <= 4]
            if rp_cerca and rp_prev:
                rp1, rp2 = rp_prev[-1], rp_cerca[-1]
                if highs[pp2] > highs[pp1] and rsi_vals[rp2] < rsi_vals[rp1]:
                    f = fuerza_div(rsi_vals[rp1], rsi_vals[rp2], 30)
                    resultado['div_bajista_rsi'] = True; resultado['div_bajista_rsi_fuerza'] = f
                    resultado['resumen'].append(f"Div bajista RSI (f={f:.2f})")
                elif highs[pp2] < highs[pp1] and rsi_vals[rp2] > rsi_vals[rp1]:
                    resultado['div_bajista_oculta'] = True; resultado['resumen'].append("Div bajista RSI oculta")
        if len(valles_precio) >= 2 and len(valles_rsi) >= 2:
            vp1, vp2 = valles_precio[-2], valles_precio[-1]
            rv_cerca = [r for r in valles_rsi if abs(r-vp2) <= 3]
            rv_prev  = [r for r in valles_rsi if abs(r-vp1) <= 4]
            if rv_cerca and rv_prev:
                rv1, rv2 = rv_prev[-1], rv_cerca[-1]
                if lows[vp2] < lows[vp1] and rsi_vals[rv2] > rsi_vals[rv1]:
                    f = fuerza_div(rsi_vals[rv1], rsi_vals[rv2], 30)
                    resultado['div_alcista_rsi'] = True; resultado['div_alcista_rsi_fuerza'] = f
                    resultado['resumen'].append(f"Div alcista RSI (f={f:.2f})")
                elif lows[vp2] > lows[vp1] and rsi_vals[rv2] < rsi_vals[rv1]:
                    resultado['div_alcista_oculta'] = True; resultado['resumen'].append("Div alcista RSI oculta")

    if macd_vals is not None and not np.all(np.isnan(macd_vals)):
        mc = np.nan_to_num(macd_vals, nan=0.0)
        picos_macd = picos_locales(mc); valles_macd = valles_locales(mc)
        escala_m = float(np.nanstd(macd_vals)) * 2 if np.nanstd(macd_vals) > 0 else 1
        if len(picos_precio) >= 2 and len(picos_macd) >= 2:
            pp1, pp2 = picos_precio[-2], picos_precio[-1]
            mp_cerca = [m for m in picos_macd if abs(m-pp2) <= 3]
            mp_prev  = [m for m in picos_macd if abs(m-pp1) <= 4]
            if mp_cerca and mp_prev:
                mp1, mp2 = mp_prev[-1], mp_cerca[-1]
                if highs[pp2] > highs[pp1] and mc[mp2] < mc[mp1]:
                    f = fuerza_div(mc[mp1], mc[mp2], escala_m)
                    resultado['div_bajista_macd'] = True; resultado['div_bajista_macd_fuerza'] = f
                    resultado['resumen'].append(f"Div bajista MACD (f={f:.2f})")
        if len(valles_precio) >= 2 and len(valles_macd) >= 2:
            vp1, vp2 = valles_precio[-2], valles_precio[-1]
            mv_cerca = [m for m in valles_macd if abs(m-vp2) <= 3]
            mv_prev  = [m for m in valles_macd if abs(m-vp1) <= 4]
            if mv_cerca and mv_prev:
                mv1, mv2 = mv_prev[-1], mv_cerca[-1]
                if lows[vp2] < lows[vp1] and mc[mv2] > mc[mv1]:
                    f = fuerza_div(mc[mv1], mc[mv2], escala_m)
                    resultado['div_alcista_macd'] = True; resultado['div_alcista_macd_fuerza'] = f
                    resultado['resumen'].append(f"Div alcista MACD (f={f:.2f})")
        if len(mc) >= 4:
            ult4 = mc[-4:]
            if ult4[-1] < 0 and all(ult4[i] > ult4[i-1] for i in range(1,4)):
                resultado['convergencia_alcista'] = True; resultado['resumen'].append("MACD convergiendo alcista")
            elif ult4[-1] > 0 and all(ult4[i] < ult4[i-1] for i in range(1,4)):
                resultado['convergencia_bajista'] = True; resultado['resumen'].append("MACD convergiendo bajista")
    return resultado


# ──────────────────────────────────────────────
#  FUERZA DE NIVEL
# ──────────────────────────────────────────────
def evaluar_fuerza_nivel(df: pd.DataFrame, precio: float, atr: float, tipo: str = 'soporte') -> Dict:
    tolerancia = atr * 0.6
    n = min(80, len(df))
    sub = df.iloc[-n:].reset_index(drop=True)
    extremos = sub['low'].values.astype(float) if tipo == 'soporte' else sub['high'].values.astype(float)
    rebote   = sub['close'].values.astype(float)
    volumen  = sub['volume'].values.astype(float)
    vm_vol   = float(np.mean(volumen)) if np.mean(volumen) > 0 else 1.0
    extremos_validos = extremos[extremos < precio + tolerancia] if tipo == 'soporte' else extremos[extremos > precio - tolerancia]
    if len(extremos_validos) == 0:
        return {'fuerza': 0.0, 'n_toques': 0, 'nivel': precio, 'bloquear_contra': False}
    candidato = float(np.median(extremos_validos[-10:])) if len(extremos_validos) >= 3 else float(extremos_validos[-1])
    toques, vol_toques, bounce_ok = [], [], []
    for i in range(len(extremos)):
        if abs(extremos[i] - candidato) <= tolerancia:
            if tipo == 'soporte' and i + 2 < len(rebote):
                bounced = rebote[i+2] > candidato + tolerancia * 0.5
            elif tipo == 'resistencia' and i + 2 < len(rebote):
                bounced = rebote[i+2] < candidato - tolerancia * 0.5
            else: bounced = False
            toques.append(i); vol_toques.append(volumen[i] / vm_vol); bounce_ok.append(bounced)
    n_toques = len(toques)
    if n_toques == 0:
        return {'fuerza': 0.0, 'n_toques': 0, 'nivel': candidato, 'bloquear_contra': False}
    fuerza = min(n_toques * 0.5, 1.5)
    if n_toques >= 2:
        seps = [toques[i+1] - toques[i] for i in range(len(toques)-1)]
        fuerza += 0.5 if np.mean(seps) >= 8 else (0.25 if np.mean(seps) >= 4 else 0)
    vol_medio = float(np.mean(vol_toques)) if vol_toques else 1.0
    fuerza += 0.5 if vol_medio >= 1.5 else (0.25 if vol_medio >= 1.0 else 0)
    pct_bounce = sum(bounce_ok) / n_toques if n_toques > 0 else 0
    fuerza += pct_bounce * 0.5
    fuerza = float(np.clip(fuerza, 0.0, 3.0))
    en_nivel = abs(precio - candidato) / precio < atr / precio * 1.2
    bloquear = fuerza >= 2.0 and en_nivel
    return {'fuerza': round(fuerza, 2), 'n_toques': n_toques, 'nivel': round(candidato, 6),
            'bloquear_contra': bloquear, 'vol_medio': round(vol_medio, 2), 'pct_bounce': round(pct_bounce, 2)}


# ──────────────────────────────────────────────
#  CONTEXTO GLOBAL DE LA SEÑAL
# ──────────────────────────────────────────────
def evaluar_contexto_señal(df: pd.DataFrame, ultima: pd.Series,
                            ctx4h: Dict, direccion: str,
                            divs: Dict, patrones: Dict) -> Dict:
    precio  = float(ultima.get('close', 0))
    atr     = float(ultima.get('atr_14', precio * 0.01))
    mult_score = 1.0; mult_margen = 1.0; bloquear = False
    razon_bloqueo = ''; contexto_log = []

    tipo_nivel_contra = 'soporte' if direccion == 'short' else 'resistencia'
    nivel_contra = evaluar_fuerza_nivel(df, precio, atr, tipo=tipo_nivel_contra)
    if nivel_contra['bloquear_contra']:
        señales_fuertes = (divs.get('div_bajista_rsi', False) and direccion == 'short' or
                           divs.get('div_alcista_rsi', False) and direccion == 'long')
        señales_macd = (divs.get('convergencia_bajista', False) and direccion == 'short' or
                        divs.get('convergencia_alcista', False) and direccion == 'long')
        if not señales_fuertes and not señales_macd:
            bloquear = True
            razon_bloqueo = (f"Nivel {tipo_nivel_contra} fuerte en contra "
                             f"(fuerza={nivel_contra['fuerza']:.1f}, {nivel_contra['n_toques']} toques) sin señales que lo rompan")
        else:
            mult_score *= 0.7
            contexto_log.append(f"⚠️ {tipo_nivel_contra.capitalize()} fuerte pero con señales de ruptura")
    elif nivel_contra['fuerza'] >= 1.0:
        mult_score *= 0.85
        contexto_log.append(f"Nivel {tipo_nivel_contra} moderado (f={nivel_contra['fuerza']:.1f})")

    if bloquear:
        return {'multiplicador_score': 0.0, 'multiplicador_margen': 1.0, 'bloquear': True,
                'razon_bloqueo': razon_bloqueo, 'nivel_confianza': 'bloqueado',
                'contexto_log': contexto_log, 'nivel_contra': nivel_contra}

    if direccion == 'short':
        div_favor = divs.get('div_bajista_rsi', False) or divs.get('div_bajista_macd', False)
        div_contra = divs.get('div_alcista_rsi', False) or divs.get('div_alcista_macd', False)
        div_fuerza = max(divs.get('div_bajista_rsi_fuerza', 0), divs.get('div_bajista_macd_fuerza', 0))
        conv_favor = divs.get('convergencia_bajista', False); conv_contra = divs.get('convergencia_alcista', False)
    else:
        div_favor = divs.get('div_alcista_rsi', False) or divs.get('div_alcista_macd', False)
        div_contra = divs.get('div_bajista_rsi', False) or divs.get('div_bajista_macd', False)
        div_fuerza = max(divs.get('div_alcista_rsi_fuerza', 0), divs.get('div_alcista_macd_fuerza', 0))
        conv_favor = divs.get('convergencia_alcista', False); conv_contra = divs.get('convergencia_bajista', False)

    if div_favor:
        bonus = 0.15 + div_fuerza * 0.15
        mult_score *= (1 + bonus); mult_margen *= min(1 + bonus * 0.5, 1.20)
        contexto_log.append(f"✅ Divergencia real a favor (fuerza={div_fuerza:.2f})")
    if div_contra:
        mult_score *= 0.65; contexto_log.append("⚠️ Divergencia real EN CONTRA de la dirección")
    if conv_favor:
        mult_score *= 1.12; mult_margen *= 1.08; contexto_log.append("✅ MACD convergiendo a favor")
    if conv_contra:
        mult_score *= 0.80; contexto_log.append("⚠️ MACD convergiendo EN CONTRA")

    t4h = ctx4h.get('tendencia', 'lateral'); m4h = ctx4h.get('momentum', 'neutral')
    if (direccion == 'short' and t4h == 'bajista') or (direccion == 'long' and t4h == 'alcista'):
        bonus_4h = 1.15 if 'acelerando' in m4h else 1.08
        mult_score *= bonus_4h; mult_margen *= min(bonus_4h * 0.6 + 0.4, 1.12)
        contexto_log.append(f"✅ 4H {'bajista' if direccion == 'short' else 'alcista'} ({m4h})")
    elif (direccion == 'short' and t4h == 'alcista') or (direccion == 'long' and t4h == 'bajista'):
        mult_score *= 0.80; contexto_log.append(f"⚠️ 4H en contra ({t4h})")

    tipo_nivel_favor = 'resistencia' if direccion == 'short' else 'soporte'
    nivel_favor = evaluar_fuerza_nivel(df, precio, atr, tipo=tipo_nivel_favor)
    if nivel_favor['fuerza'] >= 2.0:
        mult_score *= 1.10; mult_margen *= 1.05
        contexto_log.append(f"✅ {tipo_nivel_favor.capitalize()} fuerte a favor (f={nivel_favor['fuerza']:.1f})")
    elif nivel_favor['fuerza'] >= 1.0:
        mult_score *= 1.05

    mult_margen = float(np.clip(mult_margen, 0.60, 1.40))
    mult_score  = float(np.clip(mult_score, 0.20, 2.50))
    if mult_margen >= 1.25 and mult_score >= 1.30:   nivel_confianza = 'maximo'
    elif mult_margen >= 1.10 or mult_score >= 1.15:   nivel_confianza = 'alto'
    elif mult_score <= 0.70:                           nivel_confianza = 'bajo'
    else:                                              nivel_confianza = 'normal'

    return {'multiplicador_score': round(mult_score, 3), 'multiplicador_margen': round(mult_margen, 3),
            'bloquear': False, 'razon_bloqueo': '', 'nivel_confianza': nivel_confianza,
            'contexto_log': contexto_log, 'nivel_contra': nivel_contra, 'nivel_favor': nivel_favor}


# ──────────────────────────────────────────────
#  SCORE PREDICTIVO
# ──────────────────────────────────────────────
def calcular_rsi_umbral(rsi_historico: pd.Series, direccion: str) -> tuple:
    if rsi_historico is None or len(rsi_historico.dropna()) < 20:
        return 65, 35
    rsi_clean = rsi_historico.dropna().tail(50)
    rsi_med = float(rsi_clean.median())
    rsi_p75 = float(rsi_clean.quantile(0.75))
    rsi_p25 = float(rsi_clean.quantile(0.25))
    if rsi_med > 58:
        return float(np.clip(rsi_p75, 68, 80)), float(np.clip(rsi_p25, 35, 48))
    elif rsi_med < 42:
        return float(np.clip(rsi_p75, 55, 68)), float(np.clip(rsi_p25, 22, 35))
    return 65, 35


def score_predictivo_short(df: pd.DataFrame, ultima: pd.Series,
                            ctx4h: Dict, patrones: Dict = None,
                            divs: Dict = None, prob_ct: float = None) -> Tuple[float, List[str]]:
    score = 0.0; razones = []
    c = float(ultima.get('close', 0))
    rsi = float(ultima.get('rsi_14', 50))
    rsi_serie = df['rsi_14'] if 'rsi_14' in df.columns else None
    umbral_sc, _ = calcular_rsi_umbral(rsi_serie, 'short')
    if rsi >= umbral_sc:        score += 1.0; razones.append(f"RSI={rsi:.0f} sobrecomprado (umbral={umbral_sc:.0f})")
    elif rsi >= umbral_sc - 5:  score += 0.5

    if divs:
        if divs.get('div_bajista_rsi'):
            f = divs.get('div_bajista_rsi_fuerza', 0.5); score += 1.0 + f * 0.5
            razones.append(f"Divergencia bajista RSI real (fuerza={f:.2f})")
        if divs.get('div_bajista_macd'):
            f = divs.get('div_bajista_macd_fuerza', 0.5); score += 1.0 + f * 0.5
            razones.append(f"Divergencia bajista MACD real (fuerza={f:.2f})")
        if divs.get('convergencia_bajista'):
            score += 0.8; razones.append("MACD hist convergiendo bajista")
        if divs.get('div_alcista_rsi') or divs.get('div_alcista_macd'):
            score -= 1.5; razones.append("⚠️ Divergencia alcista real en contra del short")
        if divs.get('convergencia_alcista'):
            score -= 0.6; razones.append("⚠️ MACD convergiendo alcista (en contra)")
    else:
        if int(ultima.get('divergencia_bajista_rsi', 0)) == 1:
            score += 0.8; razones.append("Div bajista RSI (señal básica)")

    stk = float(ultima.get('stoch_k', 50)); std_val = float(ultima.get('stoch_d', 50))
    if stk >= 78 and stk < std_val + 2:   score += 1.0; razones.append(f"Stoch={stk:.0f} sobrecomprado y girando")
    elif stk >= 70:                         score += 0.5

    mh = df['macd_hist'].dropna()
    if len(mh) >= 3:
        mh_act, mh_prev, mh_ant = float(mh.iloc[-1]), float(mh.iloc[-2]), float(mh.iloc[-3])
        if mh_act > 0 and mh_act < mh_prev < mh_ant:   score += 1.5; razones.append("MACD hist 2 velas decelerando (pico doble)")
        elif mh_act > 0 and mh_act < mh_prev:            score += 1.0; razones.append("MACD hist decelerando (pico)")

    dist_res = float(ultima.get('dist_resistencia_pct', 999))
    cerca_618 = int(ultima.get('cerca_fib_618', 0))
    bb_pos = float(ultima.get('bb_posicion', 0.5))
    if dist_res < 0.5:      score += 1.0; razones.append(f"Precio en resistencia ({dist_res:.2f}%)")
    elif cerca_618 == 1:    score += 1.0; razones.append("Fibonacci 0.618")
    elif bb_pos > 0.90:     score += 0.5; razones.append(f"BB superior ({bb_pos:.2f})")

    vr = df['volumen_ratio'].dropna()
    precio_sube = c > float(df['close'].iloc[-3]) if len(df) >= 3 else False
    if len(vr) >= 3 and precio_sube and float(vr.iloc[-1]) < 0.80 and float(vr.iloc[-2]) < 0.80:
        score += 1.0; razones.append("Volumen bajo en subida")

    if ctx4h.get('disponible'):
        t4, m4, r4 = ctx4h['tendencia'], ctx4h['momentum'], ctx4h.get('rsi', 50)
        if t4 == 'bajista':                         score += 2.0; razones.append("4H BAJISTA")
        elif t4 == 'lateral' and 'bajista' in m4:   score += 1.0; razones.append("4H lateral momentum bajista")
        if r4 >= 65:                                score += 0.5; razones.append(f"RSI 4H={r4:.0f} sobrecomprado")
        if m4 == 'alcista_desacelerando':           score += 1.0; razones.append("4H alcista desacelerando")

    reg = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', np.nan)) else 0
    if reg <= -1:   score += 0.5
    elif reg >= 2:  score -= 0.5

    if patrones and patrones['score_short'] > 0:
        score += patrones['score_short']
        for p in patrones['patrones_short']: razones.append(f"📐 {p}")

    return score, razones


def score_predictivo_long(df: pd.DataFrame, ultima: pd.Series,
                           ctx4h: Dict, patrones: Dict = None,
                           divs: Dict = None, prob_ct: float = None) -> Tuple[float, List[str]]:
    score = 0.0; razones = []
    rsi = float(ultima.get('rsi_14', 50))
    rsi_serie = df['rsi_14'] if 'rsi_14' in df.columns else None
    _, umbral_sv = calcular_rsi_umbral(rsi_serie, 'long')
    if rsi <= umbral_sv:        score += 1.0; razones.append(f"RSI={rsi:.0f} sobrevendido (umbral={umbral_sv:.0f})")
    elif rsi <= umbral_sv + 5:  score += 0.5

    if divs:
        if divs.get('div_alcista_rsi'):
            f = divs.get('div_alcista_rsi_fuerza', 0.5); score += 1.0 + f * 0.5
            razones.append(f"Divergencia alcista RSI real (fuerza={f:.2f})")
        if divs.get('div_alcista_macd'):
            f = divs.get('div_alcista_macd_fuerza', 0.5); score += 1.0 + f * 0.5
            razones.append(f"Divergencia alcista MACD real (fuerza={f:.2f})")
        if divs.get('convergencia_alcista'):
            score += 0.8; razones.append("MACD convergiendo alcista")
        if divs.get('div_bajista_rsi') or divs.get('div_bajista_macd'):
            score -= 1.5; razones.append("⚠️ Divergencia bajista real en contra del long")
        if divs.get('convergencia_bajista'):
            score -= 0.6; razones.append("⚠️ MACD convergiendo bajista (en contra)")
    else:
        if int(ultima.get('divergencia_alcista_rsi', 0)) == 1:
            score += 0.8; razones.append("Div alcista RSI (señal básica)")

    stk = float(ultima.get('stoch_k', 50)); std_val = float(ultima.get('stoch_d', 50))
    if stk <= 22 and stk > std_val - 2:    score += 1.0; razones.append(f"Stoch={stk:.0f} sobrevendido y girando")
    elif stk <= 30:                          score += 0.5

    mh = df['macd_hist'].dropna()
    if len(mh) >= 3:
        mh_act, mh_prev, mh_ant = float(mh.iloc[-1]), float(mh.iloc[-2]), float(mh.iloc[-3])
        if mh_act < 0 and mh_act > mh_prev > mh_ant:   score += 1.5; razones.append("MACD hist 2 velas mejorando (suelo doble)")
        elif mh_act < 0 and mh_act > mh_prev:            score += 1.0; razones.append("MACD hist mejorando desde suelo")

    dist_sop = float(ultima.get('dist_soporte_pct', 999))
    cerca_382 = int(ultima.get('cerca_fib_382', 0)); cerca_500 = int(ultima.get('cerca_fib_500', 0))
    bb_pos = float(ultima.get('bb_posicion', 0.5))
    if dist_sop < 0.5:              score += 1.0; razones.append(f"Precio en soporte ({dist_sop:.2f}%)")
    elif cerca_382 == 1 or cerca_500 == 1: score += 1.0; razones.append("Fibonacci 0.382/0.5")
    elif bb_pos < 0.10:             score += 0.5; razones.append(f"BB inferior ({bb_pos:.2f})")

    vr = df['volumen_ratio'].dropna()
    precio_baja = float(ultima.get('close', 0)) < float(df['close'].iloc[-3]) if len(df) >= 3 else False
    if len(vr) >= 3 and precio_baja and float(vr.iloc[-1]) < 0.80 and float(vr.iloc[-2]) < 0.80:
        score += 1.0; razones.append("Volumen bajo en bajada (vendedores agotados)")

    if ctx4h.get('disponible'):
        t4, m4, r4 = ctx4h['tendencia'], ctx4h['momentum'], ctx4h.get('rsi', 50)
        if t4 == 'alcista':                         score += 2.0; razones.append("4H ALCISTA")
        elif t4 == 'lateral' and 'alcista' in m4:   score += 1.0; razones.append("4H lateral momentum alcista")
        if r4 <= 35:                                score += 0.5; razones.append(f"RSI 4H={r4:.0f} sobrevendido")
        if m4 == 'bajista_desacelerando':           score += 1.0; razones.append("4H bajista desacelerando")

    reg = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', np.nan)) else 0
    if reg >= 1:    score += 0.5
    elif reg <= -2: score -= 0.5

    if patrones and patrones['score_long'] > 0:
        score += patrones['score_long']
        for p in patrones['patrones_long']: razones.append(f"📐 {p}")

    return score, razones


# ──────────────────────────────────────────────
#  ADAPTIVE SIZING
# ──────────────────────────────────────────────
def actualizar_adaptive(estado: Dict, pnl: float):
    h = estado.setdefault('historial_pnl', [])
    h.append(1.0 if pnl > 0 else 0.0)
    if len(h) > ADAPTIVE_VENTANA: h.pop(0)
    if len(h) >= ADAPTIVE_VENTANA:
        wr = sum(h) / len(h)
        af = estado.get('adaptive_factor', 1.0)
        tl = estado.get('adaptive_trades_left', 0)
        if wr < ADAPTIVE_WR_BAJO and af != ADAPTIVE_FACTOR_C:
            estado['adaptive_factor'] = ADAPTIVE_FACTOR_C
            estado['adaptive_trades_left'] = ADAPTIVE_DURACION
            telegram(f"⚠️ <b>{VERSION} MODO CONSERVADOR</b>\nWR-20: {wr*100:.0f}%")
        elif wr > ADAPTIVE_WR_ALTO:
            estado['adaptive_factor'] = ADAPTIVE_FACTOR_A
        elif tl > 0:
            estado['adaptive_trades_left'] = tl - 1
            if tl - 1 == 0: estado['adaptive_factor'] = 1.0
        elif af == ADAPTIVE_FACTOR_C:
            estado['adaptive_factor'] = 1.0


# ──────────────────────────────────────────────
#  MOTOR DE SL/TP
# ──────────────────────────────────────────────
def _niveles_mercado(df: pd.DataFrame, ultima: pd.Series, precio: float, atr: float) -> Dict:
    candidatos_res = []; candidatos_sop = []
    def añadir(nivel, etiqueta, lista_res, lista_sop, zona=0.0015):
        if pd.isna(nivel) or nivel <= 0: return
        dist = (nivel - precio) / precio
        if dist > zona:   lista_res.append((nivel, etiqueta, dist))
        elif dist < -zona: lista_sop.append((nivel, etiqueta, -dist))
    for p in [9, 21, 50, 100, 200]:
        v = float(ultima.get(f'ema_{p}', 0)); añadir(v, f'EMA{p}', candidatos_res, candidatos_sop)
    for p in [20, 50, 200]:
        v = float(ultima.get(f'sma_{p}', 0)); añadir(v, f'SMA{p}', candidatos_res, candidatos_sop)
    añadir(float(ultima.get('bb_upper', 0)), 'BB_sup', candidatos_res, candidatos_sop)
    añadir(float(ultima.get('bb_lower', 0)), 'BB_inf', candidatos_res, candidatos_sop)
    for nv in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
        v = float(ultima.get(nv, 0)); añadir(v, nv.upper(), candidatos_res, candidatos_sop)
    h50 = float(df['high'].rolling(50).max().iloc[-1]); l50 = float(df['low'].rolling(50).min().iloc[-1])
    h20 = float(df['high'].rolling(20).max().iloc[-1]); l20 = float(df['low'].rolling(20).min().iloc[-1])
    añadir(h50, 'MAX50', candidatos_res, candidatos_sop); añadir(l50, 'MIN50', candidatos_res, candidatos_sop)
    añadir(h20, 'MAX20', candidatos_res, candidatos_sop); añadir(l20, 'MIN20', candidatos_res, candidatos_sop)
    highs = df['high'].values[-40:].astype(float); lows = df['low'].values[-40:].astype(float)
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            añadir(highs[i], 'pico_local', candidatos_res, candidatos_sop, zona=0.001)
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            añadir(lows[i], 'valle_local', candidatos_res, candidatos_sop, zona=0.001)
    candidatos_res.sort(key=lambda x: x[2]); candidatos_sop.sort(key=lambda x: x[2])
    return {'resistencias': candidatos_res, 'soportes': candidatos_sop}


def calcular_sl_tp_niveles(precio: float, atr: float, cfg: dict,
                            direccion: str, niveles: Dict, ctx4h: Dict) -> Dict:
    resistencias = niveles['resistencias']; soportes = niveles['soportes']
    sl_min_dist = atr * SL_ATR_MIN_MULT; sl_max_dist = atr * SL_ATR_MAX_MULT
    tp_max_dist = atr * TP_ATR_MAX_MULT; buffer = precio * BUFFER_NIVEL_PCT
    rr_min = cfg.get('rr_min', RR_MINIMO)
    es_swing = (ctx4h.get('disponible', False) and
                ((direccion == 'short' and ctx4h.get('tendencia') == 'bajista') or
                 (direccion == 'long'  and ctx4h.get('tendencia') == 'alcista')) and
                ctx4h.get('momentum', '') in ('bajista_acelerando', 'alcista_acelerando',
                                               'bajista_desacelerando', 'alcista_desacelerando'))

    if direccion == 'short':
        sl_precio, sl_etiqueta = None, 'ATR_fallback'
        for nivel, etiqueta, dist_pct in resistencias:
            dist_abs = nivel - precio
            if dist_abs < sl_min_dist: continue
            if dist_abs > sl_max_dist: break
            sl_precio = nivel + buffer; sl_etiqueta = etiqueta; break
        if sl_precio is None:
            sl_precio = precio + atr * min(cfg['sl_mult'], SL_ATR_MAX_MULT); sl_etiqueta = 'ATR_fallback'
        sl_dist = sl_precio - precio
        tp_precio, tp_etiqueta = None, 'ATR_fallback'
        n_sop = 1 if not es_swing else min(2, len(soportes))
        for i, (nivel, etiqueta, dist_pct) in enumerate(soportes):
            if i < n_sop - 1: continue
            tp_candidato = nivel + buffer; tp_dist = precio - tp_candidato
            if tp_dist < sl_min_dist: continue
            if tp_dist > tp_max_dist: break
            if tp_dist / sl_dist < rr_min: continue
            tp_precio = tp_candidato; tp_etiqueta = etiqueta; break
        if tp_precio is None:
            tp_precio = precio - min(sl_dist * max(cfg['tp_ratio'], rr_min + 0.1), tp_max_dist)
            tp_etiqueta = 'ATR_fallback'
    else:
        sl_precio, sl_etiqueta = None, 'ATR_fallback'
        for nivel, etiqueta, dist_pct in soportes:
            dist_abs = precio - nivel
            if dist_abs < sl_min_dist: continue
            if dist_abs > sl_max_dist: break
            sl_precio = nivel - buffer; sl_etiqueta = etiqueta; break
        if sl_precio is None:
            sl_precio = precio - atr * min(cfg['sl_mult'], SL_ATR_MAX_MULT); sl_etiqueta = 'ATR_fallback'
        sl_dist = precio - sl_precio
        tp_precio, tp_etiqueta = None, 'ATR_fallback'
        n_res = 1 if not es_swing else min(2, len(resistencias))
        for i, (nivel, etiqueta, dist_pct) in enumerate(resistencias):
            if i < n_res - 1: continue
            tp_candidato = nivel - buffer; tp_dist = tp_candidato - precio
            if tp_dist < sl_min_dist: continue
            if tp_dist > tp_max_dist: break
            if tp_dist / sl_dist < rr_min: continue
            tp_precio = tp_candidato; tp_etiqueta = etiqueta; break
        if tp_precio is None:
            tp_precio = precio + min(sl_dist * max(cfg['tp_ratio'], rr_min + 0.1), tp_max_dist)
            tp_etiqueta = 'ATR_fallback'

    sl_dist_final = (sl_precio - precio) if direccion == 'short' else (precio - sl_precio)
    tp_dist_final = (precio - tp_precio) if direccion == 'short' else (tp_precio - precio)
    if sl_dist_final <= 0 or tp_dist_final <= 0:
        return {'viable': False, 'motivo': 'SL o TP en lado incorrecto del precio'}
    rr_real = tp_dist_final / sl_dist_final
    viable  = rr_real >= rr_min
    return {'viable': viable, 'sl': round(sl_precio, 8), 'tp': round(tp_precio, 8),
            'sl_dist_pct': round(sl_dist_final / precio * 100, 3),
            'tp_dist_pct': round(tp_dist_final / precio * 100, 3),
            'rr': round(rr_real, 2), 'sl_nivel': sl_etiqueta, 'tp_nivel': tp_etiqueta,
            'es_swing': es_swing, 'modo': 'swing' if es_swing else ('nivel' if 'fallback' not in sl_etiqueta else 'atr_fallback'),
            'motivo': '' if viable else f'R:R={rr_real:.2f} < mínimo {rr_min}'}


def calcular_max_horas(atr_pct: float, direccion: str) -> int:
    base = MAX_HORAS_BASE_LONG if direccion == 'long' else MAX_HORAS_BASE_SHORT
    if atr_pct <= 0: return base
    factor = MAX_HORAS_ATR_REF / atr_pct
    return int(np.clip(base * factor, MAX_HORAS_MIN, MAX_HORAS_MAX))


def calcular_margen_ajustado(cfg_margen: float, atr_pct: float) -> float:
    if atr_pct <= 0: return cfg_margen
    factor = float(np.clip((MAX_HORAS_ATR_REF / atr_pct) ** 0.6, 0.40, 1.20))
    return cfg_margen * factor


def calcular_liquidacion(precio: float, cfg: dict, direccion: str, margen: float) -> float:
    lev = cfg['leverage']; margen_ratio = 1.0 / lev
    return precio * (1 - margen_ratio * 0.90) if direccion == 'long' else precio * (1 + margen_ratio * 0.90)


def margen_efectivo(capital: float, cfg_margen: float, af: float,
                    atr_pct: float = 0.012, margen_f: float = 1.0) -> float:
    cap = min(capital, capital * CAP_MULT_FUNDING)
    return cap * calcular_margen_ajustado(cfg_margen, atr_pct) * af * margen_f


def margen_total_usado(estado: Dict) -> float:
    return sum(p['margen'] for p in estado['posiciones'])


def limite_margen_ok(estado: Dict, margen_nuevo: float) -> bool:
    return (margen_total_usado(estado) + margen_nuevo) <= estado['capital']


# ──────────────────────────────────────────────
#  GESTIÓN DE POSICIONES
# ──────────────────────────────────────────────
def gestionar_posiciones(estado: Dict, precio: float, atr: float,
                          hora: int, simbolo: str, ts: str,
                          high: float = None, low: float = None) -> List[Dict]:
    cerradas = []; nuevas = []
    for pos in estado['posiciones']:
        if pos['simbolo'] != simbolo:
            nuevas.append(pos); continue
        horas = hora - pos['hora_entrada']
        max_h = pos.get('max_horas', calcular_max_horas(atr / precio if precio > 0 else MAX_HORAS_ATR_REF, pos['dir']))

        if pos['dir'] == 'long':
            if precio > pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                if (precio - pos['precio_entrada']) >= atr * TRAILING_ATR_ACTIVACION:
                    pos['trailing'] = True
                    pos['sl'] = max(pos['sl'], pos['precio_ref'] - atr * TRAILING_ATR_DISTANCIA)
            cerrar = precio <= pos.get('liq', 0) and 'liq' in pos; motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio <= pos['sl']:     cerrar = True; motivo = 'trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio >= pos['tp'] or (high is not None and high >= pos['tp']): cerrar = True; motivo = 'take_profit'
                elif horas >= max_h:        cerrar = True; motivo = 'tiempo_maximo'
            if cerrar:
                ps = precio * (1 - SLIPPAGE_PCT); exp = pos['margen'] * pos['lev']
                ret = (ps - pos['precio_entrada']) / pos['precio_entrada']
                fund = exp * FUNDING_RATE_HORA * horas
                pnl = exp * ret - exp * COMISION_PCT * 2 - fund
                estado['capital'] += pnl; estado['pnl_total'] += pnl; estado['funding_total'] += fund
                estado['n_ops'] += 1
                if pnl > 0: estado['n_wins'] += 1
                else:        estado['n_loses'] += 1
                actualizar_adaptive(estado, pnl)
                op = {'ts': ts, 'simbolo': simbolo, 'dir': 'long', 'motivo': motivo,
                      'entrada': round(pos['precio_entrada'], 4), 'salida': round(ps, 4),
                      'pnl': round(pnl, 2), 'capital': round(estado['capital'], 2)}
                cerradas.append(op)
                emoji = '✅' if pnl > 0 else '❌'
                rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
                expo_now = margen_total_usado(estado)
                sl_info = f"SL={pos['sl_nivel']}" if 'sl_nivel' in pos else ''
                tp_info = f"TP={pos['tp_nivel']}" if 'tp_nivel' in pos else ''
                telegram(f"{emoji} <b>CIERRE LONG {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo} | {sl_info} | {tp_info}\n"
                         f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Funding: ${fund:.2f}\n"
                         f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_now>=0 else ''}{rent_now:.2f}%)\n"
                         f"🔒 Margen restante: ${expo_now:.2f} | Posiciones: {len(estado['posiciones'])}")
            else:
                nuevas.append(pos)
        else:  # short
            if precio < pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                if (pos['precio_entrada'] - precio) >= atr * TRAILING_ATR_ACTIVACION:
                    pos['trailing'] = True
                    pos['sl'] = min(pos['sl'], pos['precio_ref'] + atr * TRAILING_ATR_DISTANCIA)
            cerrar = precio >= pos.get('liq', float('inf')); motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio >= pos['sl']:     cerrar = True; motivo = 'trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio <= pos['tp'] or (low is not None and low <= pos['tp']): cerrar = True; motivo = 'take_profit'
                elif horas >= max_h:        cerrar = True; motivo = 'tiempo_maximo'
            if cerrar:
                ps = precio * (1 + SLIPPAGE_PCT); exp = pos['margen'] * pos['lev']
                ret = (pos['precio_entrada'] - ps) / pos['precio_entrada']
                fund = exp * FUNDING_RATE_HORA * horas
                pnl = exp * ret - exp * COMISION_PCT * 2 - fund
                estado['capital'] += pnl; estado['pnl_total'] += pnl; estado['funding_total'] += fund
                estado['n_ops'] += 1
                if pnl > 0: estado['n_wins'] += 1
                else:        estado['n_loses'] += 1
                actualizar_adaptive(estado, pnl)
                op = {'ts': ts, 'simbolo': simbolo, 'dir': 'short', 'motivo': motivo,
                      'entrada': round(pos['precio_entrada'], 4), 'salida': round(ps, 4),
                      'pnl': round(pnl, 2), 'capital': round(estado['capital'], 2)}
                cerradas.append(op)
                emoji = '✅' if pnl > 0 else '❌'
                rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
                expo_now = margen_total_usado(estado)
                sl_info = f"SL={pos['sl_nivel']}" if 'sl_nivel' in pos else ''
                tp_info = f"TP={pos['tp_nivel']}" if 'tp_nivel' in pos else ''
                telegram(f"{emoji} <b>CIERRE SHORT {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo} | {sl_info} | {tp_info}\n"
                         f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Funding: ${fund:.2f}\n"
                         f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_now>=0 else ''}{rent_now:.2f}%)\n"
                         f"🔒 Margen restante: ${expo_now:.2f} | Posiciones: {len(estado['posiciones'])}")
            else:
                nuevas.append(pos)

    estado['posiciones'] = nuevas
    return cerradas


def abrir_posicion(estado: Dict, simbolo: str, precio: float, atr: float,
                    cfg_nivel: dict, dir_: str, ts: str, hora: int,
                    razones: List[str], df_completo: pd.DataFrame = None,
                    ctx4h: Dict = None, ultima_row: pd.Series = None,
                    score_adj: float = 0.0, señal_ia: Dict = None):
    """
    Abre una posición aplicando el filtro IA y ajuste de SL/TP por fuerza IA 4h.

    Diferencias clave respecto a v14:
      1. Llama a filtrar_entrada_con_ia() antes de operar.
      2. Llama a ajustar_cfg_por_ia() para modificar tp_ratio/sl_mult según IA 4h.
      3. La nota de ajuste IA aparece en el mensaje de Telegram.
    """
    nombre = simbolo.replace('/', '_')
    af     = estado.get('adaptive_factor', 1.0)

    # ── Guard anti-duplicados ──
    ya_existe = any(p['simbolo'] == nombre and p['dir'] == dir_ for p in estado['posiciones'])
    if ya_existe:
        print(f"  🚫 BLOQUEADO: ya hay un {dir_.upper()} abierto en {nombre}")
        return

    # ── Asignar nivel por score ──
    if score_adj >= 9.0:    nivel_str, cfg = 'nivel4', TRADE_CONFIG['nivel4']
    elif score_adj >= 6.0:  nivel_str, cfg = 'nivel3', TRADE_CONFIG['nivel3']
    elif score_adj >= 4.0:  nivel_str, cfg = 'nivel2', TRADE_CONFIG['nivel2']
    else:                   nivel_str, cfg = 'nivel1', TRADE_CONFIG['nivel1']

    print(f"  📊 Nivel por score ({score_adj:.1f}): {nivel_str.upper()} "
          f"(x{cfg['leverage']} lev, {cfg['margen']*100:.0f}% margen)")

    # ── Ajuste IA: modificar tp_ratio/sl_mult según fuerza 4h ──
    cfg_ia, nota_ia = ajustar_cfg_por_ia(cfg, señal_ia or {}, dir_)
    print(f"  🤖 Ajuste IA SL/TP: {nota_ia}")

    # ── ATR relativo ──
    atr_pct = atr / precio if precio > 0 else MAX_HORAS_ATR_REF

    # ── SL/TP desde niveles reales (usando cfg ajustado por IA) ──
    ctx_sltp = None
    if df_completo is not None:
        u = df_completo.iloc[-1]
        niveles  = _niveles_mercado(df_completo, u, precio, atr)
        ctx_sltp = calcular_sl_tp_niveles(precio, atr, cfg_ia, dir_, niveles, ctx4h or {})

    if ctx_sltp is None or not ctx_sltp['viable']:
        motivo = ctx_sltp['motivo'] if ctx_sltp else 'sin contexto de mercado'
        print(f"  ❌ {nombre} {dir_.upper()} rechazado: {motivo}")
        return

    sl  = ctx_sltp['sl']
    tp  = ctx_sltp['tp']
    liq = calcular_liquidacion(precio, cfg, dir_, 1.0)
    pe  = precio * (1 + SLIPPAGE_PCT) if dir_ == 'long' else precio * (1 - SLIPPAGE_PCT)

    # ── Margen ──
    margen_des = margen_efectivo(estado['capital'], cfg['margen'], af, atr_pct,
                                 margen_f=0.80 if ctx_sltp['es_swing'] else 1.0)
    margen_max = estado['capital'] * MAX_MARGEN_TOTAL_PCT
    margen_uso = sum(p['margen'] for p in estado['posiciones'] if p['simbolo'] == nombre)
    margen_r   = min(margen_des, margen_max - margen_uso)
    if margen_r < estado['capital'] * 0.02:
        print(f"  ⚠️  Margen insuficiente para {nombre} {dir_}, skip"); return
    if not limite_margen_ok(estado, margen_r):
        print(f"  ⚠️  Margen total agotado, skip"); return

    max_horas = calcular_max_horas(atr_pct, dir_)

    estado['posiciones'].append({
        'simbolo': nombre, 'dir': dir_, 'calidad': nivel_str,
        'precio_entrada': pe, 'precio_ref': pe,
        'sl': sl, 'tp': tp, 'liq': liq,
        'margen': margen_r, 'lev': cfg['leverage'],
        'hora_entrada': hora, 'max_horas': max_horas,
        'trailing': False, 'ts_entrada': ts,
        'adaptive_factor': af, 'modo_sltp': ctx_sltp['modo'].upper(),
        'sl_nivel': ctx_sltp['sl_nivel'], 'tp_nivel': ctx_sltp['tp_nivel'],
        'atr_entrada': round(atr, 6),
        'ia_ajuste': nota_ia,  # guardamos el ajuste IA para trazabilidad
    })

    razones_str  = '\n'.join(f"  · {r}" for r in razones[:4])
    sl_pct       = ctx_sltp['sl_dist_pct']
    tp_pct       = ctx_sltp['tp_dist_pct']
    rr           = ctx_sltp['rr']
    modo         = ctx_sltp['modo'].upper()
    swing_tag    = ' ⚡SWING' if ctx_sltp['es_swing'] else ''
    margen_usado = margen_total_usado(estado)
    margen_pct   = margen_usado / estado['capital'] * 100 if estado['capital'] > 0 else 0
    rent_act     = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100

    # Resumen IA para Telegram
    ia_tag = ''
    if señal_ia and señal_ia.get('disponible'):
        ia_dir = señal_ia['direccion']
        ia_conf = señal_ia.get('confianza', 0)
        ia_tag = (f"\n🤖 IA: {ia_dir} conf={ia_conf:.2f} | {nota_ia}")

    telegram(f"🚀 <b>APERTURA {dir_.upper()} {simbolo}</b> [{VERSION}]{swing_tag}\n"
             f"Modo: {modo} | Nivel: {nivel_str.upper()} (score={score_adj:.1f})\n"
             f"Razones:\n{razones_str}\n"
             f"Precio entrada: ${pe:,.4f}\n"
             f"SL: ${sl:,.4f} (-{sl_pct:.2f}%) ← {ctx_sltp['sl_nivel']}\n"
             f"TP: ${tp:,.4f} (+{tp_pct:.2f}%) ← {ctx_sltp['tp_nivel']}\n"
             f"R:R = 1:{rr:.1f} | Max horas: {max_horas}h\n"
             f"Margen: ${margen_r:.2f} | x{cfg['leverage']} | AF: x{af:.1f} | ATR: {atr_pct*100:.2f}%"
             f"{ia_tag}\n"
             f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_act>=0 else ''}{rent_act:.2f}%)\n"
             f"🔒 Margen usado: ${margen_usado:.2f} ({margen_pct:.0f}% del capital)\n"
             f"Posiciones abiertas: {len(estado['posiciones'])}")


# ──────────────────────────────────────────────
#  REPORTE DIARIO
# ──────────────────────────────────────────────
def enviar_reporte_diario(estado: Dict):
    capital = estado['capital']
    rent    = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    n       = estado['n_ops']; wr = estado['n_wins'] / max(n, 1) * 100
    ops_hoy = estado.get('ops_hoy', []); af = estado.get('adaptive_factor', 1.0)
    h       = estado.get('historial_pnl', []); wr20 = sum(h) / len(h) * 100 if h else 0
    expo_act = margen_total_usado(estado); expo_pct = expo_act / capital * 100 if capital > 0 else 0
    msg = (f"📊 <b>REPORTE DIARIO {VERSION.upper()}</b>\n{'─'*30}\n"
           f"💰 Capital: ${capital:,.2f} ({'+' if rent>=0 else ''}{rent:.2f}%)\n"
           f"📈 Total ops: {n} | WR: {wr:.1f}%\n📊 WR últimas 20: {wr20:.1f}%\n"
           f"💵 P&L total: ${estado['pnl_total']:+.2f}\n📉 Funding total: -${abs(estado['funding_total']):.2f}\n"
           f"🔧 Adaptive: x{af:.1f}\n🔓 Posiciones abiertas: {len(estado['posiciones'])}\n"
           f"🔒 Margen en uso: ${expo_act:.2f} ({expo_pct:.0f}% del capital)\n{'─'*30}\n")
    if ops_hoy:
        wins = sum(1 for o in ops_hoy if o.get('pnl', 0) > 0)
        loses = len(ops_hoy) - wins
        pnl_d = sum(o.get('pnl', 0) for o in ops_hoy)
        msg += f"<b>Hoy ({len(ops_hoy)} ops):</b>\n✅ {wins} | ❌ {loses} | P&L: ${pnl_d:+.2f}\n"
        for op in ops_hoy[-5:]:
            e = '✅' if op.get('pnl', 0) > 0 else '❌'
            msg += f"  {e} {op.get('dir','').upper()} {op.get('simbolo','')} ${op.get('pnl',0):+.2f}\n"
    else:
        msg += "Sin operaciones hoy.\n"
    if not estado['bot_activo']:   msg += "\n⚠️ BOT DETENIDO"
    elif estado['en_pausa']:       msg += "\n⏸️ BOT EN PAUSA"
    telegram(msg)
    estado['ops_hoy'] = []


# ──────────────────────────────────────────────
#  LOOP PRINCIPAL
# ──────────────────────────────────────────────
def ejecutar():
    print(f"\n{'='*55}")
    print(f"  Paper Trading {VERSION} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Estrategia: PREDICTIVA + IA NUEVA + SL/TP INTELIGENTE")
    print(f"{'='*55}")

    estado    = cargar_estado()
    hora_unix = int(time.time() / 3600)

    if estado.get('primera_ejecucion', True):
        telegram(f"🤖 <b>BOT {VERSION} INICIADO</b>\n"
                 f"Capital: ${CAPITAL_INICIAL:,.2f}\n"
                 f"IA NUEVA ACTIVADA:\n"
                 f"  · Modelos: exito_long/short 1h + 4h\n"
                 f"  · Filtro: veto si IA contraria conf ≥ {UMBRAL_IA_VETO}\n"
                 f"  · FLAT: entra si score técnico ≥ {SCORE_MINIMO_SIN_IA}\n"
                 f"  · SL/TP ajustado según fuerza 4h del modelo\n"
                 f"✅ Conexión OK — bot activo")
        estado['primera_ejecucion'] = False

    ahora_utc = datetime.now(timezone.utc)
    if ahora_utc.hour == 0 and ahora_utc.minute < 65:
        enviar_reporte_diario(estado)

    if not estado['bot_activo']:
        print("  🛑 Bot detenido"); guardar_estado(estado); return

    if estado['en_pausa']:
        fin = estado.get('fin_pausa_hora', 0)
        if hora_unix < fin:
            print(f"  ⏸️  Pausa. Quedan {fin - hora_unix}h")
            guardar_estado(estado); return
        estado['en_pausa'] = False

    SEP = "─" * 60
    resumen_ciclo = []
    SCORE_DIFF_MINIMA = 2.0
    SCORE_MINIMO_GIRO = 6.0

    for simbolo in CRIPTOS:
        nombre = simbolo.replace('/', '_')
        print(f"\n{SEP}")
        print(f"  📊  {simbolo}  |  {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        print(SEP)

        # ── Datos e indicadores ──
        df = obtener_velas(simbolo, intervalo=60, limite=VELAS_N)
        if df is None:
            print("  ❌  Sin datos, skip")
            resumen_ciclo.append(f"{simbolo:<12} ❌  sin datos"); continue
        try:
            df = calcular_indicadores(df)
        except Exception as e:
            print(f"  ❌  Indicadores: {e}")
            resumen_ciclo.append(f"{simbolo:<12} ❌  error indicadores"); continue

        df.dropna(subset=['ema_200', 'rsi_14', 'atr_14'], inplace=True)
        if len(df) < 50:
            resumen_ciclo.append(f"{simbolo:<12} ❌  datos insuficientes"); continue

        ultima = df.iloc[-2]
        precio = float(ultima['close']); high = float(ultima['high']); low = float(ultima['low'])
        atr    = float(ultima.get('atr_14', precio * 0.015))
        if pd.isna(atr) or atr <= 0: atr = precio * 0.015
        atr_pct = atr / precio * 100; ts = str(ultima['timestamp'])
        vela_actual    = df.iloc[-1]
        precio_entrada = float(vela_actual['close'])

        rsi   = float(ultima.get('rsi_14', 50))
        macd  = float(ultima.get('macd_hist', 0))
        stk   = float(ultima.get('stoch_k', 50))
        adx   = float(ultima.get('adx', 0))
        bb_p  = float(ultima.get('bb_posicion', 0.5))
        vol_r = float(ultima.get('volumen_ratio', 1.0))
        reg   = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', np.nan)) else 0

        tendencia_1h = "ALCISTA" if reg >= 1 else ("BAJISTA" if reg <= -1 else "LATERAL")
        macd_estado  = "▲ positivo" if macd > 0 else "▼ negativo"
        print(f"  Precio: ${precio:,.4f}  |  ATR: {atr_pct:.2f}%  |  1H: {tendencia_1h}")
        print(f"  RSI={rsi:.0f}  Stoch={stk:.0f}  MACD_hist={macd:.4f}({macd_estado})"
              f"  ADX={adx:.0f}  BB={bb_p:.2f}  Vol×{vol_r:.1f}")

        # ── Contexto 4H ──
        ctx4h = contexto_4h(simbolo)
        if ctx4h.get('disponible'):
            t4 = ctx4h['tendencia'].upper(); m4 = ctx4h['momentum']; r4 = ctx4h.get('rsi', 50)
            flecha4 = "⬆" if t4 == "ALCISTA" else ("⬇" if t4 == "BAJISTA" else "↔")
            print(f"  4H: {flecha4} {t4}  momentum={m4}  RSI={r4:.0f}")
        else:
            print("  4H: no disponible")

        # ── Contexto 1D ──
        ctx1d = contexto_diario(simbolo)
        if ctx1d.get('disponible'):
            t1d = ctx1d['tendencia'].upper(); m1d = ctx1d['momentum']; r1d = ctx1d.get('rsi', 50)
            flecha1d = "⬆" if t1d == "ALCISTA" else ("⬇" if t1d == "BAJISTA" else "↔")
            print(f"  1D: {flecha1d} {t1d}  momentum={m1d}  RSI={r1d:.0f}")
        else:
            print("  1D: no disponible")

        # ── IA NUEVA: obtener señal ──
        señal_ia = obtener_señal_ia(simbolo)
        if señal_ia['disponible']:
            ia_dir   = señal_ia['direccion']
            ia_conf  = señal_ia.get('confianza', 0)
            ia_fuerza = señal_ia.get('fuerza', 0.5)
            ia_color = "🟢" if ia_dir == "LONG" else ("🔴" if ia_dir == "SHORT" else "⚪")
            print(f"  🤖 IA nueva: {ia_color} {ia_dir}  conf_1h={ia_conf:.2f}  fuerza_4h={ia_fuerza:.2f}")
            print(f"     long_1h={señal_ia.get('prob_long_1h')}  short_1h={señal_ia.get('prob_short_1h')}"
                  f"  long_4h={señal_ia.get('prob_long_4h')}  short_4h={señal_ia.get('prob_short_4h')}")
        else:
            print(f"  🤖 IA nueva: no disponible — {señal_ia['razon']}")

        # ── Divergencias ──
        df_pred = df.iloc[:-1]
        divs = detectar_divergencias(df_pred, ventana=40)
        if divs['resumen']:
            for d in divs['resumen']:
                icon = "🔴" if "bajista" in d.lower() else "🟢"
                print(f"  {icon} Div: {d}")
        else:
            print("  ⚪ Sin divergencias detectadas")

        # ── Patrones chartistas ──
        df4h_raw = ctx4h.get('df') if ctx4h.get('disponible') else None
        patrones = detectar_patrones(df, df4h_raw)
        if patrones['patrones_short']:
            print(f"  📐 Patrones BAJISTAS: {', '.join(patrones['patrones_short'])} (score={patrones['score_short']:.1f})")
        if patrones['patrones_long']:
            print(f"  📐 Patrones ALCISTAS: {', '.join(patrones['patrones_long'])} (score={patrones['score_long']:.1f})")
        if not patrones['patrones_short'] and not patrones['patrones_long']:
            print("  📐 Sin patrones chartistas activos")
        if patrones['niveles_clave']:
            niveles_str = '  |  '.join(f"{k}=${v:,.2f}" for k, v in patrones['niveles_clave'][:4])
            print(f"  🎯 Niveles: {niveles_str}")

        # ── Gestionar posiciones abiertas ──
        cerradas = gestionar_posiciones(estado, precio, atr, hora_unix, nombre, ts, high=high, low=low)
        for op in cerradas:
            registrar_op(op); estado.setdefault('ops_hoy', []).append(op)

        if int(ultima.get('volumen_muy_bajo', 0)) == 1:
            print("  ⚠️  Volumen muy bajo — skip análisis de entrada")
            resumen_ciclo.append(f"{simbolo:<12} ⚠️  volumen bajo"); continue

        # ── Contadores de posiciones abiertas ──
        n_longs  = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'long')
        n_shorts = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'short')

        def pnl_neto_posicion(pos, precio_actual):
            exp  = pos['margen'] * pos['lev']
            horas = max((hora_unix - pos.get('ts_apertura', hora_unix)) / 3600, 0)
            fund = exp * FUNDING_RATE_HORA * horas
            if pos['dir'] == 'long':
                ps = precio_actual * (1 - SLIPPAGE_PCT); ret = (ps - pos['precio_entrada']) / pos['precio_entrada']
            else:
                ps = precio_actual * (1 + SLIPPAGE_PCT); ret = (pos['precio_entrada'] - ps) / pos['precio_entrada']
            return exp * ret - exp * COMISION_PCT * 2 - fund

        def cerrar_posicion_por_giro(pos, precio_actual):
            exp  = pos['margen'] * pos['lev']
            horas = max((hora_unix - pos.get('ts_apertura', hora_unix)) / 3600, 0)
            fund = exp * FUNDING_RATE_HORA * horas
            if pos['dir'] == 'long':
                ps = precio_actual * (1 - SLIPPAGE_PCT); ret = (ps - pos['precio_entrada']) / pos['precio_entrada']
            else:
                ps = precio_actual * (1 + SLIPPAGE_PCT); ret = (pos['precio_entrada'] - ps) / pos['precio_entrada']
            pnl = exp * ret - exp * COMISION_PCT * 2 - fund
            estado['capital'] += pnl; estado['pnl_total'] += pnl; estado['funding_total'] += fund
            estado['posiciones'] = [p for p in estado['posiciones'] if p is not pos]
            rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
            telegram(f"🔄 <b>GIRO {pos['dir'].upper()} → {'SHORT' if pos['dir']=='long' else 'LONG'} {nombre}</b> [{VERSION}]\n"
                     f"Cierre por señal contraria nivel3+\n"
                     f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                     f"P&L: ${pnl:+.2f} | Funding: ${fund:.2f}\n"
                     f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_now>=0 else ''}{rent_now:.2f}%)")
            print(f"  🔄 GIRO: cerrado {pos['dir'].upper()} con P&L=${pnl:+.2f}")
            return pnl

        # ── Pre-cálculo scores ──
        score_l_adj = score_s_adj = None
        ctx_señal_l = ctx_señal_s = None
        razones_l = razones_s = []
        bloqueado_l = bloqueado_s = False

        if n_longs == 0:
            score_l, razones_l = score_predictivo_long(df_pred, ultima, ctx4h, patrones, divs)
            ctx_señal_l = evaluar_contexto_señal(df_pred, ultima, ctx4h, 'long', divs, patrones)
            if ctx_señal_l['bloquear']:
                bloqueado_l = True
            else:
                score_l_adj = score_l * ctx_señal_l['multiplicador_score']

        if n_shorts == 0:
            score_s, razones_s = score_predictivo_short(df_pred, ultima, ctx4h, patrones, divs)
            ctx_señal_s = evaluar_contexto_señal(df_pred, ultima, ctx4h, 'short', divs, patrones)
            if ctx_señal_s['bloquear']:
                bloqueado_s = True
            else:
                score_s_adj = score_s * ctx_señal_s['multiplicador_score']

        # ── Filtro de contradicción ──
        min_s = score_minimo(1.0, 'long')
        l_valido = (score_l_adj is not None) and (score_l_adj >= min_s)
        s_valido = (score_s_adj is not None) and (score_s_adj >= min_s)
        mercado_ambiguo = False
        if l_valido and s_valido:
            diff = abs(score_l_adj - score_s_adj)
            if diff < SCORE_DIFF_MINIMA:
                mercado_ambiguo = True
                print(f"  ⚖️  MERCADO AMBIGUO — LONG={score_l_adj:.1f} vs SHORT={score_s_adj:.1f} "
                      f"(diff={diff:.1f} < {SCORE_DIFF_MINIMA}) — no se abre ninguna posición")
            else:
                if score_l_adj >= score_s_adj:
                    s_valido = False
                    print(f"  🏆 LONG domina ({score_l_adj:.1f} vs {score_s_adj:.1f}) — SHORT descartado")
                else:
                    l_valido = False
                    print(f"  🏆 SHORT domina ({score_s_adj:.1f} vs {score_l_adj:.1f}) — LONG descartado")

        # ══════════════════════════════════════════
        #  BLOQUE LONG
        # ══════════════════════════════════════════
        print(f"\n  {'─'*25} ANÁLISIS LONG {'─'*20}")
        resumen_l = "—"

        if n_longs > 0:
            pos_l = next(p for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'long')
            pnl_pct = (precio - pos_l['precio_entrada']) / pos_l['precio_entrada'] * 100
            print(f"  📌 Ya hay LONG abierto en ${pos_l['precio_entrada']:,.4f}  P&L={pnl_pct:+.2f}%"
                  f"  SL=${pos_l['sl']:,.4f}  TP=${pos_l['tp']:,.4f}")
            resumen_l = f"LONG abierto {pnl_pct:+.2f}%"
        elif bloqueado_l:
            print(f"  🚫 LONG BLOQUEADO (ctx) — {ctx_señal_l['razon_bloqueo']}")
            resumen_l = "BLOQUEADO (nivel fuerte)"
        elif score_l_adj is None:
            resumen_l = "—"
        else:
            # ── Giro desde SHORT ──
            pos_short_abierto = next(
                (p for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'short'), None)
            if pos_short_abierto is not None:
                print(f"  Score LONG: {score_l:.1f} × {ctx_señal_l['multiplicador_score']:.2f} = {score_l_adj:.1f} "
                      f"(mínimo giro={SCORE_MINIMO_GIRO})")
                if score_l_adj >= SCORE_MINIMO_GIRO:
                    # Aplicar filtro IA antes del giro
                    ia_ok, ia_motivo = filtrar_entrada_con_ia(señal_ia, 'long', score_l_adj)
                    print(f"  🤖 Filtro IA: {'✅' if ia_ok else '❌'} {ia_motivo}")
                    if not ia_ok:
                        print(f"  ❌ Giro LONG bloqueado por IA")
                        resumen_l = f"no gira — IA bloqueó ({ia_motivo[:50]})"
                    else:
                        pnl_actual = pnl_neto_posicion(pos_short_abierto, precio)
                        if pnl_actual > 0:
                            cerrar_posicion_por_giro(pos_short_abierto, precio)
                            n_shorts = 0
                            abrir_posicion(estado, simbolo, precio_entrada, atr, {}, 'long',
                                           ts, hora_unix, razones_l + ctx_señal_l['contexto_log'],
                                           df_pred, ctx4h, ultima, score_l_adj, señal_ia)
                            resumen_l = "✅ LONG abierto (giro desde SHORT)"
                        else:
                            print(f"  ❌ Giro denegado — SHORT en pérdida neta (${pnl_actual:+.2f})")
                            resumen_l = f"no gira — SHORT en pérdida (${pnl_actual:+.2f})"
                else:
                    print(f"  ❌ Score insuficiente para giro ({score_l_adj:.1f} < {SCORE_MINIMO_GIRO})")
                    resumen_l = f"no gira — score {score_l_adj:.1f}"
            else:
                # ── Flujo normal LONG ──
                razones_favor = [r for r in razones_l if not r.startswith("⚠")]
                razones_contra = list(dict.fromkeys(
                    [r for r in razones_l if r.startswith("⚠")] + ctx_señal_l['contexto_log']))
                print(f"  Score LONG: {score_l:.1f} × {ctx_señal_l['multiplicador_score']:.2f} = {score_l_adj:.1f} (mínimo={min_s})")
                if razones_favor:
                    print("  Señales a favor del LONG:")
                    for r in razones_favor: print(f"    ✅ {r}")
                if razones_contra:
                    print("  En contra / penalizaciones:")
                    for r in razones_contra: print(f"    ⚠️  {r}")

                if mercado_ambiguo:
                    resumen_l = f"no abre — mercado ambiguo (L={score_l_adj:.1f} vs S={score_s_adj:.1f})"
                    print(f"  ⚖️  No abre — mercado ambiguo")
                elif not l_valido:
                    resumen_l = "no abre — SHORT domina"
                    print(f"  ❌ No abre — SHORT tiene mayor score")
                elif score_l_adj < min_s:
                    resumen_l = f"no abre — score {score_l_adj:.1f} < {min_s}"
                    print(f"  ❌ Score insuficiente ({score_l_adj:.1f} < {min_s})")
                else:
                    # ── FILTRO IA ──────────────────────────────────────────
                    ia_ok, ia_motivo = filtrar_entrada_con_ia(señal_ia, 'long', score_l_adj)
                    print(f"  🤖 Filtro IA: {'✅' if ia_ok else '❌'} {ia_motivo}")
                    if not ia_ok:
                        resumen_l = f"no abre — IA bloqueó ({ia_motivo[:60]})"
                        print(f"  ❌ LONG bloqueado por IA nueva")
                    else:
                        print(f"  🟢 SEÑAL LONG — ABRIENDO POSICIÓN")
                        abrir_posicion(estado, simbolo, precio_entrada, atr, {}, 'long',
                                       ts, hora_unix, razones_l + ctx_señal_l['contexto_log'],
                                       df_pred, ctx4h, ultima, score_l_adj, señal_ia)
                        resumen_l = "✅ LONG abierto"

        # ══════════════════════════════════════════
        #  BLOQUE SHORT
        # ══════════════════════════════════════════
        print(f"\n  {'─'*25} ANÁLISIS SHORT {'─'*19}")
        resumen_s = "—"

        if n_shorts > 0:
            pos_s = next(p for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'short')
            pnl_pct = (pos_s['precio_entrada'] - precio) / pos_s['precio_entrada'] * 100
            print(f"  📌 Ya hay SHORT abierto en ${pos_s['precio_entrada']:,.4f}  P&L={pnl_pct:+.2f}%"
                  f"  SL=${pos_s['sl']:,.4f}  TP=${pos_s['tp']:,.4f}")
            resumen_s = f"SHORT abierto {pnl_pct:+.2f}%"
        elif bloqueado_s:
            print(f"  🚫 SHORT BLOQUEADO (ctx) — {ctx_señal_s['razon_bloqueo']}")
            resumen_s = "BLOQUEADO (nivel fuerte)"
        elif score_s_adj is None:
            resumen_s = "—"
        else:
            # ── Giro desde LONG ──
            pos_long_abierto = next(
                (p for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'long'), None)
            if pos_long_abierto is not None:
                print(f"  Score SHORT: {score_s:.1f} × {ctx_señal_s['multiplicador_score']:.2f} = {score_s_adj:.1f} "
                      f"(mínimo giro={SCORE_MINIMO_GIRO})")
                if score_s_adj >= SCORE_MINIMO_GIRO:
                    ia_ok, ia_motivo = filtrar_entrada_con_ia(señal_ia, 'short', score_s_adj)
                    print(f"  🤖 Filtro IA: {'✅' if ia_ok else '❌'} {ia_motivo}")
                    if not ia_ok:
                        print(f"  ❌ Giro SHORT bloqueado por IA")
                        resumen_s = f"no gira — IA bloqueó ({ia_motivo[:50]})"
                    else:
                        pnl_actual = pnl_neto_posicion(pos_long_abierto, precio)
                        if pnl_actual > 0:
                            cerrar_posicion_por_giro(pos_long_abierto, precio)
                            n_longs = 0
                            abrir_posicion(estado, simbolo, precio_entrada, atr, {}, 'short',
                                           ts, hora_unix, razones_s + ctx_señal_s['contexto_log'],
                                           df_pred, ctx4h, ultima, score_s_adj, señal_ia)
                            resumen_s = "✅ SHORT abierto (giro desde LONG)"
                        else:
                            print(f"  ❌ Giro denegado — LONG en pérdida neta (${pnl_actual:+.2f})")
                            resumen_s = f"no gira — LONG en pérdida (${pnl_actual:+.2f})"
                else:
                    print(f"  ❌ Score insuficiente para giro ({score_s_adj:.1f} < {SCORE_MINIMO_GIRO})")
                    resumen_s = f"no gira — score {score_s_adj:.1f}"
            else:
                # ── Flujo normal SHORT ──
                razones_favor_s = [r for r in razones_s if not r.startswith("⚠")]
                razones_contra_s = list(dict.fromkeys(
                    [r for r in razones_s if r.startswith("⚠")] + ctx_señal_s['contexto_log']))
                print(f"  Score SHORT: {score_s:.1f} × {ctx_señal_s['multiplicador_score']:.2f} = {score_s_adj:.1f} (mínimo={min_s})")
                if razones_favor_s:
                    print("  Señales a favor del SHORT:")
                    for r in razones_favor_s: print(f"    ✅ {r}")
                if razones_contra_s:
                    print("  En contra / penalizaciones:")
                    for r in razones_contra_s: print(f"    ⚠️  {r}")

                if mercado_ambiguo:
                    resumen_s = f"no abre — mercado ambiguo (L={score_l_adj:.1f} vs S={score_s_adj:.1f})"
                    print(f"  ⚖️  No abre — mercado ambiguo")
                elif not s_valido:
                    resumen_s = "no abre — LONG domina"
                    print(f"  ❌ No abre — LONG tiene mayor score")
                elif score_s_adj < min_s:
                    resumen_s = f"no abre — score {score_s_adj:.1f} < {min_s}"
                    print(f"  ❌ Score insuficiente ({score_s_adj:.1f} < {min_s})")
                else:
                    # ── FILTRO IA ──────────────────────────────────────────
                    ia_ok, ia_motivo = filtrar_entrada_con_ia(señal_ia, 'short', score_s_adj)
                    print(f"  🤖 Filtro IA: {'✅' if ia_ok else '❌'} {ia_motivo}")
                    if not ia_ok:
                        resumen_s = f"no abre — IA bloqueó ({ia_motivo[:60]})"
                        print(f"  ❌ SHORT bloqueado por IA nueva")
                    else:
                        print(f"  🔴 SEÑAL SHORT — ABRIENDO POSICIÓN")
                        abrir_posicion(estado, simbolo, precio_entrada, atr, {}, 'short',
                                       ts, hora_unix, razones_s + ctx_señal_s['contexto_log'],
                                       df_pred, ctx4h, ultima, score_s_adj, señal_ia)
                        resumen_s = "✅ SHORT abierto"

        resumen_ciclo.append(
            f"{simbolo:<12} ${precio:>10,.2f}  RSI={rsi:>3.0f}  "
            f"LONG: {resumen_l:<35}  SHORT: {resumen_s}"
        )
        time.sleep(0.5)

    # ── RESUMEN FINAL ──
    guardar_estado(estado)
    m = guardar_metricas(estado)
    s = '+' if m['rentabilidad_pct'] >= 0 else ''
    expo_fin     = margen_total_usado(estado)
    expo_fin_pct = expo_fin / estado['capital'] * 100 if estado['capital'] > 0 else 0

    print(f"\n{'═'*60}")
    print(f"  RESUMEN CICLO — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*60}")
    for linea in resumen_ciclo: print(f"  {linea}")
    print(f"{'─'*60}")
    print(f"  💰 Capital: ${m['capital_actual']:,.2f} ({s}{m['rentabilidad_pct']:.2f}%)")
    print(f"  🔒 Margen en uso: ${expo_fin:.2f} ({expo_fin_pct:.0f}% del capital)")
    print(f"  📈 Ops: {m['n_operaciones']} | WR: {m['win_rate_pct']:.1f}% | Adaptive: ×{m['adaptive_factor']:.2f}")
    print(f"  📂 Posiciones abiertas: {len(estado['posiciones'])}")
    if estado['posiciones']:
        for p in estado['posiciones']:
            pnl_est = (precio - p['precio_entrada']) / p['precio_entrada'] * 100
            if p['dir'] == 'short': pnl_est = -pnl_est
            print(f"     {p['dir'].upper():5} {p['simbolo']:<10} entrada=${p['precio_entrada']:,.4f}"
                  f"  SL=${p['sl']:,.4f}  TP=${p['tp']:,.4f}  P&L≈{pnl_est:+.2f}%")
    print(f"{'═'*60}")


if __name__ == '__main__':
    token   = os.environ.get('TELEGRAM_TOKEN_V15', '') or os.environ.get('TELEGRAM_TOKEN_V14', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_V15', '') or os.environ.get('TELEGRAM_CHAT_ID_V14', '')
    if not token:   print("  ⚠️  TELEGRAM_TOKEN_V15 no configurado (también acepta _V14)")
    elif not chat_id: print("  ⚠️  TELEGRAM_CHAT_ID_V15 no configurado")
    else:
        print(f"  ✅ Telegram configurado — chat_id: {chat_id[:6]}...{chat_id[-3:] if len(chat_id)>6 else ''}")
    ejecutar()
