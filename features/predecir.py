"""
predecir.py  (v1)
=================
Carga los modelos entrenados y genera una señal de trading para cada
moneda, basada en los últimos datos disponibles.

ARQUITECTURA DE DECISIÓN:
  - Olvida label_oportunidad (AUC ~0.50, sin señal real)
  - Usa exito_long y exito_short como señales principales
  - El modelo 4h define la tendencia (amplificador de capital)
  - El modelo 1h da el timing de entrada

SALIDA por moneda:
  {
    'simbolo':        'BTCUSDT',
    'direccion':      'LONG' | 'SHORT' | 'FLAT',
    'confianza':      0.0 - 1.0,   # prob del modelo 1h
    'fuerza':         0.0 - 1.0,   # prob del modelo 4h (amplifica capital)
    'apalancamiento': float,        # calculado con fuerza
    'capital_pct':    float,        # % del capital a usar
    'razon':          str,          # explicación legible
  }

USO:
  python predecir.py                      # señal actual para todas las monedas
  python predecir.py --simbolo BTCUSDT    # solo BTC
  python predecir.py --json               # output en JSON (para el bot)
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

# ── Configuración ──────────────────────────────────────────────────────
MODELS_DIR    = 'models'
RAW_DIR       = 'models/data/raw'
PROCESSED_DIR = 'models/data/processed'

SIMBOLOS  = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
TIMEFRAMES = ['1h', '4h']

# Umbrales mínimos de confianza para operar
# Por debajo de estos valores → FLAT (no operar)
UMBRAL_MINIMO_1H = 0.55   # confianza mínima del modelo 1h para entrar
UMBRAL_MINIMO_4H = 0.50   # el 4h puede ser más bajo, su rol es amplificar no filtrar

# Parámetros de gestión de capital
CAPITAL_BASE     = 0.10   # % base del capital por operación (10%)
CAPITAL_MAX      = 0.25   # % máximo permitido (25%)
APALANCAMIENTO_BASE = 3.0
APALANCAMIENTO_MAX  = 10.0

# Features que usan los modelos (debe coincidir con entrenar.py)
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
FEATURES = FEATURES_BASE + FEATURES_TEMPORALES


# ──────────────────────────────────────────────────────────────────────
#  CARGA DE MODELOS
# ──────────────────────────────────────────────────────────────────────
def cargar_modelo(simbolo: str, nombre: str):
    """
    Carga modelo + scaler + umbral para un símbolo dado.
    nombre puede ser: 'exito_long_1h', 'exito_short_1h', 'exito_long_4h', etc.
    Devuelve (modelo, scaler, umbral) o (None, None, None) si no existe.
    """
    carpeta = os.path.join(MODELS_DIR, simbolo)
    ruta_m  = os.path.join(carpeta, f'modelo_{nombre}.pkl')
    ruta_s  = os.path.join(carpeta, f'scaler_modelo_{nombre}.pkl')
    ruta_t  = os.path.join(carpeta, f'threshold_modelo_{nombre}.json')

    if not os.path.exists(ruta_m):
        return None, None, None

    modelo  = joblib.load(ruta_m)
    scaler  = joblib.load(ruta_s) if os.path.exists(ruta_s) else None
    umbral  = 0.5
    if os.path.exists(ruta_t):
        with open(ruta_t) as f:
            umbral = json.load(f).get('umbral', 0.5)

    return modelo, scaler, umbral


def cargar_features_guardadas(simbolo: str) -> list:
    """Lee la lista de features con la que se entrenó el modelo."""
    ruta = os.path.join(MODELS_DIR, simbolo, 'features.json')
    if os.path.exists(ruta):
        with open(ruta) as f:
            return json.load(f)
    return FEATURES


# ──────────────────────────────────────────────────────────────────────
#  PREPARAR DATOS PARA PREDICCIÓN
# ──────────────────────────────────────────────────────────────────────
def añadir_features_temporales(df: pd.DataFrame) -> pd.DataFrame:
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


def cargar_ultima_fila(simbolo: str, tf: str) -> pd.Series | None:
    """
    Carga el CSV procesado y devuelve la última fila con features completas.
    La última fila = vela más reciente cerrada.
    """
    ruta = os.path.join(PROCESSED_DIR, f"{simbolo}_{tf}.csv")
    if not os.path.exists(ruta):
        print(f"  ⚠️  No encontrado: {ruta} — ejecuta preparar_datos.py primero")
        return None

    df = pd.read_csv(ruta, parse_dates=['timestamp'])
    df = añadir_features_temporales(df)

    # Buscar la última fila que tenga todos los features sin NaN
    feats_disponibles = [f for f in FEATURES if f in df.columns]
    df_ok = df.dropna(subset=feats_disponibles)

    if df_ok.empty:
        print(f"  ⚠️  {simbolo} {tf}: no hay filas con features completos")
        return None

    return df_ok.iloc[-1]


def predecir_prob(modelo, scaler, fila: pd.Series, features: list) -> float | None:
    """Aplica scaler y modelo a una fila, devuelve prob de clase 1."""
    feats_ok = [f for f in features if f in fila.index]
    if not feats_ok:
        return None

    x = fila[feats_ok].values.reshape(1, -1).astype(float)

    if np.any(np.isnan(x)):
        return None

    if scaler is not None:
        x = scaler.transform(x)

    prob = modelo.predict_proba(x)[0][1]
    return float(prob)


# ──────────────────────────────────────────────────────────────────────
#  LÓGICA DE DECISIÓN
# ──────────────────────────────────────────────────────────────────────
def calcular_capital_y_apalancamiento(fuerza_4h: float) -> tuple[float, float]:
    """
    fuerza_4h: probabilidad del modelo exito_long o exito_short en 4h (0-1).
    Cuanto mayor sea, más capital y apalancamiento se usa.
    Por debajo de 0.5 → valores base (el 4h no apoya la operación).
    """
    # Normalizar fuerza entre 0 y 1 a partir de 0.5
    # fuerza_4h=0.50 → factor=0.0, fuerza_4h=1.0 → factor=1.0
    factor = max(0.0, (fuerza_4h - 0.50) / 0.50)

    capital = CAPITAL_BASE + factor * (CAPITAL_MAX - CAPITAL_BASE)
    apalancamiento = APALANCAMIENTO_BASE + factor * (APALANCAMIENTO_MAX - APALANCAMIENTO_BASE)

    capital        = round(min(capital, CAPITAL_MAX), 3)
    apalancamiento = round(min(apalancamiento, APALANCAMIENTO_MAX), 1)

    return capital, apalancamiento


def generar_señal(simbolo: str) -> dict:
    """
    Lógica principal de decisión para un símbolo.

    Pasos:
      1. Cargar la última vela de 1h y 4h con features completos
      2. Obtener prob de exito_long y exito_short en ambos timeframes
      3. Filtrar por umbral mínimo en 1h (timing)
      4. Usar 4h para amplificar capital si apoya la dirección
      5. Si el 4h contradice el 1h, reducir capital o cancelar
    """
    resultado = {
        'simbolo':        simbolo,
        'direccion':      'FLAT',
        'confianza':      0.0,
        'fuerza':         0.0,
        'apalancamiento': APALANCAMIENTO_BASE,
        'capital_pct':    CAPITAL_BASE,
        'prob_long_1h':   None,
        'prob_short_1h':  None,
        'prob_long_4h':   None,
        'prob_short_4h':  None,
        'razon':          'sin señal',
    }

    features = cargar_features_guardadas(simbolo)

    # ── Cargar datos ──
    fila_1h = cargar_ultima_fila(simbolo, '1h')
    fila_4h = cargar_ultima_fila(simbolo, '4h')

    if fila_1h is None:
        resultado['razon'] = 'sin datos 1h'
        return resultado

    # ── Cargar modelos ──
    m_long_1h,  s_long_1h,  u_long_1h  = cargar_modelo(simbolo, 'exito_long_1h')
    m_short_1h, s_short_1h, u_short_1h = cargar_modelo(simbolo, 'exito_short_1h')
    m_long_4h,  s_long_4h,  u_long_4h  = cargar_modelo(simbolo, 'exito_long_4h')
    m_short_4h, s_short_4h, u_short_4h = cargar_modelo(simbolo, 'exito_short_4h')

    if m_long_1h is None or m_short_1h is None:
        resultado['razon'] = 'modelos 1h no encontrados — ejecuta entrenar.py'
        return resultado

    # ── Probabilidades 1h ──
    prob_long_1h  = predecir_prob(m_long_1h,  s_long_1h,  fila_1h, features)
    prob_short_1h = predecir_prob(m_short_1h, s_short_1h, fila_1h, features)
    resultado['prob_long_1h']  = round(prob_long_1h,  3) if prob_long_1h  is not None else None
    resultado['prob_short_1h'] = round(prob_short_1h, 3) if prob_short_1h is not None else None

    # ── Probabilidades 4h ──
    prob_long_4h  = None
    prob_short_4h = None
    if fila_4h is not None and m_long_4h is not None and m_short_4h is not None:
        prob_long_4h  = predecir_prob(m_long_4h,  s_long_4h,  fila_4h, features)
        prob_short_4h = predecir_prob(m_short_4h, s_short_4h, fila_4h, features)
        resultado['prob_long_4h']  = round(prob_long_4h,  3) if prob_long_4h  is not None else None
        resultado['prob_short_4h'] = round(prob_short_4h, 3) if prob_short_4h is not None else None

    # ── Decisión de dirección basada en 1h ──
    if prob_long_1h is None or prob_short_1h is None:
        resultado['razon'] = 'NaN en features — revisar datos'
        return resultado

    # El 1h decide si hay señal y en qué dirección
    # Condición: la prob de una dirección supera el umbral mínimo
    # Y es claramente mayor que la contraria (diferencia > 0.05)
    hay_long  = prob_long_1h  >= UMBRAL_MINIMO_1H
    hay_short = prob_short_1h >= UMBRAL_MINIMO_1H
    diferencia = abs(prob_long_1h - prob_short_1h)

    if not hay_long and not hay_short:
        resultado['razon'] = (f'confianza insuficiente '
                              f'(long={prob_long_1h:.2f}, short={prob_short_1h:.2f}, '
                              f'mínimo={UMBRAL_MINIMO_1H})')
        return resultado

    if hay_long and hay_short:
        # Ambas señales altas → mercado ambiguo, no operar
        resultado['razon'] = (f'señal ambigua — ambas altas '
                              f'(long={prob_long_1h:.2f}, short={prob_short_1h:.2f})')
        return resultado

    # Dirección ganadora según 1h
    if hay_long:
        direccion    = 'LONG'
        confianza_1h = prob_long_1h
        fuerza_4h    = prob_long_4h  if prob_long_4h  is not None else 0.5
        prob_contra  = prob_short_4h if prob_short_4h is not None else 0.5
    else:
        direccion    = 'SHORT'
        confianza_1h = prob_short_1h
        fuerza_4h    = prob_short_4h if prob_short_4h is not None else 0.5
        prob_contra  = prob_long_4h  if prob_long_4h  is not None else 0.5

    # ── Modificador 4h ──
    # Si el 4h contradice activamente la dirección del 1h → reducir o cancelar
    if prob_contra > fuerza_4h and (prob_contra - fuerza_4h) > 0.10:
        resultado['razon'] = (f'4h contradice al 1h ({direccion}): '
                              f'fuerza_4h={fuerza_4h:.2f} vs contra={prob_contra:.2f}')
        return resultado

    # ── Calcular capital y apalancamiento ──
    capital, apalancamiento = calcular_capital_y_apalancamiento(fuerza_4h)

    # Si el 4h no tiene modelos disponibles, usar valores base
    if prob_long_4h is None:
        capital        = CAPITAL_BASE
        apalancamiento = APALANCAMIENTO_BASE
        nota_4h        = 'sin modelo 4h (valores base)'
    else:
        nota_4h = f'4h fuerza={fuerza_4h:.2f}'

    resultado.update({
        'direccion':      direccion,
        'confianza':      round(confianza_1h, 3),
        'fuerza':         round(fuerza_4h, 3),
        'apalancamiento': apalancamiento,
        'capital_pct':    capital,
        'razon':          (f'{direccion} — 1h={confianza_1h:.2f} | {nota_4h} | '
                           f'capital={capital*100:.0f}% x{apalancamiento}'),
    })

    return resultado


# ──────────────────────────────────────────────────────────────────────
#  OUTPUT
# ──────────────────────────────────────────────────────────────────────
COLORES = {
    'LONG':  '\033[92m',  # verde
    'SHORT': '\033[91m',  # rojo
    'FLAT':  '\033[90m',  # gris
    'RESET': '\033[0m',
}

def imprimir_señal(s: dict):
    color = COLORES.get(s['direccion'], '')
    reset = COLORES['RESET']

    dir_str = f"{color}{s['direccion']:5}{reset}"

    print(f"\n  {'─'*50}")
    print(f"  {s['simbolo']:<10}  {dir_str}  confianza={s['confianza']:.2f}  "
          f"fuerza4h={s['fuerza']:.2f}")

    if s['direccion'] != 'FLAT':
        print(f"             capital={s['capital_pct']*100:.0f}%  "
              f"apalancamiento=x{s['apalancamiento']}")

    print(f"             {s['razon']}")

    probs = []
    if s['prob_long_1h']  is not None: probs.append(f"long_1h={s['prob_long_1h']:.3f}")
    if s['prob_short_1h'] is not None: probs.append(f"short_1h={s['prob_short_1h']:.3f}")
    if s['prob_long_4h']  is not None: probs.append(f"long_4h={s['prob_long_4h']:.3f}")
    if s['prob_short_4h'] is not None: probs.append(f"short_4h={s['prob_short_4h']:.3f}")
    if probs:
        print(f"             probs: {' | '.join(probs)}")


def main():
    parser = argparse.ArgumentParser(description='Generador de señales de trading')
    parser.add_argument('--simbolo', type=str, default=None,
                        help='Símbolo concreto (ej: BTCUSDT). Sin esto, procesa todos.')
    parser.add_argument('--json', action='store_true',
                        help='Salida en JSON puro (para integración con el bot)')
    parser.add_argument('--umbral', type=float, default=None,
                        help='Umbral mínimo de confianza 1h (sobreescribe el default)')
    args = parser.parse_args()

    global UMBRAL_MINIMO_1H
    if args.umbral is not None:
        UMBRAL_MINIMO_1H = args.umbral

    simbolos = [args.simbolo.upper()] if args.simbolo else SIMBOLOS

    señales = []
    for simbolo in simbolos:
        señal = generar_señal(simbolo)
        señales.append(señal)

    if args.json:
        print(json.dumps(señales, indent=2, ensure_ascii=False))
        return

    # Output legible
    print(f"\n{'═'*52}")
    print(f"  SEÑALES DE TRADING")
    print(f"{'═'*52}")
    for s in señales:
        imprimir_señal(s)
    print(f"\n{'═'*52}")

    # Resumen rápido
    activas = [s for s in señales if s['direccion'] != 'FLAT']
    if activas:
        print(f"\n  Operaciones activas: {len(activas)}")
        for s in activas:
            print(f"    → {s['simbolo']}: {s['direccion']} "
                  f"x{s['apalancamiento']} ({s['capital_pct']*100:.0f}% capital)")
    else:
        print(f"\n  Sin señales activas — mercado sin oportunidad clara")
    print()


if __name__ == '__main__':
    main()
