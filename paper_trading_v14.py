"""
=============================================================
  CRYPTO AI BOT — Paper Trading v15
=============================================================
  Filosofía central: ANTICIPAR, no confirmar.

  Problemas de v15 resueltos:
  ────────────────────────────
  5. POSICIONES DUPLICADAS → comprobación ya existía pero reforzada
     Nunca abre long/short si ya hay uno abierto para ese símbolo.

  6. LÍMITE DE MARGEN REAL
     Nueva función limite_margen_ok(): la suma de márgenes (dinero tuyo
     puesto como garantía) nunca puede superar el capital disponible.
     Con 1000$ → máximo 1000$ en márgenes (con cualquier apalancamiento).
     Con 2000$ (si el bot gana) → el límite sube a 2000$ automáticamente.

  7. CAPITAL DINÁMICO
     margen_efectivo() ya no capea al CAPITAL_INICIAL; usa el capital
     actual. Si el bot pasa de 1000 a 2000, opera con tamaños acordes.

  8. TP INTRADÍA DETECTADO
     gestionar_posiciones() ahora recibe high/low de la vela.
     Si el high/low toca el TP aunque el close no lo alcance, se cierra.

  9. MENSAJES ENRIQUECIDOS
     Apertura, cierre y reporte diario muestran:
     · Capital actual con % rentabilidad acumulada
     · Exposición total en mercado (en $ y % del capital)
     · Funding cobrado en cada cierre
     · Número de posiciones restantes tras cierre

  Problemas de v14 resueltos:
  ────────────────────────────
  1. FILTROS REACTIVOS → filtros predictivos
     Antes: short solo si RSI < 45 y MACD ya negativo (ya bajó)
     Ahora: short si RSI sobrecomprado + divergencia + momentum
            decelerando = VA A BAJAR, no YA BAJÓ

  2. RÉGIMEN BLOQUEANTE → régimen como contexto
     Antes: short bloqueado si régimen >= 0 (demasiado estricto)
     Ahora: el régimen modula el tamaño y umbral, no bloquea

  3. SIN MULTI-TIMEFRAME → contexto de 4H integrado
     Antes: solo velas de 1H
     Ahora: se descarga el 4H para cada cripto y se usa como
            contexto de tendencia superior. Si 4H es bajista y
            1H rebota un poco → ideal para short

  4. PROBABILIDAD COMO FILTRO PRINCIPAL
     Si prob >= 0.90: se abre con UN solo indicador confirmando
     Si prob >= 0.75: necesita 2 indicadores
     Si prob >= 0.64: necesita 3 indicadores (estándar)
     La IA ya hizo el trabajo pesado, no filtrarla en exceso.

  Sistema de señales v15:
  ────────────────────────
  Para SHORT necesita:
    a) prob_short alta (del modelo entrenado en bajadas)
    b) Score de condiciones predictivas >= umbral según prob
    c) 4H confirma dirección (opcional pero reduce umbral)
  Para LONG: lógica simétrica.

  Condiciones predictivas (ANTICIPAN el movimiento):
    SHORT predictivo:
      · RSI sobrecomprado (>65) → va a retroceder
      · Divergencia bajista RSI (precio sube, RSI baja)
      · Stochastic >80 y girando → momentum agotado
      · MACD hist positivo pero decelerando (pico de momentum)
      · Precio en resistencia o cerca de Fibonacci 0.618
      · Volumen decreciente en subida (sin convicción)
    LONG predictivo (simétrico):
      · RSI sobrevendido (<35) → va a rebotar
      · Divergencia alcista RSI
      · Stochastic <20 y girando
      · MACD hist negativo pero acelerando hacia 0
      · Precio en soporte o cerca de Fibonacci 0.382/0.5
      · Volumen decreciente en bajada (vendedores agotados)
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

MAX_MARGEN_TOTAL_PCT = 0.80  # ampliado para soportar más posiciones simultáneas

# ── Tiempo máximo: base en horas, escalado dinámico por ATR ──
# tiempo_max = BASE / (atr_pct_actual / atr_ref)
# BTC con ATR 0.8% → ~25h. SOL con ATR 3% → ~6h.
MAX_HORAS_BASE_LONG  = 20
MAX_HORAS_BASE_SHORT = 16
MAX_HORAS_ATR_REF    = 0.012  # ATR de referencia (1.2% = BTC en condiciones normales)
MAX_HORAS_MIN        = 6
MAX_HORAS_MAX        = 48

# ── Trailing dinámico por ATR ──
# Se activa cuando la ganancia supera TRAILING_ATR_ACTIVACION × ATR actual.
# El stop sigue a TRAILING_ATR_DISTANCIA × ATR del precio de referencia.
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

# ── Config por calidad — 4 niveles escalonados ──
# sl_mult y tp_ratio son fallback cuando no hay niveles de mercado usables.
# El apalancamiento y margen crecen con la calidad de la señal.
#
# Puntuación de calidad (calidad_score):
#   +3  4H alineado con la dirección
#   +2  MACD histograma a favor
#   +2  Divergencia confirmada RSI/MACD
#   +1  ADX > 20
#   +1  RSI en zona extrema (>65 short / <35 long)
#   +1  Nivel S/R con 4+ toques en 4H
#   -2  MACD convergiendo en contra
#   -2  4H en contra de la dirección
#   -1  ADX < 12 (mercado sin tendencia)
#
#   0-2 pts → nivel1 (x4)  |  3-4 pts → nivel2 (x12)
#   5-6 pts → nivel3 (x20) |  7+  pts → nivel4 (x30)

TRADE_CONFIG = {
    'nivel1': {'leverage': 4,  'margen': 0.08, 'sl_mult': 2.0, 'tp_ratio': 1.8, 'rr_min': 1.6},
    'nivel2': {'leverage': 12, 'margen': 0.14, 'sl_mult': 1.8, 'tp_ratio': 2.2, 'rr_min': 1.8},
    'nivel3': {'leverage': 20, 'margen': 0.18, 'sl_mult': 1.6, 'tp_ratio': 2.6, 'rr_min': 2.0},
    'nivel4': {'leverage': 30, 'margen': 0.22, 'sl_mult': 1.4, 'tp_ratio': 3.0, 'rr_min': 2.2},
}

# ── Parámetros de Risk/Reward ──
RR_MINIMO          = 1.6    # mínimo global (nivel1 puede operar con RR más bajo)
SL_ATR_MIN_MULT    = 0.8    # SL nunca menor a 0.8×ATR — evita que el ruido lo toque
SL_ATR_MAX_MULT    = 5.0    # SL nunca mayor a 5×ATR
TP_ATR_MAX_MULT    = 12.0   # TP nunca mayor a 12×ATR
BUFFER_NIVEL_PCT   = 0.004  # 0.4% extra más allá del nivel S/R para el SL

# ── Umbrales de probabilidad ──
# El portero ahora son los scores técnicos + calidad.
# La IA actúa como amplificador de nivel, no como filtro duro.
# Umbrales reducidos para permitir entrar cuando el score técnico es sólido.
PROB_ELITE_LONG    = 0.74
PROB_PREMIUM_LONG  = 0.35   # reducido: portero = score técnico
PROB_ELITE_SHORT   = 0.70
PROB_PREMIUM_SHORT = 0.35   # reducido: portero = score técnico

# ── Amplificador de nivel por IA (tendencia diaria 1D) ──
# El modelo 1H se alimenta con velas diarias para evaluar tendencia macro.
# Su prob modifica el nivel asignado por calidad_score ANTES de abrir.
#   prob_1d > 0.65  → sube un nivel
#   prob_1d 0.45-0.65 → nivel sin cambios
#   prob_1d 0.30-0.45 → baja un nivel
#   prob_1d < 0.30  → baja dos niveles
IA_AMP_SUBE_UN_NIVEL   = 0.65
IA_AMP_SIN_CAMBIO_INF  = 0.45
IA_AMP_BAJA_UN_NIVEL   = 0.30

# ── Indicador IA corto plazo (próximas 4-8h) ──
# El mismo modelo 1H pero se usa en el score_predictivo como señal más.
# Velas de 1H usadas como ventana de contexto reciente para la predicción.
# El modelo ya entrenado predice si habrá movimiento en las próximas horas.
#   prob_ct > 0.60  → +1.5 pts al score (señal fuerte a favor)
#   prob_ct 0.50-0.60 → +0.5 pts (señal débil a favor)
#   prob_ct < 0.40  → -1.0 pts al score (señal en contra)
IA_CT_FUERTE   = 0.60   # prob corto plazo que suma 1.5 pts
IA_CT_DEBIL    = 0.50   # prob que suma 0.5 pts
IA_CT_CONTRA   = 0.40   # prob que resta 1.0 pts

# Score mínimo de condiciones predictivas según probabilidad
# A mayor prob, menos condiciones adicionales necesarias
def score_minimo(prob: float, direccion: str) -> int:
    umbral_elite = PROB_ELITE_SHORT if direccion == 'short' else PROB_ELITE_LONG
    if prob >= 0.90:  return 1   # casi certeza → 1 condición basta
    if prob >= 0.80:  return 2   # alta confianza → 2 condiciones
    if prob >= umbral_elite: return 2  # elite → 2 condiciones
    return 3                     # premium → 3 condiciones

DIR_MODELOS   = 'models'
ESTADO_FILE   = f'paper_trading/{VERSION}/estado.json'
LOG_FILE      = f'paper_trading/{VERSION}/operaciones.csv'
METRICAS_FILE = f'paper_trading/{VERSION}/metricas.json'

KRAKEN_MAP = {
    'BTC/USDT': 'XBTUSD', 'ETH/USDT': 'ETHUSD',
    'BNB/USDT': 'BNBUSD', 'SOL/USDT': 'SOLUSD',
}


# ──────────────────────────────────────────────
#  TELEGRAM
# ──────────────────────────────────────────────
def telegram(msg: str):
    token   = os.environ.get('TELEGRAM_TOKEN_V14', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_V14', '')
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
    """Descarga velas de Kraken. intervalo en minutos (60=1h, 240=4h)."""
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
    """
    Descarga velas diarias (1D) y extrae la tendencia macro.
    Devuelve dict con tendencia, momentum y el DataFrame crudo
    para pasárselo al modelo como amplificador.

    Usa el intervalo 1440 min (24h) de Kraken.
    Con 120 velas diarias tenemos ~4 meses de historia, suficiente
    para que el modelo tenga contexto de tendencia estructural.
    """
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

        precio1d  = float(c1d.iloc[-1])
        ema50_val = float(ema50_1d.iloc[-1])  if ema50_1d  is not None else precio1d
        ema200_val= float(ema200_1d.iloc[-1]) if ema200_1d is not None else precio1d
        rsi1d_val = float(rsi1d.iloc[-1])     if not rsi1d.empty else 50.0
        mh1d      = float(macd1d_h.iloc[-1])  if not macd1d_h.empty else 0.0
        mh1d_prev = float(macd1d_h.iloc[-2])  if len(macd1d_h) > 1 else 0.0

        # Tendencia diaria
        if precio1d > ema50_val > ema200_val:
            tendencia1d = 'alcista'
        elif precio1d < ema50_val < ema200_val:
            tendencia1d = 'bajista'
        else:
            tendencia1d = 'lateral'

        # Momentum diario (MACD hist)
        if mh1d > 0 and mh1d > mh1d_prev:
            momentum1d = 'alcista_acelerando'
        elif mh1d > 0 and mh1d < mh1d_prev:
            momentum1d = 'alcista_desacelerando'
        elif mh1d < 0 and mh1d < mh1d_prev:
            momentum1d = 'bajista_acelerando'
        elif mh1d < 0 and mh1d > mh1d_prev:
            momentum1d = 'bajista_desacelerando'
        else:
            momentum1d = 'neutral'

        return {
            'disponible': True,
            'tendencia':  tendencia1d,
            'momentum':   momentum1d,
            'rsi':        rsi1d_val,
            'precio':     precio1d,
            'ema50':      ema50_val,
            'ema200':     ema200_val,
            'df':         df1d,   # DataFrame crudo para pasarlo al modelo
        }
    except Exception as e:
        print(f"  ⚠️  1D contexto error: {e}")
        return {'disponible': False}


def predecir_tendencia_diaria(df1d: pd.DataFrame, sub: Dict) -> float:
    """
    Usa el modelo 1H (sub) con datos de velas diarias para obtener
    la probabilidad de tendencia macro (amplificador de largo plazo).

    Calcula los mismos indicadores que calcular_indicadores() pero sobre
    el DataFrame diario, luego pasa las features al modelo ya entrenado.
    Devuelve float 0-1 (prob de que el movimiento sea en la dirección del modelo).
    """
    if df1d is None or len(df1d) < 30 or sub is None:
        return 0.50   # neutral si no hay datos
    try:
        df_d = calcular_indicadores(df1d.copy())
        df_d = df_d.dropna(subset=['rsi_14', 'macd_hist', 'ema_50']).reset_index(drop=True)
        if len(df_d) < 10:
            return 0.50
        return predecir(df_d, sub)
    except Exception as e:
        print(f"  ⚠️  predecir_tendencia_diaria: {e}")
        return 0.50


def amplificar_nivel_por_ia(nivel_str: str, prob_1d: float, direccion: str,
                             tendencia_1d: str) -> Tuple[str, str]:
    """
    Modifica el nivel asignado por calidad_score usando la probabilidad
    del modelo sobre datos diarios (tendencia macro).

    Lógica:
      · La probabilidad del modelo es direccional: el modelo 'long' da alta
        prob cuando espera subida, el modelo 'short' cuando espera bajada.
      · Si la macro acompaña → sube nivel (más apalancamiento).
      · Si la macro va en contra → baja nivel (más conservador).
      · Si la macro es neutral → sin cambio.

    Retorna (nivel_nuevo, nota_para_log).
    """
    orden = ['nivel1', 'nivel2', 'nivel3', 'nivel4']
    idx   = orden.index(nivel_str)

    # Determinar si la macro está a favor o en contra según dirección
    macro_a_favor = (
        (direccion == 'long'  and tendencia_1d == 'alcista') or
        (direccion == 'short' and tendencia_1d == 'bajista')
    )
    macro_en_contra = (
        (direccion == 'long'  and tendencia_1d == 'bajista') or
        (direccion == 'short' and tendencia_1d == 'alcista')
    )

    if prob_1d > IA_AMP_SUBE_UN_NIVEL and macro_a_favor:
        idx_nuevo = min(idx + 1, 3)
        nota = (f"📈 IA diaria prob={prob_1d:.3f} + tendencia {tendencia_1d} "
                f"→ SUBE nivel {nivel_str}→{orden[idx_nuevo]}")
    elif prob_1d < IA_AMP_BAJA_UN_NIVEL or macro_en_contra:
        # Baja dos niveles si prob muy baja Y tendencia en contra
        # Baja un nivel si solo una condición se cumple
        bajada = 2 if (prob_1d < IA_AMP_BAJA_UN_NIVEL and macro_en_contra) else 1
        idx_nuevo = max(idx - bajada, 0)
        nota = (f"📉 IA diaria prob={prob_1d:.3f} tendencia={tendencia_1d} "
                f"→ BAJA {bajada} nivel(es) {nivel_str}→{orden[idx_nuevo]}")
    else:
        idx_nuevo = idx
        nota = (f"➡️  IA diaria prob={prob_1d:.3f} tendencia={tendencia_1d} "
                f"→ nivel sin cambio ({nivel_str})")

    return orden[idx_nuevo], nota


def contexto_4h(simbolo: str) -> Dict:
    """
    Descarga velas de 4H y extrae tendencia, RSI y régimen.
    Devuelve dict con contexto simplificado.
    """
    df4 = obtener_velas(simbolo, intervalo=240, limite=100)
    if df4 is None or len(df4) < 30:
        return {'disponible': False}
    try:
        import pandas_ta as ta
        c4 = df4['close']
        h4 = df4['high']
        l4 = df4['low']

        rsi4  = ta.rsi(c4, length=14)
        ema50_4 = ta.ema(c4, length=50)
        ema200_4 = ta.ema(c4, length=200)
        macd4_df = ta.macd(c4, fast=12, slow=26, signal=9)
        macd4_hist = macd4_df.iloc[:, 2] if (macd4_df is not None and not macd4_df.empty and macd4_df.shape[1] > 2) else pd.Series([0])

        ultima4 = df4.iloc[-1]
        rsi4_val   = float(rsi4.iloc[-1]) if not rsi4.empty else 50
        ema50_val  = float(ema50_4.iloc[-1]) if ema50_4 is not None else float(c4.iloc[-1])
        ema200_val = float(ema200_4.iloc[-1]) if ema200_4 is not None else float(c4.iloc[-1])
        precio4    = float(ultima4['close'])
        mh4        = float(macd4_hist.iloc[-1]) if not macd4_hist.empty else 0
        mh4_prev   = float(macd4_hist.iloc[-2]) if len(macd4_hist) > 1 else 0

        # Tendencia 4H
        if precio4 > ema50_val > ema200_val:
            tendencia4 = 'alcista'
        elif precio4 < ema50_val < ema200_val:
            tendencia4 = 'bajista'
        else:
            tendencia4 = 'lateral'

        # Momentum 4H
        if mh4 > 0 and mh4 > mh4_prev:
            momentum4 = 'alcista_acelerando'
        elif mh4 > 0 and mh4 < mh4_prev:
            momentum4 = 'alcista_desacelerando'
        elif mh4 < 0 and mh4 < mh4_prev:
            momentum4 = 'bajista_acelerando'
        elif mh4 < 0 and mh4 > mh4_prev:
            momentum4 = 'bajista_desacelerando'
        else:
            momentum4 = 'neutral'

        return {
            'disponible':  True,
            'tendencia':   tendencia4,
            'momentum':    momentum4,
            'rsi':         rsi4_val,
            'precio':      precio4,
            'ema50':       ema50_val,
            'ema200':      ema200_val,
            'df':          df4,          # df raw para detectar_patrones
        }
    except Exception as e:
        print(f"  ⚠️  4H contexto error: {e}")
        return {'disponible': False}



# ──────────────────────────────────────────────
#  DETECCIÓN DE PATRONES CHARTISTAS
# ──────────────────────────────────────────────

def detectar_patrones(df: pd.DataFrame, df4h: pd.DataFrame = None) -> Dict:
    """
    Detecta patrones chartistas en 1H y confirma con 4H cuando está disponible.
    Devuelve un dict con los patrones encontrados y puntuación para short/long.

    Patrones detectados:
      · Rangos (soporte/resistencia horizontales)
      · HCH y HCH invertido
      · Doble techo / Doble suelo
      · Canales alcistas y bajistas
      · Cuñas alcistas y bajistas
      · Banderas alcistas y bajistas
    """
    resultado = {
        'patrones_short': [],   # patrones que favorecen short
        'patrones_long':  [],   # patrones que favorecen long
        'score_short':    0,
        'score_long':     0,
        'niveles_clave':  [],   # niveles de precio donde el precio puede frenarse
    }

    if df is None or len(df) < 50:
        return resultado

    c   = df['close'].values.astype(float)
    h   = df['high'].values.astype(float)
    l   = df['low'].values.astype(float)
    n   = len(c)
    precio_actual = c[-1]

    # ── 1. ZONAS DE SOPORTE Y RESISTENCIA (niveles donde el precio gira) ──
    # Metodología:
    #   1. Detectar picos locales (resistencias) y valles locales (soportes) en 1H
    #   2. Agrupar toques cercanos en zonas (banda ±0.3%) y contar giros reales
    #   3. Confirmar en 4H: si el mismo nivel aparece allí, la zona es más fuerte
    #   4. Puntuar según proximidad del precio actual y fuerza de la zona

    def detectar_zonas_sr(highs_arr, lows_arr, closes_arr, banda_pct=0.003):
        """
        Encuentra zonas de precio donde ha habido múltiples giros de dirección.
        Agrupa toques cercanos (dentro de banda_pct) en una sola zona.
        Devuelve lista de (nivel, n_toques, tipo) ordenada por fuerza.
        """
        # Identificar picos y valles locales (giros reales)
        picos  = []
        valles = []
        for i in range(2, len(highs_arr) - 2):
            # Pico: máximo local con confirmación de 2 velas a cada lado
            if (highs_arr[i] > highs_arr[i-1] and highs_arr[i] > highs_arr[i-2]
                    and highs_arr[i] > highs_arr[i+1] and highs_arr[i] > highs_arr[i+2]):
                # Verificar que realmente hubo giro (close siguiente baja)
                if closes_arr[i+1] < closes_arr[i]:
                    picos.append(highs_arr[i])
            # Valle: mínimo local con confirmación de 2 velas a cada lado
            if (lows_arr[i] < lows_arr[i-1] and lows_arr[i] < lows_arr[i-2]
                    and lows_arr[i] < lows_arr[i+1] and lows_arr[i] < lows_arr[i+2]):
                # Verificar que realmente hubo giro (close siguiente sube)
                if closes_arr[i+1] > closes_arr[i]:
                    valles.append(lows_arr[i])

        # Agrupar picos en zonas de resistencia
        zonas_res = []
        for precio_pico in sorted(picos):
            agrupado = False
            for zona in zonas_res:
                if abs(precio_pico - zona['nivel']) / zona['nivel'] < banda_pct:
                    # Actualizar nivel como media ponderada
                    zona['nivel'] = (zona['nivel'] * zona['toques'] + precio_pico) / (zona['toques'] + 1)
                    zona['toques'] += 1
                    agrupado = True
                    break
            if not agrupado:
                zonas_res.append({'nivel': precio_pico, 'toques': 1, 'tipo': 'resistencia'})

        # Agrupar valles en zonas de soporte
        zonas_sop = []
        for precio_valle in sorted(valles):
            agrupado = False
            for zona in zonas_sop:
                if abs(precio_valle - zona['nivel']) / zona['nivel'] < banda_pct:
                    zona['nivel'] = (zona['nivel'] * zona['toques'] + precio_valle) / (zona['toques'] + 1)
                    zona['toques'] += 1
                    agrupado = True
                    break
            if not agrupado:
                zonas_sop.append({'nivel': precio_valle, 'toques': 1, 'tipo': 'soporte'})

        # Solo zonas con al menos 2 toques (el precio ha rebotado ahí más de una vez)
        zonas_res = [z for z in zonas_res if z['toques'] >= 2]
        zonas_sop = [z for z in zonas_sop if z['toques'] >= 2]

        return zonas_res, zonas_sop

    # Detectar zonas en 1H (últimas 60 velas para tener historia suficiente)
    ventana_1h = min(60, n)
    zonas_res_1h, zonas_sop_1h = detectar_zonas_sr(
        h[-ventana_1h:], l[-ventana_1h:], c[-ventana_1h:]
    )

    # Detectar zonas en 4H si disponible
    zonas_res_4h, zonas_sop_4h = [], []
    if df4h is not None and len(df4h) >= 30:
        h4 = df4h['high'].values.astype(float)
        l4 = df4h['low'].values.astype(float)
        c4 = df4h['close'].values.astype(float)
        zonas_res_4h, zonas_sop_4h = detectar_zonas_sr(h4, l4, c4, banda_pct=0.004)

    def zona_confirmada_4h(nivel_1h, zonas_4h, banda=0.006):
        """Devuelve True si el nivel de 1H coincide con alguna zona de 4H."""
        for z4 in zonas_4h:
            if abs(nivel_1h - z4['nivel']) / z4['nivel'] < banda:
                return True
        return False

    # Evaluar zonas de resistencia (favorecen short si precio está cerca)
    for zona in sorted(zonas_res_1h, key=lambda z: -z['toques']):
        nivel = zona['nivel']
        toques = zona['toques']
        dist   = (nivel - precio_actual) / precio_actual  # positivo = zona arriba

        # Solo nos interesa si el precio está cerca (dentro del 1.5%)
        if -0.005 < dist < 0.015:
            confirma_4h = zona_confirmada_4h(nivel, zonas_res_4h)
            score_zona  = 1.0
            if toques >= 3:    score_zona += 0.5   # más toques = zona más fuerte
            if confirma_4h:    score_zona += 1.0   # coincide en 4H = muy relevante
            label = f"Resistencia ${nivel:,.2f} ({toques} toques{'  ✅4H' if confirma_4h else ''})"
            resultado['niveles_clave'].append(('resistencia', nivel))
            resultado['patrones_short'].append(label)
            resultado['score_short'] += score_zona
            break  # solo la zona más relevante para no inflar el score

    # Evaluar zonas de soporte (favorecen long si precio está cerca)
    for zona in sorted(zonas_sop_1h, key=lambda z: -z['toques']):
        nivel  = zona['nivel']
        toques = zona['toques']
        dist   = (precio_actual - nivel) / precio_actual  # positivo = zona abajo

        if -0.005 < dist < 0.015:
            confirma_4h = zona_confirmada_4h(nivel, zonas_sop_4h)
            score_zona  = 1.0
            if toques >= 3:    score_zona += 0.5
            if confirma_4h:    score_zona += 1.0
            label = f"Soporte ${nivel:,.2f} ({toques} toques{'  ✅4H' if confirma_4h else ''})"
            resultado['niveles_clave'].append(('soporte', nivel))
            resultado['patrones_long'].append(label)
            resultado['score_long'] += score_zona
            break

    # ── 2. DOBLE TECHO ──
    # Condiciones válidas:
    #   a) Tendencia alcista PREVIA (el doble techo solo tiene sentido tras subida)
    #   b) Dos picos similares con separación mínima
    #   c) Neckline perforada o precio muy cerca de perforarla (confirmación real)
    ventana_dt = min(30, n)
    highs_dt   = h[-ventana_dt:]
    lows_dt    = l[-ventana_dt:]
    idx_picos  = []
    for i in range(2, len(highs_dt) - 2):
        if (highs_dt[i] > highs_dt[i-1] and highs_dt[i] > highs_dt[i-2]
                and highs_dt[i] > highs_dt[i+1] and highs_dt[i] > highs_dt[i+2]):
            idx_picos.append(i)

    if len(idx_picos) >= 2:
        p1 = highs_dt[idx_picos[-2]]
        p2 = highs_dt[idx_picos[-1]]
        separacion = idx_picos[-1] - idx_picos[-2]
        similitud  = abs(p1 - p2) / p1

        if similitud < 0.015 and separacion >= 5:
            # Neckline = mínimo de los lows entre los dos picos
            tramo_entre = lows_dt[idx_picos[-2]:idx_picos[-1]]
            nivel_neckline_dt = float(np.min(tramo_entre)) if len(tramo_entre) > 0 else min(p1, p2) * 0.98

            resultado['niveles_clave'].append(('doble_techo', (p1 + p2) / 2))
            resultado['niveles_clave'].append(('neckline_dt', nivel_neckline_dt))

            # a) Verificar tendencia alcista PREVIA al primer pico
            inicio_previo_dt = max(0, n - ventana_dt - 15)
            fin_previo_dt    = max(0, n - ventana_dt + idx_picos[-2])
            tendencia_alcista_dt = False
            if fin_previo_dt > inicio_previo_dt + 5:
                tramo_previo_dt = c[inicio_previo_dt:fin_previo_dt]
                try:
                    coef_dt = np.polyfit(np.arange(len(tramo_previo_dt)), tramo_previo_dt, 1)
                    subida_pct_dt = (tramo_previo_dt[-1] - tramo_previo_dt[0]) / tramo_previo_dt[0]
                    tendencia_alcista_dt = coef_dt[0] > 0 and subida_pct_dt > 0.02
                except Exception:
                    tendencia_alcista_dt = False

            # b) Confirmación neckline: precio ha perforado o está a punto de perforar
            # Perforada: precio actual <= neckline (con margen del 0.5%)
            # Cerca:     precio está entre neckline y neckline+1.5%
            neckline_perforada = precio_actual <= nivel_neckline_dt * 1.005
            neckline_cerca     = precio_actual <= nivel_neckline_dt * 1.020

            if tendencia_alcista_dt and neckline_cerca:
                score_dt = 2.0
                if neckline_perforada:
                    score_dt = 2.5  # confirmación más fuerte con perforación real
                label_dt = (f"Doble techo ${(p1+p2)/2:,.2f} | "
                            f"Neckline ${nivel_neckline_dt:,.2f} "
                            f"{'⚡PERFORADA' if neckline_perforada else '⚠️cerca'}")
                resultado['patrones_short'].append(label_dt)
                resultado['score_short'] += score_dt

    # ── 3. DOBLE SUELO ──
    # Condiciones válidas:
    #   a) Tendencia bajista PREVIA (el doble suelo solo tiene sentido tras caída)
    #   b) Dos valles similares con separación mínima
    #   c) Neckline perforada al alza o precio cerca de perforarla
    lows_ds    = l[-ventana_dt:]
    highs_ds   = h[-ventana_dt:]
    idx_valles = []
    for i in range(2, len(lows_ds) - 2):
        if (lows_ds[i] < lows_ds[i-1] and lows_ds[i] < lows_ds[i-2]
                and lows_ds[i] < lows_ds[i+1] and lows_ds[i] < lows_ds[i+2]):
            idx_valles.append(i)

    if len(idx_valles) >= 2:
        v1 = lows_ds[idx_valles[-2]]
        v2 = lows_ds[idx_valles[-1]]
        separacion = idx_valles[-1] - idx_valles[-2]
        similitud  = abs(v1 - v2) / v1

        if similitud < 0.015 and separacion >= 5:
            # Neckline = máximo de los highs entre los dos valles
            tramo_entre_ds = highs_ds[idx_valles[-2]:idx_valles[-1]]
            nivel_neckline_ds = float(np.max(tramo_entre_ds)) if len(tramo_entre_ds) > 0 else max(v1, v2) * 1.02

            resultado['niveles_clave'].append(('doble_suelo', (v1 + v2) / 2))
            resultado['niveles_clave'].append(('neckline_ds', nivel_neckline_ds))

            # a) Verificar tendencia bajista PREVIA al primer valle
            inicio_previo_ds = max(0, n - ventana_dt - 15)
            fin_previo_ds    = max(0, n - ventana_dt + idx_valles[-2])
            tendencia_bajista_ds = False
            if fin_previo_ds > inicio_previo_ds + 5:
                tramo_previo_ds = c[inicio_previo_ds:fin_previo_ds]
                try:
                    coef_ds = np.polyfit(np.arange(len(tramo_previo_ds)), tramo_previo_ds, 1)
                    bajada_pct_ds = (tramo_previo_ds[0] - tramo_previo_ds[-1]) / tramo_previo_ds[0]
                    tendencia_bajista_ds = coef_ds[0] < 0 and bajada_pct_ds > 0.02
                except Exception:
                    tendencia_bajista_ds = False

            # b) Confirmación neckline: precio ha perforado al alza o está a punto
            neckline_perforada_ds = precio_actual >= nivel_neckline_ds * 0.995
            neckline_cerca_ds     = precio_actual >= nivel_neckline_ds * 0.980

            if tendencia_bajista_ds and neckline_cerca_ds:
                score_ds = 2.0
                if neckline_perforada_ds:
                    score_ds = 2.5
                label_ds = (f"Doble suelo ${(v1+v2)/2:,.2f} | "
                            f"Neckline ${nivel_neckline_ds:,.2f} "
                            f"{'⚡PERFORADA' if neckline_perforada_ds else '⚠️cerca'}")
                resultado['patrones_long'].append(label_ds)
                resultado['score_long'] += score_ds

    # ── 4. HCH (Hombro-Cabeza-Hombro) — señal bajista ──
    # Condiciones estrictas:
    #   a) Tendencia alcista PREVIA al hombro izquierdo (20 velas antes del patrón)
    #   b) 3 picos: cabeza > ambos hombros, hombros similares en altura
    #   c) Neckline aproximadamente horizontal (< 1.5% de inclinación)
    #   d) Volumen del hombro derecho menor que el del hombro izquierdo
    if len(idx_picos) >= 3:
        idx_h1  = idx_picos[-3]
        idx_cab = idx_picos[-2]
        idx_h2  = idx_picos[-1]
        h1  = highs_dt[idx_h1]
        cab = highs_dt[idx_cab]
        h2  = highs_dt[idx_h2]

        hombros_similares = abs(h1 - h2) / h1 < 0.03
        cabeza_mayor      = cab > h1 * 1.01 and cab > h2 * 1.01
        h2_no_supera_cab  = h2 < cab  # hombro derecho nunca supera la cabeza

        # Neckline: valle entre H.izq-Cabeza y valle entre Cabeza-H.der
        valle_izq = np.min(highs_dt[idx_h1:idx_cab]) if idx_cab > idx_h1 else h1
        valle_der = np.min(highs_dt[idx_cab:idx_h2]) if idx_h2 > idx_cab else h2
        # Usar lows para neckline más precisa
        lows_tramo = l[-ventana_dt:]
        nk_izq = np.min(lows_tramo[idx_h1:idx_cab]) if idx_cab > idx_h1 else h1 * 0.98
        nk_der = np.min(lows_tramo[idx_cab:idx_h2]) if idx_h2 > idx_cab else h2 * 0.98
        inclinacion_nk = abs(nk_der - nk_izq) / nk_izq if nk_izq > 0 else 1
        neckline_horizontal = inclinacion_nk < 0.015

        # Tendencia alcista previa: 20 velas antes del hombro izquierdo
        inicio_previo = max(0, n - ventana_dt - 20)
        fin_previo    = max(0, n - ventana_dt + idx_h1)
        tendencia_alcista_previa = False
        if fin_previo > inicio_previo + 5:
            tramo_previo = c[inicio_previo:fin_previo]
            x_prev = np.arange(len(tramo_previo))
            try:
                coef_prev = np.polyfit(x_prev, tramo_previo, 1)
                pendiente_previa = coef_prev[0] / tramo_previo[0] if tramo_previo[0] > 0 else 0
                subida_pct = (tramo_previo[-1] - tramo_previo[0]) / tramo_previo[0]
                # Máximos y mínimos crecientes: al menos 2 de 3 condiciones
                cond1 = pendiente_previa > 0.001   # EMA del tramo con pendiente positiva
                cond2 = subida_pct > 0.03           # precio sube >3% en ese tramo
                highs_prev = h[inicio_previo:fin_previo]
                cond3 = len(highs_prev) > 4 and highs_prev[-1] > highs_prev[0]  # máximos crecientes
                tendencia_alcista_previa = sum([cond1, cond2, cond3]) >= 2
            except Exception:
                tendencia_alcista_previa = False

        # Volumen decreciente del H.izq al H.der
        vol_ok = True
        if 'volume' in df.columns or 'volumen' in df.columns:
            col_vol = 'volume' if 'volume' in df.columns else 'volumen'
            vols = df[col_vol].values[-ventana_dt:]
            ancho_hombro = max(3, (idx_cab - idx_h1) // 2)
            vol_hizq = np.mean(vols[max(0, idx_h1 - ancho_hombro):idx_h1 + ancho_hombro + 1])
            vol_hder = np.mean(vols[max(0, idx_h2 - ancho_hombro):idx_h2 + ancho_hombro + 1])
            vol_ok = vol_hder < vol_hizq * 1.1  # hombro der con menor o igual volumen

        if (hombros_similares and cabeza_mayor and h2_no_supera_cab
                and neckline_horizontal and tendencia_alcista_previa and vol_ok):
            neckline_media = (nk_izq + nk_der) / 2
            resultado['niveles_clave'].append(('hch_cabeza',   cab))
            resultado['niveles_clave'].append(('hch_neckline', neckline_media))
            resultado['niveles_clave'].append(('hch_hombro',   (h1 + h2) / 2))
            dist_hombro = abs(precio_actual - h2) / precio_actual
            if dist_hombro < 0.02:
                resultado['patrones_short'].append(
                    f"HCH válido: H.izq=${h1:,.2f} Cab=${cab:,.2f} H.der=${h2:,.2f} "
                    f"Neckline=${neckline_media:,.2f}")
                resultado['score_short'] += 2.5

    # ── 5. HCH INVERTIDO — señal alcista ──
    # Condiciones estrictas (simétricas al HCH):
    #   a) Tendencia bajista PREVIA al hombro izquierdo
    #   b) 3 valles: cabeza < ambos hombros, hombros similares
    #   c) Neckline aproximadamente horizontal
    #   d) Volumen del hombro derecho menor que el del hombro izquierdo
    if len(idx_valles) >= 3:
        idx_v1  = idx_valles[-3]
        idx_cabi = idx_valles[-2]
        idx_v2  = idx_valles[-1]
        v1   = lows_ds[idx_v1]
        cab_i = lows_ds[idx_cabi]
        v2   = lows_ds[idx_v2]

        hombros_sim_i  = abs(v1 - v2) / v1 < 0.03
        cabeza_menor_i = cab_i < v1 * 0.99 and cab_i < v2 * 0.99
        v2_no_baja_cab = v2 > cab_i  # hombro derecho nunca baja de la cabeza

        # Neckline del HCHi: máximos entre los valles
        highs_tramo_i = h[-ventana_dt:]
        nki_izq = np.max(highs_tramo_i[idx_v1:idx_cabi]) if idx_cabi > idx_v1 else v1 * 1.02
        nki_der = np.max(highs_tramo_i[idx_cabi:idx_v2]) if idx_v2 > idx_cabi else v2 * 1.02
        inclinacion_nki = abs(nki_der - nki_izq) / nki_izq if nki_izq > 0 else 1
        neckline_horiz_i = inclinacion_nki < 0.015

        # Tendencia bajista previa
        inicio_previo_i = max(0, n - ventana_dt - 20)
        fin_previo_i    = max(0, n - ventana_dt + idx_v1)
        tendencia_bajista_previa = False
        if fin_previo_i > inicio_previo_i + 5:
            tramo_previo_i = c[inicio_previo_i:fin_previo_i]
            x_prev_i = np.arange(len(tramo_previo_i))
            try:
                coef_prev_i = np.polyfit(x_prev_i, tramo_previo_i, 1)
                pendiente_previa_i = coef_prev_i[0] / tramo_previo_i[0] if tramo_previo_i[0] > 0 else 0
                bajada_pct = (tramo_previo_i[0] - tramo_previo_i[-1]) / tramo_previo_i[0]
                lows_prev_i = l[inicio_previo_i:fin_previo_i]
                cond1_i = pendiente_previa_i < -0.001
                cond2_i = bajada_pct > 0.03
                cond3_i = len(lows_prev_i) > 4 and lows_prev_i[-1] < lows_prev_i[0]
                tendencia_bajista_previa = sum([cond1_i, cond2_i, cond3_i]) >= 2
            except Exception:
                tendencia_bajista_previa = False

        # Volumen decreciente del V.izq al V.der
        vol_ok_i = True
        if 'volume' in df.columns or 'volumen' in df.columns:
            col_vol = 'volume' if 'volume' in df.columns else 'volumen'
            vols_i = df[col_vol].values[-ventana_dt:]
            ancho_h_i = max(3, (idx_cabi - idx_v1) // 2)
            vol_vizq = np.mean(vols_i[max(0, idx_v1 - ancho_h_i):idx_v1 + ancho_h_i + 1])
            vol_vder = np.mean(vols_i[max(0, idx_v2 - ancho_h_i):idx_v2 + ancho_h_i + 1])
            vol_ok_i = vol_vder < vol_vizq * 1.1

        if (hombros_sim_i and cabeza_menor_i and v2_no_baja_cab
                and neckline_horiz_i and tendencia_bajista_previa and vol_ok_i):
            neckline_media_i = (nki_izq + nki_der) / 2
            resultado['niveles_clave'].append(('hchi_cabeza',   cab_i))
            resultado['niveles_clave'].append(('hchi_neckline', neckline_media_i))
            resultado['niveles_clave'].append(('hchi_hombro',   (v1 + v2) / 2))
            dist_hombro_i = abs(precio_actual - v2) / precio_actual
            if dist_hombro_i < 0.02:
                resultado['patrones_long'].append(
                    f"HCH invertido válido: H.izq=${v1:,.2f} Cab=${cab_i:,.2f} H.der=${v2:,.2f} "
                    f"Neckline=${neckline_media_i:,.2f}")
                resultado['score_long'] += 2.5

    # ── 6. CANAL (alcista o bajista) ──
    # Un canal real requiere que el precio haya tocado AMBAS líneas al menos 2 veces.
    # Método: ajustamos líneas sobre picos y valles locales, luego verificamos toques.
    ventana_canal = min(25, n)
    highs_canal = h[-ventana_canal:]
    lows_canal  = l[-ventana_canal:]
    closes_canal = c[-ventana_canal:]
    x_canal = np.arange(ventana_canal)

    # Variables para reusar en bloque de Cuña (definir siempre para evitar NameError)
    pend_h_norm = None
    pend_l_norm = None
    ancho_canal  = None
    canal_valido = False

    try:
        coef_h = np.polyfit(x_canal, highs_canal, 1)
        coef_l = np.polyfit(x_canal, lows_canal,  1)
        pend_h_norm = coef_h[0] / precio_actual  # pendiente normalizada por precio
        pend_l_norm = coef_l[0] / precio_actual

        techo_canal = np.polyval(coef_h, ventana_canal - 1)
        suelo_canal = np.polyval(coef_l, ventana_canal - 1)
        ancho_canal = (techo_canal - suelo_canal) / precio_actual

        # Contar toques reales en la línea de máximos (precio estuvo cerca del techo)
        # Un "toque" = high de esa vela dentro del 0.8% de la línea proyectada
        toques_techo = 0
        toques_suelo = 0
        for i in x_canal:
            nivel_techo_i = np.polyval(coef_h, i)
            nivel_suelo_i = np.polyval(coef_l, i)
            if abs(highs_canal[i] - nivel_techo_i) / nivel_techo_i < 0.008:
                toques_techo += 1
            if abs(lows_canal[i] - nivel_suelo_i) / nivel_suelo_i < 0.008:
                toques_suelo += 1

        # Canal válido: pendientes aproximadamente paralelas Y al menos 2 toques en cada línea
        pendientes_paralelas = abs(pend_h_norm - pend_l_norm) < 0.0015
        canal_valido = (ancho_canal > 0.01 and pendientes_paralelas
                        and toques_techo >= 2 and toques_suelo >= 2)

        if canal_valido:
            dist_techo = (techo_canal - precio_actual) / precio_actual
            dist_suelo = (precio_actual - suelo_canal) / precio_actual
            info_toques = f"({toques_techo}t techo / {toques_suelo}t suelo)"

            if pend_h_norm > 0.001 and pend_l_norm > 0.001:
                # Canal alcista
                resultado['niveles_clave'].append(('techo_canal_alcista', techo_canal))
                resultado['niveles_clave'].append(('suelo_canal_alcista', suelo_canal))
                if dist_techo < 0.015:
                    resultado['patrones_short'].append(
                        f"Canal alcista: precio en techo ~${techo_canal:,.2f} {info_toques}")
                    resultado['score_short'] += 1.5
                if dist_suelo < 0.015:
                    resultado['patrones_long'].append(
                        f"Canal alcista: precio en suelo ~${suelo_canal:,.2f} {info_toques}")
                    resultado['score_long'] += 1.0

            elif pend_h_norm < -0.001 and pend_l_norm < -0.001:
                # Canal bajista
                resultado['niveles_clave'].append(('techo_canal_bajista', techo_canal))
                resultado['niveles_clave'].append(('suelo_canal_bajista', suelo_canal))
                if dist_techo < 0.015:
                    resultado['patrones_short'].append(
                        f"Canal bajista: precio en techo ~${techo_canal:,.2f} {info_toques}")
                    resultado['score_short'] += 1.5
                if dist_suelo < 0.015:
                    resultado['patrones_long'].append(
                        f"Canal bajista: precio en suelo ~${suelo_canal:,.2f} {info_toques}")
                    resultado['score_long'] += 1.0
    except Exception:
        pass

    # ── 7. CUÑA (convergente, más estrecha que un canal) ──
    # La cuña usa su propia ventana y variables independientes del canal.
    # Lógica correcta de convergencia:
    #   Cuña alcista: máximos suben MENOS que mínimos (pend_h < pend_l, ambas positivas)
    #                 → señal bajista (ruptura probable hacia abajo)
    #   Cuña bajista: mínimos bajan MENOS que máximos (pend_l > pend_h, ambas negativas)
    #                 → señal alcista (ruptura probable hacia arriba)
    ventana_cuna = min(20, n)
    highs_cuna = h[-ventana_cuna:]
    lows_cuna  = l[-ventana_cuna:]
    x_cuna = np.arange(ventana_cuna)

    try:
        coef_hc = np.polyfit(x_cuna, highs_cuna, 1)
        coef_lc = np.polyfit(x_cuna, lows_cuna,  1)
        ph = coef_hc[0] / precio_actual  # pendiente máximos normalizada
        pl = coef_lc[0] / precio_actual  # pendiente mínimos normalizada

        techo_cuna = np.polyval(coef_hc, ventana_cuna - 1)
        suelo_cuna = np.polyval(coef_lc, ventana_cuna - 1)
        ancho_cuna = (techo_cuna - suelo_cuna) / precio_actual

        # Contar toques reales en cada línea de la cuña
        toques_t_cuna = sum(
            1 for i in x_cuna
            if abs(highs_cuna[i] - np.polyval(coef_hc, i)) / np.polyval(coef_hc, i) < 0.008
        )
        toques_s_cuna = sum(
            1 for i in x_cuna
            if abs(lows_cuna[i] - np.polyval(coef_lc, i)) / np.polyval(coef_lc, i) < 0.008
        )

        convergencia = abs(ph - pl)
        toques_ok = toques_t_cuna >= 2 and toques_s_cuna >= 2

        if 0.0008 < convergencia < 0.006 and ancho_cuna > 0.007 and toques_ok:
            info_c = f"({toques_t_cuna}t/{toques_s_cuna}t)"
            if ph > 0 and pl > 0 and ph < pl:
                # Cuña alcista: máximos suben menos que mínimos → convergencia hacia arriba
                # Señal bajista: ruptura habitual es hacia abajo
                resultado['patrones_short'].append(
                    f"Cuña alcista convergente {info_c} (ruptura bajista probable)")
                resultado['score_short'] += 1.5
            elif ph < 0 and pl < 0 and pl > ph:
                # Cuña bajista: mínimos bajan menos que máximos → convergencia hacia abajo
                # Señal alcista: ruptura habitual es hacia arriba
                resultado['patrones_long'].append(
                    f"Cuña bajista convergente {info_c} (ruptura alcista probable)")
                resultado['score_long'] += 1.5
    except Exception:
        pass

    # ── 8. BANDERA (impulso + consolidación en dirección contraria) ──
    # Lógica correcta:
    #   1. Detectar impulso fuerte buscando la ventana de mayor movimiento en las últimas 30 velas
    #   2. La consolidación POSTERIOR debe ir en dirección CONTRARIA al impulso
    #      (si el impulso fue alcista, la consolidación retrocede ligeramente = bandera bajando)
    #      (si el impulso fue bajista, la consolidación sube ligeramente = bandera subiendo)
    #   3. El rango de la consolidación debe ser < 40% del rango del impulso
    if n >= 25:
        try:
            # Buscar el mejor impulso en las últimas 30 velas (ventana deslizante de 5-10 velas)
            mejor_impulso_mov  = 0.0
            mejor_impulso_fin  = -1
            mejor_impulso_ini  = -1
            ventana_impulso    = 7   # tamaño del tramo de impulso
            ventana_consol     = 8   # tamaño del tramo de consolidación posterior
            min_inicio         = max(0, n - 30)

            for ini in range(min_inicio, n - ventana_impulso - ventana_consol):
                fin = ini + ventana_impulso
                if fin + ventana_consol > n:
                    break
                rango_imp = (np.max(h[ini:fin]) - np.min(l[ini:fin])) / np.min(l[ini:fin])
                if rango_imp > mejor_impulso_mov:
                    mejor_impulso_mov = rango_imp
                    mejor_impulso_ini = ini
                    mejor_impulso_fin = fin

            if mejor_impulso_mov > 0.025 and mejor_impulso_fin > 0:
                consol_ini = mejor_impulso_fin
                consol_fin = min(n, consol_ini + ventana_consol)

                imp_h  = h[mejor_impulso_ini:mejor_impulso_fin]
                imp_l  = l[mejor_impulso_ini:mejor_impulso_fin]
                imp_c  = c[mejor_impulso_ini:mejor_impulso_fin]
                con_h  = h[consol_ini:consol_fin]
                con_l  = l[consol_ini:consol_fin]
                con_c  = c[consol_ini:consol_fin]

                if len(con_c) >= 3:
                    rango_consol = (np.max(con_h) - np.min(con_l)) / np.min(con_l)
                    impulso_alc  = imp_c[-1] > imp_c[0]   # impulso fue alcista
                    impulso_baj  = imp_c[-1] < imp_c[0]   # impulso fue bajista

                    # Dirección de la consolidación (slope de los closes)
                    slope_consol = (float(con_c[-1]) - float(con_c[0])) / float(con_c[0])

                    # Bandera alcista: impulso ALCISTA + consolidación retrocediendo (slope negativo)
                    # Bandera bajista: impulso BAJISTA + consolidación recuperando (slope positivo)
                    consol_contraimpulso_alc = impulso_alc and slope_consol < -0.003
                    consol_contraimpulso_baj = impulso_baj and slope_consol > 0.003

                    if rango_consol < mejor_impulso_mov * 0.40:
                        if consol_contraimpulso_alc:
                            resultado['patrones_long'].append(
                                f"Bandera alcista (impulso +{mejor_impulso_mov*100:.1f}%, "
                                f"consol {slope_consol*100:+.1f}%)")
                            resultado['score_long'] += 1.5
                        elif consol_contraimpulso_baj:
                            resultado['patrones_short'].append(
                                f"Bandera bajista (impulso -{mejor_impulso_mov*100:.1f}%, "
                                f"consol {slope_consol*100:+.1f}%)")
                            resultado['score_short'] += 1.5
        except Exception:
            pass

    # ── CONFIRMACIÓN 4H ──
    # Si hay datos de 4H, multiplicamos el score si la tendencia 4H confirma
    if df4h is not None and len(df4h) >= 30:
        try:
            import pandas_ta as ta
            c4  = df4h['close']
            ema50_4  = ta.ema(c4, length=50)
            ema200_4 = ta.ema(c4, length=200)
            precio4  = float(c4.iloc[-1])
            e50      = float(ema50_4.iloc[-1])
            e200     = float(ema200_4.iloc[-1])

            tendencia4 = 'alcista' if precio4 > e50 > e200 else                          'bajista' if precio4 < e50 < e200 else 'lateral'

            if tendencia4 == 'bajista' and resultado['score_short'] > 0:
                resultado['score_short'] *= 1.3
                resultado['patrones_short'].append("4H confirma dirección bajista del patrón")
            elif tendencia4 == 'alcista' and resultado['score_long'] > 0:
                resultado['score_long'] *= 1.3
                resultado['patrones_long'].append("4H confirma dirección alcista del patrón")
        except Exception:
            pass

    # Redondear scores
    resultado['score_short'] = round(resultado['score_short'], 1)
    resultado['score_long']  = round(resultado['score_long'],  1)

    return resultado


# ──────────────────────────────────────────────
#  DIVERGENCIAS REALES (RSI Y MACD)
# ──────────────────────────────────────────────

def detectar_divergencias(df: pd.DataFrame, ventana: int = 40) -> Dict:
    """
    Detecta divergencias reales entre precio e indicadores buscando
    picos y valles locales — no comparando dos puntos fijos.

    Divergencia BAJISTA regular (señal de giro a la baja):
      · Precio hace máximo más ALTO, indicador hace máximo más BAJO
      · Indica que el movimiento alcista pierde fuerza interna

    Divergencia ALCISTA regular (señal de giro al alza):
      · Precio hace mínimo más BAJO, indicador hace mínimo más ALTO
      · Indica que el movimiento bajista pierde fuerza interna

    Divergencia OCULTA (señal de continuación):
      · Bajista oculta: precio max más bajo, indicador max más alto → continúa bajando
      · Alcista oculta: precio min más alto, indicador min más bajo → continúa subiendo

    Retorna dict con clasificación y fuerza (0.0 a 1.0) de cada divergencia.
    """
    resultado = {
        'div_bajista_rsi':    False, 'div_bajista_rsi_fuerza':    0.0,
        'div_alcista_rsi':    False, 'div_alcista_rsi_fuerza':    0.0,
        'div_bajista_macd':   False, 'div_bajista_macd_fuerza':   0.0,
        'div_alcista_macd':   False, 'div_alcista_macd_fuerza':   0.0,
        'div_bajista_oculta': False, 'div_alcista_oculta':        False,
        'convergencia_bajista': False, 'convergencia_alcista': False,
        'resumen': [],
    }

    n = min(ventana, len(df))
    if n < 10:
        return resultado

    sub      = df.iloc[-n:].reset_index(drop=True)
    precios  = sub['close'].values.astype(float)
    highs    = sub['high'].values.astype(float)
    lows     = sub['low'].values.astype(float)
    rsi_vals = sub['rsi_14'].values.astype(float) if 'rsi_14' in sub else None
    macd_vals = sub['macd_hist'].values.astype(float) if 'macd_hist' in sub else None

    def picos_locales(arr, radio=2):
        idx = []
        for i in range(radio, len(arr) - radio):
            if all(arr[i] > arr[i-j] for j in range(1, radio+1)) and                all(arr[i] > arr[i+j] for j in range(1, radio+1)):
                idx.append(i)
        return idx

    def valles_locales(arr, radio=2):
        idx = []
        for i in range(radio, len(arr) - radio):
            if all(arr[i] < arr[i-j] for j in range(1, radio+1)) and                all(arr[i] < arr[i+j] for j in range(1, radio+1)):
                idx.append(i)
        return idx

    def fuerza_div(v1, v2, escala):
        """Fuerza de la divergencia: qué tan distanciados están los dos máx/mín."""
        if escala == 0:
            return 0.0
        return float(np.clip(abs(v1 - v2) / escala, 0, 1))

    picos_precio  = picos_locales(highs)
    valles_precio = valles_locales(lows)

    # ── Divergencias en RSI ────────────────────────────────────────────
    if rsi_vals is not None and not np.all(np.isnan(rsi_vals)):
        picos_rsi  = picos_locales(np.nan_to_num(rsi_vals, nan=50))
        valles_rsi = valles_locales(np.nan_to_num(rsi_vals, nan=50))

        # Divergencia bajista regular: precio hace máx más alto, RSI máx más bajo
        if len(picos_precio) >= 2 and len(picos_rsi) >= 2:
            pp1, pp2 = picos_precio[-2], picos_precio[-1]
            # Buscar picos RSI cercanos a los picos de precio (±3 velas)
            rp_cerca = [r for r in picos_rsi if abs(r - pp2) <= 3]
            rp_prev  = [r for r in picos_rsi if abs(r - pp1) <= 4]
            if rp_cerca and rp_prev:
                rp1, rp2 = rp_prev[-1], rp_cerca[-1]
                precio_sube = highs[pp2] > highs[pp1]
                rsi_baja    = rsi_vals[rp2] < rsi_vals[rp1]
                if precio_sube and rsi_baja:
                    f = fuerza_div(rsi_vals[rp1], rsi_vals[rp2], 30)
                    resultado['div_bajista_rsi']       = True
                    resultado['div_bajista_rsi_fuerza'] = f
                    resultado['resumen'].append(f"Div bajista RSI (f={f:.2f})")
                # Divergencia bajista OCULTA: precio max más bajo, RSI max más alto
                elif not precio_sube and not rsi_baja:
                    resultado['div_bajista_oculta'] = True
                    resultado['resumen'].append("Div bajista RSI oculta (continuación)")

        # Divergencia alcista regular: precio hace mín más bajo, RSI mín más alto
        if len(valles_precio) >= 2 and len(valles_rsi) >= 2:
            vp1, vp2 = valles_precio[-2], valles_precio[-1]
            rv_cerca = [r for r in valles_rsi if abs(r - vp2) <= 3]
            rv_prev  = [r for r in valles_rsi if abs(r - vp1) <= 4]
            if rv_cerca and rv_prev:
                rv1, rv2 = rv_prev[-1], rv_cerca[-1]
                precio_baja = lows[vp2] < lows[vp1]
                rsi_sube    = rsi_vals[rv2] > rsi_vals[rv1]
                if precio_baja and rsi_sube:
                    f = fuerza_div(rsi_vals[rv1], rsi_vals[rv2], 30)
                    resultado['div_alcista_rsi']       = True
                    resultado['div_alcista_rsi_fuerza'] = f
                    resultado['resumen'].append(f"Div alcista RSI (f={f:.2f})")
                elif not precio_baja and not rsi_sube:
                    resultado['div_alcista_oculta'] = True
                    resultado['resumen'].append("Div alcista RSI oculta (continuación)")

    # ── Divergencias en MACD hist ──────────────────────────────────────
    if macd_vals is not None and not np.all(np.isnan(macd_vals)):
        macd_clean   = np.nan_to_num(macd_vals, nan=0.0)
        picos_macd   = picos_locales(macd_clean)
        valles_macd  = valles_locales(macd_clean)
        macd_escala  = float(np.nanstd(macd_vals)) * 2 if np.nanstd(macd_vals) > 0 else 1

        # Divergencia bajista MACD
        if len(picos_precio) >= 2 and len(picos_macd) >= 2:
            pp1, pp2 = picos_precio[-2], picos_precio[-1]
            mp_cerca = [m for m in picos_macd if abs(m - pp2) <= 3]
            mp_prev  = [m for m in picos_macd if abs(m - pp1) <= 4]
            if mp_cerca and mp_prev:
                mp1, mp2 = mp_prev[-1], mp_cerca[-1]
                if highs[pp2] > highs[pp1] and macd_clean[mp2] < macd_clean[mp1]:
                    f = fuerza_div(macd_clean[mp1], macd_clean[mp2], macd_escala)
                    resultado['div_bajista_macd']       = True
                    resultado['div_bajista_macd_fuerza'] = f
                    resultado['resumen'].append(f"Div bajista MACD (f={f:.2f})")

        # Divergencia alcista MACD
        if len(valles_precio) >= 2 and len(valles_macd) >= 2:
            vp1, vp2 = valles_precio[-2], valles_precio[-1]
            mv_cerca = [m for m in valles_macd if abs(m - vp2) <= 3]
            mv_prev  = [m for m in valles_macd if abs(m - vp1) <= 4]
            if mv_cerca and mv_prev:
                mv1, mv2 = mv_prev[-1], mv_cerca[-1]
                if lows[vp2] < lows[vp1] and macd_clean[mv2] > macd_clean[mv1]:
                    f = fuerza_div(macd_clean[mv1], macd_clean[mv2], macd_escala)
                    resultado['div_alcista_macd']       = True
                    resultado['div_alcista_macd_fuerza'] = f
                    resultado['resumen'].append(f"Div alcista MACD (f={f:.2f})")

        # ── Convergencia: MACD a punto de cruzar la línea 0 ──────────────
        # Si el hist lleva N velas acercándose a 0 desde abajo → convergencia alcista
        if len(macd_clean) >= 4:
            ult4 = macd_clean[-4:]
            if ult4[-1] < 0 and all(ult4[i] > ult4[i-1] for i in range(1, 4)):
                resultado['convergencia_alcista'] = True
                resultado['resumen'].append("MACD convergiendo alcista (cruce 0 próximo)")
            elif ult4[-1] > 0 and all(ult4[i] < ult4[i-1] for i in range(1, 4)):
                resultado['convergencia_bajista'] = True
                resultado['resumen'].append("MACD convergiendo bajista (cruce 0 próximo)")

    return resultado


# ──────────────────────────────────────────────
#  FUERZA DE NIVEL (SOPORTE / RESISTENCIA)
# ──────────────────────────────────────────────

def evaluar_fuerza_nivel(df: pd.DataFrame, precio: float, atr: float,
                          tipo: str = 'soporte') -> Dict:
    """
    Evalúa la fuerza de un nivel de soporte o resistencia cercano.

    Un nivel fuerte tiene:
      · Muchos toques (≥3 con exactitud razonable)
      · Toques separados en el tiempo (no todos seguidos)
      · Volumen alto en los toques (el mercado respetó el nivel)
      · Bounce real después de cada toque (el precio rebotó)

    Devuelve:
      · fuerza: 0.0 a 3.0 (0 = sin nivel, 3 = nivel muy fuerte)
      · n_toques: número de toques detectados
      · nivel: precio exacto del nivel
      · bloquear_contra: True si la fuerza es tan alta que contraoperar es peligroso
    """
    tolerancia = atr * 0.6   # toque = precio estuvo a menos de 0.6 ATR del nivel
    n = min(80, len(df))
    sub = df.iloc[-n:].reset_index(drop=True)

    # Usar lows para soporte, highs para resistencia
    if tipo == 'soporte':
        extremos = sub['low'].values.astype(float)
        rebote   = sub['close'].values.astype(float)
    else:
        extremos = sub['high'].values.astype(float)
        rebote   = sub['close'].values.astype(float)

    volumen  = sub['volume'].values.astype(float)
    vm_vol   = float(np.mean(volumen)) if np.mean(volumen) > 0 else 1.0

    # Identificar el nivel: precio del extremo más repetido en zona cercana
    candidato = precio  # empezamos en el precio actual
    if tipo == 'soporte':
        # el soporte más cercano bajo el precio actual
        extremos_validos = extremos[extremos < precio + tolerancia]
    else:
        extremos_validos = extremos[extremos > precio - tolerancia]

    if len(extremos_validos) == 0:
        return {'fuerza': 0.0, 'n_toques': 0, 'nivel': precio, 'bloquear_contra': False}

    candidato = float(np.median(extremos_validos[-10:])) if len(extremos_validos) >= 3 else float(extremos_validos[-1])

    # Contar toques reales al nivel candidato
    toques = []
    vol_toques = []
    bounce_ok  = []
    for i in range(len(extremos)):
        if abs(extremos[i] - candidato) <= tolerancia:
            # Verificar que hay un bounce real (close alejándose del nivel tras el toque)
            if tipo == 'soporte' and i + 2 < len(rebote):
                bounced = rebote[i+2] > candidato + tolerancia * 0.5
            elif tipo == 'resistencia' and i + 2 < len(rebote):
                bounced = rebote[i+2] < candidato - tolerancia * 0.5
            else:
                bounced = False
            toques.append(i)
            vol_toques.append(volumen[i] / vm_vol)
            bounce_ok.append(bounced)

    n_toques = len(toques)
    if n_toques == 0:
        return {'fuerza': 0.0, 'n_toques': 0, 'nivel': candidato, 'bloquear_contra': False}

    # ── Score de fuerza ───────────────────────────────────────────────
    fuerza = 0.0

    # a) Número de toques (más toques = más fuerte, con rendimiento decreciente)
    fuerza += min(n_toques * 0.5, 1.5)

    # b) Dispersión temporal (toques bien separados = nivel más fiable)
    if n_toques >= 2:
        separaciones = [toques[i+1] - toques[i] for i in range(len(toques)-1)]
        sep_media = np.mean(separaciones)
        if sep_media >= 8:    fuerza += 0.5   # muy separados
        elif sep_media >= 4:  fuerza += 0.25

    # c) Volumen en los toques
    vol_medio_toques = float(np.mean(vol_toques)) if vol_toques else 1.0
    if vol_medio_toques >= 1.5:  fuerza += 0.5   # volumen alto en los toques
    elif vol_medio_toques >= 1.0: fuerza += 0.25

    # d) Bounce confirmado
    pct_bounce = sum(bounce_ok) / n_toques if n_toques > 0 else 0
    fuerza += pct_bounce * 0.5

    fuerza = float(np.clip(fuerza, 0.0, 3.0))

    # Bloquear si el nivel es muy fuerte Y el precio está encima (soporte) o debajo (resistencia)
    dist_al_nivel = abs(precio - candidato) / precio
    en_nivel      = dist_al_nivel < atr / precio * 1.2  # dentro de 1.2 ATR del nivel
    bloquear      = fuerza >= 2.0 and en_nivel

    return {
        'fuerza':          round(fuerza, 2),
        'n_toques':        n_toques,
        'nivel':           round(candidato, 6),
        'bloquear_contra': bloquear,
        'vol_medio':       round(vol_medio_toques, 2),
        'pct_bounce':      round(pct_bounce, 2),
    }


# ──────────────────────────────────────────────
#  CONTEXTO GLOBAL DE LA SEÑAL
# ──────────────────────────────────────────────

def evaluar_contexto_señal(df: pd.DataFrame, ultima: pd.Series,
                            ctx4h: Dict, direccion: str,
                            divs: Dict, patrones: Dict) -> Dict:
    """
    Evalúa el contexto global de la señal y devuelve:
      · multiplicador_score: cuánto amplificar/reducir el score base
      · multiplicador_margen: cuánto amplificar el margen si la señal es muy clara
      · bloquear: True si hay una razón estructural para no entrar
      · razon_bloqueo: explicación si se bloquea
      · nivel_confianza: 'bajo', 'normal', 'alto', 'maximo'
      · contexto_log: lista de strings para el log

    El multiplicador de margen puede llegar a 1.4× si TODO está muy claro:
      · Divergencia real + confirmación MACD + patrón + 4H alineado
    Pero nunca más de 1.4× para mantener la gestión de riesgo.
    """
    precio  = float(ultima.get('close', 0))
    atr     = float(ultima.get('atr_14', precio * 0.01))
    atr_pct = atr / precio if precio > 0 else 0.012

    mult_score  = 1.0
    mult_margen = 1.0
    bloquear    = False
    razon_bloqueo = ''
    contexto_log  = []

    # ── 1. Fuerza del nivel opuesto (contra quién va la operación) ────
    # Si hay un soporte fuerte y queremos hacer short, evaluamos el riesgo
    tipo_nivel_contra = 'soporte' if direccion == 'short' else 'resistencia'
    nivel_contra = evaluar_fuerza_nivel(df, precio, atr, tipo=tipo_nivel_contra)

    if nivel_contra['bloquear_contra']:
        # Nivel muy fuerte en nuestra contra — solo bloqueamos si además
        # NO hay señales muy claras que justifiquen romperlo.
        señales_fuertes = (
            divs.get('div_bajista_rsi', False) and direccion == 'short' or
            divs.get('div_alcista_rsi', False) and direccion == 'long'
        )
        señales_macd = (
            divs.get('convergencia_bajista', False) and direccion == 'short' or
            divs.get('convergencia_alcista', False) and direccion == 'long'
        )
        if not señales_fuertes and not señales_macd:
            bloquear = True
            razon_bloqueo = (f"Nivel {tipo_nivel_contra} fuerte en contra "
                             f"(fuerza={nivel_contra['fuerza']:.1f}, "
                             f"{nivel_contra['n_toques']} toques) sin señales que lo rompan")
        else:
            # Hay señales pero el nivel es fuerte → penalización de score, no bloqueo
            mult_score *= 0.7
            contexto_log.append(f"⚠️ {tipo_nivel_contra.capitalize()} fuerte "
                                 f"(f={nivel_contra['fuerza']:.1f}) pero con señales de ruptura")
    elif nivel_contra['fuerza'] >= 1.0:
        # Nivel moderado → pequeña penalización
        mult_score *= 0.85
        contexto_log.append(f"Nivel {tipo_nivel_contra} moderado (f={nivel_contra['fuerza']:.1f})")

    if bloquear:
        return {
            'multiplicador_score': 0.0,
            'multiplicador_margen': 1.0,
            'bloquear': True,
            'razon_bloqueo': razon_bloqueo,
            'nivel_confianza': 'bloqueado',
            'contexto_log': contexto_log,
            'nivel_contra': nivel_contra,
        }

    # ── 2. Divergencias — amplifican si están a favor, penalizan si están en contra ──
    if direccion == 'short':
        div_favor   = divs.get('div_bajista_rsi', False) or divs.get('div_bajista_macd', False)
        div_contra  = divs.get('div_alcista_rsi', False) or divs.get('div_alcista_macd', False)
        div_fuerza  = max(divs.get('div_bajista_rsi_fuerza', 0),
                          divs.get('div_bajista_macd_fuerza', 0))
        conv_favor  = divs.get('convergencia_bajista', False)
        conv_contra = divs.get('convergencia_alcista', False)
    else:
        div_favor   = divs.get('div_alcista_rsi', False) or divs.get('div_alcista_macd', False)
        div_contra  = divs.get('div_bajista_rsi', False) or divs.get('div_bajista_macd', False)
        div_fuerza  = max(divs.get('div_alcista_rsi_fuerza', 0),
                          divs.get('div_alcista_macd_fuerza', 0))
        conv_favor  = divs.get('convergencia_alcista', False)
        conv_contra = divs.get('convergencia_bajista', False)

    if div_favor:
        bonus = 0.15 + div_fuerza * 0.15  # entre +15% y +30% de amplificación
        mult_score  *= (1 + bonus)
        mult_margen *= min(1 + bonus * 0.5, 1.20)  # el margen sube más suavemente
        contexto_log.append(f"✅ Divergencia real a favor (fuerza={div_fuerza:.2f})")

    if div_contra:
        # Divergencia en contra: penalización importante pero no bloqueo
        mult_score  *= 0.65
        contexto_log.append("⚠️ Divergencia real EN CONTRA de la dirección")

    if conv_favor:
        mult_score  *= 1.12
        mult_margen *= 1.08
        contexto_log.append("✅ MACD convergiendo a favor (cruce próximo)")

    if conv_contra:
        mult_score  *= 0.80
        contexto_log.append("⚠️ MACD convergiendo EN CONTRA")

    # ── 3. Alineación 4H ──────────────────────────────────────────────
    t4h = ctx4h.get('tendencia', 'lateral')
    m4h = ctx4h.get('momentum', 'neutral')
    if direccion == 'short' and t4h == 'bajista':
        bonus_4h = 1.15 if 'acelerando' in m4h else 1.08
        mult_score  *= bonus_4h
        mult_margen *= min(bonus_4h * 0.6 + 0.4, 1.12)
        contexto_log.append(f"✅ 4H bajista ({m4h})")
    elif direccion == 'long' and t4h == 'alcista':
        bonus_4h = 1.15 if 'acelerando' in m4h else 1.08
        mult_score  *= bonus_4h
        mult_margen *= min(bonus_4h * 0.6 + 0.4, 1.12)
        contexto_log.append(f"✅ 4H alcista ({m4h})")
    elif (direccion == 'short' and t4h == 'alcista') or (direccion == 'long' and t4h == 'bajista'):
        mult_score  *= 0.80
        contexto_log.append(f"⚠️ 4H en contra ({t4h})")

    # ── 4. Fuerza del nivel a favor ───────────────────────────────────
    tipo_nivel_favor = 'resistencia' if direccion == 'short' else 'soporte'
    nivel_favor = evaluar_fuerza_nivel(df, precio, atr, tipo=tipo_nivel_favor)
    if nivel_favor['fuerza'] >= 2.0:
        mult_score  *= 1.10
        mult_margen *= 1.05
        contexto_log.append(f"✅ {tipo_nivel_favor.capitalize()} fuerte a favor "
                             f"(f={nivel_favor['fuerza']:.1f})")
    elif nivel_favor['fuerza'] >= 1.0:
        mult_score *= 1.05

    # ── 5. Confianza acumulada y multiplicador de margen ─────────────
    # mult_margen solo sube de 1.0 si hay varias señales muy claras.
    # Tope duro en 1.40× para no sobreexponer el capital.
    mult_margen = float(np.clip(mult_margen, 0.60, 1.40))
    mult_score  = float(np.clip(mult_score, 0.20, 2.50))

    # Nivel de confianza global
    if mult_margen >= 1.25 and mult_score >= 1.30:
        nivel_confianza = 'maximo'
    elif mult_margen >= 1.10 or mult_score >= 1.15:
        nivel_confianza = 'alto'
    elif mult_score <= 0.70:
        nivel_confianza = 'bajo'
    else:
        nivel_confianza = 'normal'

    return {
        'multiplicador_score':  round(mult_score, 3),
        'multiplicador_margen': round(mult_margen, 3),
        'bloquear':             False,
        'razon_bloqueo':        '',
        'nivel_confianza':      nivel_confianza,
        'contexto_log':         contexto_log,
        'nivel_contra':         nivel_contra,
        'nivel_favor':          nivel_favor,
    }

# ──────────────────────────────────────────────
#  SCORE PREDICTIVO — NÚCLEO DE LA ESTRATEGIA
# ──────────────────────────────────────────────

def score_predictivo_short(df: pd.DataFrame, ultima: pd.Series,
                            ctx4h: Dict, patrones: Dict = None,
                            divs: Dict = None,
                            prob_ct: float = None) -> Tuple[float, List[str]]:
    """
    Score predictivo para SHORT con pesos contextuales.
    Devuelve score float (ya multiplicado por contexto) y razones.

    prob_ct: probabilidad del modelo en velas 1H recientes (corto plazo 4-8h).
             Si se pasa, actúa como indicador adicional igual que RSI o MACD.
    """
    score   = 0.0
    razones = []
    c       = float(ultima.get('close', 0))

    # 1. RSI sobrecomprado (umbral dinámico al régimen)
    rsi = float(ultima.get('rsi_14', 50))
    rsi_serie = df['rsi_14'] if 'rsi_14' in df.columns else None
    umbral_sc, _ = calcular_rsi_umbral(rsi_serie, 'short')
    if rsi >= umbral_sc:
        score += 1.0
        razones.append(f"RSI={rsi:.0f} sobrecomprado (umbral={umbral_sc:.0f})")
    elif rsi >= umbral_sc - 5:
        score += 0.5

    # 2. Divergencias reales (sistema nuevo — peso mayor que RSI simple)
    if divs:
        if divs.get('div_bajista_rsi'):
            f = divs.get('div_bajista_rsi_fuerza', 0.5)
            score += 1.0 + f * 0.5   # entre 1.0 y 1.5 según fuerza
            razones.append(f"Divergencia bajista RSI real (fuerza={f:.2f})")
        if divs.get('div_bajista_macd'):
            f = divs.get('div_bajista_macd_fuerza', 0.5)
            score += 1.0 + f * 0.5
            razones.append(f"Divergencia bajista MACD real (fuerza={f:.2f})")
        if divs.get('convergencia_bajista'):
            score += 0.8
            razones.append("MACD hist convergiendo bajista (cruce 0 próximo)")
        # Divergencias alcistas EN CONTRA → penalización
        if divs.get('div_alcista_rsi') or divs.get('div_alcista_macd'):
            score -= 1.5
            razones.append("⚠️ Divergencia alcista real en contra del short")
        if divs.get('convergencia_alcista'):
            score -= 0.6
            razones.append("⚠️ MACD convergiendo alcista (en contra)")
    else:
        # Fallback: divergencia del sistema antiguo (comparación 5 velas)
        if int(ultima.get('divergencia_bajista_rsi', 0)) == 1:
            score += 0.8
            razones.append("Div bajista RSI (señal básica)")

    # 3. Stochastic sobrecomprado y girando
    stk = float(ultima.get('stoch_k', 50))
    std_val = float(ultima.get('stoch_d', 50))
    if stk >= 78 and stk < std_val + 2:
        score += 1.0
        razones.append(f"Stoch={stk:.0f} sobrecomprado y girando")
    elif stk >= 70:
        score += 0.5

    # 4. MACD hist decelerando
    mh = df['macd_hist'].dropna()
    if len(mh) >= 3:
        mh_act, mh_prev, mh_ant = float(mh.iloc[-1]), float(mh.iloc[-2]), float(mh.iloc[-3])
        if mh_act > 0 and mh_act < mh_prev < mh_ant:
            score += 1.5
            razones.append("MACD hist 2 velas decelerando (pico doble)")
        elif mh_act > 0 and mh_act < mh_prev:
            score += 1.0
            razones.append("MACD hist decelerando (pico)")

    # 5. Precio en resistencia / Fibonacci
    dist_res  = float(ultima.get('dist_resistencia_pct', 999))
    cerca_618 = int(ultima.get('cerca_fib_618', 0))
    bb_pos    = float(ultima.get('bb_posicion', 0.5))
    if dist_res < 0.5:
        score += 1.0; razones.append(f"Precio en resistencia ({dist_res:.2f}%)")
    elif cerca_618 == 1:
        score += 1.0; razones.append("Fibonacci 0.618")
    elif bb_pos > 0.90:
        score += 0.5; razones.append(f"BB superior ({bb_pos:.2f})")

    # 6. Volumen decreciente en subida
    vr = df['volumen_ratio'].dropna()
    precio_sube = c > float(df['close'].iloc[-3]) if len(df) >= 3 else False
    if len(vr) >= 3 and precio_sube:
        if float(vr.iloc[-1]) < 0.80 and float(vr.iloc[-2]) < 0.80:
            score += 1.0; razones.append("Volumen bajo en subida")

    # 7. Contexto 4H
    if ctx4h.get('disponible'):
        t4, m4, r4 = ctx4h['tendencia'], ctx4h['momentum'], ctx4h.get('rsi', 50)
        if t4 == 'bajista':
            score += 2.0; razones.append("4H BAJISTA")
        elif t4 == 'lateral' and 'bajista' in m4:
            score += 1.0; razones.append("4H lateral momentum bajista")
        if r4 >= 65:
            score += 0.5; razones.append(f"RSI 4H={r4:.0f} sobrecomprado")
        if m4 == 'alcista_desacelerando':
            score += 1.0; razones.append("4H alcista desacelerando")

    # 8. Régimen 1H
    reg = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', np.nan)) else 0
    if reg <= -1:   score += 0.5
    elif reg >= 2:  score -= 0.5

    # 9. Patrones chartistas (con contexto de nivel)
    if patrones and patrones['score_short'] > 0:
        score += patrones['score_short']
        for p in patrones['patrones_short']:
            razones.append(f"📐 {p}")

    # 10. IA corto plazo (modelo 1H — predicción próximas 4-8h)
    # Actúa como indicador adicional en el score, igual que RSI o MACD.
    if prob_ct is not None:
        if prob_ct > IA_CT_FUERTE:
            score += 1.5
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → señal FUERTE a favor short")
        elif prob_ct > IA_CT_DEBIL:
            score += 0.5
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → señal débil a favor short")
        elif prob_ct < IA_CT_CONTRA:
            score -= 1.0
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → señal EN CONTRA del short")
        else:
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → neutral")

    return score, razones


def score_predictivo_long(df: pd.DataFrame, ultima: pd.Series,
                           ctx4h: Dict, patrones: Dict = None,
                           divs: Dict = None,
                           prob_ct: float = None) -> Tuple[float, List[str]]:
    """
    Score predictivo para LONG con pesos contextuales.
    Simétrico al short. Devuelve score float y razones.

    prob_ct: probabilidad del modelo en velas 1H recientes (corto plazo 4-8h).
             Si se pasa, actúa como indicador adicional igual que RSI o MACD.
    """
    score   = 0.0
    razones = []

    # 1. RSI sobrevendido (umbral dinámico)
    rsi = float(ultima.get('rsi_14', 50))
    rsi_serie = df['rsi_14'] if 'rsi_14' in df.columns else None
    _, umbral_sv = calcular_rsi_umbral(rsi_serie, 'long')
    if rsi <= umbral_sv:
        score += 1.0; razones.append(f"RSI={rsi:.0f} sobrevendido (umbral={umbral_sv:.0f})")
    elif rsi <= umbral_sv + 5:
        score += 0.5

    # 2. Divergencias reales
    if divs:
        if divs.get('div_alcista_rsi'):
            f = divs.get('div_alcista_rsi_fuerza', 0.5)
            score += 1.0 + f * 0.5
            razones.append(f"Divergencia alcista RSI real (fuerza={f:.2f})")
        if divs.get('div_alcista_macd'):
            f = divs.get('div_alcista_macd_fuerza', 0.5)
            score += 1.0 + f * 0.5
            razones.append(f"Divergencia alcista MACD real (fuerza={f:.2f})")
        if divs.get('convergencia_alcista'):
            score += 0.8; razones.append("MACD convergiendo alcista (cruce 0 próximo)")
        # Divergencias bajistas EN CONTRA
        if divs.get('div_bajista_rsi') or divs.get('div_bajista_macd'):
            score -= 1.5; razones.append("⚠️ Divergencia bajista real en contra del long")
        if divs.get('convergencia_bajista'):
            score -= 0.6; razones.append("⚠️ MACD convergiendo bajista (en contra)")
    else:
        if int(ultima.get('divergencia_alcista_rsi', 0)) == 1:
            score += 0.8; razones.append("Div alcista RSI (señal básica)")

    # 3. Stochastic sobrevendido y girando
    stk = float(ultima.get('stoch_k', 50))
    std_val = float(ultima.get('stoch_d', 50))
    if stk <= 22 and stk > std_val - 2:
        score += 1.0; razones.append(f"Stoch={stk:.0f} sobrevendido y girando")
    elif stk <= 30:
        score += 0.5

    # 4. MACD hist mejorando desde suelo
    mh = df['macd_hist'].dropna()
    if len(mh) >= 3:
        mh_act, mh_prev, mh_ant = float(mh.iloc[-1]), float(mh.iloc[-2]), float(mh.iloc[-3])
        if mh_act < 0 and mh_act > mh_prev > mh_ant:
            score += 1.5; razones.append("MACD hist 2 velas mejorando (suelo doble)")
        elif mh_act < 0 and mh_act > mh_prev:
            score += 1.0; razones.append("MACD hist mejorando desde suelo")

    # 5. Precio en soporte / Fibonacci
    dist_sop  = float(ultima.get('dist_soporte_pct', 999))
    cerca_382 = int(ultima.get('cerca_fib_382', 0))
    cerca_500 = int(ultima.get('cerca_fib_500', 0))
    bb_pos    = float(ultima.get('bb_posicion', 0.5))
    if dist_sop < 0.5:
        score += 1.0; razones.append(f"Precio en soporte ({dist_sop:.2f}%)")
    elif cerca_382 == 1 or cerca_500 == 1:
        score += 1.0; razones.append("Fibonacci 0.382/0.5")
    elif bb_pos < 0.10:
        score += 0.5; razones.append(f"BB inferior ({bb_pos:.2f})")

    # 6. Volumen decreciente en bajada
    vr = df['volumen_ratio'].dropna()
    precio_baja = float(ultima.get('close', 0)) < float(df['close'].iloc[-3]) if len(df) >= 3 else False
    if len(vr) >= 3 and precio_baja:
        if float(vr.iloc[-1]) < 0.80 and float(vr.iloc[-2]) < 0.80:
            score += 1.0; razones.append("Volumen bajo en bajada (vendedores agotados)")

    # 7. Contexto 4H
    if ctx4h.get('disponible'):
        t4, m4, r4 = ctx4h['tendencia'], ctx4h['momentum'], ctx4h.get('rsi', 50)
        if t4 == 'alcista':
            score += 2.0; razones.append("4H ALCISTA")
        elif t4 == 'lateral' and 'alcista' in m4:
            score += 1.0; razones.append("4H lateral momentum alcista")
        if r4 <= 35:
            score += 0.5; razones.append(f"RSI 4H={r4:.0f} sobrevendido")
        if m4 == 'bajista_desacelerando':
            score += 1.0; razones.append("4H bajista desacelerando")

    # 8. Régimen 1H
    reg = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', np.nan)) else 0
    if reg >= 1:    score += 0.5
    elif reg <= -2: score -= 0.5

    # 9. Patrones chartistas
    if patrones and patrones['score_long'] > 0:
        score += patrones['score_long']
        for p in patrones['patrones_long']:
            razones.append(f"📐 {p}")

    # 10. IA corto plazo (modelo 1H — predicción próximas 4-8h)
    # Actúa como indicador adicional en el score, igual que RSI o MACD.
    if prob_ct is not None:
        if prob_ct > IA_CT_FUERTE:
            score += 1.5
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → señal FUERTE a favor long")
        elif prob_ct > IA_CT_DEBIL:
            score += 0.5
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → señal débil a favor long")
        elif prob_ct < IA_CT_CONTRA:
            score -= 1.0
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → señal EN CONTRA del long")
        else:
            razones.append(f"🤖 IA corto plazo={prob_ct:.3f} → neutral")

    return score, razones
# ──────────────────────────────────────────────

def calcular_calidad_score(ultima: pd.Series, ctx4h: Dict,
                            divs: Dict, direccion: str) -> Tuple[int, List[str]]:
    """
    Calcula la puntuación de calidad de la señal (0..10+) para asignar el nivel.
    Independiente del score predictivo — éste mide la confluencia de factores
    de calidad externos que justifican mayor o menor apalancamiento.

    Puntos positivos (señales que refuerzan la entrada):
      +3  4H alineado con la dirección
      +2  MACD histograma claramente a favor (mismo signo y creciendo)
      +2  Divergencia confirmada RSI o MACD a favor
      +1  ADX > 20 (tendencia real en curso)
      +1  RSI en zona extrema (>65 para short, <35 para long)
      +1  Nivel S/R con 4+ toques en 4H

    Puntos negativos (contradicciones que reducen el nivel):
      -2  MACD convergiendo en dirección contraria
      -2  4H en contra de la dirección
      -1  ADX < 12 (mercado lateral sin tendencia)

    Niveles resultantes:
      0-2 → nivel1 (x4)  |  3-4 → nivel2 (x12)
      5-6 → nivel3 (x20) |  7+  → nivel4 (x30)
    """
    pts   = 0
    notas = []

    # ── 4H alineado (+3) o en contra (-2) ──
    t4h = ctx4h.get('tendencia', 'lateral')
    if (direccion == 'short' and t4h == 'bajista') or (direccion == 'long' and t4h == 'alcista'):
        pts += 3; notas.append(f"+3 4H alineado ({t4h})")
    elif (direccion == 'short' and t4h == 'alcista') or (direccion == 'long' and t4h == 'bajista'):
        pts -= 2; notas.append(f"-2 4H en contra ({t4h})")

    # ── MACD histograma (+2 a favor, -2 en contra) ──
    mh_val  = float(ultima.get('macd_hist', 0))
    mh_prev = float(ultima.get('macd_hist_prev', mh_val))  # calculado si existe
    if direccion == 'short':
        if mh_val < 0 and mh_val <= mh_prev:  # negativo y decreciendo (más bajista)
            pts += 2; notas.append("+2 MACD bajista y acelerando")
        elif mh_val > 0 and mh_val > mh_prev:  # positivo y subiendo (en contra)
            pts -= 2; notas.append("-2 MACD alcista en contra")
    else:
        if mh_val > 0 and mh_val >= mh_prev:  # positivo y creciendo (más alcista)
            pts += 2; notas.append("+2 MACD alcista y acelerando")
        elif mh_val < 0 and mh_val < mh_prev:  # negativo y cayendo (en contra)
            pts -= 2; notas.append("-2 MACD bajista en contra")

    # ── Divergencias confirmadas (+2) ──
    if divs:
        div_favor = (
            (direccion == 'short' and (divs.get('div_bajista_rsi') or divs.get('div_bajista_macd'))) or
            (direccion == 'long'  and (divs.get('div_alcista_rsi') or divs.get('div_alcista_macd')))
        )
        if div_favor:
            pts += 2; notas.append("+2 divergencia confirmada a favor")

    # ── ADX (+1 si > 20, -1 si < 12) ──
    adx = float(ultima.get('adx', 0))
    if adx > 20:
        pts += 1; notas.append(f"+1 ADX={adx:.0f} tendencia fuerte")
    elif adx < 12:
        pts -= 1; notas.append(f"-1 ADX={adx:.0f} mercado sin tendencia")

    # ── RSI en zona extrema (+1) ──
    rsi = float(ultima.get('rsi_14', 50))
    if (direccion == 'short' and rsi > 65) or (direccion == 'long' and rsi < 35):
        pts += 1; notas.append(f"+1 RSI={rsi:.0f} zona extrema")

    # ── Nivel S/R fuerte en 4H (+1) ──
    # Si el evaluador de contexto detectó un nivel de fuerza alta cerca
    adx4h = ctx4h.get('adx', 0)
    if adx4h > 15 and ctx4h.get('disponible'):
        pts += 1; notas.append("+1 nivel S/R confirmado en 4H")

    return pts, notas


def asignar_nivel(pts: int) -> Tuple[str, dict]:
    """Asigna nivel y config según la puntuación de calidad."""
    if pts >= 7:
        return 'nivel4', TRADE_CONFIG['nivel4']
    if pts >= 5:
        return 'nivel3', TRADE_CONFIG['nivel3']
    if pts >= 3:
        return 'nivel2', TRADE_CONFIG['nivel2']
    return 'nivel1', TRADE_CONFIG['nivel1']


def clasificar_señal(prob: float, score_adj: float, prob_minimo: float) -> bool:
    """Devuelve True si la señal supera los umbrales mínimos para abrir."""
    min_score = score_minimo(prob, 'short')  # simétrico para ambas direcciones
    return prob >= prob_minimo and score_adj >= min_score


# ──────────────────────────────────────────────
#  MODELO
# ──────────────────────────────────────────────

def añadir_features_neutras(df: pd.DataFrame, features: list) -> pd.DataFrame:
    neutros = {
        'fear_greed': 0, 'fear_greed_num': 0, 'fear_greed_cambio': 0,
        'mcap_btc': 0, 'mcap_btc_log': 0, 'mcap_btc_cambio_7d': 0,
        'correlacion_btc': 1.0, 'descorrelacionado': 0,
        'tendencia_4h': 0, 'tendencia_diario': 0, 'tendencia_semanal': 0,
        'rsi_4h': 50, 'rsi_diario': 50, 'rsi_semanal': 50,
        'ema50_4h': 0, 'ema50_diario': 0, 'ema50_semanal': 0,
        'alineacion_timeframes': 0, 'señal_multitf_fuerte': 0,
        'patron_doji': 0, 'patron_martillo': 0, 'patron_estrella_fugaz': 0,
        'patron_marubozu_alcista': 0, 'patron_marubozu_bajista': 0,
        'patron_engulfing_alcista': 0, 'patron_engulfing_bajista': 0,
        'patron_harami_alcista': 0, 'patron_morning_star': 0,
        'patron_evening_star': 0, 'patron_tres_soldados': 0,
        'patron_tres_cuervos': 0, 'señal_velas': 0,
        'patron_doble_suelo': 0, 'patron_doble_techo': 0,
        'patron_hch': 0, 'patron_hch_invertido': 0,
        'patron_triangulo': 0, 'patron_flag': 0, 'patron_cuña': 0,
        'en_canal': 0, 'posicion_canal': 0.5, 'señal_patrones_graficos': 0,
        'ichi_tenkan': 0, 'ichi_kijun': 0, 'ichi_senkou_a': 0,
        'ichi_senkou_b': 0, 'ichi_chikou': 0,
    }
    for f in features:
        if f not in df.columns:
            df[f] = neutros.get(f, 0)
    return df


def cargar_submodelo(simbolo: str, tipo: str) -> Optional[Dict]:
    nombre = simbolo.replace('/', '_')
    dir_m  = os.path.join(DIR_MODELOS, nombre, tipo)
    if not os.path.exists(os.path.join(dir_m, 'xgboost.pkl')):
        return None
    try:
        from models.ensemble import EnsembleTrading
        sub = {
            'xgboost':  joblib.load(os.path.join(dir_m, 'xgboost.pkl')),
            'scaler':   joblib.load(os.path.join(dir_m, 'scaler.pkl')),
            'features': joblib.load(os.path.join(dir_m, 'features.pkl')),
            'lstm': None,
        }
        ens = EnsembleTrading()
        ens.cargar(dir_m)
        sub['ensemble'] = ens
        if nombre not in SOLO_XGBOOST:
            try:
                from models.lstm_model import cargar_lstm
                lstm = cargar_lstm(dir_m)
                if lstm:
                    sub['lstm'] = lstm
            except:
                pass
        return sub
    except Exception as e:
        print(f"  ❌ {nombre}/{tipo}: {e}")
        return None


def predecir(df: pd.DataFrame, sub: Dict) -> float:
    df = añadir_features_neutras(df.copy(), sub['features'])
    fd = [f for f in sub['features'] if f in df.columns]
    X  = sub['scaler'].transform(df[fd].values)
    np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
    px = sub['xgboost'].predict_proba(X)[:, 1]
    pl = None
    if sub['lstm']:
        try:
            from models.lstm_model import predecir_lstm
            from models.preparar_datos import crear_secuencias
            Xs, _ = crear_secuencias(X, np.zeros(len(X)), 48)
            if len(Xs) > 0:
                r  = predecir_lstm(sub['lstm'], Xs)
                pd_ = np.full(len(px) - len(r), 0.5)
                pl  = np.concatenate([pd_, r])
        except:
            pass
    return float(sub['ensemble'].predecir_proba(px, pl)[-1])


# ──────────────────────────────────────────────
#  ADAPTIVE SIZING
# ──────────────────────────────────────────────

def actualizar_adaptive(estado: Dict, pnl: float):
    h = estado.setdefault('historial_pnl', [])
    h.append(1.0 if pnl > 0 else 0.0)
    if len(h) > ADAPTIVE_VENTANA:
        h.pop(0)
    if len(h) >= ADAPTIVE_VENTANA:
        wr = sum(h) / len(h)
        af = estado.get('adaptive_factor', 1.0)
        tl = estado.get('adaptive_trades_left', 0)
        if wr < ADAPTIVE_WR_BAJO and af != ADAPTIVE_FACTOR_C:
            estado['adaptive_factor']      = ADAPTIVE_FACTOR_C
            estado['adaptive_trades_left'] = ADAPTIVE_DURACION
            telegram(f"⚠️ <b>{VERSION} MODO CONSERVADOR</b>\nWR-20: {wr*100:.0f}%")
        elif wr > ADAPTIVE_WR_ALTO:
            estado['adaptive_factor'] = ADAPTIVE_FACTOR_A
        elif tl > 0:
            estado['adaptive_trades_left'] = tl - 1
            if tl - 1 == 0:
                estado['adaptive_factor'] = 1.0
        elif af == ADAPTIVE_FACTOR_C:
            estado['adaptive_factor'] = 1.0


# ──────────────────────────────────────────────
#  MOTOR DE SL/TP BASADO EN NIVELES REALES
# ──────────────────────────────────────────────

def _niveles_mercado(df: pd.DataFrame, ultima: pd.Series, precio: float, atr: float) -> Dict:
    """
    Extrae todos los niveles de mercado relevantes y los ordena
    en resistencias (por encima del precio) y soportes (por debajo).
    Combina: EMAs, Bollinger, Fibonacci, S/R locales de picos/valles.
    Devuelve listas ordenadas de menor a mayor distancia al precio.
    """
    candidatos_res = []  # niveles por encima del precio
    candidatos_sop = []  # niveles por debajo del precio

    def añadir(nivel, etiqueta, lista_res, lista_sop, zona=0.0015):
        if pd.isna(nivel) or nivel <= 0:
            return
        dist = (nivel - precio) / precio
        if dist > zona:
            lista_res.append((nivel, etiqueta, dist))
        elif dist < -zona:
            lista_sop.append((nivel, etiqueta, -dist))

    # Medias móviles
    for p in [9, 21, 50, 100, 200]:
        v = float(ultima.get(f'ema_{p}', 0))
        añadir(v, f'EMA{p}', candidatos_res, candidatos_sop)
    for p in [20, 50, 200]:
        v = float(ultima.get(f'sma_{p}', 0))
        añadir(v, f'SMA{p}', candidatos_res, candidatos_sop)

    # Bollinger Bands
    añadir(float(ultima.get('bb_upper', 0)), 'BB_sup', candidatos_res, candidatos_sop)
    añadir(float(ultima.get('bb_lower', 0)), 'BB_inf', candidatos_res, candidatos_sop)

    # Fibonacci
    for nv in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
        v = float(ultima.get(nv, 0))
        añadir(v, nv.upper(), candidatos_res, candidatos_sop)

    # Máximo y mínimo de últimas 50 y 20 velas
    h50 = float(df['high'].rolling(50).max().iloc[-1])
    l50 = float(df['low'].rolling(50).min().iloc[-1])
    h20 = float(df['high'].rolling(20).max().iloc[-1])
    l20 = float(df['low'].rolling(20).min().iloc[-1])
    añadir(h50, 'MAX50', candidatos_res, candidatos_sop)
    añadir(l50, 'MIN50', candidatos_res, candidatos_sop)
    añadir(h20, 'MAX20', candidatos_res, candidatos_sop)
    añadir(l20, 'MIN20', candidatos_res, candidatos_sop)

    # Picos/valles locales como S/R (últimas 40 velas, ya calculados en detect_patrones
    # pero los recalculamos aquí de forma ligera para no depender del orden de llamadas)
    highs = df['high'].values[-40:].astype(float)
    lows  = df['low'].values[-40:].astype(float)
    for i in range(2, len(highs) - 2):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2]
                and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            añadir(highs[i], f'pico_local', candidatos_res, candidatos_sop, zona=0.001)
        if (lows[i] < lows[i-1] and lows[i] < lows[i-2]
                and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
            añadir(lows[i], f'valle_local', candidatos_res, candidatos_sop, zona=0.001)

    # Ordenar por distancia al precio (menor distancia primero)
    candidatos_res.sort(key=lambda x: x[2])
    candidatos_sop.sort(key=lambda x: x[2])

    return {'resistencias': candidatos_res, 'soportes': candidatos_sop}


def calcular_sl_tp_niveles(precio: float, atr: float, cfg: dict,
                            direccion: str, niveles: Dict, ctx4h: Dict) -> Dict:
    """
    Calcula SL y TP en precio absoluto usando niveles reales de mercado.

    El SL se coloca SIEMPRE detrás de la estructura de mercado más cercana:
      · SHORT: SL por encima de la resistencia más próxima (precio ya subió ahí
               y rebotó — es una barrera probada). Se añade BUFFER_NIVEL_PCT para
               que el ruido no lo toque antes de que el precio realmente rompa.
      · LONG:  SL por debajo del soporte más próximo (simétrico).

    Si no hay nivel S/R en rango útil se usa ATR × sl_mult como fallback, pero
    siempre respetando el mínimo SL_ATR_MIN_MULT para evitar wicks de ruido.

    El RR mínimo se lee del cfg del nivel asignado (nivel1=1.6 … nivel4=2.2),
    no del global RR_MINIMO, para que el nivel1 pueda operar con RRs menores.
    """
    resistencias = niveles['resistencias']
    soportes     = niveles['soportes']

    sl_min_dist = atr * SL_ATR_MIN_MULT
    sl_max_dist = atr * SL_ATR_MAX_MULT
    tp_max_dist = atr * TP_ATR_MAX_MULT
    buffer      = precio * BUFFER_NIVEL_PCT
    rr_min      = cfg.get('rr_min', RR_MINIMO)

    es_swing = (
        ctx4h.get('disponible', False) and
        ((direccion == 'short' and ctx4h.get('tendencia') == 'bajista') or
         (direccion == 'long'  and ctx4h.get('tendencia') == 'alcista')) and
        ctx4h.get('momentum', '') in ('bajista_acelerando', 'alcista_acelerando',
                                       'bajista_desacelerando', 'alcista_desacelerando')
    )

    if direccion == 'short':
        # ── SL: por encima de la resistencia más cercana ──────────────────
        # Buscamos la resistencia más próxima que esté lo suficientemente lejos
        # para no ser tocada por ruido normal (>= SL_ATR_MIN_MULT×ATR).
        sl_precio, sl_etiqueta = None, 'ATR_fallback'
        for nivel, etiqueta, dist_pct in resistencias:
            dist_abs = nivel - precio
            if dist_abs < sl_min_dist:
                continue   # demasiado cerca: cualquier wick lo toca
            if dist_abs > sl_max_dist:
                break      # demasiado lejos: riesgo excesivo
            sl_precio   = nivel + buffer  # SL por encima de la resistencia + margen
            sl_etiqueta = etiqueta
            break
        if sl_precio is None:
            sl_precio   = precio + atr * min(cfg['sl_mult'], SL_ATR_MAX_MULT)
            sl_etiqueta = 'ATR_fallback'
        sl_dist = sl_precio - precio

        # ── TP: por encima del soporte más cercano (salimos antes del rebote) ──
        tp_precio, tp_etiqueta = None, 'ATR_fallback'
        n_sop = 1 if not es_swing else min(2, len(soportes))
        for i, (nivel, etiqueta, dist_pct) in enumerate(soportes):
            if i < n_sop - 1:
                continue
            tp_candidato = nivel + buffer
            tp_dist = precio - tp_candidato
            if tp_dist < sl_min_dist:
                continue
            if tp_dist > tp_max_dist:
                break
            if tp_dist / sl_dist < rr_min:
                continue
            tp_precio, tp_etiqueta = tp_candidato, etiqueta
            break
        if tp_precio is None:
            tp_precio   = precio - min(sl_dist * max(cfg['tp_ratio'], rr_min + 0.1), tp_max_dist)
            tp_etiqueta = 'ATR_fallback'

    else:  # long
        # ── SL: por debajo del soporte más cercano ───────────────────────
        sl_precio, sl_etiqueta = None, 'ATR_fallback'
        for nivel, etiqueta, dist_pct in soportes:
            dist_abs = precio - nivel
            if dist_abs < sl_min_dist:
                continue
            if dist_abs > sl_max_dist:
                break
            sl_precio   = nivel - buffer  # SL por debajo del soporte + margen
            sl_etiqueta = etiqueta
            break
        if sl_precio is None:
            sl_precio   = precio - atr * min(cfg['sl_mult'], SL_ATR_MAX_MULT)
            sl_etiqueta = 'ATR_fallback'
        sl_dist = precio - sl_precio

        # ── TP: por debajo de la resistencia más cercana ─────────────────
        tp_precio, tp_etiqueta = None, 'ATR_fallback'
        n_res = 1 if not es_swing else min(2, len(resistencias))
        for i, (nivel, etiqueta, dist_pct) in enumerate(resistencias):
            if i < n_res - 1:
                continue
            tp_candidato = nivel - buffer
            tp_dist = tp_candidato - precio
            if tp_dist < sl_min_dist:
                continue
            if tp_dist > tp_max_dist:
                break
            if tp_dist / sl_dist < rr_min:
                continue
            tp_precio, tp_etiqueta = tp_candidato, etiqueta
            break
        if tp_precio is None:
            tp_precio   = precio + min(sl_dist * max(cfg['tp_ratio'], rr_min + 0.1), tp_max_dist)
            tp_etiqueta = 'ATR_fallback'

    # ── Verificación final ────────────────────────────────────────────────
    sl_dist_final = (sl_precio - precio) if direccion == 'short' else (precio - sl_precio)
    tp_dist_final = (precio - tp_precio) if direccion == 'short' else (tp_precio - precio)

    if sl_dist_final <= 0 or tp_dist_final <= 0:
        return {'viable': False, 'motivo': 'SL o TP en lado incorrecto del precio'}

    rr_real = tp_dist_final / sl_dist_final
    viable  = rr_real >= rr_min

    return {
        'viable':      viable,
        'sl':          round(sl_precio, 8),
        'tp':          round(tp_precio, 8),
        'sl_dist_pct': round(sl_dist_final / precio * 100, 3),
        'tp_dist_pct': round(tp_dist_final / precio * 100, 3),
        'rr':          round(rr_real, 2),
        'sl_nivel':    sl_etiqueta,
        'tp_nivel':    tp_etiqueta,
        'es_swing':    es_swing,
        'modo': 'swing' if es_swing else ('nivel' if 'fallback' not in sl_etiqueta else 'atr_fallback'),
        'motivo': '' if viable else f'R:R={rr_real:.2f} < mínimo {rr_min}',
    }


def calcular_max_horas(atr_pct: float, direccion: str) -> int:
    """
    Tiempo máximo dinámico: inversamente proporcional a la volatilidad.
    Alta volatilidad → movimiento más rápido → menos tiempo necesario.
    """
    base = MAX_HORAS_BASE_LONG if direccion == 'long' else MAX_HORAS_BASE_SHORT
    if atr_pct <= 0:
        return base
    factor = MAX_HORAS_ATR_REF / atr_pct
    return int(np.clip(base * factor, MAX_HORAS_MIN, MAX_HORAS_MAX))


def calcular_margen_ajustado(cfg_margen: float, atr_pct: float) -> float:
    """
    Reduce el margen en criptos más volátiles.
    BTC (~1% ATR) → margen base. SOL (~3% ATR) → margen × 0.55 aprox.
    Fórmula: factor = (ATR_ref / atr_pct) ^ 0.6  (potencia suavizada)
    """
    if atr_pct <= 0:
        return cfg_margen
    factor = (MAX_HORAS_ATR_REF / atr_pct) ** 0.6
    factor = float(np.clip(factor, 0.40, 1.20))  # nunca < 40% ni > 120% del base
    return cfg_margen * factor


def calcular_liquidacion(precio: float, cfg: dict, direccion: str,
                          margen: float) -> float:
    """
    Precio de liquidación aproximado considerando el margen real usado.
    liq = precio ∓ (margen / exposición) × precio × 0.90
    donde exposición = margen × leverage.
    """
    lev = cfg['leverage']
    margen_ratio = 1.0 / lev  # fracción del precio que cubre el margen
    if direccion == 'long':
        return precio * (1 - margen_ratio * 0.90)
    else:
        return precio * (1 + margen_ratio * 0.90)


def calcular_rsi_umbral(rsi_historico: pd.Series, direccion: str) -> tuple:
    """
    Ajusta los umbrales de RSI al régimen actual.
    En tendencia alcista el RSI tiende a oscilar entre 40–80 (no 30–70),
    así que el umbral de sobrecompra para un short debe subir.
    Devuelve (umbral_sobrecompra, umbral_sobreventa).
    """
    if rsi_historico is None or len(rsi_historico.dropna()) < 20:
        return 65, 35  # defaults originales

    rsi_clean = rsi_historico.dropna().tail(50)
    rsi_med   = float(rsi_clean.median())
    rsi_p75   = float(rsi_clean.quantile(0.75))
    rsi_p25   = float(rsi_clean.quantile(0.25))

    # Si el RSI mediano está alto (tendencia alcista fuerte), sube el umbral de sobrecompra
    if rsi_med > 58:
        umbral_sc = float(np.clip(rsi_p75, 68, 80))  # sobrecompra más alta
        umbral_sv = float(np.clip(rsi_p25, 35, 48))
    elif rsi_med < 42:
        umbral_sc = float(np.clip(rsi_p75, 55, 68))  # sobrecompra más baja
        umbral_sv = float(np.clip(rsi_p25, 22, 35))
    else:
        umbral_sc = 65
        umbral_sv = 35

    return umbral_sc, umbral_sv


def margen_efectivo(capital: float, cfg_margen: float, af: float,
                    atr_pct: float = 0.012, margen_f: float = 1.0) -> float:
    """
    Margen usando capital ACTUAL, escalado por la volatilidad real de la cripto.
    BTC (ATR ~1%) → margen base. SOL (ATR ~3%) → margen reducido aprox. 55%.
    """
    cap            = min(capital, capital * CAP_MULT_FUNDING)
    margen_vol_adj = calcular_margen_ajustado(cfg_margen, atr_pct)
    return cap * margen_vol_adj * af * margen_f


def margen_total_usado(estado: Dict) -> float:
    """Suma de todos los márgenes puestos como garantía en posiciones abiertas."""
    return sum(p['margen'] for p in estado['posiciones'])


def limite_margen_ok(estado: Dict, margen_nuevo: float) -> bool:
    """La suma de márgenes no puede superar el capital real disponible."""
    usado  = margen_total_usado(estado)
    limite = estado['capital']
    return (usado + margen_nuevo) <= limite


# ──────────────────────────────────────────────
#  GESTIÓN DE POSICIONES
# ──────────────────────────────────────────────

def gestionar_posiciones(estado: Dict, precio: float, atr: float,
                          hora: int, simbolo: str, ts: str,
                          high: float = None, low: float = None) -> List[Dict]:
    cerradas = []
    nuevas   = []
    for pos in estado['posiciones']:
        if pos['simbolo'] != simbolo:
            nuevas.append(pos)
            continue
        horas = hora - pos['hora_entrada']

        # ── Tiempo máximo dinámico guardado en la posición al abrirla ──
        # Si no está (posiciones antiguas), lo recalculamos desde el ATR actual.
        if 'max_horas' in pos:
            max_h = pos['max_horas']
        else:
            atr_pct_pos = atr / precio if precio > 0 else MAX_HORAS_ATR_REF
            max_h = calcular_max_horas(atr_pct_pos, pos['dir'])

        # ── Trailing dinámico por ATR ──────────────────────────────────
        # La activación y la distancia del trailing dependen del ATR actual,
        # no de un porcentaje fijo. Esto adapta el trailing a la volatilidad
        # real de la cripto en el momento.
        if pos['dir'] == 'long':
            if precio > pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                ganancia_abs = precio - pos['precio_entrada']
                if ganancia_abs >= atr * TRAILING_ATR_ACTIVACION:
                    pos['trailing'] = True
                    nuevo_sl = pos['precio_ref'] - atr * TRAILING_ATR_DISTANCIA
                    pos['sl'] = max(pos['sl'], nuevo_sl)

            cerrar = precio <= pos.get('liq', 0) and 'liq' in pos
            motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio <= pos['sl']:
                    cerrar = True; motivo = 'trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio >= pos['tp'] or (high is not None and high >= pos['tp']):
                    cerrar = True; motivo = 'take_profit'
                elif horas >= max_h:
                    cerrar = True; motivo = 'tiempo_maximo'
            if cerrar:
                ps   = precio * (1 - SLIPPAGE_PCT)
                exp  = pos['margen'] * pos['lev']
                ret  = (ps - pos['precio_entrada']) / pos['precio_entrada']
                fund = exp * FUNDING_RATE_HORA * horas
                pnl  = exp * ret - exp * COMISION_PCT * 2 - fund
                estado['capital']       += pnl
                estado['pnl_total']     += pnl
                estado['funding_total'] += fund
                estado['n_ops']         += 1
                if pnl > 0: estado['n_wins']  += 1
                else:        estado['n_loses'] += 1
                actualizar_adaptive(estado, pnl)
                op = {'ts': ts, 'simbolo': simbolo, 'dir': 'long', 'motivo': motivo,
                      'entrada': round(pos['precio_entrada'], 4), 'salida': round(ps, 4),
                      'pnl': round(pnl, 2), 'capital': round(estado['capital'], 2)}
                cerradas.append(op)
                emoji = '✅' if pnl > 0 else '❌'
                rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
                expo_now = margen_total_usado(estado)
                sl_info  = f"SL={pos['sl_nivel']}" if 'sl_nivel' in pos else ''
                tp_info  = f"TP={pos['tp_nivel']}" if 'tp_nivel' in pos else ''
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
                ganancia_abs = pos['precio_entrada'] - precio
                if ganancia_abs >= atr * TRAILING_ATR_ACTIVACION:
                    pos['trailing'] = True
                    nuevo_sl = pos['precio_ref'] + atr * TRAILING_ATR_DISTANCIA
                    pos['sl'] = min(pos['sl'], nuevo_sl)

            cerrar = precio >= pos.get('liq', float('inf'))
            motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio >= pos['sl']:
                    cerrar = True; motivo = 'trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio <= pos['tp'] or (low is not None and low <= pos['tp']):
                    cerrar = True; motivo = 'take_profit'
                elif horas >= max_h:
                    cerrar = True; motivo = 'tiempo_maximo'
            if cerrar:
                ps   = precio * (1 + SLIPPAGE_PCT)
                exp  = pos['margen'] * pos['lev']
                ret  = (pos['precio_entrada'] - ps) / pos['precio_entrada']
                fund = exp * FUNDING_RATE_HORA * horas
                pnl  = exp * ret - exp * COMISION_PCT * 2 - fund
                estado['capital']       += pnl
                estado['pnl_total']     += pnl
                estado['funding_total'] += fund
                estado['n_ops']         += 1
                if pnl > 0: estado['n_wins']  += 1
                else:        estado['n_loses'] += 1
                actualizar_adaptive(estado, pnl)
                op = {'ts': ts, 'simbolo': simbolo, 'dir': 'short', 'motivo': motivo,
                      'entrada': round(pos['precio_entrada'], 4), 'salida': round(ps, 4),
                      'pnl': round(pnl, 2), 'capital': round(estado['capital'], 2)}
                cerradas.append(op)
                emoji = '✅' if pnl > 0 else '❌'
                rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
                expo_now = margen_total_usado(estado)
                sl_info  = f"SL={pos['sl_nivel']}" if 'sl_nivel' in pos else ''
                tp_info  = f"TP={pos['tp_nivel']}" if 'tp_nivel' in pos else ''
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
                    prob: float, cfg: dict, calidad: str, dir_: str,
                    ts: str, hora: int, razones: List[str],
                    df_completo: pd.DataFrame = None, ctx4h: Dict = None,
                    ultima_row: pd.Series = None, divs: Dict = None,
                    ctx1d: Dict = None, sub_modelo: Dict = None):
    nombre = simbolo.replace('/', '_')
    af     = estado.get('adaptive_factor', 1.0)

    # ── GUARD ANTI-DUPLICADOS (mismo símbolo + dirección) ──
    ya_existe = any(p['simbolo'] == nombre and p['dir'] == dir_ for p in estado['posiciones'])
    if ya_existe:
        print(f"  🚫 BLOQUEADO: ya hay un {dir_.upper()} abierto en {nombre}")
        return

    # ── Calcular calidad escalonada y asignar nivel ──
    pts_calidad, notas_calidad = calcular_calidad_score(
        ultima_row if ultima_row is not None else pd.Series(),
        ctx4h or {}, divs or {}, dir_
    )
    nivel_str, cfg = asignar_nivel(pts_calidad)
    print(f"  📊 Calidad: {pts_calidad} pts → {nivel_str.upper()} "
          f"(x{cfg['leverage']} lev, {cfg['margen']*100:.0f}% margen)")
    for n in notas_calidad:
        print(f"     {n}")

    # ── Amplificador IA diario (tendencia macro 1D) ──
    # El modelo 1H se alimenta con velas diarias para evaluar el contexto macro.
    # Modifica el nivel ANTES de calcular SL/TP y margen.
    nivel_str_antes = nivel_str
    nota_amp = None
    if ctx1d is not None and ctx1d.get('disponible') and sub_modelo is not None:
        df1d_raw = ctx1d.get('df')
        prob_1d  = predecir_tendencia_diaria(df1d_raw, sub_modelo)
        tendencia_1d = ctx1d.get('tendencia', 'lateral')
        nivel_str, nota_amp = amplificar_nivel_por_ia(nivel_str, prob_1d, dir_, tendencia_1d)
        cfg = TRADE_CONFIG[nivel_str]
        print(f"  🌐 {nota_amp}")
        if nivel_str != nivel_str_antes:
            print(f"  📊 Nivel ajustado por IA diaria: {nivel_str_antes.upper()} → {nivel_str.upper()} "
                  f"(x{cfg['leverage']} lev, {cfg['margen']*100:.0f}% margen)")
    else:
        print(f"  🌐 IA diaria: no disponible (nivel sin cambio)")

    # ── ATR relativo ──
    atr_pct = atr / precio if precio > 0 else MAX_HORAS_ATR_REF

    # ── SL/TP desde niveles reales ──
    ctx_sltp = None
    if df_completo is not None:
        u = df_completo.iloc[-1]
        niveles  = _niveles_mercado(df_completo, u, precio, atr)
        ctx_sltp = calcular_sl_tp_niveles(precio, atr, cfg, dir_, niveles, ctx4h or {})

    if ctx_sltp is None or not ctx_sltp['viable']:
        motivo = ctx_sltp['motivo'] if ctx_sltp else 'sin contexto de mercado'
        print(f"  ❌ {nombre} {dir_.upper()} rechazado: {motivo}")
        return

    sl  = ctx_sltp['sl']
    tp  = ctx_sltp['tp']
    liq = calcular_liquidacion(precio, cfg, dir_, 1.0)
    pe  = precio * (1 + SLIPPAGE_PCT) if dir_ == 'long' else precio * (1 - SLIPPAGE_PCT)

    # ── Margen ajustado por volatilidad y factor adaptativo ──
    margen_des = margen_efectivo(estado['capital'], cfg['margen'], af, atr_pct,
                                 margen_f=0.80 if ctx_sltp['es_swing'] else 1.0)
    margen_max = estado['capital'] * MAX_MARGEN_TOTAL_PCT
    margen_uso = sum(p['margen'] for p in estado['posiciones'] if p['simbolo'] == nombre)
    margen_r   = min(margen_des, margen_max - margen_uso)

    if margen_r < estado['capital'] * 0.02:
        print(f"  ⚠️  Margen insuficiente para {nombre} {dir_}, skip")
        return
    if not limite_margen_ok(estado, margen_r):
        print(f"  ⚠️  Margen total agotado: ${margen_total_usado(estado):.2f} / ${estado['capital']:.2f}, skip")
        return

    max_horas = calcular_max_horas(atr_pct, dir_)

    estado['posiciones'].append({
        'simbolo':        nombre,
        'dir':            dir_,
        'calidad':        nivel_str,
        'precio_entrada': pe,
        'precio_ref':     pe,
        'sl':             sl,
        'tp':             tp,
        'liq':            liq,
        'margen':         margen_r,
        'lev':            cfg['leverage'],
        'hora_entrada':   hora,
        'max_horas':      max_horas,
        'trailing':       False,
        'ts_entrada':     ts,
        'adaptive_factor': af,
        'modo_sltp':      ctx_sltp['modo'].upper(),
        'sl_nivel':       ctx_sltp['sl_nivel'],
        'tp_nivel':       ctx_sltp['tp_nivel'],
        'atr_entrada':    round(atr, 6),
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

    telegram(f"🚀 <b>APERTURA {dir_.upper()} {simbolo}</b> [{VERSION}]{swing_tag}\n"
             f"Modo: {modo} | Nivel: {nivel_str.upper()} ({pts_calidad} pts) | Prob: {prob:.3f}\n"
             f"Razones:\n{razones_str}\n"
             f"Precio entrada: ${pe:,.4f}\n"
             f"SL: ${sl:,.4f} (-{sl_pct:.2f}%) ← {ctx_sltp['sl_nivel']}\n"
             f"TP: ${tp:,.4f} (+{tp_pct:.2f}%) ← {ctx_sltp['tp_nivel']}\n"
             f"R:R = 1:{rr:.1f} | Max horas: {max_horas}h\n"
             f"Margen: ${margen_r:.2f} | x{cfg['leverage']} | AF: x{af:.1f} | ATR: {atr_pct*100:.2f}%\n"
             f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_act>=0 else ''}{rent_act:.2f}%)\n"
             f"🔒 Margen usado: ${margen_usado:.2f} ({margen_pct:.0f}% del capital)\n"
             f"Posiciones abiertas: {len(estado['posiciones'])}")


# ──────────────────────────────────────────────
#  REPORTE DIARIO
# ──────────────────────────────────────────────

def enviar_reporte_diario(estado: Dict):
    capital  = estado['capital']
    rent     = (capital - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
    n        = estado['n_ops']
    wr       = estado['n_wins'] / max(n, 1) * 100
    ops_hoy  = estado.get('ops_hoy', [])
    af       = estado.get('adaptive_factor', 1.0)
    h        = estado.get('historial_pnl', [])
    wr20     = sum(h) / len(h) * 100 if h else 0

    expo_act = margen_total_usado(estado)
    expo_pct = expo_act / capital * 100 if capital > 0 else 0

    msg = (f"📊 <b>REPORTE DIARIO {VERSION.upper()}</b>\n"
           f"{'─'*30}\n"
           f"💰 Capital: ${capital:,.2f} ({'+' if rent>=0 else ''}{rent:.2f}%)\n"
           f"📈 Total ops: {n} | WR: {wr:.1f}%\n"
           f"📊 WR últimas 20: {wr20:.1f}%\n"
           f"💵 P&L total: ${estado['pnl_total']:+.2f}\n"
           f"📉 Funding total: -${abs(estado['funding_total']):.2f}\n"
           f"🔧 Adaptive: x{af:.1f}\n"
           f"🔓 Posiciones abiertas: {len(estado['posiciones'])}\n"
           f"🔒 Margen en uso: ${expo_act:.2f} ({expo_pct:.0f}% del capital)\n"
           f"{'─'*30}\n")

    if ops_hoy:
        wins  = sum(1 for o in ops_hoy if o.get('pnl', 0) > 0)
        loses = len(ops_hoy) - wins
        pnl_d = sum(o.get('pnl', 0) for o in ops_hoy)
        msg  += f"<b>Hoy ({len(ops_hoy)} ops):</b>\n"
        msg  += f"✅ {wins} | ❌ {loses} | P&L: ${pnl_d:+.2f}\n"
        for op in ops_hoy[-5:]:
            e    = '✅' if op.get('pnl', 0) > 0 else '❌'
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
    print(f"  Estrategia: PREDICTIVA + SL/TP INTELIGENTE por contexto")
    print(f"{'='*55}")

    estado    = cargar_estado()
    hora_unix = int(time.time() / 3600)

    if estado.get('primera_ejecucion', True):
        telegram(f"🤖 <b>BOT {VERSION} INICIADO</b>\n"
                 f"Capital: ${CAPITAL_INICIAL:,.2f}\n"
                 f"Estrategia PREDICTIVA + SL/TP INTELIGENTE:\n"
                 f"  · Modo SWING: tendencia 4H clara → SL lejos, TP lejos\n"
                 f"  · Modo REBOTE: precio en nivel clave → SL justo tras nivel, TP en siguiente\n"
                 f"  · Modo ESTANDAR: contexto normal → ATR estándar\n"
                 f"  · Ratio R:R mostrado en cada apertura\n"
                 f"✅ Conexión OK — bot activo")
        estado['primera_ejecucion'] = False

    ahora_utc = datetime.now(timezone.utc)
    if ahora_utc.hour == 0 and ahora_utc.minute < 65:
        enviar_reporte_diario(estado)

    if not estado['bot_activo']:
        print("  🛑 Bot detenido")
        guardar_estado(estado)
        return

    if estado['en_pausa']:
        fin = estado.get('fin_pausa_hora', 0)
        if hora_unix < fin:
            print(f"  ⏸️  Pausa. Quedan {fin - hora_unix}h")
            guardar_estado(estado)
            return
        estado['en_pausa'] = False

    SEP  = "─" * 60
    resumen_ciclo = []   # acumula una línea por cripto para el resumen final

    for simbolo in CRIPTOS:
        nombre = simbolo.replace('/', '_')
        print(f"\n{SEP}")
        print(f"  📊  {simbolo}  |  {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        print(SEP)

        # ── Datos e indicadores ───────────────────────────────────────
        df = obtener_velas(simbolo, intervalo=60, limite=VELAS_N)
        if df is None:
            print("  ❌  Sin datos, skip")
            resumen_ciclo.append(f"{simbolo:<12} ❌  sin datos")
            continue

        try:
            df = calcular_indicadores(df)
        except Exception as e:
            print(f"  ❌  Indicadores: {e}")
            resumen_ciclo.append(f"{simbolo:<12} ❌  error indicadores")
            continue

        df.dropna(subset=['ema_200', 'rsi_14', 'atr_14'], inplace=True)
        if len(df) < 50:
            resumen_ciclo.append(f"{simbolo:<12} ❌  datos insuficientes")
            continue

        ultima = df.iloc[-2]
        precio = float(ultima['close'])
        high   = float(ultima['high'])
        low    = float(ultima['low'])
        atr    = float(ultima.get('atr_14', precio * 0.015))
        if pd.isna(atr) or atr <= 0:
            atr = precio * 0.015
        atr_pct = atr / precio * 100
        ts = str(ultima['timestamp'])

        vela_actual    = df.iloc[-1]
        precio_entrada = float(vela_actual['close'])

        rsi   = float(ultima.get('rsi_14', 50))
        macd  = float(ultima.get('macd_hist', 0))
        stk   = float(ultima.get('stoch_k', 50))
        adx   = float(ultima.get('adx', 0))
        bb_p  = float(ultima.get('bb_posicion', 0.5))
        vol_r = float(ultima.get('volumen_ratio', 1.0))
        reg   = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', np.nan)) else 0

        # ── Bloque 1: precio e indicadores clave ─────────────────────
        tendencia_1h = "ALCISTA" if reg >= 1 else ("BAJISTA" if reg <= -1 else "LATERAL")
        macd_estado  = "▲ positivo" if macd > 0 else "▼ negativo"
        print(f"  Precio: ${precio:,.4f}  |  ATR: {atr_pct:.2f}%  |  1H: {tendencia_1h}")
        print(f"  RSI={rsi:.0f}  Stoch={stk:.0f}  MACD_hist={macd:.4f}({macd_estado})"
              f"  ADX={adx:.0f}  BB={bb_p:.2f}  Vol×{vol_r:.1f}")

        # ── Bloque 2: contexto 4H ────────────────────────────────────
        ctx4h = contexto_4h(simbolo)
        if ctx4h.get('disponible'):
            t4 = ctx4h['tendencia'].upper()
            m4 = ctx4h['momentum']
            r4 = ctx4h.get('rsi', 50)
            adx4 = ctx4h.get('adx', 0)
            flecha4 = "⬆" if t4 == "ALCISTA" else ("⬇" if t4 == "BAJISTA" else "↔")
            print(f"  4H: {flecha4} {t4}  momentum={m4}  RSI={r4:.0f}  ADX={adx4:.0f}")
        else:
            print("  4H: no disponible")

        # ── Bloque 2b: contexto diario 1D (amplificador macro) ───────
        ctx1d = contexto_diario(simbolo)
        if ctx1d.get('disponible'):
            t1d = ctx1d['tendencia'].upper()
            m1d = ctx1d['momentum']
            r1d = ctx1d.get('rsi', 50)
            flecha1d = "⬆" if t1d == "ALCISTA" else ("⬇" if t1d == "BAJISTA" else "↔")
            print(f"  1D: {flecha1d} {t1d}  momentum={m1d}  RSI={r1d:.0f}  (amplificador macro)")
        else:
            print("  1D: no disponible")

        # ── Bloque 3: divergencias ───────────────────────────────────
        df_pred = df.iloc[:-1]
        divs = detectar_divergencias(df_pred, ventana=40)
        if divs['resumen']:
            for d in divs['resumen']:
                icon = "🔴" if "bajista" in d.lower() else "🟢"
                print(f"  {icon} Div: {d}")
        else:
            print("  ⚪ Sin divergencias detectadas")

        # ── Bloque 4: patrones chartistas ────────────────────────────
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

        # ── Gestionar posiciones abiertas ────────────────────────────
        cerradas = gestionar_posiciones(estado, precio, atr, hora_unix, nombre, ts,
                                        high=high, low=low)
        for op in cerradas:
            registrar_op(op)
            estado.setdefault('ops_hoy', []).append(op)

        if int(ultima.get('volumen_muy_bajo', 0)) == 1:
            print("  ⚠️  Volumen muy bajo — skip análisis de entrada")
            resumen_ciclo.append(f"{simbolo:<12} ⚠️  volumen bajo")
            continue

        # ── Posiciones por símbolo y dirección ──────────────────────
        n_longs  = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'long')
        n_shorts = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'short')

        # ── Bloque 5: análisis LONG ──────────────────────────────────
        print(f"\n  {'─'*25} ANÁLISIS LONG {'─'*20}")
        resumen_l = "—"
        if n_longs > 0:
            pos_l = next(p for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'long')
            pnl_pct = (precio - pos_l['precio_entrada']) / pos_l['precio_entrada'] * 100
            print(f"  📌 Ya hay LONG abierto en ${pos_l['precio_entrada']:,.4f}  P&L={pnl_pct:+.2f}%"
                  f"  SL=${pos_l['sl']:,.4f}  TP=${pos_l['tp']:,.4f}")
            resumen_l = f"LONG abierto {pnl_pct:+.2f}%"
        else:
            sub_long = cargar_submodelo(simbolo, 'long')
            if sub_long:
                # prob_long: portero de la señal (umbral bajo, el score técnico filtra)
                prob_long = predecir(df_pred, sub_long)
                # prob_ct_long: indicador corto plazo 4-8h — se añade al score predictivo
                # Usa el mismo modelo pero sobre las últimas velas 1H (ventana reciente)
                prob_ct_long = predecir(df_pred.tail(50).reset_index(drop=True), sub_long)
                score_l, razones_l = score_predictivo_long(
                    df_pred, ultima, ctx4h, patrones, divs, prob_ct=prob_ct_long)
                ctx_señal_l = evaluar_contexto_señal(df_pred, ultima, ctx4h, 'long', divs, patrones)

                print(f"  IA prob={prob_long:.4f}  IA_ct={prob_ct_long:.4f}  score_base={score_l:.1f}  "
                      f"mult_ctx=×{ctx_señal_l['multiplicador_score']:.2f}  "
                      f"marg=×{ctx_señal_l['multiplicador_margen']:.2f}  "
                      f"confianza={ctx_señal_l['nivel_confianza'].upper()}")

                # Señales a favor
                print("  Señales a favor del LONG:")
                razones_favor = [r for r in razones_l if not r.startswith("⚠")]
                if razones_favor:
                    for r in razones_favor:
                        print(f"    ✅ {r}")
                else:
                    print("    (ninguna)")

                # Señales en contra
                razones_contra = [r for r in razones_l if r.startswith("⚠")]
                razones_contra += ctx_señal_l['contexto_log']
                razones_contra_unicas = list(dict.fromkeys(razones_contra))
                if razones_contra_unicas:
                    print("  En contra / penalizaciones:")
                    for r in razones_contra_unicas:
                        print(f"    ⚠️  {r}")

                if ctx_señal_l['bloquear']:
                    print(f"  🚫 LONG BLOQUEADO — {ctx_señal_l['razon_bloqueo']}")
                    resumen_l = f"BLOQUEADO (soporte/resistencia fuerte)"
                else:
                    score_l_adj = score_l * ctx_señal_l['multiplicador_score']
                    min_s_l     = score_minimo(prob_long, 'long')
                    print(f"  Score ajustado: {score_l:.1f} × {ctx_señal_l['multiplicador_score']:.2f}"
                          f" = {score_l_adj:.1f}  (mínimo={min_s_l})")

                    if prob_long < PROB_PREMIUM_LONG:
                        resumen_l = f"no abre — prob {prob_long:.4f} < {PROB_PREMIUM_LONG}"
                        print(f"  ❌ Prob insuficiente ({prob_long:.4f} < {PROB_PREMIUM_LONG})")
                    elif score_l_adj < min_s_l:
                        resumen_l = f"no abre — score {score_l_adj:.1f} < {min_s_l}"
                        print(f"  ❌ Score insuficiente ({score_l_adj:.1f} < {min_s_l})")
                    else:
                        print(f"  🟢 SEÑAL LONG — ABRIENDO POSICIÓN")
                        abrir_posicion(estado, simbolo, precio_entrada, atr,
                                       prob_long, {}, 'long', 'long', ts, hora_unix,
                                       razones_l + ctx_señal_l['contexto_log'],
                                       df_pred, ctx4h, ultima, divs,
                                       ctx1d=ctx1d, sub_modelo=sub_long)
                        resumen_l = f"✅ LONG abierto"
            else:
                print("  ⚠️  Sin modelo LONG disponible")
                resumen_l = "sin modelo"

        # ── Bloque 6: análisis SHORT ─────────────────────────────────
        print(f"\n  {'─'*25} ANÁLISIS SHORT {'─'*19}")
        resumen_s = "—"
        if n_shorts > 0:
            pos_s = next(p for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'short')
            pnl_pct = (pos_s['precio_entrada'] - precio) / pos_s['precio_entrada'] * 100
            print(f"  📌 Ya hay SHORT abierto en ${pos_s['precio_entrada']:,.4f}  P&L={pnl_pct:+.2f}%"
                  f"  SL=${pos_s['sl']:,.4f}  TP=${pos_s['tp']:,.4f}")
            resumen_s = f"SHORT abierto {pnl_pct:+.2f}%"
        else:
            sub_short = cargar_submodelo(simbolo, 'short')
            if sub_short:
                prob_short = predecir(df_pred, sub_short)
                # prob_ct_short: indicador corto plazo 4-8h sobre ventana reciente
                prob_ct_short = predecir(df_pred.tail(50).reset_index(drop=True), sub_short)
                score_s, razones_s = score_predictivo_short(
                    df_pred, ultima, ctx4h, patrones, divs, prob_ct=prob_ct_short)
                ctx_señal_s = evaluar_contexto_señal(df_pred, ultima, ctx4h, 'short', divs, patrones)

                print(f"  IA prob={prob_short:.4f}  IA_ct={prob_ct_short:.4f}  score_base={score_s:.1f}  "
                      f"mult_ctx=×{ctx_señal_s['multiplicador_score']:.2f}  "
                      f"marg=×{ctx_señal_s['multiplicador_margen']:.2f}  "
                      f"confianza={ctx_señal_s['nivel_confianza'].upper()}")

                print("  Señales a favor del SHORT:")
                razones_favor_s = [r for r in razones_s if not r.startswith("⚠")]
                if razones_favor_s:
                    for r in razones_favor_s:
                        print(f"    ✅ {r}")
                else:
                    print("    (ninguna)")

                razones_contra_s = [r for r in razones_s if r.startswith("⚠")]
                razones_contra_s += ctx_señal_s['contexto_log']
                razones_contra_s_unicas = list(dict.fromkeys(razones_contra_s))
                if razones_contra_s_unicas:
                    print("  En contra / penalizaciones:")
                    for r in razones_contra_s_unicas:
                        print(f"    ⚠️  {r}")

                if ctx_señal_s['bloquear']:
                    print(f"  🚫 SHORT BLOQUEADO — {ctx_señal_s['razon_bloqueo']}")
                    resumen_s = "BLOQUEADO (soporte/resistencia fuerte)"
                else:
                    score_s_adj = score_s * ctx_señal_s['multiplicador_score']
                    min_s_s     = score_minimo(prob_short, 'short')
                    print(f"  Score ajustado: {score_s:.1f} × {ctx_señal_s['multiplicador_score']:.2f}"
                          f" = {score_s_adj:.1f}  (mínimo={min_s_s})")

                    if prob_short < PROB_PREMIUM_SHORT:
                        resumen_s = f"no abre — prob {prob_short:.4f} < {PROB_PREMIUM_SHORT}"
                        print(f"  ❌ Prob insuficiente ({prob_short:.4f} < {PROB_PREMIUM_SHORT})")
                    elif score_s_adj < min_s_s:
                        resumen_s = f"no abre — score {score_s_adj:.1f} < {min_s_s}"
                        print(f"  ❌ Score insuficiente ({score_s_adj:.1f} < {min_s_s})")
                    else:
                        print(f"  🔴 SEÑAL SHORT — ABRIENDO POSICIÓN")
                        abrir_posicion(estado, simbolo, precio_entrada, atr,
                                       prob_short, {}, 'short', 'short', ts, hora_unix,
                                       razones_s + ctx_señal_s['contexto_log'],
                                       df_pred, ctx4h, ultima, divs,
                                       ctx1d=ctx1d, sub_modelo=sub_short)
                        resumen_s = f"✅ SHORT abierto"
            else:
                print("  ⚠️  Sin modelo SHORT disponible")
                resumen_s = "sin modelo"

        resumen_ciclo.append(
            f"{simbolo:<12} ${precio:>10,.2f}  RSI={rsi:>3.0f}  "
            f"LONG: {resumen_l:<35}  SHORT: {resumen_s}"
        )
        time.sleep(0.5)

    # ── RESUMEN FINAL DEL CICLO ───────────────────────────────────────
    guardar_estado(estado)
    m = guardar_metricas(estado)
    s = '+' if m['rentabilidad_pct'] >= 0 else ''
    expo_fin     = margen_total_usado(estado)
    expo_fin_pct = expo_fin / estado['capital'] * 100 if estado['capital'] > 0 else 0

    print(f"\n{'═'*60}")
    print(f"  RESUMEN CICLO — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═'*60}")
    for linea in resumen_ciclo:
        print(f"  {linea}")
    print(f"{'─'*60}")
    print(f"  💰 Capital: ${m['capital_actual']:,.2f} ({s}{m['rentabilidad_pct']:.2f}%)")
    print(f"  🔒 Margen en uso: ${expo_fin:.2f} ({expo_fin_pct:.0f}% del capital)")
    print(f"  📈 Ops: {m['n_operaciones']} | WR: {m['win_rate_pct']:.1f}% | Adaptive: ×{m['adaptive_factor']:.2f}")
    print(f"  📂 Posiciones abiertas: {len(estado['posiciones'])}")
    if estado['posiciones']:
        for p in estado['posiciones']:
            pnl_est = (precio - p['precio_entrada']) / p['precio_entrada'] * 100
            if p['dir'] == 'short':
                pnl_est = -pnl_est
            print(f"     {p['dir'].upper():5} {p['simbolo']:<10} entrada=${p['precio_entrada']:,.4f}"
                  f"  SL=${p['sl']:,.4f}  TP=${p['tp']:,.4f}  P&L≈{pnl_est:+.2f}%")
    print(f"{'═'*60}")


if __name__ == '__main__':
    token   = os.environ.get('TELEGRAM_TOKEN_V14', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_V14', '')
    if not token:
        print("  ⚠️  TELEGRAM_TOKEN_V14 no configurado")
    elif not chat_id:
        print("  ⚠️  TELEGRAM_CHAT_ID_V14 no configurado")
    else:
        print(f"  ✅ Telegram configurado — chat_id: {chat_id[:6]}...{chat_id[-3:] if len(chat_id)>6 else ''}")
    ejecutar()
