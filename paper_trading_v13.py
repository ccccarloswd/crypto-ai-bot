"""
=============================================================
  CRYPTO AI BOT — Paper Trading v13  [MODO AGRESIVO]
=============================================================
  Basado en v14 (estrategia predictiva + SL/TP inteligente)
  con todos los parámetros llevados al límite:

  Diferencias clave respecto a v14:
  ──────────────────────────────────
  · Apalancamiento x2-x3 mayor en todos los tiers
  · Márgenes mayores (más capital en juego por operación)
  · Umbrales de probabilidad más bajos (entra antes)
  · Score mínimo reducido (menos condiciones necesarias)
  · TPs más ambiciosos (deja correr más las ganadoras)
  · SLs ligeramente más amplios (aguanta más ruido)
  · TP máximo en modo swing duplicado
  · Trailing activation más bajo (protege antes)
  · Máximo de posiciones: 3 (en v14 eran 2)
  · Margen máximo total: 60% (en v14 era 40%)

  ADVERTENCIA: Este bot asume riesgos significativamente
  mayores. Una mala racha puede borrar el capital rápido.
  Úsalo solo en paper trading hasta validar la estrategia.
=============================================================
"""

import os, json, time, urllib.request, numpy as np, pandas as pd
import joblib, warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

VERSION = 'v13'

# ──────────────────────────────────────────────
#  CONFIGURACIÓN
# ──────────────────────────────────────────────
CRIPTOS  = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
VELAS_N  = 300

CAPITAL_INICIAL   = 1000.0
COMISION_PCT      = 0.001
SLIPPAGE_PCT      = 0.0005
FUNDING_RATE_HORA = 0.0003

DD_PAUSA_PCT = 0.15
DD_STOP_PCT  = 0.25
HORAS_PAUSA  = 24

MAX_HORAS_LONG  = 24   # v14: 20 — más tiempo para que los trades se desarrollen
MAX_HORAS_SHORT = 20   # v14: 16
MAX_POSICIONES       = 3      # v14: 2 — más posiciones simultáneas
MAX_MARGEN_TOTAL_PCT = 0.60   # v14: 0.40 — más capital expuesto
TRAILING_ACTIVACION  = 0.012  # v14: 0.018 — activa trailing antes (protege ganancias)
TRAILING_ATR_MULT    = 1.2    # v14: 1.0 — trailing algo más holgado para no salir temprano

SOLO_XGBOOST     = ['BTC_USDT', 'BNB_USDT']
CAP_MULT_FUNDING = 8.0

ADAPTIVE_VENTANA  = 20
ADAPTIVE_WR_BAJO  = 0.45   # v14: 0.48 — tarda más en entrar en modo conservador
ADAPTIVE_WR_ALTO  = 0.60   # v14: 0.65 — entra antes en modo agresivo
ADAPTIVE_FACTOR_C = 0.60   # v14: 0.50 — modo conservador menos restrictivo
ADAPTIVE_FACTOR_A = 1.40   # v14: 1.20 — modo agresivo más potente
ADAPTIVE_DURACION = 12     # v14: 15 — sale antes del modo conservador

# ── Config por calidad — MODO AGRESIVO ──────────────────────────────
# v14 LONG elite:   lev=8,  margen=0.22, sl_max=0.020, tp_max=0.060
# v13 LONG elite:   lev=20, margen=0.30, sl_max=0.025, tp_max=0.090
# v14 LONG premium: lev=5,  margen=0.20, sl_max=0.025, tp_max=0.050
# v13 LONG premium: lev=12, margen=0.25, sl_max=0.028, tp_max=0.075
LONG_CONFIG = {
    'elite':   {'leverage': 20, 'margen': 0.30, 'sl_mult': 1.2,
                'sl_max': 0.025, 'tp_ratio': 3.2, 'tp_max': 0.090},
    'premium': {'leverage': 12, 'margen': 0.25, 'sl_mult': 1.4,
                'sl_max': 0.028, 'tp_ratio': 2.8, 'tp_max': 0.075},
}
# v14 SHORT elite:   lev=6,  margen=0.18, sl_max=0.022, tp_max=0.055
# v13 SHORT elite:   lev=15, margen=0.26, sl_max=0.026, tp_max=0.085
# v14 SHORT premium: lev=4,  margen=0.15, sl_max=0.025, tp_max=0.045
# v13 SHORT premium: lev=10, margen=0.22, sl_max=0.030, tp_max=0.068
SHORT_CONFIG = {
    'elite':   {'leverage': 15, 'margen': 0.26, 'sl_mult': 1.3,
                'sl_max': 0.026, 'tp_ratio': 3.2, 'tp_max': 0.085},
    'premium': {'leverage': 10, 'margen': 0.22, 'sl_mult': 1.4,
                'sl_max': 0.030, 'tp_ratio': 2.8, 'tp_max': 0.068},
}

# ── Umbrales de probabilidad — más bajos = entra más a menudo ───────
# v14: ELITE_LONG=0.74, PREMIUM_LONG=0.64, ELITE_SHORT=0.70, PREMIUM_SHORT=0.62
PROB_ELITE_LONG    = 0.68   # v14: 0.74
PROB_PREMIUM_LONG  = 0.58   # v14: 0.64
PROB_ELITE_SHORT   = 0.64   # v14: 0.70
PROB_PREMIUM_SHORT = 0.56   # v14: 0.62

# Score mínimo de condiciones predictivas según probabilidad
# v13: un punto menos en cada nivel → más trades abiertos
def score_minimo(prob: float, direccion: str) -> int:
    umbral_elite = PROB_ELITE_SHORT if direccion == 'short' else PROB_ELITE_LONG
    if prob >= 0.90:  return 1
    if prob >= 0.80:  return 1   # v14 pedía 2 aquí
    if prob >= umbral_elite: return 1   # v14 pedía 2
    return 2                     # v14 pedía 3

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
        }
    except Exception as e:
        print(f"  ⚠️  4H contexto error: {e}")
        return {'disponible': False}


# ──────────────────────────────────────────────
#  SCORE PREDICTIVO — NÚCLEO DE LA ESTRATEGIA
# ──────────────────────────────────────────────

def score_predictivo_short(df: pd.DataFrame, ultima: pd.Series,
                            ctx4h: Dict) -> Tuple[int, List[str]]:
    """
    Calcula el score predictivo para un SHORT.
    Cada condición que ANTICIPA una bajada suma 1 punto.
    No necesitamos que ya esté bajando, necesitamos que vaya a bajar.

    Retorna (score, lista de razones para el log).
    """
    score   = 0
    razones = []
    c       = float(ultima.get('close', 0))

    # 1. RSI sobrecomprado → va a retroceder
    rsi = float(ultima.get('rsi_14', 50))
    if rsi >= 65:
        score += 1
        razones.append(f"RSI={rsi:.0f} (sobrecompra)")
    elif rsi >= 60:
        score += 0.5

    # 2. Divergencia bajista RSI (precio sube, RSI no acompaña)
    div_baj = int(ultima.get('divergencia_bajista_rsi', 0))
    if div_baj == 1:
        score += 1
        razones.append("Divergencia bajista RSI")

    # 3. Stochastic sobrecomprado y girando
    stk = float(ultima.get('stoch_k', 50))
    std = float(ultima.get('stoch_d', 50))
    if stk >= 78 and stk < std + 2:  # >78 y cruzando hacia abajo
        score += 1
        razones.append(f"Stoch={stk:.0f} sobrecomprado")
    elif stk >= 70:
        score += 0.5

    # 4. MACD hist positivo pero DECELERANDO (pico de momentum)
    mh = df['macd_hist'].dropna()
    if len(mh) >= 3:
        mh_act  = float(mh.iloc[-1])
        mh_prev = float(mh.iloc[-2])
        mh_ant  = float(mh.iloc[-3])
        if mh_act > 0 and mh_act < mh_prev:
            score += 1
            razones.append("MACD hist decelerando (pico)")
        elif mh_act > 0 and mh_prev > 0 and mh_act < mh_prev < mh_ant:
            score += 1.5  # 2 velas decelerando = más fuerte
            razones.append("MACD hist 2 velas decelerando")

    # 5. Precio en zona de resistencia o Fibonacci 0.618
    dist_res = float(ultima.get('dist_resistencia_pct', 999))
    cerca_618 = int(ultima.get('cerca_fib_618', 0))
    bb_pos    = float(ultima.get('bb_posicion', 0.5))
    if dist_res < 0.5:
        score += 1
        razones.append(f"Precio en resistencia ({dist_res:.2f}%)")
    elif cerca_618 == 1:
        score += 1
        razones.append("Precio en Fibonacci 0.618")
    elif bb_pos > 0.90:
        score += 0.5
        razones.append(f"Precio en banda superior BB ({bb_pos:.2f})")

    # 6. Volumen decreciente en subida (falta convicción alcista)
    vr = df['volumen_ratio'].dropna()
    precio_sube = float(ultima.get('close', 0)) > float(df['close'].iloc[-3]) if len(df) >= 3 else False
    if len(vr) >= 3 and precio_sube:
        vr_act  = float(vr.iloc[-1])
        vr_prev = float(vr.iloc[-2])
        if vr_act < 0.80 and vr_prev < 0.80:
            score += 1
            razones.append("Volumen bajo en subida (sin convicción)")

    # 7. Contexto 4H — BONUS importante
    if ctx4h.get('disponible'):
        t4 = ctx4h['tendencia']
        m4 = ctx4h['momentum']
        r4 = ctx4h.get('rsi', 50)

        if t4 == 'bajista':
            score += 2  # 4H bajista + 1H rebote = oportunidad ideal
            razones.append(f"4H BAJISTA (tendencia superior)")
        elif t4 == 'lateral' and m4 in ('bajista_acelerando', 'bajista_desacelerando'):
            score += 1
            razones.append(f"4H lateral momentum bajista")

        if r4 >= 65:
            score += 0.5
            razones.append(f"RSI 4H={r4:.0f} sobrecomprado")

        if m4 == 'alcista_desacelerando':
            score += 1
            razones.append("4H momentum alcista desacelerando")

    # 8. Régimen 1H como contexto (no bloquea, modula)
    reg = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', 0)) else 0
    if reg <= -1:
        score += 0.5   # refuerzo si ya es bajista en 1H
    elif reg >= 2:
        score -= 0.5   # pequeña penalización si 1H es muy alcista

    return int(score), razones


def score_predictivo_long(df: pd.DataFrame, ultima: pd.Series,
                           ctx4h: Dict) -> Tuple[int, List[str]]:
    """
    Simétrico al score de short. Calcula condiciones que ANTICIPAN una subida.
    """
    score   = 0
    razones = []

    # 1. RSI sobrevendido → va a rebotar
    rsi = float(ultima.get('rsi_14', 50))
    if rsi <= 35:
        score += 1
        razones.append(f"RSI={rsi:.0f} (sobreventa)")
    elif rsi <= 40:
        score += 0.5

    # 2. Divergencia alcista RSI
    div_alc = int(ultima.get('divergencia_alcista_rsi', 0))
    if div_alc == 1:
        score += 1
        razones.append("Divergencia alcista RSI")

    # 3. Stochastic sobrevendido y girando
    stk = float(ultima.get('stoch_k', 50))
    std = float(ultima.get('stoch_d', 50))
    if stk <= 22 and stk > std - 2:
        score += 1
        razones.append(f"Stoch={stk:.0f} sobrevendido")
    elif stk <= 30:
        score += 0.5

    # 4. MACD hist negativo pero ACELERANDO hacia 0 (piso de momentum)
    mh = df['macd_hist'].dropna()
    if len(mh) >= 3:
        mh_act  = float(mh.iloc[-1])
        mh_prev = float(mh.iloc[-2])
        mh_ant  = float(mh.iloc[-3])
        if mh_act < 0 and mh_act > mh_prev:
            score += 1
            razones.append("MACD hist acelerando desde suelo")
        elif mh_act < 0 and mh_prev < 0 and mh_act > mh_prev > mh_ant:
            score += 1.5
            razones.append("MACD hist 2 velas mejorando")

    # 5. Precio en zona de soporte o Fibonacci 0.382/0.5
    dist_sop  = float(ultima.get('dist_soporte_pct', 999))
    cerca_382 = int(ultima.get('cerca_fib_382', 0))
    cerca_500 = int(ultima.get('cerca_fib_500', 0))
    bb_pos    = float(ultima.get('bb_posicion', 0.5))
    if dist_sop < 0.5:
        score += 1
        razones.append(f"Precio en soporte ({dist_sop:.2f}%)")
    elif cerca_382 == 1 or cerca_500 == 1:
        score += 1
        razones.append("Precio en Fibonacci 0.382/0.5")
    elif bb_pos < 0.10:
        score += 0.5
        razones.append(f"Precio en banda inferior BB ({bb_pos:.2f})")

    # 6. Volumen decreciente en bajada (vendedores agotados)
    vr = df['volumen_ratio'].dropna()
    precio_baja = float(ultima.get('close', 0)) < float(df['close'].iloc[-3]) if len(df) >= 3 else False
    if len(vr) >= 3 and precio_baja:
        vr_act  = float(vr.iloc[-1])
        vr_prev = float(vr.iloc[-2])
        if vr_act < 0.80 and vr_prev < 0.80:
            score += 1
            razones.append("Volumen bajo en bajada (vendedores agotados)")

    # 7. Contexto 4H — BONUS importante
    if ctx4h.get('disponible'):
        t4 = ctx4h['tendencia']
        m4 = ctx4h['momentum']
        r4 = ctx4h.get('rsi', 50)

        if t4 == 'alcista':
            score += 2
            razones.append("4H ALCISTA (tendencia superior)")
        elif t4 == 'lateral' and m4 in ('alcista_acelerando', 'alcista_desacelerando'):
            score += 1
            razones.append("4H lateral momentum alcista")

        if r4 <= 35:
            score += 0.5
            razones.append(f"RSI 4H={r4:.0f} sobrevendido")

        if m4 == 'bajista_desacelerando':
            score += 1
            razones.append("4H momentum bajista desacelerando")

    # 8. Régimen 1H como contexto
    reg = int(ultima.get('regimen_mercado', 0)) if not pd.isna(ultima.get('regimen_mercado', 0)) else 0
    if reg >= 1:
        score += 0.5
    elif reg <= -2:
        score -= 0.5

    return int(score), razones


# ──────────────────────────────────────────────
#  CLASIFICACIÓN DE SEÑALES
# ──────────────────────────────────────────────

def clasificar_long(prob: float, score: int) -> Optional[Tuple[str, dict]]:
    """
    Clasifica señal long según prob y score predictivo.
    No bloquea por régimen, el score ya incorpora el contexto.
    """
    min_score = score_minimo(prob, 'long')
    if score < min_score:
        return None
    if prob >= PROB_ELITE_LONG:
        return 'elite', LONG_CONFIG['elite']
    if prob >= PROB_PREMIUM_LONG:
        return 'premium', LONG_CONFIG['premium']
    return None


def clasificar_short(prob: float, score: int) -> Optional[Tuple[str, dict]]:
    """Clasifica señal short según prob y score predictivo."""
    min_score = score_minimo(prob, 'short')
    if score < min_score:
        return None
    if prob >= PROB_ELITE_SHORT:
        return 'elite', SHORT_CONFIG['elite']
    if prob >= PROB_PREMIUM_SHORT:
        return 'premium', SHORT_CONFIG['premium']
    return None


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
#  ANÁLISIS DE CONTEXTO PARA SL/TP INTELIGENTE
# ──────────────────────────────────────────────

def analizar_contexto_sltp(df: pd.DataFrame, ultima: pd.Series,
                            ctx4h: Dict, direccion: str) -> Dict:
    """
    Analiza el contexto de mercado para determinar el modo de SL/TP:

    Modos:
      'swing'    → Movimiento amplio esperado (bear/bull flag, tendencia clara
                   en 4H). SL más lejos, TP más lejos, margen reducido.
                   Objetivo: capturar el movimiento completo.

      'rebote'   → Rebote técnico desde soporte/resistencia o media móvil.
                   SL justo detrás del nivel, TP en el siguiente nivel.
                   Objetivo: trade limpio, corto y preciso.

      'estandar' → Situación normal sin contexto especial.
                   SL/TP basados en ATR estándar.

    Para cada modo devuelve multiplicadores de SL y TP que se aplicarán
    sobre el ATR base del cfg.
    """
    c      = df['close']
    precio = float(ultima.get('close', 0))

    # ── Detectar niveles clave cercanos ──────────────────────────────
    # Soportes y resistencias recientes (últimas 50 velas)
    h_50   = df['high'].rolling(50).max().iloc[-1]
    l_50   = df['low'].rolling(50).min().iloc[-1]
    rango  = h_50 - l_50

    # Medias móviles como niveles
    ema50  = float(ultima.get('ema_50',  precio))
    ema200 = float(ultima.get('ema_200', precio))
    sma20  = float(ultima.get('sma_20',  precio))
    bb_mid = float(ultima.get('bb_mid',  precio))
    bb_up  = float(ultima.get('bb_upper', precio * 1.02))
    bb_low = float(ultima.get('bb_lower', precio * 0.98))

    # Fibonacci recientes
    fib_618 = float(ultima.get('fib_618', 0))
    fib_382 = float(ultima.get('fib_382', 0))
    fib_500 = float(ultima.get('fib_500', 0))

    # Distancia a niveles clave como % del precio
    niveles_resistencia = [n for n in [ema50, ema200, sma20, bb_mid, bb_up, h_50, fib_618]
                           if n > precio * 1.001]
    niveles_soporte     = [n for n in [ema50, ema200, sma20, bb_mid, bb_low, l_50, fib_382, fib_500]
                           if n < precio * 0.999]

    # Nivel de resistencia más cercano por encima
    res_cercana = min(niveles_resistencia, default=precio * 1.05)
    dist_res    = (res_cercana - precio) / precio  # % hacia arriba

    # Nivel de soporte más cercano por debajo
    sop_cercano = max(niveles_soporte, default=precio * 0.95)
    dist_sop    = (precio - sop_cercano) / precio  # % hacia abajo

    # ── Detectar patrón de contexto ──────────────────────────────────
    ctx4h_disponible = ctx4h.get('disponible', False)
    tendencia4h      = ctx4h.get('tendencia', 'lateral')
    momentum4h       = ctx4h.get('momentum', 'neutral')
    rsi4h            = ctx4h.get('rsi', 50)

    # Indicadores de movimiento amplio en 4H
    swing_bajista = (
        ctx4h_disponible and
        tendencia4h == 'bajista' and
        momentum4h in ('bajista_acelerando', 'bajista_desacelerando') and
        rsi4h < 55
    )
    swing_alcista = (
        ctx4h_disponible and
        tendencia4h == 'alcista' and
        momentum4h in ('alcista_acelerando', 'alcista_desacelerando') and
        rsi4h > 45
    )

    # Rebote desde nivel clave
    atr_pct = float(ultima.get('atr_pct', 1.5)) / 100  # ATR como % del precio
    if direccion == 'long':
        rebote_desde_nivel = dist_sop < atr_pct * 1.5   # precio a <1.5×ATR del soporte
    else:
        rebote_desde_nivel = dist_res < atr_pct * 1.5   # precio a <1.5×ATR de la resistencia

    # ── Decidir modo ─────────────────────────────────────────────────
    if direccion == 'short' and swing_bajista:
        modo = 'swing'
    elif direccion == 'long' and swing_alcista:
        modo = 'swing'
    elif rebote_desde_nivel:
        modo = 'rebote'
    else:
        modo = 'estandar'

    # ── Calcular multiplicadores según modo ──────────────────────────
    if modo == 'swing':
        # AGRESIVO: SL más lejos para aguantar más pullbacks,
        # TP muy lejos para capturar movimientos completos.
        # v14: sl_mult=2.2, tp_mult=5.0, margen_f=0.75
        # v13: sl_mult=2.5, tp_mult=8.0, margen_f=0.70
        sl_mult  = 2.5
        tp_mult  = 8.0
        margen_f = 0.70

    elif modo == 'rebote':
        # En rebote también somos más agresivos con el margen
        # v14: margen_f=1.10 | v13: margen_f=1.25
        if direccion == 'long':
            sl_dist  = dist_sop + 0.003
            sl_mult  = sl_dist / atr_pct if atr_pct > 0 else 1.5
            sl_mult  = float(np.clip(sl_mult, 1.0, 2.0))
            tp_dist  = dist_res * 0.85 if dist_res > 0.01 else atr_pct * 3.0
            tp_mult  = tp_dist / atr_pct if atr_pct > 0 else 3.0
            tp_mult  = float(np.clip(tp_mult, 2.0, 5.5))
        else:
            sl_dist  = dist_res + 0.003
            sl_mult  = sl_dist / atr_pct if atr_pct > 0 else 1.5
            sl_mult  = float(np.clip(sl_mult, 1.0, 2.0))
            tp_dist  = dist_sop * 0.85 if dist_sop > 0.01 else atr_pct * 3.0
            tp_mult  = tp_dist / atr_pct if atr_pct > 0 else 3.0
            tp_mult  = float(np.clip(tp_mult, 2.0, 5.5))
        margen_f = 1.25

    else:  # estandar — también más agresivo que v14
        # v14: sl_mult=1.5, tp_mult=3.0, margen_f=1.0
        # v13: sl_mult=1.6, tp_mult=4.0, margen_f=1.15
        sl_mult  = 1.6
        tp_mult  = 4.0
        margen_f = 1.15

    return {
        'modo':        modo,
        'sl_mult':     sl_mult,
        'tp_mult':     tp_mult,
        'margen_f':    margen_f,
        'dist_res':    round(dist_res * 100, 2),
        'dist_sop':    round(dist_sop * 100, 2),
        'res_cercana': round(res_cercana, 4),
        'sop_cercano': round(sop_cercano, 4),
    }


# ──────────────────────────────────────────────
#  SL / TP INTELIGENTES (usan contexto)
# ──────────────────────────────────────────────

def sl_tp_long(p, atr, cfg, ctx_sltp: Dict = None):
    """
    SL/TP para LONG con contexto inteligente.
    v13: tp_max en modo swing multiplicado por 3.0 (v14 usaba 2.0)
    """
    if ctx_sltp:
        sl_mult = ctx_sltp['sl_mult']
        tp_mult = ctx_sltp['tp_mult']
        sl_max  = cfg['sl_max'] * (1.5 if ctx_sltp['modo'] == 'swing' else 1.0)
    else:
        sl_mult = cfg['sl_mult']
        tp_mult = cfg['tp_ratio']
        sl_max  = cfg['sl_max']

    sl_pct = float(np.clip((atr * sl_mult) / p, 0.008, sl_max))
    tp_max_eff = cfg['tp_max'] * (3.0 if ctx_sltp and ctx_sltp['modo'] == 'swing' else 1.0)
    tp_pct = float(np.clip((atr * tp_mult) / p, sl_pct * 2.0, tp_max_eff))
    liq    = p * (1 - (1 / cfg['leverage']) * 0.90)
    return p * (1 - sl_pct), p * (1 + tp_pct), liq


def sl_tp_short(p, atr, cfg, ctx_sltp: Dict = None):
    """SL/TP para SHORT con contexto inteligente.
    v13: tp_max en modo swing multiplicado por 3.0 (v14 usaba 2.0)
    """
    if ctx_sltp:
        sl_mult = ctx_sltp['sl_mult']
        tp_mult = ctx_sltp['tp_mult']
        sl_max  = cfg['sl_max'] * (1.5 if ctx_sltp['modo'] == 'swing' else 1.0)
    else:
        sl_mult = cfg['sl_mult']
        tp_mult = cfg['tp_ratio']
        sl_max  = cfg['sl_max']

    sl_pct = float(np.clip((atr * sl_mult) / p, 0.008, sl_max))
    tp_max_eff = cfg['tp_max'] * (3.0 if ctx_sltp and ctx_sltp['modo'] == 'swing' else 1.0)
    tp_pct = float(np.clip((atr * tp_mult) / p, sl_pct * 2.0, tp_max_eff))
    liq    = p * (1 + (1 / cfg['leverage']) * 0.90)
    return p * (1 + sl_pct), p * (1 - tp_pct), liq


def margen_efectivo(capital: float, cfg_margen: float, af: float,
                    margen_f: float = 1.0) -> float:
    """Margen con cap de funding y factor de contexto (swing reduce, rebote aumenta)."""
    cap = min(capital, CAPITAL_INICIAL * CAP_MULT_FUNDING)
    return cap * cfg_margen * af * margen_f


# ──────────────────────────────────────────────
#  GESTIÓN DE POSICIONES
# ──────────────────────────────────────────────

def gestionar_posiciones(estado: Dict, precio: float, atr: float,
                          hora: int, simbolo: str, ts: str) -> List[Dict]:
    cerradas = []
    nuevas   = []
    for pos in estado['posiciones']:
        if pos['simbolo'] != simbolo:
            nuevas.append(pos)
            continue
        horas = hora - pos['hora_entrada']
        max_h = MAX_HORAS_LONG if pos['dir'] == 'long' else MAX_HORAS_SHORT

        if pos['dir'] == 'long':
            if precio > pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                if (precio - pos['precio_entrada']) / pos['precio_entrada'] >= TRAILING_ACTIVACION:
                    pos['trailing'] = True
                    pos['sl'] = max(pos['sl'], pos['precio_ref'] - atr * TRAILING_ATR_MULT)
            cerrar = precio <= pos.get('liq', 0) and 'liq' in pos
            motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio <= pos['sl']:   cerrar = True; motivo = 'trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio >= pos['tp']: cerrar = True; motivo = 'take_profit'
                elif horas >= max_h:      cerrar = True; motivo = 'tiempo_maximo'
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
                telegram(f"{emoji} <b>CIERRE LONG {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo}\n"
                         f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Capital: ${estado['capital']:,.2f}")
            else:
                nuevas.append(pos)
        else:  # short
            if precio < pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                if (pos['precio_entrada'] - precio) / pos['precio_entrada'] >= TRAILING_ACTIVACION:
                    pos['trailing'] = True
                    nsl = pos['precio_ref'] * (1 + atr * TRAILING_ATR_MULT / pos['precio_ref'])
                    pos['sl'] = min(pos['sl'], nsl)
            cerrar = precio >= pos.get('liq', float('inf'))
            motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio >= pos['sl']:   cerrar = True; motivo = 'trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio <= pos['tp']: cerrar = True; motivo = 'take_profit'
                elif horas >= max_h:      cerrar = True; motivo = 'tiempo_maximo'
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
                telegram(f"{emoji} <b>CIERRE SHORT {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo}\n"
                         f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Capital: ${estado['capital']:,.2f}")
            else:
                nuevas.append(pos)
    estado['posiciones'] = nuevas
    return cerradas


def abrir_posicion(estado: Dict, simbolo: str, precio: float, atr: float,
                    prob: float, cfg: dict, calidad: str, dir_: str,
                    ts: str, hora: int, razones: List[str],
                    df_completo: pd.DataFrame = None, ctx4h: Dict = None):
    nombre    = simbolo.replace('/', '_')
    af        = estado.get('adaptive_factor', 1.0)

    # Analizar contexto para SL/TP inteligente
    ctx_sltp = None
    if df_completo is not None:
        ultima_row = df_completo.iloc[-1]
        ctx_sltp   = analizar_contexto_sltp(df_completo, ultima_row, ctx4h or {}, dir_)

    margen_f   = ctx_sltp['margen_f'] if ctx_sltp else 1.0
    margen_uso = sum(p['margen'] for p in estado['posiciones'] if p['simbolo'] == nombre)
    margen_des = margen_efectivo(estado['capital'], cfg['margen'], af, margen_f)
    margen_max = estado['capital'] * MAX_MARGEN_TOTAL_PCT
    margen_r   = min(margen_des, margen_max - margen_uso)
    if margen_r < estado['capital'] * 0.02:
        return

    if dir_ == 'long':
        pe = precio * (1 + SLIPPAGE_PCT)
        sl, tp, liq = sl_tp_long(pe, atr, cfg, ctx_sltp)
    else:
        pe = precio * (1 - SLIPPAGE_PCT)
        sl, tp, liq = sl_tp_short(pe, atr, cfg, ctx_sltp)

    modo_str = ctx_sltp['modo'].upper() if ctx_sltp else 'ESTANDAR'

    estado['posiciones'].append({
        'simbolo': nombre, 'dir': dir_, 'calidad': calidad,
        'precio_entrada': pe, 'precio_ref': pe,
        'sl': sl, 'tp': tp, 'liq': liq,
        'margen': margen_r, 'lev': cfg['leverage'],
        'hora_entrada': hora, 'trailing': False, 'ts_entrada': ts,
        'adaptive_factor': af,
        'modo_sltp': modo_str,
    })

    razones_str = '\n'.join(f"  · {r}" for r in razones[:4])
    sl_pct = abs(pe - sl) / pe * 100
    tp_pct = abs(tp - pe) / pe * 100

    # Info de niveles si es rebote
    nivel_info = ''
    if ctx_sltp and ctx_sltp['modo'] == 'rebote':
        if dir_ == 'long':
            nivel_info = f"\nSoporte: ${ctx_sltp['sop_cercano']:,.4f} ({ctx_sltp['dist_sop']:.2f}%)"
        else:
            nivel_info = f"\nResistencia: ${ctx_sltp['res_cercana']:,.4f} ({ctx_sltp['dist_res']:.2f}%)"

    telegram(f"🚀 <b>APERTURA {dir_.upper()} {simbolo}</b> [{VERSION}]\n"
             f"Modo: {modo_str} | Calidad: {calidad.upper()} | Prob: {prob:.3f}\n"
             f"Razones:\n{razones_str}\n"
             f"Precio: ${pe:,.4f}\n"
             f"SL: ${sl:,.4f} (-{sl_pct:.2f}%) | TP: ${tp:,.4f} (+{tp_pct:.2f}%){nivel_info}\n"
             f"Ratio R:R = 1:{tp_pct/sl_pct:.1f}\n"
             f"Margen: ${margen_r:.2f} | x{cfg['leverage']} | AF: x{af:.1f}\n"
             f"Capital: ${estado['capital']:,.2f}")


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

    msg = (f"📊 <b>REPORTE DIARIO {VERSION.upper()}</b>\n"
           f"{'─'*30}\n"
           f"💰 Capital: ${capital:,.2f} ({'+' if rent>=0 else ''}{rent:.2f}%)\n"
           f"📈 Total ops: {n} | WR: {wr:.1f}%\n"
           f"📊 WR últimas 20: {wr20:.1f}%\n"
           f"💵 P&L: ${estado['pnl_total']:+.2f}\n"
           f"📉 Funding: ${estado['funding_total']:.2f}\n"
           f"🔧 Adaptive: x{af:.1f}\n"
           f"🔓 Posiciones: {len(estado['posiciones'])}\n"
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
    print(f"  Paper Trading {VERSION} [AGRESIVO] — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Estrategia: PREDICTIVA + SL/TP INTELIGENTE — MODO AGRESIVO")
    print(f"{'='*55}")

    estado    = cargar_estado()
    hora_unix = int(time.time() / 3600)

    if estado.get('primera_ejecucion', True):
        telegram(f"🔥 <b>BOT {VERSION} INICIADO — MODO AGRESIVO</b>\n"
                 f"Capital: ${CAPITAL_INICIAL:,.2f}\n"
                 f"Diferencias vs v14 (modo conservador):\n"
                 f"  · Lev LONG: 12-20x (v14: 5-8x)\n"
                 f"  · Lev SHORT: 10-15x (v14: 4-6x)\n"
                 f"  · Margen máximo: 60% (v14: 40%)\n"
                 f"  · Posiciones máx: {MAX_POSICIONES} (v14: 2)\n"
                 f"  · TP swing: hasta 27% (v14: ~12%)\n"
                 f"  · Score mínimo: 1-2 (v14: 2-3)\n"
                 f"⚠️ Solo para paper trading hasta validar")
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

    for simbolo in CRIPTOS:
        nombre = simbolo.replace('/', '_')
        print(f"\n  📊 {simbolo}")

        # Datos 1H
        df = obtener_velas(simbolo, intervalo=60, limite=VELAS_N)
        if df is None:
            continue

        try:
            df = calcular_indicadores(df)
        except Exception as e:
            print(f"  ❌ Indicadores: {e}")
            continue

        df.dropna(subset=['ema_200', 'rsi_14', 'atr_14'], inplace=True)
        if len(df) < 50:
            continue

        # Contexto 4H
        ctx4h = contexto_4h(simbolo)
        if ctx4h.get('disponible'):
            print(f"  4H: tendencia={ctx4h['tendencia']} | momentum={ctx4h['momentum']} | RSI={ctx4h['rsi']:.0f}")

        ultima = df.iloc[-2]
        precio = float(ultima['close'])
        atr    = float(ultima.get('atr_14', precio * 0.015))
        if pd.isna(atr) or atr <= 0:
            atr = precio * 0.015
        ts = str(ultima['timestamp'])

        # Gestionar posiciones abiertas
        cerradas = gestionar_posiciones(estado, precio, atr, hora_unix, nombre, ts)
        for op in cerradas:
            registrar_op(op)
            estado.setdefault('ops_hoy', []).append(op)

        # Drawdown
        estado['capital_max'] = max(estado.get('capital_max', CAPITAL_INICIAL), estado['capital'])
        dd = (estado['capital_max'] - estado['capital']) / estado['capital_max']
        if dd >= DD_STOP_PCT:
            estado['bot_activo'] = False
            telegram(f"🛑 <b>BOT {VERSION} DETENIDO</b>\nDD: {dd*100:.1f}%\n"
                     f"Capital: ${estado['capital']:,.2f}")
            guardar_estado(estado)
            return
        if dd >= DD_PAUSA_PCT and not estado['en_pausa']:
            estado['en_pausa']       = True
            estado['fin_pausa_hora'] = hora_unix + HORAS_PAUSA
            telegram(f"⏸️ <b>PAUSA {VERSION}</b>\nDD: {dd*100:.1f}%")
            continue

        if int(ultima.get('volumen_muy_bajo', 0)) == 1:
            print(f"  ⚠️  Volumen muy bajo, skip")
            continue

        # Límites de posiciones
        n_longs  = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'long')
        n_shorts = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre and p['dir'] == 'short')
        total    = len(estado['posiciones'])

        if total >= MAX_POSICIONES:
            continue

        df_pred = df.iloc[:-1]  # excluir vela en formación

        # ── LONG ──
        if n_longs == 0:
            sub_long = cargar_submodelo(simbolo, 'long')
            if sub_long:
                prob_long = predecir(df_pred, sub_long)
                score_l, razones_l = score_predictivo_long(df_pred, ultima, ctx4h)
                min_s_l = score_minimo(prob_long, 'long')
                print(f"  LONG  prob={prob_long:.4f} score={score_l}/{min_s_l} | ${precio:,.2f}")

                if prob_long >= PROB_PREMIUM_LONG and score_l >= min_s_l:
                    res = clasificar_long(prob_long, score_l)
                    if res:
                        cal, cfg = res
                        abrir_posicion(estado, simbolo, precio, atr,
                                       prob_long, cfg, cal, 'long', ts, hora_unix,
                                       razones_l, df_pred, ctx4h)

        # ── SHORT ──
        if n_shorts == 0 and len(estado['posiciones']) < MAX_POSICIONES:
            # No abrir short si hay long con ganancia > 3%
            max_g_long = max(
                ((precio - p['precio_entrada']) / p['precio_entrada']
                 for p in estado['posiciones']
                 if p['simbolo'] == nombre and p['dir'] == 'long'),
                default=0.0)
            if max_g_long < 0.03:
                sub_short = cargar_submodelo(simbolo, 'short')
                if sub_short:
                    prob_short = predecir(df_pred, sub_short)
                    score_s, razones_s = score_predictivo_short(df_pred, ultima, ctx4h)
                    min_s_s = score_minimo(prob_short, 'short')
                    print(f"  SHORT prob={prob_short:.4f} score={score_s}/{min_s_s}")

                    if prob_short >= PROB_PREMIUM_SHORT and score_s >= min_s_s:
                        res = clasificar_short(prob_short, score_s)
                        if res:
                            cal, cfg = res
                            abrir_posicion(estado, simbolo, precio, atr,
                                           prob_short, cfg, cal, 'short', ts, hora_unix,
                                           razones_s, df_pred, ctx4h)

        time.sleep(0.5)

    guardar_estado(estado)
    m = guardar_metricas(estado)
    s = '+' if m['rentabilidad_pct'] >= 0 else ''
    print(f"\n  Capital: ${m['capital_actual']:,.2f} ({s}{m['rentabilidad_pct']:.2f}%)")
    print(f"  Ops: {m['n_operaciones']} | WR: {m['win_rate_pct']:.1f}% | Adaptive: x{m['adaptive_factor']:.1f}")


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
