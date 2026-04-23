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

MAX_HORAS_LONG  = 20
MAX_HORAS_SHORT = 16
MAX_POSICIONES       = 2
MAX_MARGEN_TOTAL_PCT = 0.40
TRAILING_ACTIVACION  = 0.018
TRAILING_ATR_MULT    = 1.0

SOLO_XGBOOST     = ['BTC_USDT', 'BNB_USDT']
CAP_MULT_FUNDING = 8.0

ADAPTIVE_VENTANA  = 20
ADAPTIVE_WR_BAJO  = 0.48
ADAPTIVE_WR_ALTO  = 0.65
ADAPTIVE_FACTOR_C = 0.50
ADAPTIVE_FACTOR_A = 1.20
ADAPTIVE_DURACION = 15

# ── Config por calidad (uniforme todas las criptos) ──
LONG_CONFIG = {
    'elite':   {'leverage': 24, 'margen': 0.22, 'sl_mult': 1.2,
                'sl_max': 0.020, 'tp_ratio': 2.8, 'tp_max': 0.060},
    'premium': {'leverage': 15, 'margen': 0.20, 'sl_mult': 1.5,
                'sl_max': 0.025, 'tp_ratio': 2.5, 'tp_max': 0.050},
}
SHORT_CONFIG = {
    'elite':   {'leverage': 18, 'margen': 0.18, 'sl_mult': 1.3,
                'sl_max': 0.022, 'tp_ratio': 2.8, 'tp_max': 0.055},
    'premium': {'leverage': 12, 'margen': 0.15, 'sl_mult': 1.5,
                'sl_max': 0.025, 'tp_ratio': 2.5, 'tp_max': 0.045},
}

# ── Umbrales de probabilidad ──
# La IA ya filtró mucho. No la filtres demasiado con indicadores.
PROB_ELITE_LONG    = 0.74
PROB_PREMIUM_LONG  = 0.64
PROB_ELITE_SHORT   = 0.70
PROB_PREMIUM_SHORT = 0.62

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
        # Movimiento amplio: SL lejos para aguantar pullbacks,
        # TP lejos para capturar el movimiento completo.
        # Margen reducido porque SL es más amplio.
        sl_mult  = 2.2    # SL a 2.2× ATR (más lejos)
        tp_mult  = 5.0    # TP a 5× ATR (captura movimiento)
        margen_f = 0.75   # reducir margen al 75% del normal

    elif modo == 'rebote':
        # Precio en nivel clave: SL justo detrás del nivel
        # (un poco más allá del soporte/resistencia),
        # TP en el siguiente nivel (antes de la siguiente resistencia/soporte).
        if direccion == 'long':
            # SL justo bajo el soporte + 0.3% de buffer
            sl_dist  = dist_sop + 0.003
            sl_mult  = sl_dist / atr_pct if atr_pct > 0 else 1.5
            sl_mult  = float(np.clip(sl_mult, 1.0, 2.0))
            # TP en la resistencia más cercana - 0.5% de buffer (salir antes)
            tp_dist  = dist_res * 0.85 if dist_res > 0.01 else atr_pct * 2.5
            tp_mult  = tp_dist / atr_pct if atr_pct > 0 else 2.5
            tp_mult  = float(np.clip(tp_mult, 1.5, 4.0))
        else:
            # Short: SL justo sobre la resistencia + 0.3% buffer
            sl_dist  = dist_res + 0.003
            sl_mult  = sl_dist / atr_pct if atr_pct > 0 else 1.5
            sl_mult  = float(np.clip(sl_mult, 1.0, 2.0))
            # TP en el soporte más cercano - 0.5% buffer
            tp_dist  = dist_sop * 0.85 if dist_sop > 0.01 else atr_pct * 2.5
            tp_mult  = tp_dist / atr_pct if atr_pct > 0 else 2.5
            tp_mult  = float(np.clip(tp_mult, 1.5, 4.0))
        margen_f = 1.10   # margen ligeramente mayor (trade preciso = más seguro)

    else:  # estandar
        sl_mult  = 1.5
        tp_mult  = 3.0
        margen_f = 1.0

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
    ctx_sltp viene de analizar_contexto_sltp().
    Si no hay contexto, usa los valores del cfg estándar.
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
    tp_pct = float(np.clip((atr * tp_mult) / p, sl_pct * 2.0, cfg['tp_max'] * (2.0 if ctx_sltp and ctx_sltp['modo'] == 'swing' else 1.0)))
    liq    = p * (1 - (1 / cfg['leverage']) * 0.90)
    return p * (1 - sl_pct), p * (1 + tp_pct), liq


def sl_tp_short(p, atr, cfg, ctx_sltp: Dict = None):
    """SL/TP para SHORT con contexto inteligente."""
    if ctx_sltp:
        sl_mult = ctx_sltp['sl_mult']
        tp_mult = ctx_sltp['tp_mult']
        sl_max  = cfg['sl_max'] * (1.5 if ctx_sltp['modo'] == 'swing' else 1.0)
    else:
        sl_mult = cfg['sl_mult']
        tp_mult = cfg['tp_ratio']
        sl_max  = cfg['sl_max']

    sl_pct = float(np.clip((atr * sl_mult) / p, 0.008, sl_max))
    tp_pct = float(np.clip((atr * tp_mult) / p, sl_pct * 2.0, cfg['tp_max'] * (2.0 if ctx_sltp and ctx_sltp['modo'] == 'swing' else 1.0)))
    liq    = p * (1 + (1 / cfg['leverage']) * 0.90)
    return p * (1 + sl_pct), p * (1 - tp_pct), liq


def margen_efectivo(capital: float, cfg_margen: float, af: float,
                    margen_f: float = 1.0) -> float:
    """Margen usando el capital ACTUAL (crece/decrece con el bot).
    El cap de funding se calcula sobre el capital real, no el inicial."""
    cap = min(capital, capital * CAP_MULT_FUNDING)  # siempre el capital actual
    return cap * cfg_margen * af * margen_f


def margen_total_usado(estado: Dict) -> float:
    """Suma de todos los márgenes puestos como garantía en posiciones abiertas.
    El margen es el dinero TUYO inmovilizado (exposición / leverage).
    Ej: posición de 1000$ con x10 → margen = 100$ tuyos."""
    return sum(p['margen'] for p in estado['posiciones'])


def limite_margen_ok(estado: Dict, margen_nuevo: float) -> bool:
    """El margen total usado (dinero tuyo en garantías) nunca puede superar
    el capital real disponible. Si tienes 1000$, puedes abrir posiciones
    apalancadas por lo que quieras, pero la suma de garantías <= 1000$.
    Esto evita operar con más dinero del que realmente tienes."""
    usado  = margen_total_usado(estado)
    limite = estado['capital']  # nunca gastar más margen del que tienes
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
                elif precio >= pos['tp'] or (high is not None and high >= pos['tp']): cerrar = True; motivo = 'take_profit'
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
                rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
                expo_now = margen_total_usado(estado)
                telegram(f"{emoji} <b>CIERRE LONG {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo}\n"
                         f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Funding: ${fund:.2f}\n"
                         f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_now>=0 else ''}{rent_now:.2f}%)\n"
                         f"🔒 Margen restante en uso: ${expo_now:.2f} | Posiciones: {len(estado['posiciones'])}")
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
                elif precio <= pos['tp'] or (low is not None and low <= pos['tp']): cerrar = True; motivo = 'take_profit'
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
                rent_now = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100
                expo_now = margen_total_usado(estado)
                telegram(f"{emoji} <b>CIERRE SHORT {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo}\n"
                         f"${pos['precio_entrada']:,.4f} → ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Funding: ${fund:.2f}\n"
                         f"💰 Capital: ${estado['capital']:,.2f} ({'+' if rent_now>=0 else ''}{rent_now:.2f}%)\n"
                         f"🔒 Margen restante en uso: ${expo_now:.2f} | Posiciones: {len(estado['posiciones'])}")
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

    # ── GUARD ANTI-DUPLICADOS ──
    # Si ya hay una posición en la misma dirección y símbolo, no abrir otra.
    # Esta comprobación vive aquí dentro para que sea imposible saltársela.
    ya_existe = any(
        p['simbolo'] == nombre and p['dir'] == dir_
        for p in estado['posiciones']
    )
    if ya_existe:
        print(f"  🚫 BLOQUEADO: ya hay un {dir_.upper()} abierto en {nombre}, no se duplica")
        return

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
        print(f"  ⚠️  Margen insuficiente para {nombre} {dir_}, skip")
        return

    # ── Control de margen total ──
    # La suma de márgenes (dinero tuyo como garantía) no puede superar el capital real.
    # Ej: con 1000$ puedes abrir posiciones apalancadas, pero tus garantías <= 1000$.
    if not limite_margen_ok(estado, margen_r):
        margen_usado = margen_total_usado(estado)
        print(f"  ⚠️  Margen total agotado: ${margen_usado:.2f} usado de ${estado['capital']:.2f} "
              f"disponibles, skip {nombre} {dir_}")
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

    margen_usado = margen_total_usado(estado)
    margen_pct   = margen_usado / estado['capital'] * 100 if estado['capital'] > 0 else 0
    rent_act     = (estado['capital'] - CAPITAL_INICIAL) / CAPITAL_INICIAL * 100

    telegram(f"🚀 <b>APERTURA {dir_.upper()} {simbolo}</b> [{VERSION}]\n"
             f"Modo: {modo_str} | Calidad: {calidad.upper()} | Prob: {prob:.3f}\n"
             f"Razones:\n{razones_str}\n"
             f"Precio: ${pe:,.4f}\n"
             f"SL: ${sl:,.4f} (-{sl_pct:.2f}%) | TP: ${tp:,.4f} (+{tp_pct:.2f}%){nivel_info}\n"
             f"Ratio R:R = 1:{tp_pct/sl_pct:.1f}\n"
             f"Margen: ${margen_r:.2f} | x{cfg['leverage']} | AF: x{af:.1f}\n"
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
        high   = float(ultima['high'])
        low    = float(ultima['low'])
        atr    = float(ultima.get('atr_14', precio * 0.015))
        if pd.isna(atr) or atr <= 0:
            atr = precio * 0.015
        ts = str(ultima['timestamp'])

        # Gestionar posiciones abiertas
        # Pasamos high/low para detectar TPs tocados intradía aunque el close no los alcance
        cerradas = gestionar_posiciones(estado, precio, atr, hora_unix, nombre, ts,
                                        high=high, low=low)
        for op in cerradas:
            registrar_op(op)
            estado.setdefault('ops_hoy', []).append(op)

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
    expo_fin = margen_total_usado(estado)
    expo_fin_pct = expo_fin / estado['capital'] * 100 if estado['capital'] > 0 else 0
    print(f"\n  💰 Capital: ${m['capital_actual']:,.2f} ({s}{m['rentabilidad_pct']:.2f}%)")
    print(f"  🔒 Margen en uso: ${expo_fin:.2f} ({expo_fin_pct:.0f}% del capital)")
    print(f"  Ops: {m['n_operaciones']} | WR: {m['win_rate_pct']:.1f}% | Adaptive: x{m['adaptive_factor']:.1f}")
    print(f"  Posiciones abiertas: {len(estado['posiciones'])}")


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
