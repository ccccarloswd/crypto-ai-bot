"""
preparar_datos.py  (v5)
========================
Cambios respecto a v4:
  - Features de flujo de órdenes reales (taker buy/sell ratio)
    → taker_ratio          : taker_buy_base / volume  (presión compradora instantánea)
    → taker_ratio_ma5      : media móvil 5 velas del ratio
    → taker_ratio_ma20     : media móvil 20 velas del ratio
    → taker_ratio_delta    : taker_ratio - taker_ratio_ma20  (presión relativa a la media)
    → taker_ratio_pendiente: diff(3) del ratio  (aceleración de la presión)
    → taker_dominance      : taker_buy_base / (volume - taker_buy_base)  (ratio buy/sell puro)
    → taker_cvd_20         : cumulative volume delta rolling 20 velas (buy_vol - sell_vol acumulado)
    → taker_cvd_tendencia  : CVD > su propia EMA20 (1/0)
    → vol_quality          : cuánto del volumen total es taker (activo vs pasivo)
  - Label oportunidad: retroceso medido con low/high en vez de close (corregido)
  - fraccion_horizonte_max = 1.0 (sin penalización por velocidad)
  - Parámetros de horizonte y umbral más tolerantes
"""

import os
import numpy as np
import pandas as pd

# ── Rutas ──────────────────────────────────────────────────────────────
RAW_DIR       = 'models/data/raw'
PROCESSED_DIR = 'models/data/processed'

# ── Monedas y timeframes ───────────────────────────────────────────────
SIMBOLOS   = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
TIMEFRAMES = ['1h', '4h', '1d']

# ── Parámetros Label A — Oportunidad operativa ─────────────────────────
HORIZONTE             = {'1h': 12,  '4h': 8,   '1d': 5}
UMBRAL_ATR_MULT       = {'1h': 0.6, '4h': 0.8, '1d': 1.0}
MAX_RETROCESO_ATR     = 0.6   # tolerancia de retroceso en múltiplos de ATR
FRACCION_HORIZONTE_MAX = 1.0  # sin penalizar por velocidad

# ── Parámetros Label B — Éxito de operación ────────────────────────────
SL_ATR_MULT  = 1.8
TP_ATR_MULT  = 1.8 * 2.2
MAX_VELAS_OP = {'1h': 24, '4h': 12, '1d': 7}


# ──────────────────────────────────────────────────────────────────────
#  CALCULAR INDICADORES
# ──────────────────────────────────────────────────────────────────────
def calcular_indicadores(df: pd.DataFrame,
                          btc_close: pd.Series = None) -> pd.DataFrame:
    # Garantizar orden cronológico
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    o = df['open']

    # ── EMAs ──
    for p in [9, 21, 50, 100, 200]:
        df[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    for p in [20, 50, 200]:
        df[f'sma_{p}'] = c.rolling(p).mean()

    # ── MACD ──
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['macd_cruce']  = np.where(
        (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
        np.where(
            (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0
        )
    )
    df['macd_hist_pendiente'] = df['macd_hist'].diff(2)

    # ── RSI ──
    def rsi(serie, p=14):
        delta = serie.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = rsi(c, 14)
    df['rsi_7']  = rsi(c, 7)
    df['rsi_sobrecompra']  = (df['rsi_14'] >= 70).astype(int)
    df['rsi_sobreventa']   = (df['rsi_14'] <= 30).astype(int)
    df['rsi_zona_neutral'] = ((df['rsi_14'] > 40) & (df['rsi_14'] < 60)).astype(int)

    ps     = (c > c.shift(5)).astype(int)
    rs_dir = (df['rsi_14'] > df['rsi_14'].shift(5)).astype(int)
    df['divergencia_bajista_rsi'] = ((ps == 1) & (rs_dir == 0)).astype(int)
    df['divergencia_alcista_rsi'] = ((ps == 0) & (rs_dir == 1)).astype(int)
    df['rsi_pendiente'] = df['rsi_14'].diff(3)

    # ── ATR ──
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14']  = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / c * 100

    # ── ADX / DMI ──
    up       = h.diff()
    down     = -l.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr14    = df['atr_14'].replace(0, np.nan)
    plus_di  = pd.Series(plus_dm,  index=df.index).rolling(14).mean() / atr14 * 100
    minus_di = pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr14 * 100
    dx       = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
    df['adx']     = dx.rolling(14).mean()
    df['dmi_pos'] = plus_di
    df['dmi_neg'] = minus_di

    # ── Cruces EMA ──
    df['precio_sobre_ema50']  = (c > df['ema_50']).astype(int)
    df['precio_sobre_ema200'] = (c > df['ema_200']).astype(int)
    df['ema50_sobre_ema200']  = (df['ema_50'] > df['ema_200']).astype(int)

    # ── Stochastic ──
    low14       = l.rolling(14).min()
    high14      = h.rolling(14).max()
    stoch_k_raw = (c - low14) / (high14 - low14).replace(0, np.nan) * 100
    df['stoch_k']           = stoch_k_raw.rolling(3).mean()
    df['stoch_d']           = df['stoch_k'].rolling(3).mean()
    df['stoch_sobrecompra'] = (df['stoch_k'] >= 80).astype(int)
    df['stoch_sobreventa']  = (df['stoch_k'] <= 20).astype(int)

    # ── CCI ──
    tp_cci  = (h + l + c) / 3
    sma_cci = tp_cci.rolling(20).mean()
    mad_cci = tp_cci.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci_20'] = (tp_cci - sma_cci) / (0.015 * mad_cci.replace(0, np.nan))

    # ── ROC / Momentum ──
    df['roc_10']  = c.pct_change(10) * 100
    df['williams_r'] = (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan) * -100

    # ── Bollinger Bands ──
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20
    bb_width_raw   = df['bb_upper'] - df['bb_lower']
    df['bb_width']    = bb_width_raw / sma20.replace(0, np.nan)
    df['bb_posicion'] = np.where(bb_width_raw > 0, (c - df['bb_lower']) / bb_width_raw, 0.5)
    df['bb_squeeze']  = (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.7).astype(int)

    # ── Volatilidad ──
    ret = c.pct_change()
    df['volatilidad_20'] = ret.rolling(20).std() * np.sqrt(24 * 365) * 100
    df['rango_vela_pct'] = (h - l) / c * 100

    # ── OBV ──
    obv_vals = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df['obv']           = obv_vals
    df['obv_ema']       = obv_vals.ewm(span=20, adjust=False).mean()
    df['obv_tendencia'] = (df['obv'] > df['obv_ema']).astype(int)

    # ── Volumen clásico ──
    vm = v.rolling(20).mean()
    vs = v.rolling(20).std()
    df['volumen_ratio']   = v / vm.replace(0, np.nan)
    df['volumen_anomalo'] = (v > vm + 2 * vs).astype(int)
    vela_alcista = (c > o).astype(int)
    df['confirmacion_alcista'] = ((vela_alcista == 1) & (df['volumen_anomalo'] == 1)).astype(int)
    df['confirmacion_bajista'] = ((vela_alcista == 0) & (df['volumen_anomalo'] == 1)).astype(int)
    df['vol_ratio_5'] = v / v.rolling(5).mean().replace(0, np.nan)

    # ── Fibonacci ──
    mx = h.rolling(50).max()
    mn = l.rolling(50).min()
    rg = mx - mn
    df['fib_236'] = mx - rg * 0.236
    df['fib_382'] = mx - rg * 0.382
    df['fib_500'] = mx - rg * 0.500
    df['fib_618'] = mx - rg * 0.618
    for nv in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
        df[f'cerca_{nv}'] = (abs(c - df[nv]) / c < 0.005).astype(int)

    # ── Régimen de mercado ──
    df['ema200_pendiente'] = df['ema_200'].diff(50) / df['ema_200'].shift(50) * 100
    se200  = (c > df['ema_200']).astype(int)
    e5_200 = (df['ema_50'] > df['ema_200']).astype(int)
    conds  = [
        (se200 == 1) & (e5_200 == 1) & (df['ema200_pendiente'] > 0.5),
        (se200 == 1) & (e5_200 == 1),
        (se200 == 0) & (e5_200 == 0) & (df['ema200_pendiente'] < -0.5),
        (se200 == 0) & (e5_200 == 0),
    ]
    df['regimen_mercado'] = np.select(conds, [2, 1, -2, -1], default=0)

    # ── Features adicionales ML ──
    rango = (h - l).replace(0, np.nan)
    df['posicion_en_vela'] = (c - l) / rango

    for p in [1, 5, 8]:
        df[f'ret_{p}'] = c.pct_change(p) * 100

    for p in [21, 50, 200]:
        df[f'dist_ema{p}_atr'] = (c - df[f'ema_{p}']) / df['atr_14']

    # ── Microestructura de vela ──
    body        = (c - o).abs()
    total_range = (h - l).replace(0, np.nan)
    df['body_ratio']    = body / total_range
    df['upper_wick']    = (h - c.clip(lower=o)) / total_range
    df['lower_wick']    = (c.clip(upper=o) - l) / total_range
    df['spread_hl_pct'] = (h - l) / l * 100

    # ── Estructura de precio ──
    df['dist_max_24h'] = (h.rolling(24).max() - c) / c * 100
    df['dist_min_24h'] = (c - l.rolling(24).min()) / c * 100
    hh48 = h.rolling(48).max()
    ll48 = l.rolling(48).min()
    df['posicion_48h'] = (c - ll48) / (hh48 - ll48).replace(0, np.nan)

    # ── Régimen de volatilidad ──
    vol_media_100 = df['atr_pct'].rolling(100).mean()
    df['regimen_volatilidad'] = (df['atr_pct'] > vol_media_100 * 1.3).astype(int)

    # ── BTC como contexto ──
    btc      = btc_close if btc_close is not None else c
    btc_vals = btc.values if hasattr(btc, 'values') else np.array(btc)
    btc_s    = pd.Series(btc_vals, index=df.index)
    df['btc_ret_1h']   = btc_s.pct_change(1) * 100
    df['btc_ret_4h']   = btc_s.pct_change(4) * 100
    df['corr_btc_24h'] = c.pct_change().rolling(24).corr(btc_s.pct_change())

    # ── TAKER FLOW — el núcleo del cambio ──────────────────────────────
    if 'taker_buy_base' in df.columns and 'volume' in df.columns:
        taker_buy  = df['taker_buy_base'].astype(float)
        taker_sell = (v - taker_buy).clip(lower=0)

        # Ratio comprador instantáneo: 0=todo sell, 1=todo buy, 0.5=neutro
        taker_ratio = taker_buy / v.replace(0, np.nan)
        df['taker_ratio'] = taker_ratio

        # Medias móviles del ratio para ver tendencia de flujo
        df['taker_ratio_ma5']  = taker_ratio.rolling(5).mean()
        df['taker_ratio_ma20'] = taker_ratio.rolling(20).mean()

        # Presión relativa: ratio actual vs su media → positivo = más compradores que la media
        df['taker_ratio_delta'] = taker_ratio - df['taker_ratio_ma20']

        # Aceleración del flujo comprador (¿está aumentando o disminuyendo la presión?)
        df['taker_ratio_pendiente'] = taker_ratio.diff(3)

        # Ratio buy/sell puro (dominancia compradora): >1 = más buy que sell
        df['taker_dominance'] = taker_buy / taker_sell.replace(0, np.nan)
        df['taker_dominance'] = df['taker_dominance'].clip(0, 10)  # cap outliers

        # Cumulative Volume Delta rolling 20 velas: suma de (buy_vol - sell_vol)
        # Mide si el dinero activo está acumulando o distribuyendo
        cvd = (taker_buy - taker_sell)
        df['taker_cvd_20'] = cvd.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

        # CVD normalizado por encima/debajo de su propia EMA → tendencia de flujo
        cvd_ema = df['taker_cvd_20'].ewm(span=10, adjust=False).mean()
        df['taker_cvd_tendencia'] = (df['taker_cvd_20'] > cvd_ema).astype(int)

        # Calidad del volumen: qué fracción es taker (activo) vs maker (pasivo)
        # Volumen alto pero todo maker = poco convicción; alto y taker = movimiento real
        df['vol_quality'] = taker_ratio * df['volumen_ratio']

    else:
        # Si no hay datos taker (CSVs viejos), rellenar con neutro para no romper el pipeline
        print("  ⚠️  Sin columnas taker — rellenando con valores neutros. "
              "Ejecuta descargar_datos.py para obtener datos completos.")
        for col in ['taker_ratio', 'taker_ratio_ma5', 'taker_ratio_ma20',
                    'taker_ratio_delta', 'taker_ratio_pendiente',
                    'taker_dominance', 'taker_cvd_20', 'taker_cvd_tendencia', 'vol_quality']:
            df[col] = np.nan

    return df


# ──────────────────────────────────────────────────────────────────────
#  LABEL A — OPORTUNIDAD OPERATIVA LIMPIA
# ──────────────────────────────────────────────────────────────────────
def crear_label_oportunidad(df: pd.DataFrame, horizonte: int,
                             umbral_atr_mult: float,
                             max_retroceso_atr: float,
                             fraccion_max: float) -> pd.Series:
    """
    Retroceso y objetivo medidos con low/high (simétrico con SL/TP reales).
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    c   = df['close'].values
    h   = df['high'].values
    l   = df['low'].values
    atr = df['atr_14'].values
    n   = len(c)

    labels       = np.full(n, np.nan)
    limite_velas = max(1, int(horizonte * fraccion_max))

    for i in range(n - horizonte):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entrada         = c[i]
        objetivo_subida = atr[i] * umbral_atr_mult
        objetivo_caida  = atr[i] * umbral_atr_mult
        retroceso_max   = atr[i] * max_retroceso_atr

        long_ok  = False
        short_ok = False

        max_caida_low   = 0.0
        max_subida_high = 0.0

        for k, j in enumerate(range(i + 1, i + 1 + horizonte)):
            if j >= n:
                break

            dentro_limite    = (k + 1) <= limite_velas
            caida_low        = entrada - l[j]
            subida_high      = h[j] - entrada
            max_caida_low    = max(max_caida_low,   caida_low)
            max_subida_high  = max(max_subida_high, subida_high)

            # LONG: high llega al objetivo, el low nunca rompió el retroceso máximo
            if not long_ok:
                if subida_high >= objetivo_subida:
                    if max_caida_low <= retroceso_max and dentro_limite:
                        long_ok = True

            # SHORT: low baja al objetivo, el high nunca superó el retroceso máximo
            if not short_ok:
                if caida_low >= objetivo_caida:
                    if max_subida_high <= retroceso_max and dentro_limite:
                        short_ok = True

            if long_ok and short_ok:
                break

        if long_ok and not short_ok:
            labels[i] = 1
        elif short_ok and not long_ok:
            labels[i] = 0
        # ambos o ninguno → NaN

    return pd.Series(labels, index=df.index, name='label_oportunidad')


# ──────────────────────────────────────────────────────────────────────
#  LABEL B — ÉXITO DE OPERACIÓN
# ──────────────────────────────────────────────────────────────────────
def crear_label_exito(df: pd.DataFrame, max_velas: int,
                       sl_mult: float, tp_mult: float) -> tuple:
    c   = df['close'].values
    h   = df['high'].values
    l   = df['low'].values
    atr = df['atr_14'].values
    n   = len(c)

    exito_long  = np.full(n, np.nan)
    exito_short = np.full(n, np.nan)

    for i in range(n - max_velas):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entrada  = c[i]
        sl_long  = entrada - atr[i] * sl_mult
        tp_long  = entrada + atr[i] * tp_mult
        sl_short = entrada + atr[i] * sl_mult
        tp_short = entrada - atr[i] * tp_mult

        resultado_long  = np.nan
        resultado_short = np.nan

        for j in range(i + 1, min(i + 1 + max_velas, n)):
            high_j = h[j]
            low_j  = l[j]

            if np.isnan(resultado_long):
                toca_sl = low_j  <= sl_long
                toca_tp = high_j >= tp_long
                if toca_tp and toca_sl:
                    resultado_long = 1 if (tp_long - entrada) < (entrada - sl_long) else 0
                elif toca_tp:
                    resultado_long = 1
                elif toca_sl:
                    resultado_long = 0

            if np.isnan(resultado_short):
                toca_sl = high_j >= sl_short
                toca_tp = low_j  <= tp_short
                if toca_tp and toca_sl:
                    resultado_short = 1 if (entrada - tp_short) < (sl_short - entrada) else 0
                elif toca_tp:
                    resultado_short = 1
                elif toca_sl:
                    resultado_short = 0

            if not np.isnan(resultado_long) and not np.isnan(resultado_short):
                break

        exito_long[i]  = resultado_long
        exito_short[i] = resultado_short

    return (
        pd.Series(exito_long,  index=df.index, name='label_exito_long'),
        pd.Series(exito_short, index=df.index, name='label_exito_short'),
    )


# ──────────────────────────────────────────────────────────────────────
#  LISTA DE FEATURES — sincronizada con entrenar.py
# ──────────────────────────────────────────────────────────────────────
FEATURES = [
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
    # ── TAKER FLOW — nuevos ──
    'taker_ratio',           # presión compradora instantánea (0-1)
    'taker_ratio_ma5',       # tendencia corta del flujo comprador
    'taker_ratio_ma20',      # tendencia media del flujo comprador
    'taker_ratio_delta',     # flujo actual vs su media (positivo = más compra que habitual)
    'taker_ratio_pendiente', # ¿está acelerando o frenando la presión compradora?
    'taker_dominance',       # ratio buy/sell puro
    'taker_cvd_20',          # CVD rolling 20: acumulación vs distribución
    'taker_cvd_tendencia',   # CVD por encima de su EMA (1/0)
    'vol_quality',           # volumen alto Y taker = convicción real
]


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────
def cargar_btc(tf: str):
    ruta = os.path.join(RAW_DIR, f"BTCUSDT_{tf}.csv")
    if not os.path.exists(ruta):
        print(f"  ⚠️  No se encontró BTCUSDT_{tf}.csv")
        return None
    btc_df = pd.read_csv(ruta, parse_dates=['timestamp'])
    return btc_df.set_index('timestamp')['close']


def procesar(simbolo: str, tf: str, btc_raw):
    ruta_raw = os.path.join(RAW_DIR, f"{simbolo}_{tf}.csv")
    if not os.path.exists(ruta_raw):
        print(f"  ❌ No encontrado: {ruta_raw}")
        return

    df = pd.read_csv(ruta_raw, parse_dates=['timestamp'])
    print(f"  📂 {simbolo} {tf}: {len(df)} velas cargadas")

    if 'taker_buy_base' not in df.columns:
        print(f"  ⚠️  {simbolo} {tf}: sin datos taker — ejecuta descargar_datos.py primero")

    # Alinear BTC con esta moneda
    btc_close_alineado = None
    if btc_raw is not None and simbolo != 'BTCUSDT':
        btc_reindexed      = btc_raw.reindex(df['timestamp']).ffill()
        btc_close_alineado = btc_reindexed.reset_index(drop=True)

    df = calcular_indicadores(df, btc_close=btc_close_alineado)

    horizonte       = HORIZONTE[tf]
    umbral_atr_mult = UMBRAL_ATR_MULT[tf]
    max_velas       = MAX_VELAS_OP[tf]

    print(f"     Calculando label oportunidad "
          f"(horizonte={horizonte}, umbral={umbral_atr_mult}xATR, "
          f"retroceso_max={MAX_RETROCESO_ATR}xATR)...")
    df['label_oportunidad'] = crear_label_oportunidad(
        df, horizonte, umbral_atr_mult, MAX_RETROCESO_ATR, FRACCION_HORIZONTE_MAX
    )

    print(f"     Calculando label éxito de operación (max_velas={max_velas})...")
    df['label_exito_long'], df['label_exito_short'] = crear_label_exito(
        df, max_velas, SL_ATR_MULT, TP_ATR_MULT
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    ruta_out = os.path.join(PROCESSED_DIR, f"{simbolo}_{tf}.csv")
    df.to_csv(ruta_out, index=False)

    for label_col in ['label_oportunidad', 'label_exito_long', 'label_exito_short']:
        if label_col not in df.columns:
            continue
        serie   = df[label_col].dropna()
        n_total = len(df)
        n_val   = len(serie)
        if n_val == 0:
            print(f"     [{label_col}]  ⚠️  0 muestras válidas")
            continue
        n_pos = int(serie.sum())
        n_neg = n_val - n_pos
        print(f"     [{label_col}]  total={n_total}  validos={n_val} ({n_val/n_total:.0%})  "
              f"1s={n_pos} ({n_pos/n_val:.1%})  0s={n_neg} ({n_neg/n_val:.1%})")

    print(f"     ✅ Guardado: {ruta_out}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"\n{'═'*55}")
    print(f"  PREPARANDO DATOS DE ENTRENAMIENTO  (v5)")
    print(f"{'═'*55}\n")

    for tf in TIMEFRAMES:
        btc_raw = cargar_btc(tf)
        for simbolo in SIMBOLOS:
            print(f"\n── {simbolo} {tf} ──")
            procesar(simbolo, tf, btc_raw)

    print(f"\n{'═'*55}")
    print(f"  ✅ Datos procesados en: {os.path.abspath(PROCESSED_DIR)}")
    print(f"  Siguiente paso: ejecutar entrenar.py")
    print(f"{'═'*55}\n")


if __name__ == '__main__':
    main()
