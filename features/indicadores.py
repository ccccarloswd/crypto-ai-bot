"""
=============================================================
  CRYPTO AI BOT — Indicadores Técnicos
=============================================================
  Calcula todos los indicadores técnicos sobre un DataFrame
  OHLCV. Cada función recibe el DataFrame y devuelve el mismo
  DataFrame con nuevas columnas añadidas.

  Categorías:
    - Tendencia   : EMA, SMA, MACD, ADX, Ichimoku
    - Momentum    : RSI, Stochastic, CCI, ROC, Williams %R
    - Volatilidad : Bollinger Bands, ATR, Keltner Channels
    - Volumen     : OBV, VWAP, MFI, anomalías de volumen
=============================================================
"""

import pandas as pd
import numpy as np
import pandas_ta as ta


# ──────────────────────────────────────────────
#  TENDENCIA
# ──────────────────────────────────────────────

def añadir_tendencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMAs, SMAs, MACD, ADX e Ichimoku.
    Añade también señales de cruce entre medias.
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    # --- Medias móviles exponenciales ---
    for periodo in [9, 21, 50, 100, 200]:
        df[f'ema_{periodo}'] = ta.ema(close, length=periodo)

    # --- Medias móviles simples ---
    for periodo in [20, 50, 200]:
        df[f'sma_{periodo}'] = ta.sma(close, length=periodo)

    # --- MACD (12, 26, 9) ---
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd']          = macd_df['MACD_12_26_9']
        df['macd_signal']   = macd_df['MACDs_12_26_9']
        df['macd_hist']     = macd_df['MACDh_12_26_9']
        df['macd_cruce']    = np.where(
            (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
            np.where(
                (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0
            )
        )

    # --- ADX — fuerza de la tendencia ---
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None:
        df['adx']    = adx_df['ADX_14']
        df['dmi_pos'] = adx_df['DMP_14']
        df['dmi_neg'] = adx_df['DMN_14']

    # --- Ichimoku Cloud ---
    ichimoku_df, _ = ta.ichimoku(high, low, close)
    if ichimoku_df is not None:
        col_map = {
            'ichi_tenkan':   'ITS_9',
            'ichi_kijun':    'IKS_26',
            'ichi_senkou_a': 'ISA_9',
            'ichi_senkou_b': 'ISB_26',
            'ichi_chikou':   'ICS_26',
        }
        for col_dest, col_src in col_map.items():
            df[col_dest] = ichimoku_df[col_src] if col_src in ichimoku_df.columns else np.nan

    # --- Señales de cruce de medias (golden cross / death cross) ---
    df['golden_cross'] = np.where(
        (df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1)), 1, 0
    )
    df['death_cross'] = np.where(
        (df['ema_50'] < df['ema_200']) & (df['ema_50'].shift(1) >= df['ema_200'].shift(1)), 1, 0
    )

    # --- Posición del precio respecto a medias ---
    df['precio_sobre_ema50']  = (close > df['ema_50']).astype(int)
    df['precio_sobre_ema200'] = (close > df['ema_200']).astype(int)
    df['ema50_sobre_ema200']  = (df['ema_50'] > df['ema_200']).astype(int)

    return df


# ──────────────────────────────────────────────
#  MOMENTUM
# ──────────────────────────────────────────────

def añadir_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI, Stochastic, CCI, ROC, Williams %R.
    Incluye zonas de sobrecompra/sobreventa.
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    # --- RSI ---
    df['rsi_14'] = ta.rsi(close, length=14)
    df['rsi_7']  = ta.rsi(close, length=7)

    # Zonas RSI
    df['rsi_sobrecompra']  = (df['rsi_14'] >= 70).astype(int)
    df['rsi_sobreventa']   = (df['rsi_14'] <= 30).astype(int)
    df['rsi_zona_neutral'] = ((df['rsi_14'] > 40) & (df['rsi_14'] < 60)).astype(int)

    # Divergencia RSI (precio sube pero RSI baja = señal bajista, y viceversa)
    precio_sube = (close > close.shift(5)).astype(int)
    rsi_sube    = (df['rsi_14'] > df['rsi_14'].shift(5)).astype(int)
    df['divergencia_bajista_rsi'] = ((precio_sube == 1) & (rsi_sube == 0)).astype(int)
    df['divergencia_alcista_rsi'] = ((precio_sube == 0) & (rsi_sube == 1)).astype(int)

    # --- Stochastic ---
    stoch_df = ta.stoch(high, low, close, k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]
        df['stoch_sobrecompra'] = (df['stoch_k'] >= 80).astype(int)
        df['stoch_sobreventa']  = (df['stoch_k'] <= 20).astype(int)

    # --- CCI (Commodity Channel Index) ---
    df['cci_20'] = ta.cci(high, low, close, length=20)

    # --- ROC (Rate of Change) — momentum puro ---
    df['roc_10'] = ta.roc(close, length=10)
    df['roc_20'] = ta.roc(close, length=20)

    # --- Williams %R ---
    df['williams_r'] = ta.willr(high, low, close, length=14)

    # --- Momentum simple ---
    df['momentum_10'] = close - close.shift(10)
    df['momentum_pct_10'] = close.pct_change(periods=10) * 100

    return df


# ──────────────────────────────────────────────
#  VOLATILIDAD
# ──────────────────────────────────────────────

def añadir_volatilidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bollinger Bands, ATR, Keltner Channels y métricas de volatilidad.
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    # --- Bollinger Bands (20, 2) ---
    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None:
        df['bb_upper'] = bb_df['BBU_20_2.0_2.0']
        df['bb_mid']   = bb_df['BBM_20_2.0_2.0']
        df['bb_lower'] = bb_df['BBL_20_2.0_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']  # normalizado

        # Posición del precio dentro de las bandas (0 = lower, 1 = upper)
        rango_bb = df['bb_upper'] - df['bb_lower']
        df['bb_posicion'] = np.where(
            rango_bb > 0,
            (close - df['bb_lower']) / rango_bb,
            0.5
        )
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.7).astype(int)

    # --- ATR (Average True Range) — volatilidad real ---
    df['atr_14'] = ta.atr(high, low, close, length=14)
    df['atr_pct'] = df['atr_14'] / close * 100  # ATR como % del precio

    # --- Keltner Channels ---
    kc_df = ta.kc(high, low, close, length=20)
    if kc_df is not None:
        df['kc_upper'] = kc_df.iloc[:, 0]
        df['kc_mid']   = kc_df.iloc[:, 1]
        df['kc_lower'] = kc_df.iloc[:, 2]

    # --- Volatilidad histórica (desviación estándar de retornos) ---
    retornos = close.pct_change()
    df['volatilidad_20']  = retornos.rolling(20).std()  * np.sqrt(24 * 365) * 100  # anualizada %
    df['volatilidad_50']  = retornos.rolling(50).std()  * np.sqrt(24 * 365) * 100

    # --- Rangos de vela ---
    df['rango_vela']     = high - low
    df['rango_vela_pct'] = (high - low) / close * 100

    return df


# ──────────────────────────────────────────────
#  VOLUMEN
# ──────────────────────────────────────────────

def añadir_volumen(df: pd.DataFrame) -> pd.DataFrame:
    """
    OBV, VWAP, MFI y detección de anomalías de volumen.
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    # --- OBV (On-Balance Volume) ---
    df['obv'] = ta.obv(close, volume)
    df['obv_ema'] = ta.ema(df['obv'], length=20)
    df['obv_tendencia'] = (df['obv'] > df['obv_ema']).astype(int)

    # --- VWAP (Volume Weighted Average Price) ---
    df_temp = df.set_index('timestamp')
    vwap_result = ta.vwap(df_temp['high'], df_temp['low'], df_temp['close'], df_temp['volume'])
    df['vwap'] = vwap_result.values if vwap_result is not None else np.nan
    df['precio_sobre_vwap'] = (close > df['vwap']).fillna(False).astype(int)

    # --- MFI (Money Flow Index) — RSI del volumen ---
    df['mfi_14'] = ta.mfi(high, low, close, volume, length=14)
    df['mfi_sobrecompra'] = (df['mfi_14'] >= 80).astype(int)
    df['mfi_sobreventa']  = (df['mfi_14'] <= 20).astype(int)

    # --- Anomalías de volumen ---
    vol_media = volume.rolling(20).mean()
    vol_std   = volume.rolling(20).std()
    df['volumen_ratio']   = volume / vol_media  # ratio respecto a media
    df['volumen_anomalo'] = (volume > vol_media + 2 * vol_std).astype(int)
    df['volumen_muy_bajo'] = (volume < vol_media * 0.3).astype(int)

    # --- Confirmación volumen-precio ---
    # Vela alcista con volumen alto = señal fuerte
    vela_alcista = (close > df['open']).astype(int)
    df['confirmacion_alcista'] = ((vela_alcista == 1) & (df['volumen_anomalo'] == 1)).astype(int)
    df['confirmacion_bajista'] = ((vela_alcista == 0) & (df['volumen_anomalo'] == 1)).astype(int)

    return df


# ──────────────────────────────────────────────
#  NIVELES DE SOPORTE Y RESISTENCIA
# ──────────────────────────────────────────────

def añadir_soportes_resistencias(df: pd.DataFrame, ventana: int = 20) -> pd.DataFrame:
    """
    Detecta niveles de soporte y resistencia como máximos y mínimos locales.
    Calcula distancia del precio actual a los niveles más cercanos.
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    # Máximos locales (resistencias)
    df['es_resistencia'] = (
        (high == high.rolling(ventana, center=True).max())
    ).astype(int)

    # Mínimos locales (soportes)
    df['es_soporte'] = (
        (low == low.rolling(ventana, center=True).min())
    ).astype(int)

    # Distancia al soporte/resistencia más reciente
    resistencias = high[df['es_resistencia'] == 1]
    soportes     = low[df['es_soporte'] == 1]

    ultima_resistencia = high.where(df['es_resistencia'] == 1).ffill()
    ultimo_soporte     = low.where(df['es_soporte'] == 1).ffill()

    df['dist_resistencia_pct'] = (ultima_resistencia - close) / close * 100
    df['dist_soporte_pct']     = (close - ultimo_soporte) / close * 100

    # Fibonacci retracements sobre el último swing significativo
    max_reciente = high.rolling(50).max()
    min_reciente = low.rolling(50).min()
    rango        = max_reciente - min_reciente

    df['fib_236'] = max_reciente - rango * 0.236
    df['fib_382'] = max_reciente - rango * 0.382
    df['fib_500'] = max_reciente - rango * 0.500
    df['fib_618'] = max_reciente - rango * 0.618

    # Precio cerca de nivel Fibonacci (dentro del 0.5%)
    tolerancia = 0.005
    for nivel in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
        df[f'cerca_{nivel}'] = (
            abs(close - df[nivel]) / close < tolerancia
        ).astype(int)

    return df


# ──────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────

def calcular_todos_los_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todos los indicadores al DataFrame.
    Entrada:  DataFrame con columnas [timestamp, open, high, low, close, volume]
    Salida:   DataFrame con 60-80 columnas adicionales
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    print("    → Tendencia...")
    df = añadir_tendencia(df)

    print("    → Momentum...")
    df = añadir_momentum(df)

    print("    → Volatilidad...")
    df = añadir_volatilidad(df)

    print("    → Volumen...")
    df = añadir_volumen(df)

    print("    → Soportes y resistencias...")
    df = añadir_soportes_resistencias(df)

    # Eliminar filas con NaN al inicio (período de calentamiento de indicadores)
    filas_antes = len(df)
    df.dropna(subset=['ema_200', 'rsi_14', 'macd', 'bb_upper', 'atr_14'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    filas_eliminadas = filas_antes - len(df)

    if filas_eliminadas > 0:
        print(f"    → Eliminadas {filas_eliminadas} filas de calentamiento (primeras velas sin suficiente histórico)")

    return df
