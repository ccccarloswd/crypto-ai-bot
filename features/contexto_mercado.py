"""
=============================================================
  CRYPTO AI BOT — Contexto de Mercado
=============================================================
  Añade información macroeconómica del mercado crypto:
    - Dominancia de Bitcoin
    - Fear & Greed Index
    - Correlaciones entre activos
    - Régimen de mercado (bull/bear/lateral)
    - Features multi-timeframe
=============================================================
"""

import pandas as pd
import numpy as np
import os


DIR_CONTEXTO = os.path.join('data', 'contexto')
DIR_RAW      = os.path.join('data', 'raw')


# ──────────────────────────────────────────────
#  DOMINANCIA DE BITCOIN
# ──────────────────────────────────────────────

def añadir_dominancia_btc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona datos de dominancia de Bitcoin con el DataFrame principal.
    Usa interpolación para rellenar huecos entre días.
    """
    ruta = os.path.join(DIR_CONTEXTO, 'dominancia_btc.csv')

    if not os.path.exists(ruta):
        print("    ⚠️  No se encontró dominancia_btc.csv, omitiendo...")
        df['mcap_btc'] = np.nan
        return df

    dom = pd.read_csv(ruta, parse_dates=['timestamp'])
    dom = dom.sort_values('timestamp').drop_duplicates('timestamp')

    # Fusión por fecha (reindex con interpolación para timeframes intra-diarios)
    dom = dom.set_index('timestamp')
    dom = dom.resample('1h').interpolate(method='time')
    dom = dom.reset_index()

    df = pd.merge_asof(
        df.sort_values('timestamp'),
        dom[['timestamp', 'mcap_btc']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    # Normalizar market cap de BTC (log escala)
    df['mcap_btc_log'] = np.log1p(df['mcap_btc'])

    # Cambio en market cap (¿está entrando o saliendo dinero?)
    df['mcap_btc_cambio_7d'] = df['mcap_btc'].pct_change(periods=7 * 24) * 100

    print("    ✅ Dominancia BTC fusionada")
    return df


# ──────────────────────────────────────────────
#  FEAR & GREED INDEX
# ──────────────────────────────────────────────

def añadir_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona el índice de miedo y codicia con el DataFrame principal.
    """
    ruta = os.path.join(DIR_CONTEXTO, 'fear_greed.csv')

    if not os.path.exists(ruta):
        print("    ⚠️  No se encontró fear_greed.csv, omitiendo...")
        df['fear_greed'] = np.nan
        return df

    fg = pd.read_csv(ruta, parse_dates=['timestamp'])
    fg = fg.sort_values('timestamp').drop_duplicates('timestamp')
    fg = fg.rename(columns={'value': 'fear_greed'})

    # El Fear & Greed es diario, replicar para timeframes menores
    fg = fg.set_index('timestamp')
    fg = fg.resample('1h').ffill()
    fg = fg.reset_index()

    df = pd.merge_asof(
        df.sort_values('timestamp'),
        fg[['timestamp', 'fear_greed']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    # Categorías del índice
    df['fear_greed_zona'] = pd.cut(
        df['fear_greed'],
        bins=[0, 25, 45, 55, 75, 100],
        labels=['miedo_extremo', 'miedo', 'neutral', 'codicia', 'codicia_extrema']
    ).astype(str)

    # Codificar zonas numéricamente
    zona_map = {
        'miedo_extremo': -2,
        'miedo': -1,
        'neutral': 0,
        'codicia': 1,
        'codicia_extrema': 2
    }
    df['fear_greed_num'] = df['fear_greed_zona'].map(zona_map)

    # Cambio de sentimiento
    df['fear_greed_cambio'] = df['fear_greed'].diff(periods=24)  # cambio en 24h

    print("    ✅ Fear & Greed fusionado")
    return df


# ──────────────────────────────────────────────
#  RÉGIMEN DE MERCADO
# ──────────────────────────────────────────────

def detectar_regimen_mercado(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica el estado general del mercado:
      2  = Bull market fuerte
      1  = Bull market moderado
      0  = Mercado lateral
     -1  = Bear market moderado
     -2  = Bear market fuerte

    Basado en posición respecto a medias móviles y tendencia de largo plazo.
    """
    close = df['close']

    # Requiere que los indicadores ya estén calculados
    if 'ema_200' not in df.columns:
        print("    ⚠️  EMA 200 no encontrada. Calcula indicadores primero.")
        return df

    # Posición del precio respecto a medias clave
    sobre_ema200 = (close > df['ema_200']).astype(int)
    sobre_ema50  = (close > df['ema_50']).astype(int)
    ema50_sobre_200 = (df['ema_50'] > df['ema_200']).astype(int)

    # Tendencia de largo plazo (pendiente de EMA 200 en las últimas 50 velas)
    df['ema200_pendiente'] = df['ema_200'].diff(periods=50) / df['ema_200'].shift(50) * 100

    # Régimen combinado
    condiciones = [
        (sobre_ema200 == 1) & (sobre_ema50 == 1) & (ema50_sobre_200 == 1) & (df['ema200_pendiente'] > 0.5),
        (sobre_ema200 == 1) & (ema50_sobre_200 == 1),
        (sobre_ema200 == 0) & (ema50_sobre_200 == 0) & (df['ema200_pendiente'] < -0.5),
        (sobre_ema200 == 0) & (ema50_sobre_200 == 0),
    ]
    valores = [2, 1, -2, -1]
    df['regimen_mercado'] = np.select(condiciones, valores, default=0)

    # Etiqueta legible
    regimen_map = {2: 'bull_fuerte', 1: 'bull_moderado', 0: 'lateral',
                   -1: 'bear_moderado', -2: 'bear_fuerte'}
    df['regimen_etiqueta'] = df['regimen_mercado'].map(regimen_map)

    return df


# ──────────────────────────────────────────────
#  CORRELACIÓN ENTRE ACTIVOS
# ──────────────────────────────────────────────

def calcular_correlacion_btc(df: pd.DataFrame, simbolo: str,
                              ventana: int = 168) -> pd.DataFrame:
    """
    Calcula la correlación rolling entre el activo actual y BTC.
    ventana: 168 horas = 1 semana por defecto.
    Si el activo es BTC, devuelve correlación = 1.
    """
    if simbolo == 'BTC_USDT':
        df['correlacion_btc'] = 1.0
        return df

    ruta_btc = os.path.join(DIR_RAW, 'BTC_USDT', '1h.csv')

    if not os.path.exists(ruta_btc):
        print("    ⚠️  No se encontró BTC 1h para calcular correlación")
        df['correlacion_btc'] = np.nan
        return df

    btc = pd.read_csv(ruta_btc, parse_dates=['timestamp'])
    btc = btc[['timestamp', 'close']].rename(columns={'close': 'close_btc'})

    df = pd.merge_asof(
        df.sort_values('timestamp'),
        btc.sort_values('timestamp'),
        on='timestamp',
        direction='nearest'
    )

    retornos_activo = df['close'].pct_change()
    retornos_btc    = df['close_btc'].pct_change()

    df['correlacion_btc'] = retornos_activo.rolling(ventana).corr(retornos_btc)

    # Alta correlación con BTC puede indicar movimiento de mercado general
    df['descorrelacionado'] = (df['correlacion_btc'] < 0.5).astype(int)

    return df


# ──────────────────────────────────────────────
#  FEATURES MULTI-TIMEFRAME
# ──────────────────────────────────────────────

def añadir_contexto_multitimeframe(df_1h: pd.DataFrame,
                                   simbolo: str) -> pd.DataFrame:
    """
    Para cada vela de 1h, añade el contexto de timeframes superiores:
      - Tendencia en 4h
      - Tendencia en diario
      - Tendencia en semanal
      - RSI en 4h y diario
      - Régimen en diario

    Esto permite al modelo saber si la señal de 1h está alineada
    con la tendencia mayor.
    """
    timeframes_sup = {
        '4h': '4h',
        '1d': 'diario',
        '1w': 'semanal',
    }

    df = df_1h.copy()

    for tf, nombre in timeframes_sup.items():
        ruta = os.path.join(DIR_RAW, simbolo, f'{tf}.csv')

        if not os.path.exists(ruta):
            print(f"    ⚠️  No se encontró {simbolo}/{tf}.csv, omitiendo contexto {nombre}")
            continue

        df_tf = pd.read_csv(ruta, parse_dates=['timestamp'])
        df_tf = df_tf.sort_values('timestamp')

        # Calcular indicadores básicos en este timeframe
        try:
            import pandas_ta as ta
            df_tf[f'ema50_{nombre}']    = ta.ema(df_tf['close'], length=50)
            df_tf[f'ema200_{nombre}']   = ta.ema(df_tf['close'], length=200)
            df_tf[f'rsi_{nombre}']      = ta.rsi(df_tf['close'], length=14)
            df_tf[f'tendencia_{nombre}'] = (
                (df_tf['close'] > df_tf[f'ema50_{nombre}']) &
                (df_tf[f'ema50_{nombre}'] > df_tf[f'ema200_{nombre}'])
            ).astype(int) - (
                (df_tf['close'] < df_tf[f'ema50_{nombre}']) &
                (df_tf[f'ema50_{nombre}'] < df_tf[f'ema200_{nombre}'])
            ).astype(int)

        except Exception as e:
            print(f"    ⚠️  Error calculando indicadores para {tf}: {e}")
            continue

        columnas_merge = ['timestamp', f'rsi_{nombre}',
                          f'tendencia_{nombre}', f'ema50_{nombre}']

        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_tf[columnas_merge].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )

        print(f"    ✅ Contexto {nombre} ({tf}) añadido")

    # Alineación de timeframes: ¿todos apuntan en la misma dirección?
    cols_tendencia = [c for c in df.columns if c.startswith('tendencia_')]
    if cols_tendencia:
        df['alineacion_timeframes'] = df[cols_tendencia].sum(axis=1)
        # +3 = todos alcistas, -3 = todos bajistas, 0 = conflicto
        df['señal_multitf_fuerte'] = (df['alineacion_timeframes'].abs() >= 2).astype(int)

    return df


# ──────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────

def añadir_contexto_mercado(df: pd.DataFrame, simbolo: str) -> pd.DataFrame:
    """
    Añade todo el contexto de mercado al DataFrame.
    simbolo: nombre de la carpeta, ej. 'BTC_USDT'
    """
    print("    → Dominancia BTC...")
    df = añadir_dominancia_btc(df)

    print("    → Fear & Greed Index...")
    df = añadir_fear_greed(df)

    print("    → Régimen de mercado...")
    df = detectar_regimen_mercado(df)

    print("    → Correlación con BTC...")
    df = calcular_correlacion_btc(df, simbolo)

    print("    → Contexto multi-timeframe...")
    df = añadir_contexto_multitimeframe(df, simbolo)

    return df
