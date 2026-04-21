"""
=============================================================
  CRYPTO AI BOT — Detección de Patrones
=============================================================
  Detecta patrones técnicos clásicos de forma algorítmica.

  Patrones de velas japonesas:
    Doji, Martillo, Estrella fugaz, Engulfing,
    Morning Star, Evening Star, Harami

  Patrones gráficos:
    Doble suelo / Doble techo
    Hombro-Cabeza-Hombro (HCH) y HCH invertido
    Triángulos (ascendente, descendente, simétrico)
    Flags alcistas y bajistas
    Cuñas (wedge) alcistas y bajistas
    Canal de precio

  Cada patrón devuelve:
    1  = patrón alcista detectado
   -1  = patrón bajista detectado
    0  = sin patrón
=============================================================
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# ──────────────────────────────────────────────
#  UTILIDADES INTERNAS
# ──────────────────────────────────────────────

def _maximos_locales(serie: pd.Series, orden: int = 5) -> np.ndarray:
    """Índices de máximos locales en una serie."""
    indices = argrelextrema(serie.values, np.greater_equal, order=orden)[0]
    return indices


def _minimos_locales(serie: pd.Series, orden: int = 5) -> np.ndarray:
    """Índices de mínimos locales en una serie."""
    indices = argrelextrema(serie.values, np.less_equal, order=orden)[0]
    return indices


def _cuerpo_vela(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Tamaño absoluto del cuerpo de la vela."""
    return abs(close - open_)


def _mecha_superior(open_: pd.Series, close: pd.Series, high: pd.Series) -> pd.Series:
    """Tamaño de la mecha superior."""
    return high - pd.concat([open_, close], axis=1).max(axis=1)


def _mecha_inferior(open_: pd.Series, close: pd.Series, low: pd.Series) -> pd.Series:
    """Tamaño de la mecha inferior."""
    return pd.concat([open_, close], axis=1).min(axis=1) - low


# ──────────────────────────────────────────────
#  PATRONES DE VELAS JAPONESAS
# ──────────────────────────────────────────────

def detectar_patrones_velas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta los patrones de velas japonesas más relevantes.
    Añade una columna por patrón (0 = ausente, 1 = presente).
    """
    o = df['open']
    h = df['high']
    l = df['low']
    c = df['close']

    cuerpo       = _cuerpo_vela(o, c)
    mecha_sup    = _mecha_superior(o, c, h)
    mecha_inf    = _mecha_inferior(o, c, l)
    rango_total  = h - l

    # Evitar división por cero
    rango_safe = rango_total.replace(0, np.nan)

    # --- Doji (indecisión) ---
    # Cuerpo muy pequeño respecto al rango total
    df['patron_doji'] = (
        (cuerpo / rango_safe < 0.1) &
        (rango_total > 0)
    ).astype(int)

    # --- Martillo (reversión alcista) ---
    # Mecha inferior larga, cuerpo pequeño arriba, sin mecha superior
    df['patron_martillo'] = (
        (mecha_inf > cuerpo * 2) &
        (mecha_sup < cuerpo * 0.5) &
        (cuerpo / rango_safe > 0.1)
    ).astype(int)

    # --- Estrella fugaz (reversión bajista) ---
    # Mecha superior larga, cuerpo pequeño abajo, sin mecha inferior
    df['patron_estrella_fugaz'] = (
        (mecha_sup > cuerpo * 2) &
        (mecha_inf < cuerpo * 0.5) &
        (cuerpo / rango_safe > 0.1)
    ).astype(int)

    # --- Marubozu alcista (impulso fuerte alcista) ---
    # Vela verde sin mechas significativas
    df['patron_marubozu_alcista'] = (
        (c > o) &
        (mecha_sup < cuerpo * 0.05) &
        (mecha_inf < cuerpo * 0.05)
    ).astype(int)

    # --- Marubozu bajista (impulso fuerte bajista) ---
    df['patron_marubozu_bajista'] = (
        (c < o) &
        (mecha_sup < cuerpo * 0.05) &
        (mecha_inf < cuerpo * 0.05)
    ).astype(int)

    # --- Engulfing alcista (reversión alcista) ---
    # Vela verde que envuelve completamente la vela roja anterior
    df['patron_engulfing_alcista'] = (
        (c > o) &
        (c.shift(1) < o.shift(1)) &
        (c > o.shift(1)) &
        (o < c.shift(1))
    ).astype(int)

    # --- Engulfing bajista (reversión bajista) ---
    df['patron_engulfing_bajista'] = (
        (c < o) &
        (c.shift(1) > o.shift(1)) &
        (c < o.shift(1)) &
        (o > c.shift(1))
    ).astype(int)

    # --- Harami alcista ---
    # Vela pequeña verde dentro del cuerpo de la vela roja anterior
    df['patron_harami_alcista'] = (
        (c > o) &
        (c.shift(1) < o.shift(1)) &
        (c < o.shift(1)) &
        (o > c.shift(1))
    ).astype(int)

    # --- Morning Star (reversión alcista de 3 velas) ---
    df['patron_morning_star'] = (
        (c.shift(2) < o.shift(2)) &                     # vela 1: bajista
        (cuerpo.shift(1) < cuerpo.shift(2) * 0.3) &     # vela 2: pequeña (estrella)
        (c > o) &                                         # vela 3: alcista
        (c > (o.shift(2) + c.shift(2)) / 2)             # cierra por encima del punto medio
    ).astype(int)

    # --- Evening Star (reversión bajista de 3 velas) ---
    df['patron_evening_star'] = (
        (c.shift(2) > o.shift(2)) &
        (cuerpo.shift(1) < cuerpo.shift(2) * 0.3) &
        (c < o) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    # --- Tres velas blancas (continuación alcista fuerte) ---
    df['patron_tres_soldados'] = (
        (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) &
        (c > c.shift(1)) & (c.shift(1) > c.shift(2)) &
        (o > o.shift(1)) & (o.shift(1) > o.shift(2))
    ).astype(int)

    # --- Tres cuervos negros (continuación bajista fuerte) ---
    df['patron_tres_cuervos'] = (
        (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) &
        (c < c.shift(1)) & (c.shift(1) < c.shift(2)) &
        (o < o.shift(1)) & (o.shift(1) < o.shift(2))
    ).astype(int)

    # --- Señal combinada de velas ---
    patrones_alcistas = ['patron_martillo', 'patron_engulfing_alcista',
                         'patron_harami_alcista', 'patron_morning_star',
                         'patron_tres_soldados', 'patron_marubozu_alcista']
    patrones_bajistas = ['patron_estrella_fugaz', 'patron_engulfing_bajista',
                         'patron_evening_star', 'patron_tres_cuervos',
                         'patron_marubozu_bajista']

    df['señal_velas'] = (
        df[patrones_alcistas].sum(axis=1).clip(0, 1) -
        df[patrones_bajistas].sum(axis=1).clip(0, 1)
    )

    return df


# ──────────────────────────────────────────────
#  PATRONES GRÁFICOS
# ──────────────────────────────────────────────

def detectar_doble_suelo_techo(df: pd.DataFrame, ventana: int = 20,
                                tolerancia: float = 0.02) -> pd.DataFrame:
    """
    Detecta patrones de Doble Suelo (W) y Doble Techo (M).
    tolerancia: diferencia máxima entre los dos suelos/techos (2% por defecto)
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    señal_doble_suelo = np.zeros(len(df))
    señal_doble_techo = np.zeros(len(df))

    min_idx = _minimos_locales(low, orden=ventana // 2)
    max_idx = _maximos_locales(high, orden=ventana // 2)

    # --- Doble suelo ---
    for i in range(1, len(min_idx)):
        idx1 = min_idx[i - 1]
        idx2 = min_idx[i]

        if idx2 - idx1 < ventana // 2:
            continue

        val1 = low.iloc[idx1]
        val2 = low.iloc[idx2]

        if abs(val1 - val2) / val1 < tolerancia:
            # Verificar que hay un pico entre los dos suelos
            entre = high.iloc[idx1:idx2]
            if len(entre) > 0 and entre.max() > max(val1, val2) * 1.02:
                señal_doble_suelo[idx2] = 1

    # --- Doble techo ---
    for i in range(1, len(max_idx)):
        idx1 = max_idx[i - 1]
        idx2 = max_idx[i]

        if idx2 - idx1 < ventana // 2:
            continue

        val1 = high.iloc[idx1]
        val2 = high.iloc[idx2]

        if abs(val1 - val2) / val1 < tolerancia:
            entre = low.iloc[idx1:idx2]
            if len(entre) > 0 and entre.min() < min(val1, val2) * 0.98:
                señal_doble_techo[idx2] = -1

    df['patron_doble_suelo'] = señal_doble_suelo.astype(int)
    df['patron_doble_techo'] = señal_doble_techo.astype(int)

    return df


def detectar_hch(df: pd.DataFrame, ventana: int = 15,
                 tolerancia: float = 0.03) -> pd.DataFrame:
    """
    Detecta patrones Hombro-Cabeza-Hombro (bajista) y HCH invertido (alcista).
    """
    high = df['high']
    low  = df['low']

    señal_hch          = np.zeros(len(df))
    señal_hch_invertido = np.zeros(len(df))

    max_idx = _maximos_locales(high, orden=ventana // 2)
    min_idx = _minimos_locales(low,  orden=ventana // 2)

    # --- HCH bajista ---
    for i in range(2, len(max_idx)):
        h_izq   = max_idx[i - 2]
        cabeza  = max_idx[i - 1]
        h_der   = max_idx[i]

        if not (high.iloc[cabeza] > high.iloc[h_izq] and
                high.iloc[cabeza] > high.iloc[h_der]):
            continue

        sim_hombros = abs(high.iloc[h_izq] - high.iloc[h_der]) / high.iloc[h_izq]
        if sim_hombros > tolerancia:
            continue

        señal_hch[h_der] = -1

    # --- HCH invertido alcista ---
    for i in range(2, len(min_idx)):
        h_izq   = min_idx[i - 2]
        cabeza  = min_idx[i - 1]
        h_der   = min_idx[i]

        if not (low.iloc[cabeza] < low.iloc[h_izq] and
                low.iloc[cabeza] < low.iloc[h_der]):
            continue

        sim_hombros = abs(low.iloc[h_izq] - low.iloc[h_der]) / low.iloc[h_izq]
        if sim_hombros > tolerancia:
            continue

        señal_hch_invertido[h_der] = 1

    df['patron_hch']          = señal_hch.astype(int)
    df['patron_hch_invertido'] = señal_hch_invertido.astype(int)

    return df


def detectar_triangulos(df: pd.DataFrame, ventana: int = 30) -> pd.DataFrame:
    """
    Detecta triángulos ascendentes, descendentes y simétricos.
    Se basa en la convergencia de máximos y mínimos locales.
    """
    high = df['high']
    low  = df['low']

    señal_triangulo = np.zeros(len(df))
    n = len(df)

    for i in range(ventana * 2, n):
        segmento_high = high.iloc[i - ventana:i]
        segmento_low  = low.iloc[i - ventana:i]

        # Pendiente de máximos y mínimos mediante regresión lineal simple
        x = np.arange(ventana)
        pend_max = np.polyfit(x, segmento_high.values, 1)[0]
        pend_min = np.polyfit(x, segmento_low.values,  1)[0]

        # Normalizar pendientes por el precio
        precio_ref = segmento_high.mean()
        pend_max_n = pend_max / precio_ref
        pend_min_n = pend_min / precio_ref

        umbral = 0.0005  # pendiente significativa

        # Triángulo ascendente: máximos planos, mínimos subiendo → alcista
        if abs(pend_max_n) < umbral and pend_min_n > umbral:
            señal_triangulo[i] = 1

        # Triángulo descendente: máximos bajando, mínimos planos → bajista
        elif pend_max_n < -umbral and abs(pend_min_n) < umbral:
            señal_triangulo[i] = -1

        # Triángulo simétrico: ambos convergiendo (neutro, depende de ruptura)
        elif pend_max_n < -umbral and pend_min_n > umbral:
            señal_triangulo[i] = 0.5  # potencial ruptura, dirección indefinida

    df['patron_triangulo'] = señal_triangulo

    return df


def detectar_flags_cunas(df: pd.DataFrame, ventana: int = 20) -> pd.DataFrame:
    """
    Detecta flags (banderas) y cuñas (wedges) alcistas y bajistas.

    Flag alcista:  impulso alcista fuerte + consolidación bajista leve → continuación alcista
    Flag bajista:  impulso bajista fuerte + consolidación alcista leve → continuación bajista
    Cuña alcista:  precio sube pero con momentum decreciente → reversión bajista
    Cuña bajista:  precio baja pero con momentum decreciente → reversión alcista
    """
    close = df['close']
    high  = df['high']
    low   = df['low']

    señal_flag  = np.zeros(len(df))
    señal_cuña  = np.zeros(len(df))

    n = len(df)
    impulso_ventana = ventana // 2

    for i in range(ventana * 2, n):
        # Fase de impulso (antes del patrón)
        seg_impulso_c = close.iloc[i - ventana - impulso_ventana: i - ventana]
        # Fase de consolidación (el patrón)
        seg_consol_h  = high.iloc[i - ventana:i]
        seg_consol_l  = low.iloc[i - ventana:i]
        seg_consol_c  = close.iloc[i - ventana:i]

        if len(seg_impulso_c) < 2 or len(seg_consol_c) < 2:
            continue

        cambio_impulso = (seg_impulso_c.iloc[-1] - seg_impulso_c.iloc[0]) / seg_impulso_c.iloc[0]

        x = np.arange(ventana)
        pend_h = np.polyfit(x, seg_consol_h.values, 1)[0] / seg_consol_h.mean()
        pend_l = np.polyfit(x, seg_consol_l.values, 1)[0] / seg_consol_l.mean()

        # --- Flag alcista ---
        if cambio_impulso > 0.03 and pend_h < 0 and pend_l < 0:
            señal_flag[i] = 1

        # --- Flag bajista ---
        elif cambio_impulso < -0.03 and pend_h > 0 and pend_l > 0:
            señal_flag[i] = -1

        # --- Cuña alcista (bearish wedge) ---
        # Precio sube pero máximos y mínimos convergen → agotamiento
        elif pend_h > 0 and pend_l > 0 and pend_l > pend_h:
            señal_cuña[i] = -1  # señal bajista

        # --- Cuña bajista (bullish wedge) ---
        elif pend_h < 0 and pend_l < 0 and pend_h < pend_l:
            señal_cuña[i] = 1  # señal alcista

    df['patron_flag'] = señal_flag
    df['patron_cuña']  = señal_cuña

    return df


def detectar_canal(df: pd.DataFrame, ventana: int = 30) -> pd.DataFrame:
    """
    Detecta si el precio está operando en un canal definido
    y la posición relativa del precio dentro del canal.
    """
    high  = df['high']
    low   = df['low']
    close = df['close']

    en_canal       = np.zeros(len(df))
    posicion_canal = np.full(len(df), np.nan)

    n = len(df)
    x = np.arange(ventana)

    for i in range(ventana, n):
        seg_h = high.iloc[i - ventana:i].values
        seg_l = low.iloc[i - ventana:i].values

        coef_h = np.polyfit(x, seg_h, 1)
        coef_l = np.polyfit(x, seg_l, 1)

        # Residuos — qué tan bien se ajustan a líneas rectas
        res_h = np.std(seg_h - np.polyval(coef_h, x)) / np.mean(seg_h)
        res_l = np.std(seg_l - np.polyval(coef_l, x)) / np.mean(seg_l)

        # Canal válido si ambos extremos son suficientemente lineales
        if res_h < 0.02 and res_l < 0.02:
            en_canal[i] = 1
            techo_actual = np.polyval(coef_h, ventana)
            suelo_actual = np.polyval(coef_l, ventana)
            rango = techo_actual - suelo_actual
            if rango > 0:
                posicion_canal[i] = (close.iloc[i] - suelo_actual) / rango

    df['en_canal']        = en_canal.astype(int)
    df['posicion_canal']  = posicion_canal  # 0 = suelo, 1 = techo del canal

    return df


# ──────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────

def calcular_todos_los_patrones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la detección de todos los patrones al DataFrame.
    """
    df = df.copy()

    print("    → Patrones de velas japonesas...")
    df = detectar_patrones_velas(df)

    print("    → Doble suelo / Doble techo...")
    df = detectar_doble_suelo_techo(df)

    print("    → Hombro-Cabeza-Hombro...")
    df = detectar_hch(df)

    print("    → Triángulos...")
    df = detectar_triangulos(df)

    print("    → Flags y cuñas...")
    df = detectar_flags_cunas(df)

    print("    → Canales de precio...")
    df = detectar_canal(df)

    # --- Señal combinada de patrones gráficos ---
    patrones_alcistas = ['patron_doble_suelo', 'patron_hch_invertido',
                         'patron_flag', 'patron_cuña']
    patrones_bajistas = ['patron_doble_techo', 'patron_hch']

    señal_alcista = df[patrones_alcistas].clip(0, 1).sum(axis=1)
    señal_bajista = df[patrones_bajistas].abs().clip(0, 1).sum(axis=1)

    df['señal_patrones_graficos'] = (señal_alcista - señal_bajista).clip(-3, 3)

    return df
