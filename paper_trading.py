"""
=============================================================
  CRYPTO AI BOT — Paper Trading
=============================================================
  Corre cada hora via GitHub Actions.
  Lee datos públicos de Binance (sin cuenta).
  Genera señales con el modelo entrenado.
  Simula operaciones con la estrategia v8.
  Guarda estado en JSON y envía alertas por Telegram.
=============================================================
"""

import os
import json
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
#  CONFIGURACIÓN — igual que backtesting v8
# ──────────────────────────────────────────────

CRIPTOS      = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
TIMEFRAME    = '1h'
VELAS_NEEDED = 300       # velas para calcular indicadores

CAPITAL_INICIAL  = 1000.0
COMISION_PCT     = 0.001
SLIPPAGE_PCT     = 0.0005
MAX_HORAS_TRADE  = 16
FUNDING_RATE_HORA = 0.0003

DD_PAUSA_PCT = 0.18
DD_STOP_PCT  = 0.30
HORAS_PAUSA  = 24

SOLO_XGBOOST = ['BTC_USDT', 'BNB_USDT']

MAX_POSICIONES       = 2
MAX_MARGEN_TOTAL_PCT = 0.40

SIGNAL_CONFIG = {
    'elite':   {'prob': 0.74, 'leverage': 8,  'margen': 0.22,
                'sl_mult': 1.2, 'sl_max': 0.020, 'tp_ratio': 3.5, 'tp_max': 0.080},
    'premium': {'prob': 0.64, 'leverage': 5,  'margen': 0.20,
                'sl_mult': 1.5, 'sl_max': 0.025, 'tp_ratio': 3.0, 'tp_max': 0.060},
}
UMBRAL_MINIMO   = min(c['prob'] for c in SIGNAL_CONFIG.values())
ELITE_REGIMEN_MIN = 0

TRAILING_ACTIVACION = 0.020
TRAILING_ATR_MULT   = 1.2

# Rutas
DIR_MODELOS  = 'models'
ESTADO_FILE  = 'paper_trading/estado.json'
LOG_FILE     = 'paper_trading/operaciones.csv'
METRICAS_FILE = 'paper_trading/metricas.json'


# ──────────────────────────────────────────────
#  TELEGRAM
# ──────────────────────────────────────────────

def enviar_telegram(mensaje: str):
    """Envía un mensaje por Telegram. Token y chat_id desde variables de entorno."""
    token   = os.environ.get('TELEGRAM_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')

    if not token or not chat_id:
        print(f"  [Telegram desactivado] {mensaje}")
        return

    try:
        import urllib.request
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({'chat_id': chat_id, 'text': mensaje,
                           'parse_mode': 'HTML'}).encode()
        req  = urllib.request.Request(url, data=data,
                                      headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  ⚠️  Error Telegram: {e}")


# ──────────────────────────────────────────────
#  ESTADO PERSISTENTE
# ──────────────────────────────────────────────

def cargar_estado() -> Dict:
    """Carga el estado del bot desde el archivo JSON."""
    if os.path.exists(ESTADO_FILE):
        with open(ESTADO_FILE, 'r') as f:
            return json.load(f)

    # Estado inicial
    return {
        'capital':          CAPITAL_INICIAL,
        'capital_max':      CAPITAL_INICIAL,
        'posiciones':       [],
        'bot_activo':       True,
        'en_pausa':         False,
        'fin_pausa_hora':   None,
        'n_operaciones':    0,
        'n_wins':           0,
        'n_loses':          0,
        'pnl_total':        0.0,
        'funding_total':    0.0,
        'ultima_ejecucion': None,
        'version':          'v8',
    }


def guardar_estado(estado: Dict):
    """Guarda el estado actual en JSON."""
    os.makedirs(os.path.dirname(ESTADO_FILE), exist_ok=True)
    estado['ultima_ejecucion'] = datetime.now(timezone.utc).isoformat()
    with open(ESTADO_FILE, 'w') as f:
        json.dump(estado, f, indent=2, default=str)


def registrar_operacion(op: Dict):
    """Añade una operación al CSV de historial."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    df_op = pd.DataFrame([op])
    if os.path.exists(LOG_FILE):
        df_op.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_op.to_csv(LOG_FILE, index=False)


def actualizar_metricas(estado: Dict):
    """Actualiza el archivo de métricas resumidas."""
    capital      = estado['capital']
    capital_ini  = CAPITAL_INICIAL
    rent         = (capital - capital_ini) / capital_ini * 100
    n_ops        = estado['n_operaciones']
    wr           = estado['n_wins'] / max(n_ops, 1) * 100

    metricas = {
        'capital_actual':    round(capital, 2),
        'capital_inicial':   capital_ini,
        'rentabilidad_pct':  round(rent, 2),
        'n_operaciones':     n_ops,
        'win_rate_pct':      round(wr, 1),
        'pnl_total':         round(estado['pnl_total'], 2),
        'funding_total':     round(estado['funding_total'], 2),
        'posiciones_abiertas': len(estado['posiciones']),
        'bot_activo':        estado['bot_activo'],
        'en_pausa':          estado['en_pausa'],
        'ultima_actualizacion': datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(METRICAS_FILE), exist_ok=True)
    with open(METRICAS_FILE, 'w') as f:
        json.dump(metricas, f, indent=2)
    return metricas


# ──────────────────────────────────────────────
#  DATOS DE MERCADO
# ──────────────────────────────────────────────

def obtener_velas(simbolo: str, exchange=None) -> Optional[pd.DataFrame]:
    """Descarga velas horarias usando CryptoCompare API (sin restricciones geográficas)."""
    import urllib.request
    import json as json_lib

    # Convertir BTC/USDT a BTC
    moneda = simbolo.replace('/USDT', '').replace('/', '')

    url = (f"https://min-api.cryptocompare.com/data/v2/histohour"
           f"?fsym={moneda}&tsym=USDT&limit={VELAS_NEEDED}&aggregate=1")

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0', 'authorization': ''}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json_lib.loads(response.read())

        if data.get('Response') != 'Success':
            print(f"  ⚠️  Error API CryptoCompare: {data.get('Message')}")
            return None

        velas = data['Data']['Data']
        if len(velas) < 200:
            print(f"  ⚠️  Pocas velas: {len(velas)}")
            return None

        df = pd.DataFrame(velas)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'open':       'open',
            'high':       'high',
            'low':        'low',
            'close':      'close',
            'volumefrom': 'volume',
        })

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    except Exception as e:
        print(f"  ❌ Error obteniendo velas {simbolo}: {e}")
        return None


# ──────────────────────────────────────────────
#  INDICADORES (versión ligera para producción)
# ──────────────────────────────────────────────

def calcular_indicadores_rapido(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los indicadores necesarios para el modelo.
    Versión optimizada para producción (solo últimas velas).
    """
    import pandas_ta as ta

    df = df.copy()

    # Los mismos indicadores que en features/indicadores.py
    close  = df['close']
    high   = df['high']
    low    = df['low']
    volume = df['volume']

    # Tendencia
    for p in [9, 21, 50, 100, 200]:
        df[f'ema_{p}'] = ta.ema(close, length=p)
    for p in [20, 50, 200]:
        df[f'sma_{p}'] = ta.sma(close, length=p)

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        df['macd']        = macd_df['MACD_12_26_9']
        df['macd_signal'] = macd_df['MACDs_12_26_9']
        df['macd_hist']   = macd_df['MACDh_12_26_9']
        df['macd_cruce']  = np.where(
            (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 1,
            np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), -1, 0)
        )

    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None:
        df['adx']     = adx_df['ADX_14']
        df['dmi_pos'] = adx_df['DMP_14']
        df['dmi_neg'] = adx_df['DMN_14']

    df['golden_cross']        = np.where((df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1)), 1, 0)
    df['death_cross']         = np.where((df['ema_50'] < df['ema_200']) & (df['ema_50'].shift(1) >= df['ema_200'].shift(1)), 1, 0)
    df['precio_sobre_ema50']  = (close > df['ema_50']).astype(int)
    df['precio_sobre_ema200'] = (close > df['ema_200']).astype(int)
    df['ema50_sobre_ema200']  = (df['ema_50'] > df['ema_200']).astype(int)

    # Momentum
    df['rsi_14'] = ta.rsi(close, length=14)
    df['rsi_7']  = ta.rsi(close, length=7)
    df['rsi_sobrecompra'] = (df['rsi_14'] >= 70).astype(int)
    df['rsi_sobreventa']  = (df['rsi_14'] <= 30).astype(int)
    df['rsi_zona_neutral'] = ((df['rsi_14'] > 40) & (df['rsi_14'] < 60)).astype(int)

    precio_sube = (close > close.shift(5)).astype(int)
    rsi_sube    = (df['rsi_14'] > df['rsi_14'].shift(5)).astype(int)
    df['divergencia_bajista_rsi'] = ((precio_sube == 1) & (rsi_sube == 0)).astype(int)
    df['divergencia_alcista_rsi'] = ((precio_sube == 0) & (rsi_sube == 1)).astype(int)

    stoch_df = ta.stoch(high, low, close, k=14, d=3)
    if stoch_df is not None:
        df['stoch_k'] = stoch_df.iloc[:, 0]
        df['stoch_d'] = stoch_df.iloc[:, 1]
        df['stoch_sobrecompra'] = (df['stoch_k'] >= 80).astype(int)
        df['stoch_sobreventa']  = (df['stoch_k'] <= 20).astype(int)

    df['cci_20']      = ta.cci(high, low, close, length=20)
    df['roc_10']      = ta.roc(close, length=10)
    df['roc_20']      = ta.roc(close, length=20)
    df['williams_r']  = ta.willr(high, low, close, length=14)
    df['momentum_10'] = close - close.shift(10)
    df['momentum_pct_10'] = close.pct_change(periods=10) * 100

    # Volatilidad
    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None:
        df['bb_upper'] = bb_df['BBU_20_2.0']
        df['bb_mid']   = bb_df['BBM_20_2.0']
        df['bb_lower'] = bb_df['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        rango_bb = df['bb_upper'] - df['bb_lower']
        df['bb_posicion'] = np.where(rango_bb > 0, (close - df['bb_lower']) / rango_bb, 0.5)
        df['bb_squeeze']  = (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.7).astype(int)

    df['atr_14']  = ta.atr(high, low, close, length=14)
    df['atr_pct'] = df['atr_14'] / close * 100

    kc_df = ta.kc(high, low, close, length=20)
    if kc_df is not None:
        df['kc_upper'] = kc_df.iloc[:, 0]
        df['kc_mid']   = kc_df.iloc[:, 1]
        df['kc_lower'] = kc_df.iloc[:, 2]

    retornos = close.pct_change()
    df['volatilidad_20'] = retornos.rolling(20).std() * np.sqrt(24 * 365) * 100
    df['volatilidad_50'] = retornos.rolling(50).std() * np.sqrt(24 * 365) * 100
    df['rango_vela']     = high - low
    df['rango_vela_pct'] = (high - low) / close * 100

    # Volumen
    df['obv']              = ta.obv(close, volume)
    df['obv_ema']          = ta.ema(df['obv'], length=20)
    df['obv_tendencia']    = (df['obv'] > df['obv_ema']).astype(int)

    df_temp = df.set_index('timestamp')
    vwap_result = ta.vwap(df_temp['high'], df_temp['low'], df_temp['close'], df_temp['volume'])
    df['vwap'] = vwap_result.values if vwap_result is not None else np.nan
    df['precio_sobre_vwap'] = (close > df['vwap']).fillna(False).astype(int)

    df['mfi_14']          = ta.mfi(high, low, close, volume, length=14)
    df['mfi_sobrecompra'] = (df['mfi_14'] >= 80).astype(int)
    df['mfi_sobreventa']  = (df['mfi_14'] <= 20).astype(int)

    vol_media = volume.rolling(20).mean()
    vol_std   = volume.rolling(20).std()
    df['volumen_ratio']    = volume / vol_media
    df['volumen_anomalo']  = (volume > vol_media + 2 * vol_std).astype(int)
    df['volumen_muy_bajo'] = (volume < vol_media * 0.3).astype(int)

    vela_alcista = (close > df['open']).astype(int)
    df['confirmacion_alcista'] = ((vela_alcista == 1) & (df['volumen_anomalo'] == 1)).astype(int)
    df['confirmacion_bajista'] = ((vela_alcista == 0) & (df['volumen_anomalo'] == 1)).astype(int)

    # Soportes y resistencias
    df['es_resistencia'] = (high == high.rolling(20, center=True).max()).astype(int)
    df['es_soporte']     = (low  == low.rolling(20, center=True).min()).astype(int)

    ultima_resistencia = high.where(df['es_resistencia'] == 1).ffill()
    ultimo_soporte     = low.where(df['es_soporte'] == 1).ffill()
    df['dist_resistencia_pct'] = (ultima_resistencia - close) / close * 100
    df['dist_soporte_pct']     = (close - ultimo_soporte) / close * 100

    max_rec = high.rolling(50).max()
    min_rec = low.rolling(50).min()
    rango   = max_rec - min_rec
    df['fib_236'] = max_rec - rango * 0.236
    df['fib_382'] = max_rec - rango * 0.382
    df['fib_500'] = max_rec - rango * 0.500
    df['fib_618'] = max_rec - rango * 0.618

    tolerancia = 0.005
    for nivel in ['fib_236', 'fib_382', 'fib_500', 'fib_618']:
        df[f'cerca_{nivel}'] = (abs(close - df[nivel]) / close < tolerancia).astype(int)

    # Régimen de mercado
    df['ema200_pendiente'] = df['ema_200'].diff(periods=50) / df['ema_200'].shift(50) * 100
    sobre_ema200    = (close > df['ema_200']).astype(int)
    sobre_ema50     = (close > df['ema_50']).astype(int)
    ema50_sobre_200 = (df['ema_50'] > df['ema_200']).astype(int)
    condiciones = [
        (sobre_ema200 == 1) & (sobre_ema50 == 1) & (ema50_sobre_200 == 1) & (df['ema200_pendiente'] > 0.5),
        (sobre_ema200 == 1) & (ema50_sobre_200 == 1),
        (sobre_ema200 == 0) & (ema50_sobre_200 == 0) & (df['ema200_pendiente'] < -0.5),
        (sobre_ema200 == 0) & (ema50_sobre_200 == 0),
    ]
    df['regimen_mercado'] = np.select(condiciones, [2, 1, -2, -1], default=0)

    return df


# ──────────────────────────────────────────────
#  MODELO
# ──────────────────────────────────────────────

def cargar_modelo(simbolo: str) -> Optional[Dict]:
    """Carga el modelo entrenado para una cripto."""
    nombre = simbolo.replace('/', '_')
    dir_m  = os.path.join(DIR_MODELOS, nombre)

    try:
        from models.ensemble import EnsembleTrading
        modelos = {
            'xgboost':  joblib.load(os.path.join(dir_m, 'xgboost.pkl')),
            'scaler':   joblib.load(os.path.join(dir_m, 'scaler.pkl')),
            'features': joblib.load(os.path.join(dir_m, 'features.pkl')),
            'lstm':     None,
        }
        ensemble = EnsembleTrading()
        ensemble.cargar(dir_m)
        modelos['ensemble'] = ensemble

        if nombre not in SOLO_XGBOOST:
            try:
                from models.lstm_model import cargar_lstm
                lstm = cargar_lstm(dir_m)
                if lstm is not None:
                    modelos['lstm'] = lstm
            except Exception:
                pass

        return modelos

    except Exception as e:
        print(f"  ❌ Error cargando modelo {simbolo}: {e}")
        return None


def predecir(df: pd.DataFrame, modelos: Dict) -> float:
    """Genera la probabilidad de subida para la última vela."""
    features      = modelos['features']
    features_disp = [f for f in features if f in df.columns]

    X = modelos['scaler'].transform(df[features_disp].values)
    np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)

    probs_xgb = modelos['xgboost'].predict_proba(X)[:, 1]

    probs_lstm = None
    if modelos['lstm'] is not None:
        try:
            from models.lstm_model import predecir_lstm
            from models.preparar_datos import crear_secuencias
            X_seq, _ = crear_secuencias(X, np.zeros(len(X)), 48)
            if len(X_seq) > 0:
                raw        = predecir_lstm(modelos['lstm'], X_seq)
                pad        = np.full(len(probs_xgb) - len(raw), 0.5)
                probs_lstm = np.concatenate([pad, raw])
        except Exception:
            pass

    probs_ens = modelos['ensemble'].predecir_proba(probs_xgb, probs_lstm)

    # Retornar probabilidad de la última vela
    return float(probs_ens[-1])


# ──────────────────────────────────────────────
#  ESTRATEGIA v8
# ──────────────────────────────────────────────

def clasificar_señal(prob: float, regimen: int) -> Optional[Tuple[str, dict]]:
    """Clasifica la señal según estrategia v8."""
    if prob >= SIGNAL_CONFIG['elite']['prob'] and regimen >= ELITE_REGIMEN_MIN:
        return 'elite', SIGNAL_CONFIG['elite']
    if prob >= SIGNAL_CONFIG['premium']['prob']:
        return 'premium', SIGNAL_CONFIG['premium']
    return None


def calcular_sl_tp(precio: float, atr: float,
                   cfg: dict) -> Tuple[float, float, float]:
    """Calcula SL, TP y precio de liquidación."""
    sl_atr = (atr * cfg['sl_mult']) / precio
    sl_pct = float(np.clip(sl_atr, 0.008, cfg['sl_max']))
    tp_pct = float(np.clip(sl_pct * cfg['tp_ratio'], sl_pct * 2.5, cfg['tp_max']))
    sl     = precio * (1 - sl_pct)
    tp     = precio * (1 + tp_pct)
    liq    = precio * (1 - (1 / cfg['leverage']) * 0.90)
    return sl, tp, liq


def filtro_entrada(regimen: int, volumen_muy_bajo: int) -> Tuple[bool, str]:
    """Filtros mínimos de entrada (probados en v5)."""
    if regimen == -2:
        return False, 'bear_extremo'
    if volumen_muy_bajo == 1:
        return False, 'volumen_muy_bajo'
    return True, 'ok'


# ──────────────────────────────────────────────
#  GESTIÓN DE POSICIONES
# ──────────────────────────────────────────────

def gestionar_posiciones(estado: Dict, precio_actual: float,
                         atr_actual: float, hora_actual: int,
                         simbolo: str, timestamp: str) -> List[Dict]:
    """
    Revisa las posiciones abiertas y cierra las que corresponda.
    Devuelve lista de operaciones cerradas.
    """
    cerradas    = []
    nuevas_pos  = []

    for pos in estado['posiciones']:
        if pos['simbolo'] != simbolo:
            nuevas_pos.append(pos)
            continue

        horas = hora_actual - pos['hora_entrada']

        # Actualizar trailing
        if precio_actual > pos.get('precio_max', pos['precio_entrada']):
            pos['precio_max'] = precio_actual
            ganancia_nom = (precio_actual - pos['precio_entrada']) / pos['precio_entrada']
            if ganancia_nom >= TRAILING_ACTIVACION:
                pos['trailing'] = True
                trailing_sl = pos['precio_max'] - atr_actual * TRAILING_ATR_MULT
                pos['stop_loss'] = max(pos['stop_loss'], trailing_sl)

        # Condiciones de cierre
        cerrar, motivo = False, ''
        if precio_actual <= pos.get('liq', 0):
            cerrar, motivo = True, 'liquidacion'
        elif precio_actual <= pos['stop_loss']:
            cerrar, motivo = True, 'trailing_stop' if pos.get('trailing') else 'stop_loss'
        elif precio_actual >= pos['take_profit']:
            cerrar, motivo = True, 'take_profit'
        elif horas >= MAX_HORAS_TRADE:
            cerrar, motivo = True, 'tiempo_maximo'

        if cerrar:
            p_sal     = precio_actual * (1 - SLIPPAGE_PCT)
            exp       = pos['margen'] * pos['leverage']
            ret       = (p_sal - pos['precio_entrada']) / pos['precio_entrada']
            funding   = exp * FUNDING_RATE_HORA * horas
            comision  = exp * COMISION_PCT * 2
            pnl       = exp * ret - funding - comision

            estado['capital']     += pnl
            estado['pnl_total']   += pnl
            estado['funding_total'] += funding
            estado['n_operaciones'] += 1

            if pnl > 0:
                estado['n_wins'] += 1
            else:
                estado['n_loses'] += 1

            op = {
                'timestamp':        timestamp,
                'simbolo':          simbolo,
                'tipo':             'cierre',
                'motivo':           motivo,
                'precio_entrada':   round(pos['precio_entrada'], 6),
                'precio_salida':    round(p_sal, 6),
                'pnl_usdt':         round(pnl, 4),
                'funding_usdt':     round(funding, 4),
                'horas':            horas,
                'leverage':         pos['leverage'],
                'calidad':          pos['calidad'],
                'capital_total':    round(estado['capital'], 2),
            }
            cerradas.append(op)

            # Alerta Telegram
            emoji = '✅' if pnl > 0 else '❌'
            enviar_telegram(
                f"{emoji} <b>CIERRE {simbolo}</b>\n"
                f"Motivo: {motivo}\n"
                f"Entrada: ${pos['precio_entrada']:,.4f}\n"
                f"Salida:  ${p_sal:,.4f}\n"
                f"P&L: ${pnl:+.2f} USDT\n"
                f"Horas: {horas}h | Calidad: {pos['calidad'].upper()}\n"
                f"Capital: ${estado['capital']:,.2f}"
            )
        else:
            nuevas_pos.append(pos)

    estado['posiciones'] = nuevas_pos
    return cerradas


def abrir_posicion(estado: Dict, simbolo: str, precio: float,
                   atr: float, prob: float, cfg: dict,
                   calidad: str, timestamp: str, hora: int):
    """Abre una nueva posición simulada."""
    margen_en_uso  = sum(p['margen'] for p in estado['posiciones'] if p['simbolo'] == simbolo)
    margen_max     = estado['capital'] * MAX_MARGEN_TOTAL_PCT
    margen_libre   = margen_max - margen_en_uso
    margen_deseado = estado['capital'] * cfg['margen']
    margen_real    = min(margen_deseado, margen_libre)

    if margen_real < estado['capital'] * 0.03:
        print(f"  ⚠️  Margen insuficiente para {simbolo}")
        return

    p_ent = precio * (1 + SLIPPAGE_PCT)
    sl, tp, liq = calcular_sl_tp(p_ent, atr, cfg)

    pos = {
        'simbolo':          simbolo,
        'timestamp_entrada': timestamp,
        'precio_entrada':   p_ent,
        'precio_max':       p_ent,
        'stop_loss':        sl,
        'take_profit':      tp,
        'liq':              liq,
        'margen':           margen_real,
        'leverage':         cfg['leverage'],
        'calidad':          calidad,
        'hora_entrada':     hora,
        'trailing':         False,
    }
    estado['posiciones'].append(pos)

    exp = margen_real * cfg['leverage']
    enviar_telegram(
        f"🚀 <b>APERTURA {simbolo}</b>\n"
        f"Calidad: {calidad.upper()} | Prob: {prob:.3f}\n"
        f"Precio: ${p_ent:,.4f}\n"
        f"SL: ${sl:,.4f} | TP: ${tp:,.4f}\n"
        f"Margen: ${margen_real:.2f} | Leverage: x{cfg['leverage']}\n"
        f"Exposición: ${exp:.2f}\n"
        f"Capital: ${estado['capital']:,.2f}"
    )


# ──────────────────────────────────────────────
#  LOOP PRINCIPAL
# ──────────────────────────────────────────────

def ejecutar():
    """Ejecuta una iteración del bot (se llama cada hora desde GitHub Actions)."""
    print(f"\n{'='*60}")
    print(f"  CRYPTO AI BOT — Paper Trading v8")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    # Cargar estado
    estado    = cargar_estado()
    hora_actual = int(time.time() / 3600)  # hora Unix para tracking

    if not estado['bot_activo']:
        print("  🛑 Bot detenido por drawdown máximo")
        return

    # Verificar pausa
    if estado['en_pausa']:
        fin_pausa = estado.get('fin_pausa_hora', 0)
        if hora_actual < fin_pausa:
            restante = fin_pausa - hora_actual
            print(f"  ⏸️  Bot en pausa. Quedan {restante}h")
            guardar_estado(estado)
            return
        else:
            estado['en_pausa'] = False
            print("  ▶️  Pausa terminada. Bot activo.")


    # Procesar cada cripto
    for simbolo in CRIPTOS:
        nombre = simbolo.replace('/', '_')
        print(f"\n  📊 {simbolo}")

        # Obtener datos
        df = obtener_velas(simbolo)
        if df is None:
            continue

        # Calcular indicadores
        try:
            df = calcular_indicadores_rapido(df)
        except Exception as e:
            print(f"  ❌ Error indicadores {simbolo}: {e}")
            continue

        # Eliminar NaN del inicio
        df.dropna(subset=['ema_200', 'rsi_14', 'atr_14'], inplace=True)
        if len(df) < 50:
            print(f"  ⚠️  Pocas velas válidas tras indicadores: {len(df)}")
            continue

        # Última vela completa (la penúltima, la última está en formación)
        ultima = df.iloc[-2]
        precio_actual = float(ultima['close'])
        atr_actual    = float(ultima.get('atr_14', precio_actual * 0.015))
        regimen       = int(ultima.get('regimen_mercado', 0))
        vol_bajo      = int(ultima.get('volumen_muy_bajo', 0))
        timestamp     = str(ultima['timestamp'])

        # Gestionar posiciones abiertas
        n_pos_simbolo = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre)
        cerradas = gestionar_posiciones(
            estado, precio_actual, atr_actual, hora_actual, nombre, timestamp
        )
        if cerradas:
            for op in cerradas:
                registrar_operacion(op)

        # Verificar drawdown
        capital_max = max(estado['capital_max'], estado['capital'])
        estado['capital_max'] = capital_max
        dd = (capital_max - estado['capital']) / capital_max

        if dd >= DD_STOP_PCT:
            estado['bot_activo'] = False
            enviar_telegram(
                f"🛑 <b>BOT DETENIDO</b>\n"
                f"Drawdown máximo alcanzado: {dd*100:.1f}%\n"
                f"Capital: ${estado['capital']:,.2f}"
            )
            guardar_estado(estado)
            return

        if dd >= DD_PAUSA_PCT and not estado['en_pausa']:
            estado['en_pausa']       = True
            estado['fin_pausa_hora'] = hora_actual + HORAS_PAUSA
            enviar_telegram(
                f"⏸️ <b>PAUSA {HORAS_PAUSA}h</b>\n"
                f"Drawdown: {dd*100:.1f}%\n"
                f"Capital: ${estado['capital']:,.2f}"
            )
            continue

        # Cargar modelo y generar señal
        n_pos_abiertas = sum(1 for p in estado['posiciones'] if p['simbolo'] == nombre)
        if n_pos_abiertas >= MAX_POSICIONES:
            print(f"  ℹ️  Máximo de posiciones abiertas para {nombre}")
            continue

        modelos = cargar_modelo(simbolo)
        if modelos is None:
            continue

        try:
            prob = predecir(df.iloc[:-1], modelos)  # excluir vela en formación
        except Exception as e:
            print(f"  ❌ Error predicción {simbolo}: {e}")
            continue

        print(f"  Probabilidad: {prob:.4f} | Régimen: {regimen} | "
              f"Precio: ${precio_actual:,.4f}")

        # Evaluar señal
        if prob < UMBRAL_MINIMO:
            print(f"  → Sin señal (prob {prob:.3f} < {UMBRAL_MINIMO})")
            continue

        ok, motivo_filtro = filtro_entrada(regimen, vol_bajo)
        if not ok:
            print(f"  → Filtrado: {motivo_filtro}")
            continue

        resultado = clasificar_señal(prob, regimen)
        if resultado is None:
            print(f"  → Condiciones insuficientes")
            continue

        calidad, cfg = resultado
        print(f"  → ✅ SEÑAL {calidad.upper()} | prob={prob:.3f} | x{cfg['leverage']}")

        abrir_posicion(
            estado, nombre, precio_actual, atr_actual,
            prob, cfg, calidad, timestamp, hora_actual
        )
        registrar_operacion({
            'timestamp':      timestamp,
            'simbolo':        nombre,
            'tipo':           'apertura',
            'motivo':         calidad,
            'precio_entrada': round(precio_actual, 6),
            'precio_salida':  None,
            'pnl_usdt':       None,
            'funding_usdt':   None,
            'horas':          None,
            'leverage':       cfg['leverage'],
            'calidad':        calidad,
            'capital_total':  round(estado['capital'], 2),
        })

        time.sleep(0.5)  # respetar rate limits

    # Guardar estado y actualizar métricas
    guardar_estado(estado)
    metricas = actualizar_metricas(estado)

    # Resumen en consola
    print(f"\n{'─'*60}")
    print(f"  Capital:      ${metricas['capital_actual']:,.2f}  "
          f"({'+' if metricas['rentabilidad_pct'] >= 0 else ''}"
          f"{metricas['rentabilidad_pct']:.2f}%)")
    print(f"  Operaciones:  {metricas['n_operaciones']}  |  "
          f"Win rate: {metricas['win_rate_pct']:.1f}%")
    print(f"  Posiciones:   {metricas['posiciones_abiertas']} abiertas")
    print(f"{'─'*60}")

    # Resumen diario por Telegram (cada 24 ejecuciones aprox)
    if estado['n_operaciones'] > 0 and estado['n_operaciones'] % 10 == 0:
        rent = metricas['rentabilidad_pct']
        enviar_telegram(
            f"📊 <b>Resumen Paper Trading</b>\n"
            f"Capital: ${metricas['capital_actual']:,.2f} "
            f"({'+' if rent >= 0 else ''}{rent:.2f}%)\n"
            f"Operaciones: {metricas['n_operaciones']} | "
            f"WR: {metricas['win_rate_pct']:.1f}%\n"
            f"P&L total: ${metricas['pnl_total']:+.2f}"
        )


if __name__ == '__main__':
    ejecutar()
