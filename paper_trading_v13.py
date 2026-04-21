"""
=============================================================
  CRYPTO AI BOT — Paper Trading v13
=============================================================
  Estrategia: Long + Short con modelos independientes
  Configuración: parámetros diferenciados por cripto
  Ejecutado cada hora por GitHub Actions
=============================================================
"""

import os, json, time, urllib.request, numpy as np, pandas as pd
import joblib, warnings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
#  IDENTIFICADOR DE VERSIÓN
# ──────────────────────────────────────────────
VERSION = 'v13'

# ──────────────────────────────────────────────
#  CONFIGURACIÓN — igual que backtesting v13
# ──────────────────────────────────────────────
CRIPTOS   = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
VELAS_N   = 300

CAPITAL_INICIAL   = 1000.0
COMISION_PCT      = 0.001
SLIPPAGE_PCT      = 0.0005
FUNDING_RATE_HORA = 0.0003

DD_PAUSA_PCT = 0.15
DD_STOP_PCT  = 0.25
HORAS_PAUSA  = 24

MAX_HORAS_LONG  = 12
MAX_HORAS_SHORT = 10
MAX_POSICIONES       = 2
MAX_MARGEN_TOTAL_PCT = 0.40
TRAILING_ACTIVACION  = 0.020
TRAILING_ATR_MULT    = 1.2

SOLO_XGBOOST = ['BTC_USDT', 'BNB_USDT']

# Config longs base
LONG_CONFIG_BASE = {
    'elite':   {'prob': 0.74, 'leverage': 8, 'margen': 0.22,
                'sl_mult': 1.2, 'sl_max': 0.020, 'tp_ratio': 3.5, 'tp_max': 0.080},
    'premium': {'prob': 0.64, 'leverage': 5, 'margen': 0.20,
                'sl_mult': 1.5, 'sl_max': 0.025, 'tp_ratio': 3.0, 'tp_max': 0.060},
}
LONG_CONFIG_ETH = {
    'elite':   LONG_CONFIG_BASE['elite'],
    'premium': {**LONG_CONFIG_BASE['premium'], 'prob': 0.67},
}
SHORT_CONFIG_SOLO_ELITE = {
    'elite':   {'prob': 0.70, 'leverage': 6, 'margen': 0.18,
                'sl_mult': 1.3, 'sl_max': 0.022, 'tp_ratio': 3.2, 'tp_max': 0.070},
    'premium': None,
    'regimen_max': -1,
}
SHORT_CONFIG_ETH = {
    'elite':   {'prob': 0.70, 'leverage': 6, 'margen': 0.18,
                'sl_mult': 1.3, 'sl_max': 0.022, 'tp_ratio': 3.2, 'tp_max': 0.070},
    'premium': {'prob': 0.60, 'leverage': 4, 'margen': 0.15,
                'sl_mult': 1.5, 'sl_max': 0.025, 'tp_ratio': 3.0, 'tp_max': 0.060},
    'regimen_max': -1,
}
SHORT_CONFIG_SOL = {
    'elite':   {'prob': 0.70, 'leverage': 6, 'margen': 0.18,
                'sl_mult': 1.3, 'sl_max': 0.022, 'tp_ratio': 3.2, 'tp_max': 0.070},
    'premium': {'prob': 0.65, 'leverage': 4, 'margen': 0.15,
                'sl_mult': 1.5, 'sl_max': 0.025, 'tp_ratio': 3.0, 'tp_max': 0.060},
    'regimen_max': -1,
}
LONG_CFG  = {'BTC_USDT': LONG_CONFIG_BASE, 'ETH_USDT': LONG_CONFIG_ETH,
             'BNB_USDT': LONG_CONFIG_BASE, 'SOL_USDT': LONG_CONFIG_BASE}
SHORT_CFG = {'BTC_USDT': SHORT_CONFIG_SOLO_ELITE, 'ETH_USDT': SHORT_CONFIG_ETH,
             'BNB_USDT': SHORT_CONFIG_SOLO_ELITE, 'SOL_USDT': SHORT_CONFIG_SOL}

DIR_MODELOS   = 'models'
ESTADO_FILE   = f'paper_trading/{VERSION}/estado.json'
LOG_FILE      = f'paper_trading/{VERSION}/operaciones.csv'
METRICAS_FILE = f'paper_trading/{VERSION}/metricas.json'


# ──────────────────────────────────────────────
#  TELEGRAM
# ──────────────────────────────────────────────
def telegram(msg: str):
    token   = os.environ.get('TELEGRAM_TOKEN_V13', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID_V13', '')
    if not token or not chat_id:
        print(f"[TG-{VERSION}] {msg[:100]}")
        return
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({'chat_id': chat_id, 'text': msg,
                           'parse_mode': 'HTML'}).encode()
        req  = urllib.request.Request(url, data=data,
                                       headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  ⚠️  TG error: {e}")


# ──────────────────────────────────────────────
#  ESTADO PERSISTENTE
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
        'bot_activo': estado['bot_activo'],
        'ultima_actualizacion': datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(METRICAS_FILE), exist_ok=True)
    with open(METRICAS_FILE, 'w') as f:
        json.dump(m, f, indent=2)
    return m


# ──────────────────────────────────────────────
#  DATOS DE MERCADO — Kraken
# ──────────────────────────────────────────────
KRAKEN_MAP = {
    'BTC/USDT': 'XBTUSD', 'ETH/USDT': 'ETHUSD',
    'BNB/USDT': 'BNBUSD', 'SOL/USDT': 'SOLUSD',
}

def obtener_velas(simbolo: str) -> Optional[pd.DataFrame]:
    par   = KRAKEN_MAP.get(simbolo, simbolo.replace('/', ''))
    desde = int(time.time()) - (VELAS_N * 3600)
    url   = f"https://api.kraken.com/0/public/OHLC?pair={par}&interval=60&since={desde}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if data.get('error'):
            print(f"  ⚠️  Kraken: {data['error']}")
            return None
        result = data.get('result', {})
        clave  = [k for k in result if k != 'last']
        if not clave:
            return None
        velas = result[clave[0]]
        if len(velas) < 200:
            return None
        df = pd.DataFrame(velas, columns=[
            'time','open','high','low','close','vwap','volume','count'])
        df['timestamp'] = pd.to_datetime(df['time'].astype(int), unit='s')
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        return df[['timestamp','open','high','low','close','volume']].reset_index(drop=True)
    except Exception as e:
        print(f"  ❌ Kraken {simbolo}: {e}")
        return None


# ──────────────────────────────────────────────
#  INDICADORES
# ──────────────────────────────────────────────
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import pandas_ta as ta
    except ImportError:
        import pandas_ta_openbb as ta

    df = df.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [9,21,50,100,200]:
        df[f'ema_{p}'] = ta.ema(c, length=p)
    for p in [20,50,200]:
        df[f'sma_{p}'] = ta.sma(c, length=p)

    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df['macd']        = macd_df.iloc[:,0]
        df['macd_signal'] = macd_df.iloc[:,1] if macd_df.shape[1]>1 else np.nan
        df['macd_hist']   = macd_df.iloc[:,2] if macd_df.shape[1]>2 else np.nan
    else:
        df['macd'] = df['macd_signal'] = df['macd_hist'] = np.nan
    df['macd_cruce'] = np.where(
        (df['macd']>df['macd_signal'])&(df['macd'].shift(1)<=df['macd_signal'].shift(1)),1,
        np.where((df['macd']<df['macd_signal'])&(df['macd'].shift(1)>=df['macd_signal'].shift(1)),-1,0))

    adx_df = ta.adx(h,l,c,length=14)
    if adx_df is not None and not adx_df.empty:
        df['adx']     = adx_df.iloc[:,0]
        df['dmi_pos'] = adx_df.iloc[:,1] if adx_df.shape[1]>1 else np.nan
        df['dmi_neg'] = adx_df.iloc[:,2] if adx_df.shape[1]>2 else np.nan
    else:
        df['adx'] = df['dmi_pos'] = df['dmi_neg'] = np.nan

    df['golden_cross']        = np.where((df['ema_50']>df['ema_200'])&(df['ema_50'].shift(1)<=df['ema_200'].shift(1)),1,0)
    df['death_cross']         = np.where((df['ema_50']<df['ema_200'])&(df['ema_50'].shift(1)>=df['ema_200'].shift(1)),1,0)
    df['precio_sobre_ema50']  = (c>df['ema_50']).astype(int)
    df['precio_sobre_ema200'] = (c>df['ema_200']).astype(int)
    df['ema50_sobre_ema200']  = (df['ema_50']>df['ema_200']).astype(int)

    df['rsi_14'] = ta.rsi(c,length=14)
    df['rsi_7']  = ta.rsi(c,length=7)
    df['rsi_sobrecompra']  = (df['rsi_14']>=70).astype(int)
    df['rsi_sobreventa']   = (df['rsi_14']<=30).astype(int)
    df['rsi_zona_neutral'] = ((df['rsi_14']>40)&(df['rsi_14']<60)).astype(int)
    ps = (c>c.shift(5)).astype(int); rs = (df['rsi_14']>df['rsi_14'].shift(5)).astype(int)
    df['divergencia_bajista_rsi'] = ((ps==1)&(rs==0)).astype(int)
    df['divergencia_alcista_rsi'] = ((ps==0)&(rs==1)).astype(int)

    st = ta.stoch(h,l,c,k=14,d=3)
    if st is not None and not st.empty:
        df['stoch_k'] = st.iloc[:,0]
        df['stoch_d'] = st.iloc[:,1] if st.shape[1]>1 else np.nan
        df['stoch_sobrecompra'] = (df['stoch_k']>=80).astype(int)
        df['stoch_sobreventa']  = (df['stoch_k']<=20).astype(int)
    else:
        df['stoch_k'] = df['stoch_d'] = np.nan
        df['stoch_sobrecompra'] = df['stoch_sobreventa'] = 0

    df['cci_20']          = ta.cci(h,l,c,length=20)
    df['roc_10']          = ta.roc(c,length=10)
    df['roc_20']          = ta.roc(c,length=20)
    df['williams_r']      = ta.willr(h,l,c,length=14)
    df['momentum_10']     = c - c.shift(10)
    df['momentum_pct_10'] = c.pct_change(periods=10)*100

    bb = ta.bbands(c,length=20,std=2)
    if bb is not None and not bb.empty:
        cols = bb.columns.tolist()
        uc = next((x for x in cols if 'U' in x or 'upper' in x.lower()),None)
        mc = next((x for x in cols if 'M' in x or 'mid' in x.lower()),None)
        lc = next((x for x in cols if 'L' in x or 'lower' in x.lower()),None)
        df['bb_upper'] = bb[uc] if uc else np.nan
        df['bb_mid']   = bb[mc] if mc else np.nan
        df['bb_lower'] = bb[lc] if lc else np.nan
        rbb = df['bb_upper']-df['bb_lower']
        df['bb_width']    = rbb/df['bb_mid'].replace(0,np.nan)
        df['bb_posicion'] = np.where(rbb>0,(c-df['bb_lower'])/rbb,0.5)
        df['bb_squeeze']  = (df['bb_width']<df['bb_width'].rolling(50).mean()*0.7).astype(int)
    else:
        for x in ['bb_upper','bb_mid','bb_lower','bb_width','bb_posicion','bb_squeeze']:
            df[x] = np.nan

    df['atr_14']  = ta.atr(h,l,c,length=14)
    df['atr_pct'] = df['atr_14']/c*100

    kc = ta.kc(h,l,c,length=20)
    if kc is not None and not kc.empty:
        df['kc_upper'] = kc.iloc[:,0]
        df['kc_mid']   = kc.iloc[:,1] if kc.shape[1]>1 else np.nan
        df['kc_lower'] = kc.iloc[:,2] if kc.shape[1]>2 else np.nan
    else:
        df['kc_upper'] = df['kc_mid'] = df['kc_lower'] = np.nan

    ret = c.pct_change()
    df['volatilidad_20'] = ret.rolling(20).std()*np.sqrt(24*365)*100
    df['volatilidad_50'] = ret.rolling(50).std()*np.sqrt(24*365)*100
    df['rango_vela']     = h-l
    df['rango_vela_pct'] = (h-l)/c*100

    df['obv']           = ta.obv(c,v)
    df['obv_ema']       = ta.ema(df['obv'],length=20)
    df['obv_tendencia'] = (df['obv']>df['obv_ema']).astype(int)
    try:
        dt = df.set_index('timestamp')
        vr = ta.vwap(dt['high'],dt['low'],dt['close'],dt['volume'])
        df['vwap'] = vr.values if vr is not None else np.nan
    except:
        df['vwap'] = np.nan
    df['precio_sobre_vwap'] = (c>df['vwap']).fillna(False).astype(int)
    df['mfi_14']          = ta.mfi(h,l,c,v,length=14)
    df['mfi_sobrecompra'] = (df['mfi_14']>=80).astype(int)
    df['mfi_sobreventa']  = (df['mfi_14']<=20).astype(int)

    vm = v.rolling(20).mean(); vs = v.rolling(20).std()
    df['volumen_ratio']    = v/vm
    df['volumen_anomalo']  = (v>vm+2*vs).astype(int)
    df['volumen_muy_bajo'] = (v<vm*0.3).astype(int)
    va = (c>df['open']).astype(int)
    df['confirmacion_alcista'] = ((va==1)&(df['volumen_anomalo']==1)).astype(int)
    df['confirmacion_bajista'] = ((va==0)&(df['volumen_anomalo']==1)).astype(int)

    df['es_resistencia'] = (h==h.rolling(20,center=True).max()).astype(int)
    df['es_soporte']     = (l==l.rolling(20,center=True).min()).astype(int)
    ur = h.where(df['es_resistencia']==1).ffill()
    us = l.where(df['es_soporte']==1).ffill()
    df['dist_resistencia_pct'] = (ur-c)/c*100
    df['dist_soporte_pct']     = (c-us)/c*100

    mx = h.rolling(50).max(); mn = l.rolling(50).min(); rg = mx-mn
    df['fib_236'] = mx-rg*0.236; df['fib_382'] = mx-rg*0.382
    df['fib_500'] = mx-rg*0.500; df['fib_618'] = mx-rg*0.618
    for nv in ['fib_236','fib_382','fib_500','fib_618']:
        df[f'cerca_{nv}'] = (abs(c-df[nv])/c<0.005).astype(int)

    df['ema200_pendiente'] = df['ema_200'].diff(50)/df['ema_200'].shift(50)*100
    se200=(c>df['ema_200']).astype(int); se50=(c>df['ema_50']).astype(int)
    e5_200=(df['ema_50']>df['ema_200']).astype(int)
    conds=[(se200==1)&(se50==1)&(e5_200==1)&(df['ema200_pendiente']>0.5),
           (se200==1)&(e5_200==1),
           (se200==0)&(e5_200==0)&(df['ema200_pendiente']<-0.5),
           (se200==0)&(e5_200==0)]
    df['regimen_mercado'] = np.select(conds,[2,1,-2,-1],default=0)

    return df


def añadir_features_neutras(df: pd.DataFrame, features_modelo: list) -> pd.DataFrame:
    """Rellena features que no se pueden calcular en tiempo real con valores neutros."""
    neutros = {
        'fear_greed':0,'fear_greed_num':0,'fear_greed_cambio':0,
        'mcap_btc':0,'mcap_btc_log':0,'mcap_btc_cambio_7d':0,
        'correlacion_btc':1.0,'descorrelacionado':0,
        'tendencia_4h':0,'tendencia_diario':0,'tendencia_semanal':0,
        'rsi_4h':50,'rsi_diario':50,'rsi_semanal':50,
        'ema50_4h':0,'ema50_diario':0,'ema50_semanal':0,
        'alineacion_timeframes':0,'señal_multitf_fuerte':0,
        'patron_doji':0,'patron_martillo':0,'patron_estrella_fugaz':0,
        'patron_marubozu_alcista':0,'patron_marubozu_bajista':0,
        'patron_engulfing_alcista':0,'patron_engulfing_bajista':0,
        'patron_harami_alcista':0,'patron_morning_star':0,
        'patron_evening_star':0,'patron_tres_soldados':0,
        'patron_tres_cuervos':0,'señal_velas':0,
        'patron_doble_suelo':0,'patron_doble_techo':0,
        'patron_hch':0,'patron_hch_invertido':0,
        'patron_triangulo':0,'patron_flag':0,'patron_cuña':0,
        'en_canal':0,'posicion_canal':0.5,'señal_patrones_graficos':0,
        'ichi_tenkan':0,'ichi_kijun':0,'ichi_senkou_a':0,
        'ichi_senkou_b':0,'ichi_chikou':0,
    }
    for f in features_modelo:
        if f not in df.columns:
            df[f] = neutros.get(f, 0)
    return df


# ──────────────────────────────────────────────
#  MODELO
# ──────────────────────────────────────────────
def cargar_submodelo(simbolo: str, tipo: str) -> Optional[Dict]:
    nombre = simbolo.replace('/','_')
    dir_m  = os.path.join(DIR_MODELOS, nombre, tipo)
    if not os.path.exists(os.path.join(dir_m,'xgboost.pkl')):
        print(f"  ❌ Modelo {nombre}/{tipo} no encontrado")
        return None
    try:
        from models.ensemble import EnsembleTrading
        sub = {
            'xgboost':  joblib.load(os.path.join(dir_m,'xgboost.pkl')),
            'scaler':   joblib.load(os.path.join(dir_m,'scaler.pkl')),
            'features': joblib.load(os.path.join(dir_m,'features.pkl')),
            'lstm': None,
        }
        ens = EnsembleTrading(); ens.cargar(dir_m); sub['ensemble'] = ens
        if nombre not in SOLO_XGBOOST:
            try:
                from models.lstm_model import cargar_lstm
                lstm = cargar_lstm(dir_m)
                if lstm: sub['lstm'] = lstm
            except: pass
        return sub
    except Exception as e:
        print(f"  ❌ Error cargando {nombre}/{tipo}: {e}")
        return None

def predecir(df: pd.DataFrame, sub: Dict) -> float:
    df = añadir_features_neutras(df.copy(), sub['features'])
    fd = [f for f in sub['features'] if f in df.columns]
    X  = sub['scaler'].transform(df[fd].values)
    np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
    px = sub['xgboost'].predict_proba(X)[:,1]
    pl = None
    if sub['lstm']:
        try:
            from models.lstm_model import predecir_lstm
            from models.preparar_datos import crear_secuencias
            Xs,_ = crear_secuencias(X, np.zeros(len(X)), 48)
            if len(Xs)>0:
                r  = predecir_lstm(sub['lstm'], Xs)
                pd_ = np.full(len(px)-len(r), 0.5)
                pl  = np.concatenate([pd_, r])
        except: pass
    return float(sub['ensemble'].predecir_proba(px, pl)[-1])


# ──────────────────────────────────────────────
#  ESTRATEGIA v13
# ──────────────────────────────────────────────
def clasificar_long(prob: float, row: pd.Series, cfg: dict) -> Optional[Tuple[str,dict]]:
    reg = int(row.get('regimen_mercado',0)) if not pd.isna(row.get('regimen_mercado',0)) else 0
    if reg == -2: return None
    if prob >= cfg['elite']['prob'] and reg >= 0: return 'elite', cfg['elite']
    if prob >= cfg['premium']['prob']: return 'premium', cfg['premium']
    return None

def confirmacion_short(row: pd.Series) -> bool:
    conds = 0
    rsi = row.get('rsi_14',50)
    if not pd.isna(rsi) and float(rsi)<45: conds+=1
    mh = row.get('macd_hist',0)
    if not pd.isna(mh) and float(mh)<0: conds+=1
    c = float(row.get('close',0))
    for col in ['sma_20','bb_mid','ema_20']:
        v = row.get(col)
        if v is not None and not pd.isna(v) and float(v)>0:
            if c < float(v): conds+=1
            break
    return conds >= 2

def clasificar_short(prob: float, row: pd.Series, cfg: dict) -> Optional[Tuple[str,dict]]:
    reg = int(row.get('regimen_mercado',0)) if not pd.isna(row.get('regimen_mercado',0)) else 0
    if reg > cfg['regimen_max']: return None
    if not confirmacion_short(row): return None
    if prob >= cfg['elite']['prob']: return 'elite', cfg['elite']
    if cfg['premium'] and prob >= cfg['premium']['prob']: return 'premium', cfg['premium']
    return None

def sl_tp_long(p: float, atr: float, cfg: dict) -> Tuple[float,float,float]:
    sl = float(np.clip((atr*cfg['sl_mult'])/p, 0.008, cfg['sl_max']))
    tp = float(np.clip(sl*cfg['tp_ratio'], sl*2.5, cfg['tp_max']))
    return p*(1-sl), p*(1+tp), p*(1-(1/cfg['leverage'])*0.90)

def sl_tp_short(p: float, atr: float, cfg: dict) -> Tuple[float,float,float]:
    sl = float(np.clip((atr*cfg['sl_mult'])/p, 0.008, cfg['sl_max']))
    tp = float(np.clip(sl*cfg['tp_ratio'], sl*2.5, cfg['tp_max']))
    return p*(1+sl), p*(1-tp), p*(1+(1/cfg['leverage'])*0.90)


# ──────────────────────────────────────────────
#  GESTIÓN DE POSICIONES
# ──────────────────────────────────────────────
def gestionar_posiciones(estado: Dict, precio: float, atr: float,
                          hora: int, simbolo: str, ts: str) -> List[Dict]:
    cerradas = []; nuevas = []
    for pos in estado['posiciones']:
        if pos['simbolo'] != simbolo:
            nuevas.append(pos); continue
        horas = hora - pos['hora_entrada']
        max_h = MAX_HORAS_LONG if pos['dir']=='long' else MAX_HORAS_SHORT

        if pos['dir']=='long':
            if precio > pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                g = (precio-pos['precio_entrada'])/pos['precio_entrada']
                if g >= TRAILING_ACTIVACION:
                    pos['trailing'] = True
                    pos['sl'] = max(pos['sl'], pos['precio_ref']-atr*TRAILING_ATR_MULT)
            cerrar = precio<=pos.get('liq',0) and 'liq' in pos
            motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio<=pos['sl']: cerrar=True; motivo='trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio>=pos['tp']: cerrar=True; motivo='take_profit'
                elif horas>=max_h: cerrar=True; motivo='tiempo_maximo'
            if cerrar:
                ps = precio*(1-SLIPPAGE_PCT)
                exp = pos['margen']*pos['lev']
                ret = (ps-pos['precio_entrada'])/pos['precio_entrada']
                fund = exp*FUNDING_RATE_HORA*horas
                pnl  = exp*ret - exp*COMISION_PCT*2 - fund
                estado['capital']+=pnl; estado['pnl_total']+=pnl
                estado['funding_total']+=fund; estado['n_ops']+=1
                if pnl>0: estado['n_wins']+=1
                else: estado['n_loses']+=1
                op = {'ts':ts,'simbolo':simbolo,'dir':'long','motivo':motivo,
                      'entrada':round(pos['precio_entrada'],4),'salida':round(ps,4),
                      'pnl':round(pnl,2),'capital':round(estado['capital'],2)}
                cerradas.append(op)
                emoji = '✅' if pnl>0 else '❌'
                telegram(f"{emoji} <b>CIERRE LONG {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo}\n"
                         f"Entrada: ${pos['precio_entrada']:,.4f} → Salida: ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Capital: ${estado['capital']:,.2f}")
            else:
                nuevas.append(pos)
        else:  # short
            if precio < pos.get('precio_ref', pos['precio_entrada']):
                pos['precio_ref'] = precio
                g = (pos['precio_entrada']-precio)/pos['precio_entrada']
                if g >= TRAILING_ACTIVACION:
                    pos['trailing'] = True
                    nuevo_sl = pos['precio_ref']*(1+atr*TRAILING_ATR_MULT/pos['precio_ref'])
                    pos['sl'] = min(pos['sl'], nuevo_sl)
            cerrar = precio>=pos.get('liq',float('inf'))
            motivo = 'liquidacion' if cerrar else ''
            if not cerrar:
                if precio>=pos['sl']: cerrar=True; motivo='trailing_stop' if pos.get('trailing') else 'stop_loss'
                elif precio<=pos['tp']: cerrar=True; motivo='take_profit'
                elif horas>=max_h: cerrar=True; motivo='tiempo_maximo'
            if cerrar:
                ps = precio*(1+SLIPPAGE_PCT)
                exp = pos['margen']*pos['lev']
                ret = (pos['precio_entrada']-ps)/pos['precio_entrada']
                fund = exp*FUNDING_RATE_HORA*horas
                pnl  = exp*ret - exp*COMISION_PCT*2 - fund
                estado['capital']+=pnl; estado['pnl_total']+=pnl
                estado['funding_total']+=fund; estado['n_ops']+=1
                if pnl>0: estado['n_wins']+=1
                else: estado['n_loses']+=1
                op = {'ts':ts,'simbolo':simbolo,'dir':'short','motivo':motivo,
                      'entrada':round(pos['precio_entrada'],4),'salida':round(ps,4),
                      'pnl':round(pnl,2),'capital':round(estado['capital'],2)}
                cerradas.append(op)
                emoji = '✅' if pnl>0 else '❌'
                telegram(f"{emoji} <b>CIERRE SHORT {simbolo}</b> [{VERSION}]\n"
                         f"Motivo: {motivo}\n"
                         f"Entrada: ${pos['precio_entrada']:,.4f} → Salida: ${ps:,.4f}\n"
                         f"P&L: ${pnl:+.2f} | Capital: ${estado['capital']:,.2f}")
            else:
                nuevas.append(pos)
    estado['posiciones'] = nuevas
    return cerradas

def abrir_posicion(estado: Dict, simbolo: str, precio: float, atr: float,
                    prob: float, cfg: dict, calidad: str, dir_: str,
                    ts: str, hora: int):
    nombre = simbolo.replace('/','_')
    margen_uso = sum(p['margen'] for p in estado['posiciones'] if p['simbolo']==nombre)
    margen_max = estado['capital']*MAX_MARGEN_TOTAL_PCT
    margen_des = estado['capital']*cfg['margen']
    margen_r   = min(margen_des, margen_max-margen_uso)
    if margen_r < estado['capital']*0.03: return

    if dir_=='long':
        pe = precio*(1+SLIPPAGE_PCT)
        sl,tp,liq = sl_tp_long(pe,atr,cfg)
    else:
        pe = precio*(1-SLIPPAGE_PCT)
        sl,tp,liq = sl_tp_short(pe,atr,cfg)

    exp = margen_r*cfg['leverage']
    estado['posiciones'].append({
        'simbolo': nombre, 'dir': dir_, 'calidad': calidad,
        'precio_entrada': pe, 'precio_ref': pe,
        'sl': sl, 'tp': tp, 'liq': liq,
        'margen': margen_r, 'lev': cfg['leverage'],
        'hora_entrada': hora, 'trailing': False,
        'ts_entrada': ts,
    })
    telegram(f"🚀 <b>APERTURA {dir_.upper()} {simbolo}</b> [{VERSION}]\n"
             f"Calidad: {calidad.upper()} | Prob: {prob:.3f}\n"
             f"Precio: ${pe:,.4f} | SL: ${sl:,.4f} | TP: ${tp:,.4f}\n"
             f"Margen: ${margen_r:.2f} | x{cfg['leverage']}\n"
             f"Capital: ${estado['capital']:,.2f}")


# ──────────────────────────────────────────────
#  REPORTE DIARIO
# ──────────────────────────────────────────────
def enviar_reporte_diario(estado: Dict):
    capital = estado['capital']
    rent    = (capital-CAPITAL_INICIAL)/CAPITAL_INICIAL*100
    n       = estado['n_ops']
    wr      = estado['n_wins']/max(n,1)*100
    ops_hoy = estado.get('ops_hoy', [])

    msg = (f"📊 <b>REPORTE DIARIO {VERSION.upper()}</b>\n"
           f"{'─'*30}\n"
           f"💰 Capital: ${capital:,.2f} ({'+' if rent>=0 else ''}{rent:.2f}%)\n"
           f"📈 Total ops: {n} | WR: {wr:.1f}%\n"
           f"💵 P&L total: ${estado['pnl_total']:+.2f}\n"
           f"📉 Funding total: ${estado['funding_total']:.2f}\n"
           f"🔓 Posiciones abiertas: {len(estado['posiciones'])}\n"
           f"{'─'*30}\n")

    if ops_hoy:
        msg += f"<b>Operaciones de hoy ({len(ops_hoy)}):</b>\n"
        wins  = sum(1 for o in ops_hoy if o.get('pnl',0)>0)
        loses = len(ops_hoy)-wins
        pnl_d = sum(o.get('pnl',0) for o in ops_hoy)
        msg  += f"✅ {wins} ganadoras | ❌ {loses} perdedoras | P&L: ${pnl_d:+.2f}\n"
        for op in ops_hoy[-5:]:
            e = '✅' if op.get('pnl',0)>0 else '❌'
            msg += f"  {e} {op.get('dir','').upper()} {op.get('simbolo','')} → ${op.get('pnl',0):+.2f}\n"
    else:
        msg += "Sin operaciones hoy.\n"

    if not estado['bot_activo']:
        msg += "\n⚠️ BOT DETENIDO por drawdown máximo"
    elif estado['en_pausa']:
        msg += "\n⏸️ BOT EN PAUSA por drawdown"

    telegram(msg)
    estado['ops_hoy'] = []


# ──────────────────────────────────────────────
#  LOOP PRINCIPAL
# ──────────────────────────────────────────────
def ejecutar():
    print(f"\n{'='*50}")
    print(f"  Paper Trading {VERSION} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}")

    estado    = cargar_estado()
    hora_unix = int(time.time()/3600)

    # Mensaje de inicio si es la primera ejecución
    if estado.get('primera_ejecucion', True):
        telegram(f"🤖 <b>BOT {VERSION} INICIADO</b>\n"
                 f"Capital inicial: ${CAPITAL_INICIAL:,.2f}\n"
                 f"Estrategia: Long + Short con modelos independientes\n"
                 f"Criptos: BTC, ETH, BNB, SOL\n"
                 f"✅ Conexión verificada — el bot está activo")
        estado['primera_ejecucion'] = False

    # Reporte diario a las 00:00 UTC
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
            print(f"  ⏸️  En pausa. Quedan {fin-hora_unix}h")
            guardar_estado(estado)
            return
        estado['en_pausa'] = False

    for simbolo in CRIPTOS:
        nombre = simbolo.replace('/','_')
        print(f"\n  📊 {simbolo}")

        df = obtener_velas(simbolo)
        if df is None: continue

        try:
            df = calcular_indicadores(df)
        except Exception as e:
            print(f"  ❌ Indicadores: {e}"); continue

        df.dropna(subset=['ema_200','rsi_14','atr_14'], inplace=True)
        if len(df) < 50: continue

        ultima        = df.iloc[-2]
        precio        = float(ultima['close'])
        atr           = float(ultima.get('atr_14', precio*0.015))
        if pd.isna(atr) or atr<=0: atr = precio*0.015
        regimen       = int(ultima.get('regimen_mercado',0))
        vol_bajo      = int(ultima.get('volumen_muy_bajo',0))
        ts            = str(ultima['timestamp'])

        # Gestionar posiciones abiertas
        cerradas = gestionar_posiciones(estado, precio, atr, hora_unix, nombre, ts)
        for op in cerradas:
            registrar_op(op)
            estado.setdefault('ops_hoy',[]).append(op)

        # Verificar drawdown
        estado['capital_max'] = max(estado.get('capital_max', CAPITAL_INICIAL), estado['capital'])
        dd = (estado['capital_max']-estado['capital'])/estado['capital_max']
        if dd >= DD_STOP_PCT:
            estado['bot_activo'] = False
            telegram(f"🛑 <b>BOT {VERSION} DETENIDO</b>\nDrawdown: {dd*100:.1f}%\n"
                     f"Capital: ${estado['capital']:,.2f}")
            guardar_estado(estado); return
        if dd >= DD_PAUSA_PCT and not estado['en_pausa']:
            estado['en_pausa']       = True
            estado['fin_pausa_hora'] = hora_unix+HORAS_PAUSA
            telegram(f"⏸️ <b>PAUSA {VERSION}</b>\nDD: {dd*100:.1f}%")
            continue

        if vol_bajo: continue

        # Contar posiciones abiertas por dirección
        n_longs  = sum(1 for p in estado['posiciones'] if p['simbolo']==nombre and p['dir']=='long')
        n_shorts = sum(1 for p in estado['posiciones'] if p['simbolo']==nombre and p['dir']=='short')
        total_pos = len(estado['posiciones'])

        # Cargar modelos y predecir
        long_cfg_cripto  = LONG_CFG[nombre]
        short_cfg_cripto = SHORT_CFG[nombre]

        if total_pos < MAX_POSICIONES:
            sub_long = cargar_submodelo(simbolo, 'long')
            if sub_long:
                prob_long = predecir(df.iloc[:-1], sub_long)
                print(f"  prob_long={prob_long:.4f} | reg={regimen} | ${precio:,.2f}")
                if n_longs == 0 and prob_long >= long_cfg_cripto['premium']['prob']:
                    res = clasificar_long(prob_long, ultima, long_cfg_cripto)
                    if res:
                        cal, cfg = res
                        abrir_posicion(estado, simbolo, precio, atr,
                                       prob_long, cfg, cal, 'long', ts, hora_unix)

            if total_pos < MAX_POSICIONES:
                sub_short = cargar_submodelo(simbolo, 'short')
                if sub_short:
                    prob_short = predecir(df.iloc[:-1], sub_short)
                    print(f"  prob_short={prob_short:.4f}")
                    max_g_long = max(
                        ((precio-p['precio_entrada'])/p['precio_entrada']
                         for p in estado['posiciones']
                         if p['simbolo']==nombre and p['dir']=='long'),
                        default=0.0)
                    if n_shorts == 0 and prob_short >= short_cfg_cripto['elite']['prob'] \
                       and max_g_long < 0.03:
                        res = clasificar_short(prob_short, ultima, short_cfg_cripto)
                        if res:
                            cal, cfg = res
                            abrir_posicion(estado, simbolo, precio, atr,
                                           prob_short, cfg, cal, 'short', ts, hora_unix)

        time.sleep(0.5)

    guardar_estado(estado)
    m = guardar_metricas(estado)
    s = '+' if m['rentabilidad_pct']>=0 else ''
    print(f"\n  Capital: ${m['capital_actual']:,.2f} ({s}{m['rentabilidad_pct']:.2f}%)")
    print(f"  Ops: {m['n_operaciones']} | WR: {m['win_rate_pct']:.1f}%")
    print(f"  Posiciones abiertas: {m['posiciones_abiertas']}")


if __name__ == '__main__':
    ejecutar()
