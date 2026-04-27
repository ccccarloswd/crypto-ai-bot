"""
descargar_datos.py  (v2)
========================
Cambios respecto a v1:
  - Conserva taker_buy_base y taker_buy_quote además de OHLCV
  - Conserva n_trades y quote_volume (útiles para features de flujo de órdenes)
  - El resto de lógica de descarga es idéntica
"""

import requests, time, os
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

SIMBOLOS  = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
TIMEFRAMES = ['1h', '4h', '1d']

DIAS_HISTORIA = 2190

CARPETA_RAW = 'models/data/raw'
BASE_URL    = 'https://api.binance.com/api/v3/klines'
LIMIT       = 1000

# Columnas que devuelve Binance en orden exacto
COLUMNAS_BINANCE = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'n_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]

# Columnas que guardamos (eliminamos close_time e ignore, guardamos el resto)
COLUMNAS_GUARDAR = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'quote_volume', 'n_trades', 'taker_buy_base', 'taker_buy_quote'
]

COLUMNAS_FLOAT = [
    'open', 'high', 'low', 'close', 'volume',
    'quote_volume', 'taker_buy_base', 'taker_buy_quote'
]


def descargar_velas(simbolo: str, intervalo: str, dias: int) -> pd.DataFrame:
    ahora_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    inicio_ms = ahora_ms - (dias * 24 * 60 * 60 * 1000)

    todas = []
    desde = inicio_ms

    ms_por_vela = {'1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000}
    ms_vela    = ms_por_vela[intervalo]
    n_requests = max(1, (ahora_ms - inicio_ms) // (LIMIT * ms_vela) + 1)

    with tqdm(total=n_requests, desc=f"{simbolo} {intervalo}", unit="req") as pbar:
        while desde < ahora_ms:
            params = {
                'symbol':    simbolo,
                'interval':  intervalo,
                'startTime': desde,
                'limit':     LIMIT,
            }
            try:
                r = requests.get(BASE_URL, params=params, timeout=15)
                r.raise_for_status()
                datos = r.json()
            except Exception as e:
                print(f"\n  ❌ Error descargando {simbolo} {intervalo}: {e}")
                time.sleep(5)
                continue

            if not datos:
                break

            todas.extend(datos)
            desde = datos[-1][0] + 1
            pbar.update(1)
            time.sleep(0.05)

            if len(datos) < LIMIT:
                break

    if not todas:
        return pd.DataFrame()

    df = pd.DataFrame(todas, columns=COLUMNAS_BINANCE)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['n_trades']  = df['n_trades'].astype(int)
    for col in COLUMNAS_FLOAT:
        df[col] = df[col].astype(float)

    df = df[COLUMNAS_GUARDAR].copy()
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)

    # Eliminar la última vela (incompleta, en formación)
    df = df.iloc[:-1]

    return df


def main():
    os.makedirs(CARPETA_RAW, exist_ok=True)

    resumen = []

    for simbolo in SIMBOLOS:
        for tf in TIMEFRAMES:
            ruta = os.path.join(CARPETA_RAW, f"{simbolo}_{tf}.csv")

            if os.path.exists(ruta):
                df_existente = pd.read_csv(ruta, parse_dates=['timestamp'])

                # Compatibilidad: si el CSV antiguo no tiene taker_buy_base, forzar redescarga
                if 'taker_buy_base' not in df_existente.columns:
                    print(f"  ⚠️  {simbolo} {tf} — CSV antiguo sin taker data, redescargando...")
                else:
                    ultima = pd.to_datetime(df_existente['timestamp'].iloc[-1], utc=True)
                    horas  = (datetime.now(timezone.utc) - ultima).total_seconds() / 3600
                    if horas < 2:
                        print(f"  ✅ {simbolo} {tf} — ya actualizado ({len(df_existente)} velas)")
                        resumen.append((simbolo, tf, len(df_existente), "ya actualizado"))
                        continue

            print(f"\n  ⬇️  Descargando {simbolo} {tf}...")
            df = descargar_velas(simbolo, tf, DIAS_HISTORIA)

            if df.empty:
                print(f"  ❌ Sin datos para {simbolo} {tf}")
                resumen.append((simbolo, tf, 0, "ERROR"))
                continue

            df.to_csv(ruta, index=False)
            print(f"  💾 Guardado: {ruta} ({len(df)} velas)")
            resumen.append((simbolo, tf, len(df), "descargado"))

            time.sleep(0.5)

    print(f"\n{'═'*55}")
    print(f"  RESUMEN DE DESCARGA")
    print(f"{'═'*55}")
    for simbolo, tf, n, estado in resumen:
        icono = "✅" if estado != "ERROR" else "❌"
        print(f"  {icono}  {simbolo:<10} {tf:<4}  {n:>5} velas  — {estado}")
    print(f"{'═'*55}")
    print(f"\n  Datos guardados en: {os.path.abspath(CARPETA_RAW)}")
    print(f"  Siguiente paso: ejecutar preparar_datos.py")


if __name__ == '__main__':
    main()
