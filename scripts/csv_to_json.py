import os, json
import pandas as pd

CSV_PATH = 'paper_trading/v14/operaciones.csv'
JSON_OUT = 'docs/data/operaciones.json'

os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    for col in ['entrada', 'salida', 'pnl', 'capital']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    with open(JSON_OUT, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, default=str)
    print(f"✅ {len(df)} operaciones convertidas")
else:
    with open(JSON_OUT, 'w') as f:
        json.dump([], f)
