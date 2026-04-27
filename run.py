import os, subprocess, sys

processed = 'models/data/processed'

# Comprobar si hay CSVs procesados
csvs = []
if os.path.exists(processed):
    csvs = [f for f in os.listdir(processed) if f.endswith('.csv')]

if len(csvs) < 8:
    print("⚙️  Sin datos procesados — ejecutando pipeline...")
    subprocess.run([sys.executable, 'descargar_datos.py'], check=True)
    subprocess.run([sys.executable, 'preparar_datos.py'], check=True)
    print("✅ Datos listos")
else:
    print(f"✅ Datos ya disponibles ({len(csvs)} CSVs)")

# Lanzar el bot
import paper_trading_v15
paper_trading_v15.ejecutar()
