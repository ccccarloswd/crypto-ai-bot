
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Tuple

def _verificar_pytorch():
    try:
        import torch
        return True
    except ImportError:
        return False

def _construir_red(n_features, longitud_seq, hidden_size=128, n_capas=2, dropout=0.3):
    import torch
    import torch.nn as nn
    class LSTMTrading(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm     = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                                    num_layers=n_capas, batch_first=True,
                                    dropout=dropout if n_capas > 1 else 0.0)
            self.dropout1 = nn.Dropout(dropout)
            self.norm     = nn.LayerNorm(hidden_size)
            self.fc1      = nn.Linear(hidden_size, 64)
            self.relu     = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout * 0.5)
            self.fc2      = nn.Linear(64, 1)
            self.sigmoid  = nn.Sigmoid()
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            ultimo = lstm_out[:, -1, :]
            out = self.dropout1(ultimo)
            out = self.norm(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.dropout2(out)
            out = self.fc2(out)
            return self.sigmoid(out).squeeze(1)
    return LSTMTrading()

def _crear_dataset_torch(X, y):
    import torch
    from torch.utils.data import TensorDataset
    return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))

def entrenar_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                  dir_modelos="models", epochs=50, batch_size=256,
                  learning_rate=0.001, paciencia=8):
    if not _verificar_pytorch():
        print("    PyTorch no instalado. Omitiendo LSTM.")
        return None, {}
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Dispositivo: {device}")

    n_features   = X_train_seq.shape[2]
    longitud_seq = X_train_seq.shape[1]

    ds_train = _crear_dataset_torch(X_train_seq, y_train_seq)
    ds_val   = _crear_dataset_torch(X_val_seq,   y_val_seq)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)

    modelo = _construir_red(n_features, longitud_seq).to(device)
    ratio  = float((y_train_seq == 0).sum() / max((y_train_seq == 1).sum(), 1))
    criterio    = nn.BCELoss(weight=torch.tensor([ratio]).to(device))
    optimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, patience=3, factor=0.5)

    historial = {"loss_train": [], "loss_val": []}
    mejor_loss_val     = float("inf")
    epocas_sin_mejora  = 0
    ruta_checkpoint    = os.path.join(dir_modelos, "lstm_mejor.pt")
    os.makedirs(dir_modelos, exist_ok=True)

    print(f"    Entrenando LSTM ({n_features} features, seq={longitud_seq}, hidden=128, capas=2)...")
    print(f"    {'Epoca':>6}  {'Loss Train':>12}  {'Loss Val':>10}  {'Estado':>10}")
    print(f"    {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}")

    for epoca in range(1, epochs + 1):
        modelo.train()
        loss_train_total = 0.0
        for X_batch, y_batch in loader_train:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizador.zero_grad()
            pred = modelo(X_batch)
            loss = criterio(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
            loss_train_total += loss.item() * len(X_batch)
        loss_train = loss_train_total / len(ds_train)

        modelo.eval()
        loss_val_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader_val:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss_val_total += criterio(modelo(X_batch), y_batch).item() * len(X_batch)
        loss_val = loss_val_total / len(ds_val)
        scheduler.step(loss_val)

        historial["loss_train"].append(loss_train)
        historial["loss_val"].append(loss_val)

        if loss_val < mejor_loss_val - 1e-4:
            mejor_loss_val    = loss_val
            epocas_sin_mejora = 0
            torch.save(modelo.state_dict(), ruta_checkpoint)
            estado = "✅ guardado"
        else:
            epocas_sin_mejora += 1
            estado = f"({epocas_sin_mejora}/{paciencia})"

        if epoca % 5 == 0 or epocas_sin_mejora == paciencia:
            print(f"    {epoca:>6}  {loss_train:>12.5f}  {loss_val:>10.5f}  {estado:>10}")

        if epocas_sin_mejora >= paciencia:
            print(f"\n    Early stopping en epoca {epoca}")
            break

    modelo.load_state_dict(torch.load(ruta_checkpoint, map_location=device))
    modelo.eval()

    ruta_final = os.path.join(dir_modelos, "lstm.pt")
    torch.save({"state_dict": modelo.state_dict(),
                "n_features": n_features,
                "longitud_seq": longitud_seq}, ruta_final)

    print(f"\n    ✅ LSTM guardado en {ruta_final}")
    print(f"    Mejor loss val: {mejor_loss_val:.5f}")
    return modelo, {"historial": historial, "mejor_loss_val": mejor_loss_val}

def predecir_lstm(modelo, X_seq, batch_size=512):
    if modelo is None:
        return np.full(len(X_seq), 0.5)
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    device   = next(modelo.parameters()).device
    dataset  = TensorDataset(torch.FloatTensor(X_seq))
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs    = []
    modelo.eval()
    with torch.no_grad():
        for (X_batch,) in loader:
            pred = modelo(X_batch.to(device))
            probs.extend(pred.cpu().numpy())
    return np.array(probs)

def cargar_lstm(dir_modelos="models"):
    if not _verificar_pytorch():
        return None
    import torch
    ruta = os.path.join(dir_modelos, "lstm.pt")
    if not os.path.exists(ruta):
        return None
    checkpoint = torch.load(ruta, map_location="cpu")
    modelo = _construir_red(checkpoint["n_features"], checkpoint["longitud_seq"])
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    return modelo
