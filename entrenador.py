import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ---- Configuracion ----
KEYPOINTS_DIR = "keypoints"
N_FRAMES = 30
EPOCHS = 100 # Aumentamos el maximo, el Early Stopping lo detendra antes si es necesario
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 7 # Epocas a esperar antes de Early Stopping

# ---- Cargar keypoints ----
print("\n[INFO] Cargando keypoints del dataset...")

X = []
y = []

if not os.path.exists(KEYPOINTS_DIR):
    print(f"[ERROR] No se encontro la carpeta '{KEYPOINTS_DIR}'")
    exit()

categorias = sorted([d for d in os.listdir(KEYPOINTS_DIR) if os.path.isdir(os.path.join(KEYPOINTS_DIR, d))])
cat_a_numero = {cat: i for i, cat in enumerate(categorias)}

print(f"Categorias encontradas: {categorias}")

for categoria in categorias:
    carpeta = os.path.join(KEYPOINTS_DIR, categoria)
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.npy')]
    print(f"  - {categoria}: {len(archivos)} secuencias")

    for archivo in archivos:
        ruta = os.path.join(carpeta, archivo)
        try:
            secuencia = np.load(ruta)
            if secuencia.shape == (N_FRAMES, 34):
                X.append(secuencia)
                y.append(cat_a_numero[categoria])
        except Exception as e:
            print(f"  Error cargando {archivo}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("[ERROR] No se cargaron datos. Verifica que los archivos .npy tengan forma (30, 34)")
    exit()

print(f"\nDataset total cargado: {X.shape[0]} secuencias")

# ---- Dividir train/val ----
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---- Preparar DataLoaders ----
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

# ---- Modelo LSTM Mejorado ----
class AnomalyLSTM(nn.Module):
    def __init__(self, n_categorias):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4 # Aumentado para reducir overfitting
        )
        self.clasificador = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_categorias)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Tomamos el ultimo estado de la secuencia
        return self.clasificador(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

modelo = AnomalyLSTM(n_categorias=len(categorias)).to(device)
optimizer = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Scheduler para reducir LR si el loss de val no baja
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ---- Entrenamiento ----
print("\nIniciando entrenamiento...")
mejor_val_loss = float('inf')
epocas_sin_mejora = 0

for epoch in range(EPOCHS):
    modelo.train()
    loss_train = 0
    
    # Barra de progreso para cada epoca
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        prediccion = modelo(X_batch)
        loss = criterion(prediccion, y_batch)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Validacion
    modelo.eval()
    loss_val = 0
    correctas = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            prediccion = modelo(X_batch)
            loss_val += criterion(prediccion, y_batch).item()
            correctas += (prediccion.argmax(dim=1) == y_batch).sum().item()

    avg_loss_train = loss_train / len(train_loader)
    avg_loss_val = loss_val / len(val_loader)
    accuracy = correctas / len(X_val)

    print(f" -> Val Loss: {avg_loss_val:.4f} | Accuracy: {accuracy*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Paso del scheduler
    scheduler.step(avg_loss_val)

    # Early Stopping y Guardar Mejor Modelo
    if avg_loss_val < mejor_val_loss:
        mejor_val_loss = avg_loss_val
        epocas_sin_mejora = 0
        torch.save({
            "model_state": modelo.state_dict(),
            "categorias": categorias,
            "cat_a_numero": cat_a_numero
        }, "anomaly_model.pt")
        print(f" [GUARDADO] Nuevo mejor modelo encontrado.")
    else:
        epocas_sin_mejora += 1
        if epocas_sin_mejora >= PATIENCE:
            print(f"\n[STOP] Early stopping activado tras {PATIENCE} epocas sin mejora.")
            break

print(f"\nProceso finalizado. Mejor Loss en validacion: {mejor_val_loss:.4f}")