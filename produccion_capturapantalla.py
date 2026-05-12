import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import mss 

# ---- Modelo (igual que en entrenamiento) ----
class AnomalyLSTM(nn.Module):
    def __init__(self, n_categorias):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.clasificador = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_categorias)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.clasificador(out[:, -1, :])


# ---- Configuracion ----
N_FRAMES = 30
UMBRAL_CONFIANZA = 0.75

# ---- Cargar modelos ----
print("Cargando modelos...")

if not os.path.exists("anomaly_model.pt"):
    print("\n[ERROR] No se encontro el archivo 'anomaly_model.pt'.")
    print("Debes entrenar el modelo primero con 'entrenador.py' o proporcionar uno.\n")
    exit()

checkpoint = torch.load("anomaly_model.pt", map_location=torch.device('cpu'))
categorias = checkpoint["categorias"]
cat_a_numero = checkpoint["cat_a_numero"]

# Las que disparan alerta — ajusta segun tu proyecto
CATEGORIAS_ANOMALAS = [c for c in categorias if c != "Normal"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

modelo = AnomalyLSTM(n_categorias=len(categorias)).to(device)
modelo.load_state_dict(checkpoint["model_state"])
modelo.eval()

yolo = YOLO("yolov8n-pose.pt")

print(f"Categorias cargadas: {categorias}")
print(f"Categorias anomalas: {CATEGORIAS_ANOMALAS}")
print("Sistema listo.\n")


 

#Para camara
cap = cv2.VideoCapture(0)

#Para video pregrabvado
#cap = cv2.VideoCapture("Database/Shoplifting/Shoplifting001_x264.mp4")

buffer = deque(maxlen=N_FRAMES)

estado = "Analizando..."
color = (255, 255, 0)
confianza_actual = 0.0
categoria_actual = ""





sct = mss.mss()

# Definís la región del navegador manualmente
# Ajusta estos valores según tu pantalla
monitor = {
    "top": 100,     # desde cuánto arriba empieza
    "left": 0,      # desde cuánto a la izquierda
    "width": 1280,  # ancho de la región
    "height": 720   # alto de la región
}

while True:
    screenshot = sct.grab(sct.monitors[0])
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
 
    #ret, frame = cap.read()
    #if not ret:
    #    print("No se pudo leer el frame.")
    #    break

    # 1. YOLO extrae keypoints del frame actual
    results = yolo(frame, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kp = results[0].keypoints.xy[0].cpu().numpy().flatten()
    else:
        kp = np.zeros(34)

    buffer.append(kp)

    # 2. Predecir cuando el buffer esta lleno
    if len(buffer) == N_FRAMES:
        secuencia = torch.tensor(
            np.array(buffer),
            dtype=torch.float32
        ).unsqueeze(0).to(device)  # (1, 30, 34)

        with torch.no_grad():
            salida = modelo(secuencia)
            probabilidades = torch.softmax(salida, dim=1)[0]
            confianza, clase_idx = probabilidades.max(dim=0)

            confianza_actual = confianza.item()
            categoria_actual = categorias[clase_idx.item()]

            if categoria_actual in CATEGORIAS_ANOMALAS and confianza_actual >= UMBRAL_CONFIANZA:
                estado = f"ALERTA: {categoria_actual}"
                color = (0, 0, 255)   # rojo
            else:
                estado = "Normal"
                color = (0, 255, 0)   # verde

    # 3. Dibujar sobre el frame
    frame_anotado = results[0].plot()

    # Fondo de la caja de texto
    cv2.rectangle(frame_anotado, (0, 0), (420, 90), (0, 0, 0), -1)

    # Estado principal
    cv2.putText(
        frame_anotado,
        estado,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, color, 2
    )

    # Confianza
    cv2.putText(
        frame_anotado,
        f"Confianza: {confianza_actual*100:.1f}%  |  {categoria_actual}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 1
    )

    cv2.imshow("Sistema de Deteccion de Anomalias", frame_anotado)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()