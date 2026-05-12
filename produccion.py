import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque

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

checkpoint = torch.load("anomaly_model.pt", map_location=torch.device('cpu')) # Asegurar carga en CPU si no hay GPU
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

# Diccionario para guardar los movimientos de cada persona por su ID
person_buffers = {}

estado = "Analizando..."
color = (255, 255, 0)
confianza_actual = 0.0
categoria_actual = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    # 1. YOLO detecta y SIGUE (track) a las personas
    results = yolo.track(frame, persist=True, verbose=False)

    alerta_detectada = False

    if results[0].keypoints is not None and results[0].boxes is not None and results[0].boxes.id is not None:
        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for i, person_id in enumerate(ids):
            # Si es una persona nueva, le creamos su propio buffer
            if person_id not in person_buffers:
                person_buffers[person_id] = deque(maxlen=N_FRAMES)
            
            # Extraer y aplanar sus puntos clave (34 valores)
            kp = keypoints_list[i].flatten()
            
            # Asegurar que tenemos 34 valores (17 puntos x,y)
            if kp.shape[0] < 34:
                temp = np.zeros(34)
                temp[:kp.shape[0]] = kp
                kp = temp
            elif kp.shape[0] > 34:
                kp = kp[:34]

            person_buffers[person_id].append(kp)

            # 2. Si esta persona ya tiene suficientes frames, analizar su comportamiento
            if len(person_buffers[person_id]) == N_FRAMES:
                secuencia = torch.tensor(
                    np.array(person_buffers[person_id]),
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    salida = modelo(secuencia)
                    probabilidades = torch.softmax(salida, dim=1)[0]
                    confianza, clase_idx = probabilidades.max(dim=0)

                    temp_confianza = confianza.item()
                    temp_categoria = categorias[clase_idx.item()]

                    # Si CUALQUIERA esta en modo alerta, activamos la alarma global
                    if temp_categoria in CATEGORIAS_ANOMALAS and temp_confianza >= UMBRAL_CONFIANZA:
                        alerta_detectada = True
                        categoria_actual = temp_categoria
                        confianza_actual = temp_confianza

        if alerta_detectada:
            estado = f"ALERTA: {categoria_actual}"
            color = (0, 0, 255)   # rojo
        else:
            estado = "Normal"
            color = (0, 255, 0)   # verde
    else:
        # Si no hay nadie, el estado es normal/analizando
        estado = "Analizando..."
        color = (255, 255, 0)


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