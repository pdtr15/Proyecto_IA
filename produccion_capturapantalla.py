import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import mss
from modelo import AnomalyLSTM


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
CATEGORIAS_ANOMALAS = [c for c in categorias if c.lower() not in ("normal", "nomal")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

modelo = AnomalyLSTM(n_categorias=len(categorias)).to(device)
modelo.load_state_dict(checkpoint["model_state"])
modelo.eval()

yolo = YOLO("yolov8n-pose.pt")

print(f"Categorias cargadas: {categorias}")
print(f"Categorias anomalas: {CATEGORIAS_ANOMALAS}")
print("Sistema listo.\n")


 

# Diccionario para guardar los movimientos de cada persona por su ID
person_buffers = {}

estado = "Analizando..."
color = (255, 255, 0)
confianza_actual = 0.0
categoria_actual = ""

sct = mss.mss()

# Definís la región del navegador manualmente
# Ajusta estos valores según tu pantalla
monitor = {
    "top": 100,
    "left": 0,
    "width": 1280,
    "height": 720
}

window_name = "Sistema de Deteccion de Anomalias"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 900, 520)
cv2.moveWindow(window_name, monitor["left"] + monitor["width"] + 20, monitor["top"])

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 1. YOLO detecta y SIGUE (track) a las personas
    results = yolo.track(frame, persist=True, verbose=False)

    alerta_detectada = False

    if results[0].keypoints is not None and results[0].boxes is not None and results[0].boxes.id is not None:
        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for i, person_id in enumerate(ids):
            if person_id not in person_buffers:
                person_buffers[person_id] = deque(maxlen=N_FRAMES)

            kp = keypoints_list[i].flatten()

            if kp.shape[0] < 34:
                temp = np.zeros(34)
                temp[:kp.shape[0]] = kp
                kp = temp
            elif kp.shape[0] > 34:
                kp = kp[:34]

            person_buffers[person_id].append(kp)

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

                    if temp_categoria in CATEGORIAS_ANOMALAS and temp_confianza >= UMBRAL_CONFIANZA:
                        alerta_detectada = True
                        categoria_actual = temp_categoria
                        confianza_actual = temp_confianza

        if alerta_detectada:
            estado = f"ALERTA: {categoria_actual}"
            color = (0, 0, 255)
        else:
            estado = "Normal"
            color = (0, 255, 0)
    else:
        estado = "Analizando..."
        color = (255, 255, 0)

    # 3. Dibujar sobre el frame
    frame_anotado = results[0].plot()

    cv2.rectangle(frame_anotado, (0, 0), (420, 90), (0, 0, 0), -1)

    cv2.putText(
        frame_anotado,
        estado,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, color, 2
    )

    cv2.putText(
        frame_anotado,
        f"Confianza: {confianza_actual*100:.1f}%  |  {categoria_actual}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, color, 1
    )

    cv2.imshow(window_name, frame_anotado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
