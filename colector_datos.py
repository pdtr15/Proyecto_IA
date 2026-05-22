import cv2
import os
import time
import torch
import numpy as np
import argparse
from ultralytics import YOLO
from collections import deque

# ---- Configuracion ----
OUTPUT_DIR = "keypoints"
N_FRAMES = 30
yolo = YOLO("yolov8n-pose.pt")

def recolectar(categoria=None):
    if categoria is None:
        categoria = input("Ingresa el nombre de la accion a grabar (ej: Normal, Alerta): ").strip()
    else:
        categoria = categoria.strip()

    if not categoria:
        print("Nombre invalido.")
        return

    target_dir = os.path.join(OUTPUT_DIR, categoria)
    os.makedirs(target_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara.")
        return

    # Diccionario para buffers por persona
    person_buffers = {}
    count = 0

    print(f"\n--- GRABANDO: {categoria} ---")
    print("Muevete frente a la camara. Presiona 'q' para dejar de grabar esta accion.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. YOLO detecta y SIGUE a las personas
        results = yolo.track(frame, persist=True, verbose=False)
        
        if results[0].keypoints is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            keypoints_list = results[0].keypoints.xy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()

            for i, person_id in enumerate(ids):
                if person_id not in person_buffers:
                    person_buffers[person_id] = deque(maxlen=N_FRAMES)
                
                kp = keypoints_list[i].flatten()
                
                # Asegurar 34 valores
                if kp.shape[0] < 34:
                    temp = np.zeros(34)
                    temp[:kp.shape[0]] = kp
                    kp = temp
                elif kp.shape[0] > 34:
                    kp = kp[:34]

                person_buffers[person_id].append(kp)

                # 2. Guardar secuencia cuando el buffer de ESTA persona este lleno
                if len(person_buffers[person_id]) == N_FRAMES:
                    file_path = os.path.join(target_dir, f"seq_{int(time.time()*1000)}_{count}.npy")
                    np.save(file_path, np.array(person_buffers[person_id]))
                    count += 1
                    # Dibujar un punto por cada persona siendo grabada
                    cv2.circle(frame, (30 + (i*20), 30), 8, (0, 0, 255), -1)

        # 3. UI Visual
        frame_anotado = results[0].plot()
        
        cv2.rectangle(frame_anotado, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame_anotado, f"Accion: {categoria} | Total Secuencias: {count}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Recolector de Datos Multi-Persona", frame_anotado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[LISTO] Se guardaron {count} secuencias en: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolector de datos para IAPF")
    parser.add_argument("--categoria", help="Categoria a grabar, por ejemplo Normal o Alerta")
    args = parser.parse_args()
    recolectar(args.categoria)
