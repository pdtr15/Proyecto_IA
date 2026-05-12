from ultralytics import YOLO
import cv2
import numpy as np
import os

# ---- Configuracion ----
DATABASE_DIR = "database"
OUTPUT_DIR = "keypoints"
N_FRAMES = 30

model = YOLO("yolov8n-pose.pt")

def procesar_video(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None

    # Samplear N frames distribuidos en todo el video
    indices = np.linspace(0, total_frames - 1, N_FRAMES, dtype=int)
    
    keypoints = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            keypoints.append(np.zeros(34))
            continue
        
        results = model(frame, verbose=False)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp = results[0].keypoints.xy[0].cpu().numpy().flatten()
        else:
            kp = np.zeros(34)
        
        keypoints.append(kp)
    
    cap.release()
    
    # Padding si faltan frames
    while len(keypoints) < N_FRAMES:
        keypoints.append(np.zeros(34))
    
    return np.array(keypoints)  # (30, 34)


# ---- Procesar toda la Database ----
categorias = os.listdir(DATABASE_DIR)
print(f"Categorias encontradas: {categorias}")

for categoria in categorias:
    input_cat = os.path.join(DATABASE_DIR, categoria)
    output_cat = os.path.join(OUTPUT_DIR, categoria)
    os.makedirs(output_cat, exist_ok=True)

    videos = os.listdir(input_cat)
    total = len(videos)
    print(f"\nProcesando {categoria} ({total} videos)...")

    for i, video in enumerate(videos):
        video_path = os.path.join(input_cat, video)
        output_path = os.path.join(output_cat, video.rsplit(".", 1)[0] + ".npy")

        # Saltar si ya fue procesado
        if os.path.exists(output_path):
            print(f"  [{i+1}/{total}] Ya existe, saltando: {video}")
            continue

        try:
            kp = procesar_video(video_path)
            if kp is not None:
                np.save(output_path, kp)
                print(f"  [{i+1}/{total}] OK: {video}")
            else:
                print(f"  [{i+1}/{total}] Video vacio, saltando: {video}")
        except Exception as e:
            print(f"  [{i+1}/{total}] Error en {video}: {e}")

print("\nExtraccion completada.")
