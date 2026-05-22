import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import mss
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modelo import AnomalyLSTM

# ---- Configuracion Global ----
N_FRAMES = 30
KEYPOINTS_DIR = "keypoints"

class IAPFApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("IAPF - Estación de Control de Inteligencia Artificial")
        self.geometry("1200x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables de Monitoreo
        self.monitoreo_running = False
        self.source_type = "Cámara"
        self.camera_index = 0
        self.umbral_confianza = 0.75
        
        # Variables de Entrenamiento
        self.entrenamiento_running = False
        self.training_history = {"loss": [], "accuracy": []}

        # Variables de Recolección
        self.recoleccion_running = False
        self.recoleccion_count = 0
        self.recoleccion_cat = "Normal"
        self.recoleccion_tipo = "Normal"
        self.recoleccion_source_type = "Cámara"
        self.recoleccion_camera_index = 0

        # Layout Principal
        self.tabview = ctk.CTkTabview(self, width=1150, height=800)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_monitoreo = self.tabview.add("🔍 Monitoreo")
        self.tab_entrenamiento = self.tabview.add("🧠 Entrenamiento")
        self.tab_recoleccion = self.tabview.add("🎥 Recolector de Datos")

        self.setup_tab_monitoreo()
        self.setup_tab_entrenamiento()
        self.setup_tab_recoleccion()
        
        # Cargar Modelos al final
        self.cargar_modelos()

    def cargar_modelos(self):
        try:
            self.yolo = YOLO("yolov8n-pose.pt")
            if os.path.exists("anomaly_model.pt"):
                checkpoint = torch.load("anomaly_model.pt", map_location=torch.device('cpu'))
                self.categorias = checkpoint["categorias"]
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.modelo_anomalia = AnomalyLSTM(n_categorias=len(self.categorias)).to(self.device)
                self.modelo_anomalia.load_state_dict(checkpoint["model_state"])
                self.modelo_anomalia.eval()
                self.log_monitoreo("Modelos cargados correctamente.")
            else:
                self.categorias = []
                self.log_monitoreo("Aviso: No se encontró anomaly_model.pt. Debes entrenar primero.")
        except Exception as e:
            print(f"Error carga: {e}")

    # ---------------------------------------------------------
    # TAB MONITOREO
    # ---------------------------------------------------------
    def setup_tab_monitoreo(self):
        self.tab_monitoreo.grid_columnconfigure(1, weight=1)
        self.tab_monitoreo.grid_rowconfigure(0, weight=1)

        self.sidebar_mon = ctk.CTkFrame(self.tab_monitoreo, width=200)
        self.sidebar_mon.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.btn_mon = ctk.CTkButton(self.sidebar_mon, text="Iniciar Monitoreo", command=self.toggle_monitoreo, fg_color="green")
        self.btn_mon.pack(pady=20, padx=10)

        ctk.CTkLabel(self.sidebar_mon, text="Fuente:").pack(pady=(10,0))
        self.opt_source = ctk.CTkOptionMenu(self.sidebar_mon, values=["Cámara 0", "Cámara 1", "Captura de Pantalla"], command=self.change_source)
        self.opt_source.pack(pady=10, padx=10)

        self.lbl_conf = ctk.CTkLabel(self.sidebar_mon, text=f"Confianza: {self.umbral_confianza:.2f}")
        self.lbl_conf.pack(pady=(10,0))
        self.sld_conf = ctk.CTkSlider(self.sidebar_mon, from_=0.1, to=0.95, command=self.update_threshold)
        self.sld_conf.set(self.umbral_confianza)
        self.sld_conf.pack(pady=10, padx=10)

        self.txt_mon = ctk.CTkTextbox(self.sidebar_mon, height=200, width=180, font=("Consolas", 11))
        self.txt_mon.pack(pady=20, padx=10)

        self.vid_container = ctk.CTkFrame(self.tab_monitoreo, bg_color="black")
        self.vid_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.vid_label = ctk.CTkLabel(self.vid_container, text="Sistema en Espera", font=("Arial", 20))
        self.vid_label.pack(expand=True, fill="both")

        self.mon_status = ctk.CTkLabel(self.tab_monitoreo, text="✅ ESTADO: NORMAL", font=("Arial", 24, "bold"), text_color="green")
        self.mon_status.grid(row=1, column=0, columnspan=2, pady=10)

    def log_monitoreo(self, msg):
        self.txt_mon.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.txt_mon.see("end")

    def change_source(self, val):
        self.source_type = "Pantalla" if "Pantalla" in val else "Cámara"
        if self.source_type == "Cámara": self.camera_index = int(val.split()[-1])

    def update_threshold(self, val):
        self.umbral_confianza = float(val)
        self.lbl_conf.configure(text=f"Confianza: {self.umbral_confianza:.2f}")

    def toggle_monitoreo(self):
        if not self.monitoreo_running:
            self.monitoreo_running = True
            self.btn_mon.configure(text="Detener", fg_color="red")
            threading.Thread(target=self.monitoreo_loop, daemon=True).start()
        else:
            self.monitoreo_running = False
            self.btn_mon.configure(text="Iniciar Monitoreo", fg_color="green")

    def monitoreo_loop(self):
        cap = None
        sct = mss.mss()
        target_monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        buffers = {}

        while self.monitoreo_running:
            try:
                if self.source_type == "Cámara":
                    if cap is None: cap = cv2.VideoCapture(self.camera_index)
                    ret, frame = cap.read()
                    if not ret: time.sleep(0.1); continue
                else:
                    if cap: cap.release(); cap = None
                    screenshot = sct.grab(target_monitor)
                    frame = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = cv2.resize(frame, (800, 450))
                time.sleep(0.01)

                results = self.yolo.track(frame, persist=True, verbose=False)
                alerta, cat_name, conf_val = False, "", 0

                if results[0].keypoints is not None and results[0].boxes.id is not None:
                    kps = results[0].keypoints.xy.cpu().numpy()
                    ids = results[0].boxes.id.int().cpu().tolist()
                    for i, pid in enumerate(ids):
                        if pid not in buffers: buffers[pid] = deque(maxlen=N_FRAMES)
                        kp = kps[i].flatten()
                        if kp.shape[0] < 34: 
                            tmp = np.zeros(34); tmp[:kp.shape[0]] = kp; kp = tmp
                        else: kp = kp[:34]
                        buffers[pid].append(kp)

                        if len(buffers[pid]) == N_FRAMES:
                            seq = torch.tensor(np.array(buffers[pid]), dtype=torch.float32).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                output = self.modelo_anomalia(seq)
                                probs = torch.softmax(output, dim=1)[0]
                                conf, idx = probs.max(dim=0)
                                cat = self.categorias[idx.item()]
                                if cat.lower() not in ("normal", "nomal") and conf >= self.umbral_confianza:
                                    alerta, cat_name, conf_val = True, cat, conf.item()

                if alerta:
                    self.mon_status.configure(text=f"⚠️ ALERTA: {cat_name} ({conf_val*100:.1f}%)", text_color="red")
                else:
                    self.mon_status.configure(text="✅ ESTADO: NORMAL", text_color="green")

                img = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450))
                self.vid_label.configure(image=ctk_img, text="")
                self.vid_label.image = ctk_img
            except Exception as e:
                self.log_monitoreo(f"Error: {e}")
                self.monitoreo_running = False
                self.btn_mon.configure(text="Iniciar Monitoreo", fg_color="green")

        if cap: cap.release()
        self.vid_label.configure(image=None, text="Monitoreo detenido")

    # ---------------------------------------------------------
    # TAB ENTRENAMIENTO
    # ---------------------------------------------------------
    def setup_tab_entrenamiento(self):
        self.tab_entrenamiento.grid_columnconfigure(0, weight=1); self.tab_entrenamiento.grid_columnconfigure(1, weight=2)
        self.tab_entrenamiento.grid_rowconfigure(0, weight=1)

        self.panel_left = ctk.CTkFrame(self.tab_entrenamiento)
        self.panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.panel_left, text="Configuración del Entrenamiento", font=("Arial", 16, "bold")).pack(pady=15)
        self.btn_train = ctk.CTkButton(self.panel_left, text="🚀 Iniciar Entrenamiento", command=self.start_training, height=40)
        self.btn_train.pack(pady=10, padx=20, fill="x")

        self.lbl_stats = ctk.CTkLabel(self.panel_left, text="Analizando dataset...", justify="left")
        self.lbl_stats.pack(pady=20, padx=20, fill="x")
        self.update_dataset_stats()

        self.train_log = ctk.CTkTextbox(self.panel_left, height=250, font=("Consolas", 10))
        self.train_log.pack(pady=10, padx=20, fill="both", expand=True)

        self.panel_right = ctk.CTkFrame(self.tab_entrenamiento)
        self.panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111); self.ax.set_facecolor('#1e1e1e'); self.ax.tick_params(colors='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.panel_right); self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def log_train(self, msg):
        self.train_log.insert("end", f"{msg}\n"); self.train_log.see("end")

    def update_dataset_stats(self):
        if not os.path.exists(KEYPOINTS_DIR): return
        cats = sorted([d for d in os.listdir(KEYPOINTS_DIR) if os.path.isdir(os.path.join(KEYPOINTS_DIR, d))])
        text = "📊 Dataset Actual:\n\n"
        total = 0
        for c in cats:
            count = len([f for f in os.listdir(os.path.join(KEYPOINTS_DIR, c)) if f.endswith('.npy')])
            text += f" • {c}: {count} secuencias\n"; total += count
        text += f"\nTotal: {total} ejemplos"
        self.lbl_stats.configure(text=text)

    def start_training(self):
        if self.entrenamiento_running: return
        self.entrenamiento_running = True; self.btn_train.configure(state="disabled", text="Entrenando...")
        self.training_history = {"loss": []}; threading.Thread(target=self.train_process, daemon=True).start()

    def train_process(self):
        try:
            self.log_train("Cargando datos..."); X, y = [], []
            cats = sorted([d for d in os.listdir(KEYPOINTS_DIR) if os.path.isdir(os.path.join(KEYPOINTS_DIR, d))])
            cat_map = {c: i for i, c in enumerate(cats)}
            for c in cats:
                path = os.path.join(KEYPOINTS_DIR, c)
                for f in os.listdir(path):
                    if f.endswith('.npy'):
                        data = np.load(os.path.join(path, f))
                        if data.shape == (30, 34): X.append(data); y.append(cat_map[c])
            X, y = np.array(X), np.array(y); X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
            class_counts = np.bincount(y_train, minlength=len(cats))
            class_weights = len(y_train) / (len(cats) * np.maximum(class_counts, 1))
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
            self.log_train("Balanceo de clases aplicado:")
            for cat_name, count, weight in zip(cats, class_counts, class_weights):
                self.log_train(f" - {cat_name}: {count} muestras train | peso {weight:.3f}")
            loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=32, sampler=sampler)
            v_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=32)
            class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            model = AnomalyLSTM(len(cats)).to(self.device); opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss(weight=class_weights_t)
            best_val_loss = float("inf")
            for epoch in range(30):
                model.train(); t_loss = 0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad(); pred = model(xb); loss = crit(pred, yb); loss.backward(); opt.step(); t_loss += loss.item()
                avg_loss = t_loss/len(loader); self.training_history["loss"].append(avg_loss)
                self.ax.clear(); self.ax.plot(self.training_history["loss"], color='#3498db'); self.canvas.draw()
                model.eval(); correct = 0; val_loss = 0
                with torch.no_grad():
                    for xb, yb in v_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        pred = model(xb)
                        val_loss += crit(pred, yb).item()
                        correct += (pred.argmax(1) == yb).sum().item()
                acc = correct / len(X_val); self.log_train(f"Época {epoch+1}/30 - Loss: {avg_loss:.4f} - Acc: {acc*100:.1f}%")
                avg_val_loss = val_loss / len(v_loader)
                acc = correct / len(X_val); self.log_train(f"Epoca {epoch+1}/30 - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Acc: {acc*100:.1f}%")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        "model_state": model.state_dict(),
                        "categorias": cats,
                        "cat_a_numero": cat_map
                    }, "anomaly_model.pt")
            self.log_train("\n✅ Entrenamiento completado!"); self.cargar_modelos()
        except Exception as e: self.log_train(f"Error: {e}")
        finally: self.entrenamiento_running = False; self.btn_train.configure(state="normal", text="🚀 Iniciar Entrenamiento")

    # ---------------------------------------------------------
    # TAB RECOLECCIÓN
    # ---------------------------------------------------------
    def setup_tab_recoleccion(self):
        self.tab_recoleccion.grid_columnconfigure(1, weight=1); self.tab_recoleccion.grid_rowconfigure(0, weight=1)
        self.sidebar_rec = ctk.CTkFrame(self.tab_recoleccion, width=200); self.sidebar_rec.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.sidebar_rec, text="Tipo de dato:").pack(pady=(20, 0), padx=10)
        self.opt_rec_tipo = ctk.CTkOptionMenu(
            self.sidebar_rec,
            values=["Normal", "Alerta", "Otra categoria"],
            command=self.change_recoleccion_tipo
        )
        self.opt_rec_tipo.set("Normal")
        self.opt_rec_tipo.pack(pady=10, padx=10, fill="x")

        ctk.CTkLabel(self.sidebar_rec, text="Categoria personalizada:").pack(pady=(10, 0), padx=10)
        self.ent_cat = ctk.CTkEntry(self.sidebar_rec, placeholder_text="Ej: Robo, Caida...")
        self.ent_cat.pack(pady=10, padx=10, fill="x")
        self.ent_cat.insert(0, "Normal")
        self.ent_cat.configure(state="disabled")

        ctk.CTkLabel(self.sidebar_rec, text="Fuente de grabación:").pack(pady=(10, 0), padx=10)
        self.opt_rec_source = ctk.CTkOptionMenu(
            self.sidebar_rec,
            values=["Cámara 0", "Cámara 1", "Captura de Pantalla"],
            command=self.change_recoleccion_source
        )
        self.opt_rec_source.set("Cámara 0")
        self.opt_rec_source.pack(pady=10, padx=10, fill="x")

        self.btn_rec = ctk.CTkButton(self.sidebar_rec, text="🔴 Iniciar Grabación", command=self.toggle_recoleccion, fg_color="red")
        self.btn_rec.pack(pady=20, padx=10)

        self.lbl_rec_count = ctk.CTkLabel(self.sidebar_rec, text="Secuencias grabadas: 0", font=("Arial", 14, "bold"))
        self.lbl_rec_count.pack(pady=20, padx=10)

        self.rec_vid_container = ctk.CTkFrame(self.tab_recoleccion, bg_color="black")
        self.rec_vid_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.rec_vid_label = ctk.CTkLabel(self.rec_vid_container, text="Feed de Grabación", font=("Arial", 20)); self.rec_vid_label.pack(expand=True, fill="both")

    def change_recoleccion_tipo(self, value):
        self.recoleccion_tipo = value
        self.ent_cat.configure(state="normal")
        self.ent_cat.delete(0, "end")

        if value == "Otra categoria":
            self.ent_cat.insert(0, "NuevaCategoria")
        else:
            self.ent_cat.insert(0, value)
            self.ent_cat.configure(state="disabled")

    def change_recoleccion_source(self, value):
        self.recoleccion_source_type = "Pantalla" if "Pantalla" in value else "Cámara"
        if self.recoleccion_source_type == "Cámara":
            self.recoleccion_camera_index = int(value.split()[-1])

    def toggle_recoleccion(self):
        if not self.recoleccion_running:
            if self.recoleccion_tipo == "Otra categoria":
                self.recoleccion_cat = self.ent_cat.get().strip()
            else:
                self.recoleccion_cat = self.recoleccion_tipo
            if not self.recoleccion_cat: return
            self.recoleccion_running = True; self.recoleccion_count = 0
            self.btn_rec.configure(text="⏹ Detener Grabación", fg_color="gray")
            os.makedirs(os.path.join(KEYPOINTS_DIR, self.recoleccion_cat), exist_ok=True)
            threading.Thread(target=self.recoleccion_loop, daemon=True).start()
        else:
            self.recoleccion_running = False; self.btn_rec.configure(text="🔴 Iniciar Grabación", fg_color="red")
            self.update_dataset_stats()

    def recoleccion_loop(self):
        cap = None
        sct = mss.mss()
        target_monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        buffers = {}
        target_dir = os.path.join(KEYPOINTS_DIR, self.recoleccion_cat)
        while self.recoleccion_running:
            if self.recoleccion_source_type == "Cámara":
                if cap is None:
                    cap = cv2.VideoCapture(self.recoleccion_camera_index)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if cap:
                    cap.release()
                    cap = None
                screenshot = sct.grab(target_monitor)
                frame = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (800, 450))
            results = self.yolo.track(frame, persist=True, verbose=False)
            if results[0].keypoints is not None and results[0].boxes.id is not None:
                kps = results[0].keypoints.xy.cpu().numpy(); ids = results[0].boxes.id.int().cpu().tolist()
                for i, pid in enumerate(ids):
                    if i > 0: continue
                    if pid not in buffers: buffers[pid] = deque(maxlen=N_FRAMES)
                    kp = kps[i].flatten()
                    if kp.shape[0] < 34: 
                        tmp = np.zeros(34); tmp[:kp.shape[0]] = kp; kp = tmp
                    else: kp = kp[:34]
                    buffers[pid].append(kp)
                    if len(buffers[pid]) == N_FRAMES:
                        np.save(os.path.join(target_dir, f"rec_{int(time.time()*1000)}.npy"), np.array(buffers[pid]))
                        self.recoleccion_count += 1
                        self.lbl_rec_count.configure(text=f"Secuencias grabadas: {self.recoleccion_count}")
            img = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450))
            self.rec_vid_label.configure(image=ctk_img, text=""); self.rec_vid_label.image = ctk_img
        if cap:
            cap.release()
        self.rec_vid_label.configure(image=None, text="Grabación terminada")

if __name__ == "__main__":
    app = IAPFApp()
    app.mainloop()
