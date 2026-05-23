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
import ctypes
import ctypes.wintypes
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

        self.title("IAPF - Estacion de Control de Inteligencia Artificial")
        self.geometry("1200x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Variables de Monitoreo
        self.monitoreo_running = False
        self.source_type = "Camara"
        self.camera_index = 0
        self.mon_window_map = {}
        self.mon_selected_window = None
        self.mon_region = {"left": 0, "top": 0, "width": 1280, "height": 720}
        self.mon_preview_on = True
        self.mon_preview_window = None
        self.mon_preview_label = None
        self.mon_preview_status_label = None
        self.mon_alert_text = "NORMAL"
        self.umbral_confianza = 0.75
        
        # Variables de Entrenamiento
        self.entrenamiento_running = False
        self.training_history = {"loss": [], "accuracy": []}

        # Variables de RecolecciÃ³n
        self.recoleccion_running = False
        self.recoleccion_count = 0
        self.recoleccion_cat = "Normal"
        self.recoleccion_tipo = "Normal"
        self.recoleccion_source_type = "Camara"
        self.recoleccion_camera_index = 0
        self.recoleccion_window_map = {}
        self.recoleccion_selected_window = None
        self.recoleccion_region = {"left": 0, "top": 0, "width": 1280, "height": 720}
        self.preview_flotante_on = True
        self.preview_window = None
        self.preview_label = None
        self.auto_train_enabled = False
        self.auto_train_threshold = 50
        self.auto_train_triggered = False
        self.balance_meta = None
        self.dataset_counts = {}

        # Layout Principal
        self.tabview = ctk.CTkTabview(self, width=1150, height=800)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_monitoreo = self.tabview.add("Monitoreo")
        self.tab_entrenamiento = self.tabview.add("Entrenamiento")
        self.tab_recoleccion = self.tabview.add("Recolector de Datos")

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
                self.log_monitoreo("Aviso: No se encontro anomaly_model.pt. Debes entrenar primero.")
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
        self.opt_source = ctk.CTkOptionMenu(
            self.sidebar_mon,
            values=["Camara 0", "Camara 1", "Captura de Pantalla", "Ventana Especifica", "Region Personalizada"],
            command=self.change_source
        )
        self.opt_source.pack(pady=10, padx=10)

        self.btn_mon_refresh_windows = ctk.CTkButton(self.sidebar_mon, text="Actualizar Ventanas", command=self.refresh_monitoreo_windows)
        self.opt_mon_window = ctk.CTkOptionMenu(self.sidebar_mon, values=["Sin ventanas"], command=self.change_monitoreo_selected_window)

        self.frame_mon_region = ctk.CTkFrame(self.sidebar_mon)
        ctk.CTkLabel(self.frame_mon_region, text="Region X Y W H").pack(pady=(6, 2), padx=8, anchor="w")
        self.ent_mon_x = ctk.CTkEntry(self.frame_mon_region, placeholder_text="X")
        self.ent_mon_y = ctk.CTkEntry(self.frame_mon_region, placeholder_text="Y")
        self.ent_mon_w = ctk.CTkEntry(self.frame_mon_region, placeholder_text="W")
        self.ent_mon_h = ctk.CTkEntry(self.frame_mon_region, placeholder_text="H")
        self.ent_mon_x.pack(pady=2, padx=8, fill="x")
        self.ent_mon_y.pack(pady=2, padx=8, fill="x")
        self.ent_mon_w.pack(pady=2, padx=8, fill="x")
        self.ent_mon_h.pack(pady=(2, 8), padx=8, fill="x")
        self.ent_mon_x.insert(0, str(self.mon_region["left"]))
        self.ent_mon_y.insert(0, str(self.mon_region["top"]))
        self.ent_mon_w.insert(0, str(self.mon_region["width"]))
        self.ent_mon_h.insert(0, str(self.mon_region["height"]))

        self.mon_preview_var = ctk.BooleanVar(value=True)
        self.chk_mon_preview = ctk.CTkCheckBox(
            self.sidebar_mon,
            text="Mini vista flotante",
            variable=self.mon_preview_var,
            command=self.toggle_mon_preview_flotante
        )
        self.chk_mon_preview.pack(pady=(8, 0), padx=10, anchor="w")

        self.lbl_conf = ctk.CTkLabel(self.sidebar_mon, text=f"Confianza: {self.umbral_confianza:.2f}")
        self.lbl_conf.pack(pady=(10,0))
        self.sld_conf = ctk.CTkSlider(self.sidebar_mon, from_=0.1, to=0.95, command=self.update_threshold)
        self.sld_conf.set(self.umbral_confianza)
        self.sld_conf.pack(pady=10, padx=10)

        self.txt_mon = ctk.CTkTextbox(self.sidebar_mon, height=200, width=180, font=("Consolas", 11))
        self.txt_mon.pack(pady=20, padx=10)

        self.lbl_alarm_compact = ctk.CTkLabel(
            self.sidebar_mon,
            text="Alarma: NORMAL",
            font=("Arial", 13, "bold"),
            text_color="green"
        )
        self.lbl_alarm_compact.pack(pady=(0, 8), padx=10)

        self.vid_container = ctk.CTkFrame(self.tab_monitoreo, bg_color="black")
        self.vid_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.vid_label = ctk.CTkLabel(self.vid_container, text="Sistema en Espera", font=("Arial", 20))
        self.vid_label.pack(expand=True, fill="both")

        self.mon_status = ctk.CTkLabel(self.tab_monitoreo, text="ESTADO: NORMAL", font=("Arial", 24, "bold"), text_color="green")
        self.mon_status.grid(row=1, column=0, columnspan=2, pady=10)
        self.update_monitoreo_source_controls()

    def log_monitoreo(self, msg):
        self.txt_mon.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.txt_mon.see("end")

    def change_source(self, val):
        if "Pantalla" in val:
            self.source_type = "Pantalla"
        elif "Ventana" in val:
            self.source_type = "Ventana"
        elif "Region" in val:
            self.source_type = "Region"
        else:
            self.source_type = "Camara"
        if self.source_type == "Camara":
            self.camera_index = int(val.split()[-1])
        self.update_monitoreo_source_controls()

    def update_monitoreo_source_controls(self):
        self.btn_mon_refresh_windows.pack_forget()
        self.opt_mon_window.pack_forget()
        self.frame_mon_region.pack_forget()
        if self.source_type == "Ventana":
            self.btn_mon_refresh_windows.pack(pady=(4, 4), padx=10, fill="x")
            self.opt_mon_window.pack(pady=(0, 8), padx=10, fill="x")
            self.refresh_monitoreo_windows()
        elif self.source_type == "Region":
            self.frame_mon_region.pack(pady=(4, 8), padx=10, fill="x")

    def refresh_monitoreo_windows(self):
        if self.source_type != "Ventana":
            return
        wins = self.get_window_list()
        if not wins:
            self.mon_window_map = {}
            self.opt_mon_window.configure(values=["Sin ventanas"])
            self.opt_mon_window.set("Sin ventanas")
            self.mon_selected_window = None
            return
        values = []
        self.mon_window_map = {}
        for hwnd, title in wins[:50]:
            label = f"{title[:60]} [{hwnd}]"
            values.append(label)
            self.mon_window_map[label] = hwnd
        self.opt_mon_window.configure(values=values)
        self.opt_mon_window.set(values[0])
        self.mon_selected_window = self.mon_window_map[values[0]]

    def change_monitoreo_selected_window(self, value):
        self.mon_selected_window = self.mon_window_map.get(value)

    def parse_monitoreo_region(self):
        try:
            left = int(self.ent_mon_x.get().strip())
            top = int(self.ent_mon_y.get().strip())
            width = int(self.ent_mon_w.get().strip())
            height = int(self.ent_mon_h.get().strip())
            if width <= 0 or height <= 0:
                return False
            self.mon_region = {"left": left, "top": top, "width": width, "height": height}
            return True
        except Exception:
            return False

    def toggle_mon_preview_flotante(self):
        self.mon_preview_on = bool(self.mon_preview_var.get())
        if not self.mon_preview_on and self.mon_preview_window is not None:
            self.mon_preview_window.destroy()
            self.mon_preview_window = None
            self.mon_preview_label = None

    def ensure_mon_preview_window(self):
        if not self.mon_preview_on:
            return
        if self.mon_preview_window is None or not self.mon_preview_window.winfo_exists():
            self.mon_preview_window = ctk.CTkToplevel(self)
            self.mon_preview_window.title("Mini Monitoreo")
            self.mon_preview_window.geometry("300x220")
            self.mon_preview_window.attributes("-topmost", True)
            self.mon_preview_label = ctk.CTkLabel(self.mon_preview_window, text="Sin senal")
            self.mon_preview_label.pack(expand=True, fill="both", padx=8, pady=(8, 4))
            self.mon_preview_status_label = ctk.CTkLabel(
                self.mon_preview_window,
                text="Estado: NORMAL",
                font=("Arial", 12, "bold"),
                text_color="green"
            )
            self.mon_preview_status_label.pack(padx=8, pady=(0, 8), fill="x")

    def update_mon_preview(self, frame_bgr):
        self.ensure_mon_preview_window()
        if self.mon_preview_label is None:
            return
        small = cv2.resize(frame_bgr, (280, 158))
        img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(280, 158))
        self.mon_preview_label.configure(image=ctk_img, text="")
        self.mon_preview_label.image = ctk_img
        if self.mon_preview_status_label is not None:
            is_alert = self.mon_alert_text != "NORMAL"
            self.mon_preview_status_label.configure(
                text=f"Estado: {self.mon_alert_text}",
                text_color="red" if is_alert else "green"
            )

    def capture_monitoreo_frame(self, cap, sct, target_monitor):
        if self.source_type == "Camara":
            if cap is None:
                cap = cv2.VideoCapture(self.camera_index)
            ret, frame = cap.read()
            if not ret:
                return None, cap
            return frame, cap

        if cap:
            cap.release()
            cap = None

        if self.source_type == "Pantalla":
            screenshot = sct.grab(target_monitor)
        elif self.source_type == "Ventana":
            if not self.mon_selected_window:
                return None, cap
            user32 = ctypes.windll.user32
            hwnd = int(self.mon_selected_window)
            if not user32.IsWindow(hwnd):
                return None, cap
            rect = ctypes.wintypes.RECT()
            ok = user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if not ok:
                return None, cap
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width <= 0 or height <= 0:
                return None, cap
            screenshot = sct.grab({"left": rect.left, "top": rect.top, "width": width, "height": height})
        else:
            if not self.parse_monitoreo_region():
                return None, cap
            screenshot = sct.grab(self.mon_region)

        frame = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, cap

    def update_threshold(self, val):
        self.umbral_confianza = float(val)
        self.lbl_conf.configure(text=f"Confianza: {self.umbral_confianza:.2f}")

    def toggle_monitoreo(self):
        if not self.monitoreo_running:
            if self.source_type == "Region" and not self.parse_monitoreo_region():
                self.lbl_alarm_compact.configure(text="Alarma: REGION INVALIDA", text_color="red")
                return
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
                frame, cap = self.capture_monitoreo_frame(cap, sct, target_monitor)
                if frame is None:
                    time.sleep(0.05)
                    continue

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
                    self.mon_status.configure(text=f"ALERTA: {cat_name} ({conf_val*100:.1f}%)", text_color="red")
                    self.lbl_alarm_compact.configure(text=f"Alarma: ALERTA ({cat_name})", text_color="red")
                    self.mon_alert_text = f"ALERTA {cat_name}"
                else:
                    self.mon_status.configure(text="ESTADO: NORMAL", text_color="green")
                    self.lbl_alarm_compact.configure(text="Alarma: NORMAL", text_color="green")
                    self.mon_alert_text = "NORMAL"

                img = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450))
                self.vid_label.configure(image=ctk_img, text="")
                self.vid_label.image = ctk_img
                self.update_mon_preview(frame)
            except Exception as e:
                self.log_monitoreo(f"Error: {e}")
                self.monitoreo_running = False
                self.btn_mon.configure(text="Iniciar Monitoreo", fg_color="green")

        if cap: cap.release()
        self.vid_label.configure(image=None, text="Monitoreo detenido")
        self.lbl_alarm_compact.configure(text="Alarma: PAUSADO", text_color="gray")
        self.mon_alert_text = "PAUSADO"

    # ---------------------------------------------------------
    # TAB ENTRENAMIENTO
    # ---------------------------------------------------------
    def setup_tab_entrenamiento(self):
        self.tab_entrenamiento.grid_columnconfigure(0, weight=1); self.tab_entrenamiento.grid_columnconfigure(1, weight=2)
        self.tab_entrenamiento.grid_rowconfigure(0, weight=1)

        self.panel_left = ctk.CTkFrame(self.tab_entrenamiento)
        self.panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.panel_left, text="Configuracion del Entrenamiento", font=("Arial", 16, "bold")).pack(pady=15)
        self.btn_train = ctk.CTkButton(self.panel_left, text="Iniciar Entrenamiento", command=self.start_training, height=40)
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
        self.dataset_counts = {}
        text = "Dataset Actual:\n\n"
        total = 0
        for c in cats:
            count = len([f for f in os.listdir(os.path.join(KEYPOINTS_DIR, c)) if f.endswith('.npy')])
            self.dataset_counts[c] = count
            text += f" - {c}: {count} secuencias\n"; total += count
        text += f"\nTotal: {total} ejemplos"
        self.lbl_stats.configure(text=text)
        self.update_balance_panel()

    def start_training(self):
        if self.entrenamiento_running: return
        self.entrenamiento_running = True; self.btn_train.configure(state="disabled", text="Entrenando...")
        self.update_operacion_status()
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
                acc = correct / len(X_val); self.log_train(f"Epoca {epoch+1}/30 - Loss: {avg_loss:.4f} - Acc: {acc*100:.1f}%")
                avg_val_loss = val_loss / len(v_loader)
                acc = correct / len(X_val); self.log_train(f"Epoca {epoch+1}/30 - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Acc: {acc*100:.1f}%")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        "model_state": model.state_dict(),
                        "categorias": cats,
                        "cat_a_numero": cat_map
                    }, "anomaly_model.pt")
            self.log_train("\nEntrenamiento completado!"); self.cargar_modelos()
        except Exception as e: self.log_train(f"Error: {e}")
        finally:
            self.entrenamiento_running = False
            self.btn_train.configure(state="normal", text="Iniciar Entrenamiento")
            self.update_operacion_status()

    # ---------------------------------------------------------
    # TAB RECOLECCIÃ“N
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

        ctk.CTkLabel(self.sidebar_rec, text="Fuente de grabacion:").pack(pady=(10, 0), padx=10)
        self.opt_rec_source = ctk.CTkOptionMenu(
            self.sidebar_rec,
            values=["Camara 0", "Camara 1", "Captura de Pantalla", "Ventana Especifica", "Region Personalizada"],
            command=self.change_recoleccion_source
        )
        self.opt_rec_source.set("Camara 0")
        self.opt_rec_source.pack(pady=10, padx=10, fill="x")

        self.btn_refresh_windows = ctk.CTkButton(self.sidebar_rec, text="Actualizar Ventanas", command=self.refresh_windows)
        self.opt_window_select = ctk.CTkOptionMenu(self.sidebar_rec, values=["Sin ventanas"], command=self.change_selected_window)

        self.frame_region = ctk.CTkFrame(self.sidebar_rec)
        ctk.CTkLabel(self.frame_region, text="Region X Y W H").pack(pady=(6, 2), padx=8, anchor="w")
        self.ent_region_x = ctk.CTkEntry(self.frame_region, placeholder_text="X")
        self.ent_region_y = ctk.CTkEntry(self.frame_region, placeholder_text="Y")
        self.ent_region_w = ctk.CTkEntry(self.frame_region, placeholder_text="W")
        self.ent_region_h = ctk.CTkEntry(self.frame_region, placeholder_text="H")
        self.ent_region_x.pack(pady=2, padx=8, fill="x")
        self.ent_region_y.pack(pady=2, padx=8, fill="x")
        self.ent_region_w.pack(pady=2, padx=8, fill="x")
        self.ent_region_h.pack(pady=(2, 8), padx=8, fill="x")
        self.ent_region_x.insert(0, str(self.recoleccion_region["left"]))
        self.ent_region_y.insert(0, str(self.recoleccion_region["top"]))
        self.ent_region_w.insert(0, str(self.recoleccion_region["width"]))
        self.ent_region_h.insert(0, str(self.recoleccion_region["height"]))

        self.btn_rec = ctk.CTkButton(self.sidebar_rec, text="Iniciar Grabacion", command=self.toggle_recoleccion, fg_color="red")
        self.btn_rec.pack(pady=20, padx=10)

        self.preview_var = ctk.BooleanVar(value=True)
        self.chk_preview = ctk.CTkCheckBox(
            self.sidebar_rec,
            text="Mini vista flotante",
            variable=self.preview_var,
            command=self.toggle_preview_flotante
        )
        self.chk_preview.pack(pady=(0, 8), padx=10, anchor="w")

        self.auto_train_var = ctk.BooleanVar(value=False)
        self.chk_auto_train = ctk.CTkCheckBox(
            self.sidebar_rec,
            text="Auto-entrenar al llegar a N",
            variable=self.auto_train_var,
            command=self.on_auto_train_toggle
        )
        self.chk_auto_train.pack(pady=(0, 6), padx=10, anchor="w")

        self.ent_auto_train_n = ctk.CTkEntry(self.sidebar_rec, placeholder_text="N secuencias (ej. 50)")
        self.ent_auto_train_n.insert(0, str(self.auto_train_threshold))
        self.ent_auto_train_n.pack(pady=(0, 10), padx=10, fill="x")

        self.lbl_rec_count = ctk.CTkLabel(self.sidebar_rec, text="Secuencias grabadas: 0", font=("Arial", 14, "bold"))
        self.lbl_rec_count.pack(pady=20, padx=10)

        ctk.CTkLabel(self.sidebar_rec, text="Balance del dataset:", font=("Arial", 12, "bold")).pack(pady=(0, 6), padx=10, anchor="w")
        self.lbl_balance_hint = ctk.CTkLabel(self.sidebar_rec, text="Calculando balance...", justify="left")
        self.lbl_balance_hint.pack(pady=(0, 8), padx=10, fill="x")
        self.btn_balance = ctk.CTkButton(self.sidebar_rec, text="Sugerir Clase Minoritaria", command=self.aplicar_balance_sugerido)
        self.btn_balance.pack(pady=(0, 10), padx=10, fill="x")

        self.lbl_operacion_estado = ctk.CTkLabel(
            self.sidebar_rec,
            text="Estado: Pausado",
            font=("Arial", 12, "bold"),
            text_color="gray"
        )
        self.lbl_operacion_estado.pack(pady=(0, 10), padx=10)

        self.rec_vid_container = ctk.CTkFrame(self.tab_recoleccion, bg_color="black")
        self.rec_vid_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.rec_vid_label = ctk.CTkLabel(self.rec_vid_container, text="Feed de Grabacion", font=("Arial", 20)); self.rec_vid_label.pack(expand=True, fill="both")
        self.update_source_controls()
        self.update_dataset_stats()
        self.update_operacion_status()

    def on_auto_train_toggle(self):
        self.auto_train_enabled = bool(self.auto_train_var.get())

    def parse_auto_train_threshold(self):
        try:
            n = int(self.ent_auto_train_n.get().strip())
            if n <= 0:
                raise ValueError()
            self.auto_train_threshold = n
        except Exception:
            self.ent_auto_train_n.delete(0, "end")
            self.ent_auto_train_n.insert(0, str(self.auto_train_threshold))

    def update_operacion_status(self):
        if not hasattr(self, "lbl_operacion_estado"):
            return
        if self.entrenamiento_running:
            self.lbl_operacion_estado.configure(text="Estado: Entrenando", text_color="orange")
        elif self.recoleccion_running:
            self.lbl_operacion_estado.configure(text="Estado: Recolectando", text_color="green")
        else:
            self.lbl_operacion_estado.configure(text="Estado: Pausado", text_color="gray")

    def get_balance_plan(self):
        if not self.dataset_counts:
            return None
        ordered = sorted(self.dataset_counts.items(), key=lambda x: x[1])
        min_cat, min_count = ordered[0]
        max_cat, max_count = ordered[-1]
        faltantes = max_count - min_count
        return {
            "min_cat": min_cat,
            "min_count": min_count,
            "max_cat": max_cat,
            "max_count": max_count,
            "faltantes": faltantes
        }

    def update_balance_panel(self):
        if not hasattr(self, "lbl_balance_hint"):
            return
        plan = self.get_balance_plan()
        if not plan:
            self.lbl_balance_hint.configure(text="Sin datos para balancear.")
            return
        if plan["faltantes"] <= 0:
            self.lbl_balance_hint.configure(text="Dataset equilibrado entre clases actuales.")
            return
        self.lbl_balance_hint.configure(
            text=(
                f"Clase con menos datos: {plan['min_cat']} ({plan['min_count']})\n"
                f"Clase mayor: {plan['max_cat']} ({plan['max_count']})\n"
                f"Faltan: {plan['faltantes']} secuencias para empatar."
            )
        )

    def aplicar_balance_sugerido(self):
        plan = self.get_balance_plan()
        if not plan:
            return
        if plan["faltantes"] <= 0:
            self.balance_meta = None
            self.lbl_balance_hint.configure(text="Dataset equilibrado entre clases actuales.")
            return
        self.balance_meta = {"categoria": plan["min_cat"], "objetivo": plan["faltantes"]}
        self.recoleccion_tipo = "Otra categoria"
        self.opt_rec_tipo.set("Otra categoria")
        self.ent_cat.configure(state="normal")
        self.ent_cat.delete(0, "end")
        self.ent_cat.insert(0, plan["min_cat"])
        self.ent_auto_train_n.delete(0, "end")
        self.ent_auto_train_n.insert(0, str(plan["faltantes"]))
        self.auto_train_var.set(True)
        self.auto_train_enabled = True
        self.lbl_balance_hint.configure(text=f"Listo: recolectar '{plan['min_cat']}'\nMeta activa: +{plan['faltantes']} secuencias.")

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
        if "Pantalla" in value:
            self.recoleccion_source_type = "Pantalla"
        elif "Ventana" in value:
            self.recoleccion_source_type = "Ventana"
        elif "Region" in value:
            self.recoleccion_source_type = "Region"
        else:
            self.recoleccion_source_type = "Camara"
        if self.recoleccion_source_type == "Camara":
            self.recoleccion_camera_index = int(value.split()[-1])
        self.update_source_controls()

    def update_source_controls(self):
        self.btn_refresh_windows.pack_forget()
        self.opt_window_select.pack_forget()
        self.frame_region.pack_forget()
        if self.recoleccion_source_type == "Ventana":
            self.btn_refresh_windows.pack(pady=(0, 6), padx=10, fill="x")
            self.opt_window_select.pack(pady=(0, 10), padx=10, fill="x")
            self.refresh_windows()
        elif self.recoleccion_source_type == "Region":
            self.frame_region.pack(pady=(0, 10), padx=10, fill="x")

    def get_window_list(self):
        user32 = ctypes.windll.user32
        windows = []

        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title:
                return True
            windows.append((int(hwnd), title))
            return True

        user32.EnumWindows(enum_proc(callback), 0)
        return windows

    def refresh_windows(self):
        if self.recoleccion_source_type != "Ventana":
            return
        try:
            wins = self.get_window_list()
            if not wins:
                self.recoleccion_window_map = {}
                self.opt_window_select.configure(values=["Sin ventanas"])
                self.opt_window_select.set("Sin ventanas")
                self.recoleccion_selected_window = None
                return
            values = []
            self.recoleccion_window_map = {}
            for hwnd, title in wins[:50]:
                label = f"{title[:60]} [{hwnd}]"
                values.append(label)
                self.recoleccion_window_map[label] = hwnd
            self.opt_window_select.configure(values=values)
            self.opt_window_select.set(values[0])
            self.recoleccion_selected_window = self.recoleccion_window_map[values[0]]
        except Exception:
            self.recoleccion_window_map = {}
            self.opt_window_select.configure(values=["Error leyendo ventanas"])
            self.opt_window_select.set("Error leyendo ventanas")
            self.recoleccion_selected_window = None

    def change_selected_window(self, value):
        self.recoleccion_selected_window = self.recoleccion_window_map.get(value)

    def parse_region(self):
        try:
            left = int(self.ent_region_x.get().strip())
            top = int(self.ent_region_y.get().strip())
            width = int(self.ent_region_w.get().strip())
            height = int(self.ent_region_h.get().strip())
            if width <= 0 or height <= 0:
                return False
            self.recoleccion_region = {"left": left, "top": top, "width": width, "height": height}
            return True
        except Exception:
            return False

    def toggle_preview_flotante(self):
        self.preview_flotante_on = bool(self.preview_var.get())
        if not self.preview_flotante_on and self.preview_window is not None:
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None

    def ensure_preview_window(self):
        if not self.preview_flotante_on:
            return
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = ctk.CTkToplevel(self)
            self.preview_window.title("Mini Vista")
            self.preview_window.geometry("300x190")
            self.preview_window.attributes("-topmost", True)
            self.preview_label = ctk.CTkLabel(self.preview_window, text="Sin senal")
            self.preview_label.pack(expand=True, fill="both", padx=8, pady=8)

    def update_preview_flotante(self, frame_bgr):
        self.ensure_preview_window()
        if self.preview_label is None or frame_bgr is None:
            return
        small = cv2.resize(frame_bgr, (280, 158))
        img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(280, 158))
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

    def capture_recoleccion_frame(self, cap, sct, target_monitor):
        if self.recoleccion_source_type == "Camara":
            if cap is None:
                cap = cv2.VideoCapture(self.recoleccion_camera_index)
            ret, frame = cap.read()
            if not ret:
                return None, cap
            return frame, cap

        if cap:
            cap.release()
            cap = None

        if self.recoleccion_source_type == "Pantalla":
            screenshot = sct.grab(target_monitor)
        elif self.recoleccion_source_type == "Ventana":
            if not self.recoleccion_selected_window:
                return None, cap
            user32 = ctypes.windll.user32
            hwnd = int(self.recoleccion_selected_window)
            if not user32.IsWindow(hwnd):
                return None, cap
            rect = ctypes.wintypes.RECT()
            ok = user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if not ok:
                return None, cap
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if width <= 0 or height <= 0:
                return None, cap
            screenshot = sct.grab({"left": rect.left, "top": rect.top, "width": width, "height": height})
        else:
            if not self.parse_region():
                return None, cap
            screenshot = sct.grab(self.recoleccion_region)

        frame = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, cap

    def toggle_recoleccion(self):
        if not self.recoleccion_running:
            if self.recoleccion_tipo == "Otra categoria":
                self.recoleccion_cat = self.ent_cat.get().strip()
            else:
                self.recoleccion_cat = self.recoleccion_tipo
            if not self.recoleccion_cat: return
            self.parse_auto_train_threshold()
            if self.recoleccion_source_type == "Region" and not self.parse_region():
                self.lbl_operacion_estado.configure(text="Estado: Region invalida", text_color="red")
                return
            self.auto_train_enabled = bool(self.auto_train_var.get())
            self.auto_train_triggered = False
            if not (self.balance_meta and self.recoleccion_cat == self.balance_meta.get("categoria")):
                self.balance_meta = None
            self.recoleccion_running = True; self.recoleccion_count = 0
            self.btn_rec.configure(text="Detener Grabacion", fg_color="gray")
            os.makedirs(os.path.join(KEYPOINTS_DIR, self.recoleccion_cat), exist_ok=True)
            self.update_operacion_status()
            threading.Thread(target=self.recoleccion_loop, daemon=True).start()
        else:
            self.recoleccion_running = False; self.btn_rec.configure(text="Iniciar Grabacion", fg_color="red")
            self.update_operacion_status()
            self.update_dataset_stats()

    def recoleccion_loop(self):
        cap = None
        sct = mss.mss()
        target_monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        buffers = {}
        target_dir = os.path.join(KEYPOINTS_DIR, self.recoleccion_cat)
        while self.recoleccion_running:
            frame, cap = self.capture_recoleccion_frame(cap, sct, target_monitor)
            if frame is None:
                time.sleep(0.05)
                continue
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
                        if self.balance_meta and self.recoleccion_cat == self.balance_meta.get("categoria"):
                            objetivo = int(self.balance_meta.get("objetivo", 0))
                            restantes = max(0, objetivo - self.recoleccion_count)
                            self.lbl_balance_hint.configure(text=f"Balanceando '{self.recoleccion_cat}'\nMeta: {objetivo} | Restantes: {restantes}")
                            if restantes <= 0:
                                self.recoleccion_running = False
                        if (
                            self.auto_train_enabled
                            and not self.auto_train_triggered
                            and self.recoleccion_count >= self.auto_train_threshold
                        ):
                            self.auto_train_triggered = True
                            if not self.entrenamiento_running:
                                self.start_training()
            img = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450))
            self.rec_vid_label.configure(image=ctk_img, text=""); self.rec_vid_label.image = ctk_img
            self.update_preview_flotante(frame)
        if cap:
            cap.release()
        if self.preview_window is not None and self.preview_window.winfo_exists() and not self.preview_flotante_on:
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None
        self.recoleccion_running = False
        self.btn_rec.configure(text="Iniciar Grabacion", fg_color="red")
        self.update_operacion_status()
        self.update_dataset_stats()
        self.rec_vid_label.configure(image=None, text="Grabacion terminada")

if __name__ == "__main__":
    app = IAPFApp()
    app.mainloop()

