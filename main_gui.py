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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- Configuracion Global ----
N_FRAMES = 30
KEYPOINTS_DIR = "keypoints"

# ---- Modelo LSTM ----
class AnomalyLSTM(nn.Module):
    def __init__(self, n_categorias):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=34,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4
        )
        self.clasificador = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_categorias)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.clasificador(out[:, -1, :])

class IAPFApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("IAPF - Estación de Control de Inteligencia Artificial")
        self.geometry("1200x800")
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

        # Layout Principal
        self.tabview = ctk.CTkTabview(self, width=1150, height=750)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_monitoreo = self.tabview.add("🔍 Monitoreo en Tiempo Real")
        self.tab_entrenamiento = self.tabview.add("🧠 Entrenamiento (Laboratorio)")

        self.setup_tab_monitoreo()
        self.setup_tab_entrenamiento()
        
        # Cargar Modelos al final para que los logs ya existan
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

        # Sidebar Monitoreo
        self.sidebar_mon = ctk.CTkFrame(self.tab_monitoreo, width=200)
        self.sidebar_mon.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.btn_mon = ctk.CTkButton(self.sidebar_mon, text="Iniciar Monitoreo", command=self.toggle_monitoreo, fg_color="green")
        self.btn_mon.pack(pady=20, padx=10)

        ctk.CTkLabel(self.sidebar_mon, text="Fuente de Video:").pack(pady=(10,0))
        self.opt_source = ctk.CTkOptionMenu(self.sidebar_mon, values=["Cámara 0", "Cámara 1", "Captura de Pantalla"], command=self.change_source)
        self.opt_source.pack(pady=10, padx=10)

        self.lbl_conf = ctk.CTkLabel(self.sidebar_mon, text=f"Confianza: {self.umbral_confianza:.2f}")
        self.lbl_conf.pack(pady=(10,0))
        self.sld_conf = ctk.CTkSlider(self.sidebar_mon, from_=0.1, to=0.95, command=self.update_threshold)
        self.sld_conf.set(self.umbral_confianza)
        self.sld_conf.pack(pady=10, padx=10)

        self.txt_mon = ctk.CTkTextbox(self.sidebar_mon, height=200, width=180, font=("Consolas", 11))
        self.txt_mon.pack(pady=20, padx=10)

        # Display Video
        self.vid_container = ctk.CTkFrame(self.tab_monitoreo, bg_color="black")
        self.vid_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.vid_label = ctk.CTkLabel(self.vid_container, text="Sistema en Espera", font=("Arial", 20))
        self.vid_label.pack(expand=True, fill="both")

        self.mon_status = ctk.CTkLabel(self.tab_monitoreo, text="ESTADO: NORMAL", font=("Arial", 24, "bold"), text_color="green")
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
        monitors = sct.monitors
        # Elegir el monitor 1 si existe, si no el 0
        target_monitor = monitors[1] if len(monitors) > 1 else monitors[0]
        
        buffers = {}

        while self.monitoreo_running:
            if self.source_type == "Cámara":
                if cap is None: cap = cv2.VideoCapture(self.camera_index)
                ret, frame = cap.read()
                if not ret: 
                    time.sleep(0.1)
                    continue
            else:
                if cap: 
                    cap.release()
                    cap = None
                # Método optimizado de captura
                screenshot = sct.grab(target_monitor)
                frame = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape((screenshot.height, screenshot.width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Redimensionar inmediatamente para aliviar carga
            frame = cv2.resize(frame, (800, 450))
            
            # Pequeño sleep para no saturar el CPU
            time.sleep(0.01)

            results = self.yolo.track(frame, persist=True, verbose=False)
            
            alerta = False
            cat_name = ""
            conf_val = 0

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
                            if cat.lower() not in ["normal", "nomal"] and conf >= self.umbral_confianza:
                                alerta, cat_name, conf_val = True, cat, conf.item()

            if alerta:
                self.mon_status.configure(text=f"⚠️ ALERTA: {cat_name} ({conf_val*100:.1f}%)", text_color="red")
            else:
                self.mon_status.configure(text="✅ ESTADO: NORMAL", text_color="green")

            # Dibujar y convertir para CTk usando CTkImage para HighDPI
            img = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 450))
            
            self.vid_label.configure(image=ctk_img, text="")
            self.vid_label.image = ctk_img

        if cap: cap.release()
        self.vid_label.configure(image=None, text="Monitoreo detenido")

    # ---------------------------------------------------------
    # TAB ENTRENAMIENTO
    # ---------------------------------------------------------
    def setup_tab_entrenamiento(self):
        self.tab_entrenamiento.grid_columnconfigure(0, weight=1)
        self.tab_entrenamiento.grid_columnconfigure(1, weight=2)
        self.tab_entrenamiento.grid_rowconfigure(0, weight=1)

        # Panel Izquierdo: Controles y Datos
        self.panel_left = ctk.CTkFrame(self.tab_entrenamiento)
        self.panel_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(self.panel_left, text="Configuración del Entrenamiento", font=("Arial", 16, "bold")).pack(pady=15)
        
        self.btn_train = ctk.CTkButton(self.panel_left, text="🚀 Iniciar Entrenamiento", command=self.start_training, height=40)
        self.btn_train.pack(pady=10, padx=20, fill="x")

        # Config Stats
        self.lbl_stats = ctk.CTkLabel(self.panel_left, text="Analizando dataset...", justify="left")
        self.lbl_stats.pack(pady=20, padx=20, fill="x")
        self.update_dataset_stats()

        self.train_log = ctk.CTkTextbox(self.panel_left, height=250, font=("Consolas", 10))
        self.train_log.pack(pady=10, padx=20, fill="both", expand=True)

        # Panel Derecho: Gráfica
        self.panel_right = ctk.CTkFrame(self.tab_entrenamiento)
        self.panel_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.set_title("Progreso de Entrenamiento (Loss)", color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.panel_right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def log_train(self, msg):
        self.train_log.insert("end", f"{msg}\n")
        self.train_log.see("end")

    def update_dataset_stats(self):
        if not os.path.exists(KEYPOINTS_DIR):
            self.lbl_stats.configure(text="Error: No existe carpeta 'keypoints'")
            return
        
        cats = sorted([d for d in os.listdir(KEYPOINTS_DIR) if os.path.isdir(os.path.join(KEYPOINTS_DIR, d))])
        text = "📊 Dataset Actual:\n\n"
        total = 0
        for c in cats:
            count = len([f for f in os.listdir(os.path.join(KEYPOINTS_DIR, c)) if f.endswith('.npy')])
            text += f" • {c}: {count} secuencias\n"
            total += count
        text += f"\nTotal: {total} ejemplos"
        self.lbl_stats.configure(text=text)

    def start_training(self):
        if self.entrenamiento_running: return
        self.entrenamiento_running = True
        self.btn_train.configure(state="disabled", text="Entrenando...")
        self.training_history = {"loss": []}
        threading.Thread(target=self.train_process, daemon=True).start()

    def train_process(self):
        try:
            self.log_train("Cargando datos...")
            X, y = [], []
            cats = sorted([d for d in os.listdir(KEYPOINTS_DIR) if os.path.isdir(os.path.join(KEYPOINTS_DIR, d))])
            cat_map = {c: i for i, c in enumerate(cats)}

            for c in cats:
                path = os.path.join(KEYPOINTS_DIR, c)
                for f in os.listdir(path):
                    if f.endswith('.npy'):
                        data = np.load(os.path.join(path, f))
                        if data.shape == (30, 34):
                            X.append(data); y.append(cat_map[c])
            
            X, y = np.array(X), np.array(y)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
            
            loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)
            v_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=32)

            model = AnomalyLSTM(len(cats)).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            
            self.log_train(f"Iniciando entrenamiento ({len(X)} secuencias)...")
            
            best_acc = 0
            for epoch in range(30):
                model.train()
                t_loss = 0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad(); pred = model(xb); loss = crit(pred, yb)
                    loss.backward(); opt.step(); t_loss += loss.item()
                
                avg_loss = t_loss/len(loader)
                self.training_history["loss"].append(avg_loss)
                
                # Update Graph
                self.ax.clear()
                self.ax.plot(self.training_history["loss"], color='#3498db', linewidth=2)
                self.ax.set_title("Pérdida (Loss) por Época", color='white')
                self.canvas.draw()

                model.eval()
                correct = 0
                with torch.no_grad():
                    for xb, yb in v_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                
                acc = correct / len(X_val)
                self.log_train(f"Época {epoch+1}/30 - Loss: {avg_loss:.4f} - Acc: {acc*100:.1f}%")

                if acc > best_acc:
                    best_acc = acc
                    torch.save({"model_state": model.state_dict(), "categorias": cats}, "anomaly_model.pt")

            self.log_train("\n✅ Entrenamiento completado!")
            self.cargar_modelos() # Recargar en produccion
            
        except Exception as e:
            self.log_train(f"Error: {e}")
        finally:
            self.entrenamiento_running = False
            self.btn_train.configure(state="normal", text="🚀 Iniciar Entrenamiento")

if __name__ == "__main__":
    app = IAPFApp()
    app.mainloop()
