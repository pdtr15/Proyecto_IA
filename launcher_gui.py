import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "launcher_gui.log"
proceso_actual = None


def mostrar_error_si_falla(proceso, nombre):
    global proceso_actual
    if proceso.poll() is None:
        lbl_estado.configure(text=f"Ejecutando: {nombre}", fg="#86efac")
        return

    detalle = ""
    if LOG_PATH.exists():
        lineas = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        detalle = "\n".join(lineas[-12:])

    lbl_estado.configure(text=f"{nombre} se cerro. Revisa el log.", fg="#fca5a5")
    proceso_actual = None
    actualizar_botones(False)
    messagebox.showerror(
        "No se pudo iniciar",
        f"{nombre} se cerro al iniciar.\n\nDetalle:\n{detalle or 'No hubo salida de error.'}",
    )


def ejecutar_script(nombre_archivo, argumentos=None, nombre=None):
    global proceso_actual
    argumentos = argumentos or []
    nombre = nombre or nombre_archivo

    if proceso_actual is not None and proceso_actual.poll() is None:
        messagebox.showinfo("Proceso activo", "Ya hay un proceso ejecutandose. Detenlo antes de iniciar otro.")
        return

    ruta = BASE_DIR / nombre_archivo
    if not ruta.exists():
        messagebox.showerror("Archivo no encontrado", f"No existe:\n{ruta}")
        return

    try:
        if "capturapantalla" in nombre_archivo:
            ventana.iconify()

        log = open(LOG_PATH, "a", encoding="utf-8")
        log.write(f"\n\n=== Ejecutando {nombre} ===\n")
        log.flush()

        proceso = subprocess.Popen(
            [sys.executable, str(ruta), *argumentos],
            cwd=str(BASE_DIR),
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
        )
        proceso_actual = proceso
        actualizar_botones(True)
        lbl_estado.configure(text=f"Iniciando: {nombre}...", fg="#fde68a")
        ventana.after(2500, lambda: mostrar_error_si_falla(proceso, nombre))
        ventana.after(1000, revisar_proceso)
    except Exception as exc:
        messagebox.showerror("Error al ejecutar", str(exc))


def actualizar_botones(ejecutando):
    estado_inicio = "disabled" if ejecutando else "normal"
    estado_detener = "normal" if ejecutando else "disabled"
    btn_camara.configure(state=estado_inicio)
    btn_pantalla.configure(state=estado_inicio)
    btn_grabar_normal.configure(state=estado_inicio)
    btn_grabar_alerta.configure(state=estado_inicio)
    btn_entrenar.configure(state=estado_inicio)
    btn_detener.configure(state=estado_detener)


def detener_proceso():
    global proceso_actual
    if proceso_actual is not None and proceso_actual.poll() is None:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/PID", str(proceso_actual.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            proceso_actual.terminate()
        lbl_estado.configure(text="Deteniendo deteccion...", fg="#fde68a")
        ventana.after(800, finalizar_detencion)
    else:
        proceso_actual = None
        actualizar_botones(False)
        lbl_estado.configure(text="No hay deteccion activa", fg="#94a3b8")


def finalizar_detencion():
    global proceso_actual
    if proceso_actual is not None and proceso_actual.poll() is None:
        try:
            proceso_actual.kill()
        except Exception:
            pass
    proceso_actual = None
    actualizar_botones(False)
    lbl_estado.configure(text="Deteccion detenida", fg="#94a3b8")


def revisar_proceso():
    global proceso_actual
    if proceso_actual is None:
        return
    if proceso_actual.poll() is None:
        ventana.after(1000, revisar_proceso)
        return
    proceso_actual = None
    actualizar_botones(False)
    lbl_estado.configure(text="Deteccion finalizada", fg="#94a3b8")
    ventana.deiconify()


def cerrar_app():
    if proceso_actual is not None and proceso_actual.poll() is None:
        detener_proceso()
    ventana.destroy()


ventana = tk.Tk()
ventana.title("IAPF - Control Rapido")
ventana.geometry("500x470")
ventana.resizable(False, False)
ventana.configure(bg="#111827")
ventana.protocol("WM_DELETE_WINDOW", cerrar_app)

contenedor = tk.Frame(ventana, bg="#111827", padx=24, pady=22)
contenedor.pack(fill="both", expand=True)

titulo = tk.Label(
    contenedor,
    text="IAPF - Control Rapido",
    bg="#111827",
    fg="#f9fafb",
    font=("Segoe UI", 18, "bold"),
)
titulo.pack(pady=(0, 8))

subtitulo = tk.Label(
    contenedor,
    text="Deteccion, recoleccion de datos y entrenamiento",
    bg="#111827",
    fg="#cbd5e1",
    font=("Segoe UI", 10),
)
subtitulo.pack(pady=(0, 22))

btn_camara = tk.Button(
    contenedor,
    text="Iniciar deteccion con camara",
    command=lambda: ejecutar_script("produccion.py"),
    bg="#16a34a",
    fg="white",
    activebackground="#15803d",
    activeforeground="white",
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
)
btn_camara.pack(fill="x", pady=6)

btn_pantalla = tk.Button(
    contenedor,
    text="Iniciar deteccion por pantalla",
    command=lambda: ejecutar_script("produccion_capturapantalla.py"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
)
btn_pantalla.pack(fill="x", pady=6)

separador = tk.Label(
    contenedor,
    text="Datos para entrenamiento",
    bg="#111827",
    fg="#e5e7eb",
    font=("Segoe UI", 11, "bold"),
)
separador.pack(pady=(16, 6))

fila_entrenamiento = tk.Frame(contenedor, bg="#111827")
fila_entrenamiento.pack(fill="x")

btn_grabar_normal = tk.Button(
    fila_entrenamiento,
    text="Grabar Normal",
    command=lambda: ejecutar_script(
        "colector_datos.py",
        ["--categoria", "Normal"],
        "grabacion Normal",
    ),
    bg="#0d9488",
    fg="white",
    activebackground="#0f766e",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
)
btn_grabar_normal.pack(side="left", fill="x", expand=True, padx=(0, 5), pady=4)

btn_grabar_alerta = tk.Button(
    fila_entrenamiento,
    text="Grabar Alerta",
    command=lambda: ejecutar_script(
        "colector_datos.py",
        ["--categoria", "Alerta"],
        "grabacion Alerta",
    ),
    bg="#f97316",
    fg="white",
    activebackground="#ea580c",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
)
btn_grabar_alerta.pack(side="left", fill="x", expand=True, padx=(5, 0), pady=4)

btn_entrenar = tk.Button(
    contenedor,
    text="Entrenar modelo con datos guardados",
    command=lambda: ejecutar_script("entrenador.py", nombre="entrenamiento del modelo"),
    bg="#7c3aed",
    fg="white",
    activebackground="#6d28d9",
    activeforeground="white",
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    height=2,
    cursor="hand2",
)
btn_entrenar.pack(fill="x", pady=(8, 6))

btn_detener = tk.Button(
    contenedor,
    text="Detener deteccion",
    command=detener_proceso,
    bg="#dc2626",
    fg="white",
    activebackground="#b91c1c",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    height=1,
    cursor="hand2",
    state="disabled",
)
btn_detener.pack(fill="x", pady=(8, 4))

nota = tk.Label(
    contenedor,
    text="Para terminar una grabacion o deteccion, presiona Q o usa Detener.",
    bg="#111827",
    fg="#94a3b8",
    font=("Segoe UI", 9),
)
nota.pack(pady=(18, 0))

lbl_estado = tk.Label(
    contenedor,
    text="Listo para iniciar",
    bg="#111827",
    fg="#94a3b8",
    font=("Segoe UI", 9),
)
lbl_estado.pack(pady=(8, 0))

ventana.mainloop()
