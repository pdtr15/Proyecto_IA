# IAPF - Sistema de Detección de Anomalías con IA

Este proyecto es una estación de control integral que utiliza Visión por Computadora y Redes Neuronales Recurrentes (LSTM) para detectar comportamientos anómalos (robos, caídas, etc.) en tiempo real, ya sea desde una cámara física o capturando la pantalla de la computadora.

## 🚀 Características Principales

- **Interfaz Dual**: Panel unificado con pestañas para Monitoreo y Entrenamiento.
- **Detección Multi-Persona**: Utiliza YOLOv8-Pose con tracking para seguir a varios individuos simultáneamente.
- **IA de Clasificación**: Modelo LSTM que analiza secuencias de movimientos (30 frames) para identificar patrones.
- **Captura Híbrida**: Cambia instantáneamente entre cámaras web y grabación de pantalla.
- **Laboratorio de IA**: Gráficas de entrenamiento en tiempo real y estadísticas del dataset.

---

## 🛠️ Requisitos Técnicos

Asegúrate de tener instalado Python 3.9 o superior y las siguientes librerías:

```bash
pip install torch torchvision ultralytics opencv-python numpy scikit-learn mss customtkinter pillow tqdm matplotlib
```

---

## 💻 Instrucciones de Ejecución

### 1. Iniciar la Aplicación Principal
Para abrir la interfaz gráfica unificada, ejecuta:
```bash
python main_gui.py
```

### 2. Cómo Monitorear
1. En la pestaña **🔍 Monitoreo**, selecciona la fuente de video (Cámara o Pantalla).
2. Ajusta el **Umbral de Confianza** (recomendado: 0.75).
3. Haz clic en **Iniciar Monitoreo**.
4. El sistema mostrará un estado **Verde (Normal)** o **Rojo (Alerta)** según lo detectado.

### 3. Cómo Entrenar el Modelo
Si tienes nuevos datos en la carpeta `keypoints/`:
1. Ve a la pestaña **🧠 Entrenamiento**.
2. Verifica que el dataset se haya detectado correctamente en el resumen estadístico.
3. Haz clic en **Iniciar Entrenamiento**.
4. Observa la gráfica de pérdida (Loss); si la línea baja, el modelo está aprendiendo correctamente.
5. Al finalizar, el nuevo modelo se cargará automáticamente para su uso inmediato.

---

## 📂 Estructura del Proyecto

- `main_gui.py`: Aplicación principal (Interfaz Gráfica).
- `entrenador.py`: Lógica de entrenamiento optimizada (Early Stopping, Scheduler).
- `yolov8n-pose.pt`: Modelo de visión de base para detectar puntos clave del cuerpo.
- `anomaly_model.pt`: Tu modelo entrenado para detectar anomalías específicas.
- `keypoints/`: Carpeta que contiene los datos de entrenamiento (.npy).

---

## 🧠 ¿Cómo funciona la IA?

1. **Visión (YOLOv8)**: Detecta 17 puntos clave del cuerpo humano (nariz, ojos, hombros, etc.) en cada frame.
2. **Memoria (LSTM)**: El sistema guarda los últimos 30 frames de movimiento. La red neuronal LSTM analiza esta secuencia completa para entender "qué está pasando" en lugar de ver solo una imagen fija.
3. **Decisión**: Si la probabilidad de una anomalía supera el umbral configurado, el sistema dispara una alerta visual.

---

## 👥 Equipo de Desarrollo
Proyecto IAPF - Inteligencia Artificial para la Seguridad Ciudadana.
