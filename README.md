# IAPF - Deteccion de Anomalias con IA

IAPF es una estacion de control con Vision por Computadora + LSTM para detectar comportamientos anormales en tiempo real.

Soporta:
- Monitoreo en vivo por camara.
- Monitoreo por captura de pantalla.
- Recoleccion de datos etiquetados (`Normal`, `Alerta` u otras categorias).
- Entrenamiento del modelo desde GUI o por script.

## Requisitos

- Python 3.9+
- Dependencias:

```bash
pip install -r requirements.txt
```

Si no usas `requirements.txt`:

```bash
pip install torch torchvision ultralytics opencv-python numpy scikit-learn mss customtkinter pillow tqdm matplotlib
```

## Archivos principales

- `main_gui.py`: interfaz completa (monitoreo, entrenamiento, recolector).
- `launcher_gui.py`: interfaz rapida con botones para produccion y entrenamiento.
- `produccion.py`: deteccion en vivo desde camara.
- `produccion_capturapantalla.py`: deteccion en vivo desde pantalla.
- `colector_datos.py`: recoleccion de secuencias en `keypoints/`.
- `entrenador.py`: entrenamiento por script con balanceo de clases.
- `modelo.py`: red LSTM para clasificacion temporal.

## Uso rapido

### Opcion 1: Interfaz completa

```bash
python main_gui.py
```

Pestanas:
- `Monitoreo`: elige fuente (camara o pantalla) y umbral.
- `Entrenamiento`: entrena el modelo con el dataset actual.
- `Recolector de Datos`:
  - `Tipo de dato`: `Normal`, `Alerta` u `Otra categoria`.
  - `Fuente de grabacion`: `Camara 0`, `Camara 1` o `Captura de Pantalla`.
  - Boton `Iniciar Grabacion` para guardar secuencias.

### Opcion 2: Lanzador rapido

```bash
python launcher_gui.py
```

Incluye botones para:
- Deteccion con camara.
- Deteccion por pantalla.
- Grabar `Normal`.
- Grabar `Alerta`.
- Entrenar modelo.

## Flujo recomendado para entrenar bien

1. Graba datos `Normal` y `Alerta` desde `Recolector de Datos`.
2. Revisa que existan archivos `.npy` en:
   - `keypoints/Normal`
   - `keypoints/Alerta`
3. Ejecuta entrenamiento desde:
   - Pestana `Entrenamiento` en `main_gui.py`, o
   - `python entrenador.py`
4. Verifica que se actualice `anomaly_model.pt`.
5. Prueba deteccion en `Monitoreo` o scripts de produccion.

## Mejora aplicada: balanceo de clases

Para reducir sesgo cuando hay mas `Normal` que `Alerta`, el entrenamiento ahora usa:
- `WeightedRandomSampler` (muestreo balanceado por clase).
- `CrossEntropyLoss(weight=...)` (penaliza mas la clase minoritaria).

Esto esta aplicado en:
- `entrenador.py`
- entrenamiento interno de `main_gui.py`

## Notas de uso

- Para cerrar una ventana OpenCV, presiona `Q`.
- En modo captura de pantalla, evita capturar la misma ventana de salida para no generar efecto espejo.
- Si no existe `anomaly_model.pt`, entrena primero.
