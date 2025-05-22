from config import ANNOTATION_BASE, SOURCE_TYPES, SENSOR_BASE, COLS, FPS, FS
from load_annotations import load_annotation_metadata
from load_signals import load_accelerometer_signals
import os
import pickle
from load_annotations import load_annotations
from sync_and_features import construir_dataset, construir_video_segments

# --- Constants ---
ANNOTATION_FILES = load_annotation_metadata(ANNOTATION_BASE, ["With_Audio"])
# Define this with your annotation metadata

print("Loading accelrometer signals....")
# --- Load signals and annotations ---
df_dict = load_accelerometer_signals(SENSOR_BASE, COLS)

print("Finished loading accelerometer signals")

# Directorio de salida
output_dir = "../datasets_by_window"
os.makedirs(output_dir, exist_ok=True)

# Estrategias de anotación
aggregation_modes = ["majority", "union"]

# Tamaños de ventana
window_sizes = [200, 100]

for mode in aggregation_modes:
    print(f"\n🧩 Procesando anotaciones con estrategia: {mode}")

    # Cargar anotaciones y segmentos
    annotations = load_annotations(ANNOTATION_FILES, aggregation=mode, debug=True)

    print("Empezando a construit segmentos")
    video_segments = construir_video_segments(annotations, fps=FPS)

    for win_size in window_sizes:
        print(f"\n🪟 Evaluando ventana de {win_size} muestras ({mode})")

        X_total, y_total, resumen_data = construir_dataset(
            annotations,
            df_dict,
            video_segments,
            FS=FS,
            FPS=FPS,
            WINDOW_SIZE=win_size,
            STEP_SIZE=max(1, win_size // 2),
            COLS=COLS
        )

        if not X_total or not y_total or sum(y_total) == 0:
            print("⚠️ Dataset vacío o sin clases positivas. Saltando...")
            continue

        # Guardar archivo .pkl
        filename = f"dataset_{mode}_win{win_size}.pkl"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "wb") as f:
            pickle.dump((X_total, y_total, resumen_data), f)
        print(f"✅ Guardado: {output_path}")
