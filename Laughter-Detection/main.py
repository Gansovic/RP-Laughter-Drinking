from config import *
from load_annotations import load_annotation_metadata, load_annotations_majority_vote
from load_signals import load_accelerometer_signals
from sync_and_features import construir_video_segments, construir_dataset
from analysis import analizar_resumen_risa
import pickle

# 1. Cargar metadatos y anotaciones
annotation_files = load_annotation_metadata(ANNOTATION_BASE, SOURCE_TYPES)
print(f"🔍 Found {len(annotation_files)} annotation files")

annotations_by_segment = load_annotations_majority_vote(annotation_files,debug=False)
print("Passed anotaciones")

# 2. Cargar señales
df_dict = load_accelerometer_signals(SENSOR_BASE, COLS)
print(f"Passed signals: {len(df_dict)}")

# 3. Estimar segmentos
video_segments = construir_video_segments(annotations_by_segment, FPS)
print("Passed segments")

# 4. Construir dataset
X_total, y_total, resumen_data = construir_dataset(
    annotations_by_segment, df_dict, video_segments, FS, FPS, WINDOW_SIZE, STEP_SIZE, COLS
)
print("Passed dataset")

# 5. Análisis descriptivo
analizar_resumen_risa(resumen_data)

# 6. Guardar dataset procesado
with open("laughter_dataset.pkl", "wb") as f:
    pickle.dump((X_total, y_total, resumen_data), f)
print("✅ Dataset guardado como laughter_dataset.pkl")