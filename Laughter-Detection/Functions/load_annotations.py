# laughter_pipeline/load_annotations.py

import os
import pandas as pd
from glob import glob
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm


def load_annotation_metadata(base_path, source_folders):
    """
    Carga metadatos de anotaciones desde subcarpetas como 'No_Audio', 'With_Audio', etc.

    Args:
        base_path (str): Ruta base a las anotaciones.
        source_folders (list of str): Lista de subcarpetas a incluir, por ejemplo: ["No_Audio", "With_Audio"]

    Returns:
        list of dicts: Metadatos de archivos de anotaciones.
    """
    annotation_files = []
    for subfolder in source_folders:
        folder = os.path.join(base_path, subfolder)
        files = sorted(glob(os.path.join(folder, '*.csv')))
        for f in files:
            annotation_files.append({
                "subfolder": subfolder,  # Ej: No_Audio
                "path": f,
                "filename": os.path.basename(f),
                "video_id": os.path.basename(f).split('_')[0],
                "segment": os.path.basename(f).split('_')[1],
                "annotator": os.path.basename(f).split('_')[2].replace('.csv', '')
            })
    return annotation_files

def print_annotation_pid_stats(annotations_by_segment):
    pids_totales = Counter()
    for df in annotations_by_segment.values():
        for col in df.columns:
            pids_totales[col] += 1

    #print("\n🔍 Participantes con anotaciones (después de majority vote):")
    #for pid, count in sorted(pids_totales.items(), key=lambda x: int(x[0][1:])):
    #    print(f"{pid}: {count} segmentos")

def load_annotations(annotation_files, aggregation="majority", debug=False):
    grouped = defaultdict(list)
    for ann in annotation_files:
        key = (ann['video_id'], ann['segment'])
        grouped[key].append(ann['path'])

    annotations_by_segment = {}
    print(f"\n🔄 Aggregating annotations from {len(grouped)} segments with aggregation='{aggregation}'...\n")

    for (video_id, segment), paths in tqdm(grouped.items(), desc="Segments", unit="seg"):
        if debug:
            print(f"➡️ Processing segment: {video_id}_{segment} ({len(paths)} files)")

        dfs = []
        for path in paths:
            try:
                with open(path, 'r', encoding='latin1', errors='ignore') as f:
                    first_line = f.readline()
                    n_cols = len(first_line.strip().split(','))
                    if n_cols < 2 or n_cols > 1000:
                        print(f"⚠️ Skipping {path} due to unexpected column count: {n_cols}")
                        continue
                df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip', dtype=str, low_memory=False)
                if df.empty or df.shape[1] < 10:
                    print(f"⚠️ Archivo inválido o vacío: {path}")
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Error reading {path}: {e}")
                continue

        if len(dfs) < 1:
            continue

        # Determinar columnas comunes
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        if not common_cols:
            continue

        try:
            common_cols = sorted(common_cols, key=lambda x: int(x))
            min_len = min(len(df) for df in dfs)
            dfs = [
                df.loc[:min_len - 1, common_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                for df in dfs
            ]
            stacked = np.stack([df.values for df in dfs], axis=0)

            if aggregation == "majority":
                combined = (np.sum(stacked, axis=0) >= (len(dfs) // 2 + 1)).astype(int)
            elif aggregation == "union":
                combined = (np.sum(stacked, axis=0) >= 1).astype(int)
            else:
                raise ValueError("Invalid aggregation method. Use 'majority' or 'union'.")

            annotations_by_segment[(video_id, segment)] = pd.DataFrame(
                combined, columns=[f'p{col}' for col in common_cols]
            )
        except Exception as e:
            print(f"⚠️ Error processing segment {video_id}_{segment}: {e}")
            continue

    print_annotation_pid_stats(annotations_by_segment)
    return annotations_by_segment

