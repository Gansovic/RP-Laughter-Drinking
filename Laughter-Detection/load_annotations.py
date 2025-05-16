# laughter_pipeline/load_annotations.py

import os
import pandas as pd
from glob import glob
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# Actualización de ruta en config.py (deberás reflejar esto ahí también)
ANNOTATION_BASE = "/Volumes/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/v4/release/annotations/actions/laughing"

def load_annotation_metadata(base_path, source_types):
    annotation_files = []
    for src_type in source_types:
        folder = os.path.join(base_path, src_type)
        files = sorted(glob(os.path.join(folder, '*.csv')))
        for f in files:
            annotation_files.append({
                "source": src_type,
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

def load_annotations_majority_vote(annotation_files, debug=False):
    grouped = defaultdict(list)
    for ann in annotation_files:
        key = (ann['video_id'], ann['segment'])
        grouped[key].append(ann['path'])

    annotations_by_segment = {}
    print(f"\n🔄 Aggregating annotations from {len(grouped)} segments...\n")

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
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ Error reading {path}: {e}")
                continue

        if len(dfs) < 2:
            continue

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
            majority = (np.sum(stacked, axis=0) >= (len(dfs) // 2 + 1)).astype(int)

            annotations_by_segment[(video_id, segment)] = pd.DataFrame(
                majority, columns=[f'p{col}' for col in common_cols]
            )
        except Exception as e:
            print(f"⚠️ Error processing segment {video_id}_{segment}: {e}")
            continue

    print_annotation_pid_stats(annotations_by_segment)
    return annotations_by_segment
