import numpy as np
from tqdm import tqdm
import time

def construir_video_segments(annotations_by_segment, fps=25):
    return {
        (video_id, segment_id): (0.0, df.shape[0] / fps)
        for (video_id, segment_id), df in annotations_by_segment.items()
    }

def segment_signal(df, window_size, step_size, columns):
    segments, starts = [], []
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        segments.append(df[columns].iloc[start:end].values)
        starts.append(df['time'].iloc[start])
    return np.array(segments), np.array(starts)

def extract_features(segment):
    x, y, z = segment[:, 0], segment[:, 1], segment[:, 2]
    features = []

    for axis_data in [x, y, z]:
        deriv1 = np.gradient(axis_data)
        deriv2 = np.gradient(deriv1)

        features.extend([
            np.mean(axis_data),
            np.std(axis_data),
            np.min(axis_data),
            np.max(axis_data),
            np.sum(axis_data ** 2),             # energy
            axis_data[-1] - axis_data[0],       # dynamic change
            np.mean(deriv1),                    # slope
            np.mean(np.abs(deriv1)),            # abs(slope)
            np.mean(deriv1),
            np.std(deriv1),
            np.mean(deriv2),
            np.std(deriv2),
        ])

    sma = np.mean(np.abs(x) + np.abs(y) + np.abs(z))
    features.append(sma)

    # Axis correlations (X-Y, X-Z, Y-Z)
    features.append(np.corrcoef(x, y)[0, 1])
    features.append(np.corrcoef(x, z)[0, 1])
    features.append(np.corrcoef(y, z)[0, 1])

    return features

def construir_dataset(annotations_by_segment, df_dict, video_segments, FS, FPS, WINDOW_SIZE, STEP_SIZE, COLS):
    X_total, y_total, resumen_data = [], [], []
    print(f"\n🔄 Building dataset from {len(annotations_by_segment)} segments...\n")

    for (video_id, segment_id), ann_df in tqdm(annotations_by_segment.items(), desc="Segments", unit="seg", colour="green"):
        segment_tuple = (video_id, segment_id)
        if segment_tuple not in video_segments:
            continue
        start_t, end_t = video_segments[segment_tuple]
        segment_key = f"{video_id}_{segment_id}"

        for col in ann_df.columns:
            pid = int(col[1:])

            if pid not in df_dict:
                continue


            df_full = df_dict[pid]
            df = df_full[(df_full['time'] >= start_t) & (df_full['time'] < end_t)].copy()
            if df.empty:
                continue

            segments, starts = segment_signal(df, WINDOW_SIZE, STEP_SIZE, [f'{c}_filt' for c in COLS])
            if len(segments) == 0:
                continue

            if WINDOW_SIZE < 50:
                segments_np = np.array(segments)  # shape: (n_windows, window_size, 3)

                means = segments_np.mean(axis=1)  # (n_windows, 3)
                stds = segments_np.std(axis=1)
                maxs = segments_np.max(axis=1)
                mins = segments_np.min(axis=1)
                energy = np.sum(segments_np ** 2, axis=1)

                # Correct SMA: average of sum of abs(X,Y,Z) per sample
                sma = np.mean(np.sum(np.abs(segments_np), axis=2), axis=1).reshape(-1, 1)  # (n_windows, 1)

                # Stack all features: shape (n_windows, total_features)
                features = np.hstack([means, stds, maxs, mins, energy, sma])


            else:
                features = [extract_features(seg) for seg in segments]

            centers = starts + (WINDOW_SIZE / (2 * FS))
            label_vector = ann_df[col].values.astype(float)
            label_times = np.arange(len(label_vector)) / FPS
            valid_idx = (centers >= label_times[0]) & (centers <= label_times[-1])
            if valid_idx.sum() == 0:
                print(f"⚠️ No valid windows for segment {segment_key}, pid {pid}")
                continue

            valid_centers = centers[valid_idx]
            valid_features = [features[i] for i in range(len(features)) if valid_idx[i]]
            labels_interp = np.interp(valid_centers, label_times, label_vector)
            labels = [int(round(x)) for x in labels_interp if not np.isnan(x)]

            if len(valid_features) != len(labels):
                print(f"⚠️ Mismatch between features and labels in {segment_key}, pid {pid}")
                continue

            X_total.extend(valid_features)
            y_total.extend(labels)
            resumen_data.append({
                "video": segment_key,
                "pid": pid,
                "n_segmentos": len(valid_features),
                "n_risa": sum(labels)
            })

    print("Salimos de construir dataset")
    return X_total, y_total, resumen_data
