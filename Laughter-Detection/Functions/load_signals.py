import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_filter(signal, lowcut=0.5, highcut=10.0, fs=50):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal)

def load_accelerometer_signals(sensor_base, cols):
    df_dict = {}
    for pid in range(1, 51):
        try:
            path = os.path.join(sensor_base, f"{pid}.csv")
            df = pd.read_csv(path, usecols=cols, encoding='latin1', on_bad_lines='skip', engine="python")
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df.dropna(subset=cols, inplace=True)
            df['time'] = np.arange(len(df)) / 50
            for axis in cols:
                df[f'{axis}_filt'] = apply_filter(df[axis])
            scaler = StandardScaler()
            df[[f'{c}_filt' for c in cols]] = scaler.fit_transform(df[[f'{c}_filt' for c in cols]])
            df_dict[pid] = df
            #print(f"✅ Loaded signal for participant {pid} — {len(df)} rows")
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            continue
    return df_dict