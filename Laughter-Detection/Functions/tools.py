from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    return butter(order, [low, high], btype='band')
#Esto sirve para bloquear freq debajo o por ensima de los parametros

def apply_filter(signal, lowcut=0.5, highcut=10.0, fs=50):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal)