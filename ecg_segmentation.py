import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import biosppy.signals.ecg as ecg
import neurokit2 as nk
from scipy.signal import butter, filtfilt


def lowpass_filter(signal, fs=1000, cutoff=15):
   
    b, a = butter(2, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, signal)

def detect_p_peak(signal, r_idx, fs, window_ms, offset_ms):

    window_samples = int(window_ms / 1000 * fs)
    offset_samples = int(offset_ms / 1000 * fs)

    start = max(r_idx - window_samples, 0)
    end = r_idx - offset_samples
    segment = signal[start:end]

    if len(segment) < 2:
       return None

    segment_filt = lowpass_filter(segment, fs=fs, cutoff=10)

    peak_idx = np.argmax(np.abs(segment_filt))
    
    return start + peak_idx

### Pour la suite : ###
### def detect_t_peak(signal, r_idx, fs, window_ms, offset_ms):

if __name__ == "__main__":
    
    
    INPUT_FOLDER = "raws_signals"
    
    DEFAULT_ECG_FS = 1000
    
    start_time = 820
    window_s = 70

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue
    
        print(f"Processing {fname} ...")
        mat = scipy.io.loadmat(path)
        data = mat['data'][0,0]
        ecg_raw = data['E_data'].squeeze()
        t = data['E_time'].squeeze()
    
        start_idx = int(start_time * DEFAULT_ECG_FS)
        end_idx = start_idx + int(window_s * DEFAULT_ECG_FS)
        if end_idx > len(ecg_raw):
            end_idx = len(ecg_raw)
    
        ecg_segment = ecg_raw[start_idx:end_idx]
        t_segment = t[start_idx:end_idx]
    
        ### Nettoyage du signal : test du module neurokit2 ###
        ecg_cleaned = nk.ecg_clean(ecg_segment, sampling_rate=DEFAULT_ECG_FS, method="neurokit")
    
        ### Détection des R-peaks avec Biosppy ###
        out = ecg.ecg(signal=ecg_cleaned, sampling_rate=DEFAULT_ECG_FS, show=False)
        r_peaks = out['rpeaks']
        print(f"Nombre de R-peaks détectés: {len(r_peaks)}")
    
        ### Détection des P-peaks ###
        p_peaks = []
        for r in r_peaks:
            p = detect_p_peak(ecg_cleaned, r, fs=DEFAULT_ECG_FS, window_ms=150, offset_ms=80)
            if p is not None:
                p_peaks.append(p)
        p_peaks = np.array(p_peaks)
        print(f"Nombre de P-peaks détectés: {len(p_peaks)}")
        
        ### Détection des T-peaks ###
        
        ### ... ###

        plt.figure(figsize=(14,4))
        plt.plot(t_segment, ecg_cleaned, color="black", label="ECG Cleaned", linewidth=1.2)
        plt.scatter(t_segment[r_peaks], ecg_cleaned[r_peaks], color="red", label="R Peaks")
        plt.scatter(t_segment[p_peaks], ecg_cleaned[p_peaks], color="green", label="P Peaks")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"ECG avec R et P peaks pour {fname}")
        plt.grid()
        plt.legend()
        plt.show()
