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

def detect_peaks_ecg(signal, r_idx, fs, window_ms, offset_ms, name_peak):

    window_samples = int(window_ms / 1000 * fs)
    offset_samples = int(offset_ms / 1000 * fs)
    
    if name_peak == "P" or  name_peak == "Q" : 
    
        start = max(r_idx - window_samples, 0)
        end = r_idx - offset_samples
        segment = signal[start:end]
        
    else : 
        start = max(r_idx + offset_samples, 0)
        end   = min(r_idx + window_samples, len(signal))
        segment = signal[start:end]

    if start >= end :
        return None

    segment_filt = lowpass_filter(segment, fs=fs, cutoff=10)

    if name_peak == "P" or name_peak == "T" : 

        peak_idx = np.argmax(np.abs(segment_filt))
        
    else : 
        peak_idx = np.argmin(segment_filt)
    
    return start + peak_idx



if __name__ == "__main__":
    
    
    INPUT_FOLDER = "raws_signals"
    
    DEFAULT_ECG_FS = 1000
    
    start_time = 820
    window_s = 30

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
    
        ### Détection des peaks (P, Q, R, S, T) ###
        p_peaks = []
        q_peaks = []
        s_peaks = []
        t_peaks = []
        
        for r in r_peaks:
            
            p = detect_peaks_ecg(ecg_cleaned, r, fs=DEFAULT_ECG_FS, window_ms=150, offset_ms=80, name_peak="P")
            q = detect_peaks_ecg(ecg_cleaned, r, fs=DEFAULT_ECG_FS, window_ms=40, offset_ms=10, name_peak="Q")
            s = detect_peaks_ecg(ecg_cleaned, r, fs=DEFAULT_ECG_FS, window_ms=40, offset_ms=10, name_peak="S")
            t = detect_peaks_ecg(ecg_cleaned, r, fs=DEFAULT_ECG_FS, window_ms=280, offset_ms=220, name_peak="T")
            
            if p is not None and q is not None and s is not None and t is not None :
                p_peaks.append(p)
                q_peaks.append(q)
                s_peaks.append(s)
                t_peaks.append(t)

        p_peaks = np.array(p_peaks)
        q_peaks = np.array(q_peaks)
        s_peaks = np.array(s_peaks)
        t_peaks = np.array(t_peaks)
        
        print(f"Nombre de P-peaks détectés: {len(p_peaks)}")
        print(f"Nombre de Q-peaks détectés: {len(q_peaks)}")
        print(f"Nombre de S-peaks détectés: {len(s_peaks)}")
        print(f"Nombre de T-peaks détectés: {len(t_peaks)}")
  
    
        plt.figure(figsize=(14,4))
        plt.plot(t_segment, ecg_cleaned, color="black", label="ECG Cleaned", linewidth=1.2)
        
        plt.scatter(t_segment[p_peaks], ecg_cleaned[p_peaks], color="green", label="P Peaks")
        plt.scatter(t_segment[q_peaks], ecg_cleaned[q_peaks], color="purple", label="Q Peaks")
        plt.scatter(t_segment[r_peaks], ecg_cleaned[r_peaks], color="red", label="R Peaks")
        plt.scatter(t_segment[s_peaks], ecg_cleaned[s_peaks], color="yellow", label="S Peaks")
        plt.scatter(t_segment[t_peaks], ecg_cleaned[t_peaks], color="skyblue", label="T Peaks")
        
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"ECG avec peaks (P, Q, R, S, T) pour {fname}")
        plt.grid()
        plt.legend()
        plt.show()
