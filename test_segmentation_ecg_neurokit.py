import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import biosppy.signals.ecg as ecg
import neurokit2 as nk


def safe_indices(peak_list):
    
    ### Définition du Docstring ###
    
    """
    On Filtre les NaN et on convertit en int pour indexer un tableau numpy.
    
    """
    return np.array([int(x) for x in peak_list if not np.isnan(x)], dtype=int)



if __name__ == "__main__":
    
    INPUT_FOLDER = "raws_signals"
    DEFAULT_ECG_FS = 1000 ### Fréquence d'échantillonnage de L'ECG ###
    
    start_time = 820   
    window_s = 30 ### Fenêtre de 30 secondes sur L'ECG ###  

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue

        print(f"Processing {fname} ...")
        
        ### Récupération des données brutes ECG et du vecteur temps ###
        mat = scipy.io.loadmat(path)
        data = mat['data'][0,0]
        ecg_raw = data['E_data'].squeeze()
        t = data['E_time'].squeeze() ###
        mat = scipy.io.loadmat(path)
        data = mat['data'][0,0]
        ecg_raw = data['E_data'].squeeze()
        t = data['E_time'].squeeze()

        ### Sélection du segment ###
        start_idx = int(start_time * DEFAULT_ECG_FS)
        end_idx = start_idx + int(window_s * DEFAULT_ECG_FS)
        if end_idx > len(ecg_raw):
            end_idx = len(ecg_raw)

        ecg_segment = ecg_raw[start_idx:end_idx]
        t_segment = t[start_idx:end_idx]

        ### Nettoyage du signal avec neurokit2 ###
        ecg_cleaned = nk.ecg_clean(ecg_segment, sampling_rate=DEFAULT_ECG_FS, method="neurokit")

        ### Détection des R-peaks avec Biosppy ###
        out = ecg.ecg(signal=ecg_cleaned, sampling_rate=DEFAULT_ECG_FS, show=False)
        r_peaks = out['rpeaks']
        print(f"Nombre de R-peaks détectés: {len(r_peaks)}")

        ### Détection automatique des pics P, Q, R, S, T avec NeuroKit ###
        signals, peaks = nk.ecg_delineate(
            ecg_cleaned,
            r_peaks,
            sampling_rate=DEFAULT_ECG_FS,
            method="dwt"
        )

        ### Filtrage des NaN et conversion en int ###
        p_peaks = safe_indices(peaks["ECG_P_Peaks"])
        q_peaks = safe_indices(peaks["ECG_Q_Peaks"])
        s_peaks = safe_indices(peaks["ECG_S_Peaks"])
        t_peaks = safe_indices(peaks["ECG_T_Peaks"])

        print(f"Nombre de P-peaks détectés: {len(p_peaks)}")
        print(f"Nombre de Q-peaks détectés: {len(q_peaks)}")
        print(f"Nombre de S-peaks détectés: {len(s_peaks)}")
        print(f"Nombre de T-peaks détectés: {len(t_peaks)}")

        ### Visualisation ###
        plt.figure(figsize=(14,4))
        plt.plot(t_segment, ecg_cleaned, color="black", label="ECG Cleaned", linewidth=1.2)
        plt.scatter(t_segment[p_peaks], ecg_cleaned[p_peaks], color="green", label="P Peaks")
        plt.scatter(t_segment[q_peaks], ecg_cleaned[q_peaks], color="purple", label="Q Peaks")
        plt.scatter(t_segment[r_peaks], ecg_cleaned[r_peaks], color="red", label="R Peaks")
        plt.scatter(t_segment[s_peaks], ecg_cleaned[s_peaks], color="navy", label="S Peaks")
        plt.scatter(t_segment[t_peaks], ecg_cleaned[t_peaks], color="skyblue", label="T Peaks")

        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"ECG avec peaks (P, Q, R, S, T) pour {fname}")
        plt.grid()
        plt.legend()
        plt.show()
