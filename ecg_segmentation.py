import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import biosppy.signals.ecg as ecg
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from graph_plot import load_mat_file


def detect_peaks_ecg(signal, r_idx, fs, window_ms, offset_ms, name_peak):
    
    ### Définition du Docstring ###

    """
    On Détecte les pics des ondes ECG (P, Q, S, T) autour d'un pic R donné.
    
    """

    ### On convertit des durées en nombre d'échantillons ###
    window_samples = int(window_ms / 1000 * fs)
    offset_samples = int(offset_ms / 1000 * fs)
    
    ### Segment de recherche selon le type de pic ###
    if name_peak in ["P", "Q"]:
        start = max(r_idx - window_samples, 0)
        end   = r_idx - offset_samples
    
    else:  # "S" ou "T"
        start = r_idx + offset_samples
        end   = min(r_idx + window_samples, len(signal))

    ### Si le segment est invalide, on renvoie None ###
    if start >= end:
        return None

    segment = signal[start:end]

    ### Détection du pic dans le segment ###
    if name_peak == "P":
        peak_idx = np.argmax(segment) ### Pic généralement positif ###

    elif name_peak == "T":
        peak_idx = np.argmax(segment) 
        for i in range(1, len(segment) - 1):
            ### On prend le premier pic local qui est supérieur à ses voisins ###
            if segment[i] > segment[i-1] and segment[i] > segment[i+1]:
                peak_idx = i
                break

    else:  ### "Q" ou "S" ###
        peak_idx = np.argmin(segment) ### Pic négatif ###

    peak_val = segment[peak_idx]

    ### On calcul l'amplitude relative par rapport au pic R ###
    r_amp = np.abs(signal[r_idx])
    baseline = np.median(segment)
    amp = np.abs(peak_val - baseline)

    ### Filtre physio pour onde P : ###
    #if name_peak == "P":
        #if amp < 0.05 * r_amp:  ### Trop faible pour être un pic P ###
            #return None
        #if amp > 0.95 * r_amp:   ### Trop grand pour être un pic P ###
            #return None
    
    ### Filtre physio pour onde T : ###
    #if name_peak == "T":
        #if amp > 0.85 * r_amp:  ### Amplitude anormale pour un pic T ###
            #return None

    ### On retourne l'indice du pic dans le signal ###
    return start + peak_idx


def detect_PR_interval(signal, peak_idx_1, peak_idx_2, fs, thresh_fraction=0.01, max_distance_ms_p=400, max_distance_ms_q=400):
    """
    Détecte l'intervalle PR avec seuil de dérivée adaptatif.
    """

    deriv = np.gradient(signal)
    max_distance_samples_p = int(max_distance_ms_p / 1000 * fs)
    max_distance_samples_q = int(max_distance_ms_q / 1000 * fs)
    
    # Seuil adaptatif pour P : basé sur la dérivée max dans la fenêtre
    window_start_p = max(0, peak_idx_1 - max_distance_samples_p)
    window_end_p = peak_idx_1
    deriv_window_p = deriv[window_start_p:window_end_p]
    if len(deriv_window_p) > 0:
        deriv_thresh_p = thresh_fraction * np.max(np.abs(deriv_window_p))
    else:
        deriv_thresh_p = 0.01
    
    # Onset de P
    start = peak_idx_1
    while start > window_start_p:
        if abs(deriv[start]) <= deriv_thresh_p:
            break
        start -= 1
    
    # Seuil adaptatif pour Q
    window_start_q = max(0, peak_idx_2 - max_distance_samples_q)
    window_end_q = peak_idx_2
    deriv_window_q = deriv[window_start_q:window_end_q]
    if len(deriv_window_q) > 0:
        deriv_thresh_q = thresh_fraction * np.max(np.abs(deriv_window_q))
    else:
        deriv_thresh_q = 0.01
    
    # Onset de Q
    end = peak_idx_2
    while end > window_start_q:
        if abs(deriv[end]) <= deriv_thresh_q:
            break
        end -= 1
    
    return start, end


def detect_QT_interval(signal, peak_idx_1, peak_idx_2, fs, thresh_fraction=0.05, max_distance_ms_t=200):
    """
    Détecte l'intervalle QT avec seuil de dérivée adaptatif.

    """

    deriv = np.gradient(signal)
    max_distance_samples_t = int(max_distance_ms_t / 1000 * fs)
    
    start = peak_idx_1
        
    # Seuil adaptatif pour T
    window_start_t = peak_idx_2
    window_end_t = min(len(signal), peak_idx_2 + max_distance_samples_t)
    deriv_window_t = deriv[window_start_t:window_end_t]
    if len(deriv_window_t) > 0:
        deriv_thresh_t = max(thresh_fraction * np.max(np.abs(deriv_window_t)), 0.002)
    else:
        deriv_thresh_t = 0.002 
    
    # Fin de l'onde T : avancer depuis le pic T
    min_end = peak_idx_2 + int(0.02 * fs)
    if min_end >= window_end_t:
        min_end = peak_idx_2 + 1
    end = peak_idx_2 + 1
    while end < window_end_t - 1:
        if end >= min_end and abs(deriv[end]) <= deriv_thresh_t:
            break
        end += 1
    
    return start, end


if __name__ == "__main__":
    
    
    INPUT_FOLDER = "processed"
    
    DEFAULT_ECG_FS = 1000  ### Fréquence d'échantillonnage de L'ECG ###
    
    start_time = 820
    window_s = 30 ### Fenêtre de 30 secondes sur L'ECG ###  

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue
    
        print(f"Processing {fname} ...")
        #mat = scipy.io.loadmat(path)
        #data = mat['data'][0,0]
        #ecg_raw = data['E_data'].squeeze()
        #t = data['E_time'].squeeze()

        ### Récupération des données filtrées de l'ECG ###
        data = load_mat_file(path)
        ecg_clean = data["ECG_clean"]
        time = data["time"]

        ### Sélection du segment ###
        start_idx = int(start_time * DEFAULT_ECG_FS)
        end_idx = start_idx + int(window_s * DEFAULT_ECG_FS)
        if end_idx > len(ecg_clean):
            end_idx = len(ecg_clean)
    
        ecg_segment = ecg_clean[start_idx:end_idx]
        t_segment = time[start_idx:end_idx]
    
        ### Nettoyage du signal : test du module neurokit2 ###
        #ecg_cleaned = nk.ecg_clean(ecg_segment, sampling_rate=DEFAULT_ECG_FS, method="neurokit")
    
        ### Détection des R-peaks avec Biosppy ###
        out = ecg.ecg(signal=ecg_segment, sampling_rate=DEFAULT_ECG_FS, show=False)
        r_peaks = out['rpeaks']
        print(f"Nombre de R-peaks détectés: {len(r_peaks)}")
    
        ### Détection des peaks (P, Q, S, T) ###
        p_peaks = []
        q_peaks = []
        s_peaks = []
        t_peaks = []
        PR_intervals_list = []
        QT_intervals_list = []
        
        for r in r_peaks:
            
            p = detect_peaks_ecg(ecg_segment, r, fs=DEFAULT_ECG_FS, window_ms=200, offset_ms=80, name_peak="P")
            q = detect_peaks_ecg(ecg_segment, r, fs=DEFAULT_ECG_FS, window_ms=80, offset_ms=10, name_peak="Q")
            s = detect_peaks_ecg(ecg_segment, r, fs=DEFAULT_ECG_FS, window_ms=80, offset_ms=10, name_peak="S")
            t = detect_peaks_ecg(ecg_segment, r, fs=DEFAULT_ECG_FS, window_ms=450, offset_ms=150, name_peak="T")

            ### Détection des intervalles PR ###
            if p is not None and q is not None:
                start_1, end_1 = detect_PR_interval(ecg_segment, p, q, DEFAULT_ECG_FS)
                PR_intervals_list.append((start_1, end_1))
            else:
                PR_interval = None

            ### Détection des intervalles QT ###
            if q is not None and t is not None and p is not None :
                start_2, end_2 = detect_QT_interval(ecg_segment, end_1, t, DEFAULT_ECG_FS)
                QT_intervals_list.append((start_2, end_2))
            else:
                QT_interval = None

            if p is not None :  
                p_peaks.append(p)
            if q is not None : 
                q_peaks.append(q)
            if s is not None : 
                s_peaks.append(s)
            if t is not None : 
                t_peaks.append(t)
        
        print(f"Nombre de P-peaks détectés: {len(p_peaks)}")
        print(f"Nombre de Q-peaks détectés: {len(q_peaks)}")
        print(f"Nombre de S-peaks détectés: {len(s_peaks)}")
        print(f"Nombre de T-peaks détectés: {len(t_peaks)}")

        print(f"\nNombre d'intervalle PR détéctés: {len(PR_intervals_list)}")

        PR_array = np.array(PR_intervals_list)

        PR_starts = PR_array[:, 0]
        PR_ends = PR_array[:, 1]

        QT_array = np.array(QT_intervals_list)

        QT_starts = QT_array[:, 0]
        QT_ends = QT_array[:, 1]

        ### Test Temps PR ###

        PR_times = t_segment[PR_array]

        PR_durations = PR_times[:, 1] - PR_times[:, 0]

        PR_mean = np.mean(PR_durations)

        PR_std = np.std(PR_durations)

        ###----###

        print(f"Durée moyenne du PR interval : {PR_mean*1000:.1f} ms")  

        print(f"Ecart-type PR : {PR_std*1000:.1f} ms")  

        print(f"Médiane du PR : {np.median(PR_durations)*1000:.1f} ms")

        print(PR_durations)

        ### Test Temps QT ###

        QT_times = t_segment[QT_array]

        QT_durations = QT_times[:, 1] - QT_times[:, 0]

        QT_mean = np.mean(QT_durations)

        QT_std = np.std(QT_durations)

        #################################################################
        
        ### Calcul de la fréquence cardiaque moyenne à partir des R-peaks détectés pour QT corrigé ###

        RR = np.diff(r_peaks) / DEFAULT_ECG_FS  ### Durée entre les R-peaks en secondes ###

        #print(RR)

        FC = 60 / RR  ### Fréquence cardiaque en bpm ###

        FC_mean = np.mean(FC)

        FC_median = np.median(FC)

        #print(f"Fréquence cardiaque médiane : {FC_median:.1f} bpm")
        
        print("")

        print(f"Nombre d'intervalle QT détéctés: {len(QT_intervals_list)}")

        print(f"Fréquence cardiaque moyenne : {FC_mean:.1f} bpm")

        QTC = QT_durations / np.sqrt(np.mean(RR))  ### QT corrigé par la formule de Bazett ###

        print(QTC)

        print(f"Durée moyenne du QT corrigé (QTc) : {np.mean(QTC)*1000:.1f} ms")

        #print(f"Ecart-type QT : {np.std(QTC)*1000:.1f} ms") ### outliers peuvent fausser l'écart-type, à revoir avec un filtrage plus strict des intervalles QT ###

        print(f"Médiane du QT corrigé (QTc) : {np.median(QTC)*1000:.1f} ms")

        ###----###
  
        ### Visualisation du segment ECG avec les pics détectés ###
        plt.figure(figsize=(14,4))
        plt.plot(t_segment, ecg_segment, color="black", label="ECG Cleaned", linewidth=1.2)
        
        plt.scatter(t_segment[p_peaks], ecg_segment[p_peaks], color="green", label="P Peaks")
        plt.scatter(t_segment[q_peaks], ecg_segment[q_peaks], color="purple", label="Q Peaks")
        plt.scatter(t_segment[r_peaks], ecg_segment[r_peaks], color="red", label="R Peaks")
        plt.scatter(t_segment[s_peaks], ecg_segment[s_peaks], color="navy", label="S Peaks")
        plt.scatter(t_segment[t_peaks], ecg_segment[t_peaks], color="skyblue", label="T Peaks")

      
        plt.scatter(t_segment[PR_starts], ecg_segment[PR_starts], color="orange", marker="x", s=50, label="PR start")
        plt.scatter(t_segment[PR_ends], ecg_segment[PR_ends], color="red", marker="x", s=50, label="PR end")

        #plt.scatter(t_segment[QT_starts], ecg_segment[QT_starts], color="cyan", marker="x", s=50, label="QT start")
        plt.scatter(t_segment[QT_ends], ecg_segment[QT_ends], color="brown", marker="x", s=50, label="QT end")

        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"ECG avec peaks (P, Q, R, S, T) pour {fname}")
        plt.grid()
        plt.legend()
        plt.show()
