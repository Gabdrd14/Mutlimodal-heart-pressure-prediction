import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import biosppy.signals.ecg as bsp_ecg
from scipy.signal import butter, filtfilt, find_peaks


def load_mat_file(path):

    ### Définition du Docstring ###

    """
    On charge le fichier .mat et on extrait les données ECG, SCG, fs, time, patient.
    
    On crée un vecteur de temps à partir de window_start, window_end et n_samples.
    
    """
    mat = scipy.io.loadmat(path)

    fs = int(mat['fs'].squeeze())
    print(f"fs détecté : {fs} Hz")
    window_start = float(mat['window_start_s'].squeeze())
    window_end = float(mat['window_end_s'].squeeze())
    n_samples = mat['ecg'].shape[1]
    time = np.linspace(window_start, window_end, n_samples)

    return {
        "patient": str(mat['patient'].squeeze()),
        "ECG_clean": mat['ecg'].squeeze(),
        "ECG_raw": mat['ecg_raw'].squeeze(),
        "time": time,
        "fs": fs,
        "SCG_clean": mat['scg'].squeeze(),
    }



def detect_pacing_spikes(signal, fs, threshold_factor=8.0, min_distance_ms=200):
    
    ### Définition du Docstring ###

    """
    On détecte les spikes de pacing via la dérivée seconde du signal ECG.
    Les spikes de pacing sont des artefacts très courts et abrupts, souvent présents chez les patients avec pacemaker ou CRT-D. 
    La dérivée seconde amplifie ces transitions rapides, permettant de les détecter même s'ils sont petits en amplitude.

    """

    deriv2 = np.abs(np.gradient(np.gradient(signal)))
    threshold = threshold_factor * np.std(deriv2)
    min_distance = int(min_distance_ms / 1000 * fs)

    candidates = np.where(deriv2 > threshold)[0]

    if len(candidates) == 0:
        return np.array([])

    ### On garde seulement les pics les plus forts, en respectant une distance minimale entre eux (min_distance) ###
    spikes = [candidates[0]]
    for idx in candidates[1:]:
        if idx - spikes[-1] > min_distance:
            spikes.append(idx)
        elif deriv2[idx] > deriv2[spikes[-1]]:
            spikes[-1] = idx 

    return np.array(spikes)



def remove_pacing_spikes(signal, spike_indices, fs, window_ms=6):
    
    ### Définition du Docstring ###

    """
    On supprime les spikes de pacing par interpolation linéaire autour de chaque spike détecté.

    """
    signal_clean = signal.copy()
    window_samples = int(window_ms / 1000 * fs) 

    for spike_idx in spike_indices:
        start = max(0, spike_idx - window_samples)
        end = min(len(signal), spike_idx + window_samples)

        ### On interpole linéairement entre les points avant et après le spike, en s'assurant de ne pas dépasser les limites du signal ###
        signal_clean[start:end] = np.linspace(
            signal_clean[start],
            signal_clean[end - 1],
            end - start
        )

    return signal_clean



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
        end = r_idx - offset_samples
    
    else:  ### "S" ou "T" ###
        start = r_idx + offset_samples
        end   = min(r_idx + window_samples, len(signal))

    ### Si le segment est invalide, on renvoie None ###
    if start >= end:
        return None

    segment = signal[start:end]

    ### Détection du pic dans le segment : Q et S sont des minima, P et T sont des maxima ###
    if name_peak in ["Q", "S"]:
        peak_idx = np.argmin(segment)

    elif name_peak == "P":
        peak_idx = np.argmax(segment)

    else:  ### "T" ####
        peak_idx = np.argmax(segment)

    ### On calcul l'amplitude relative par rapport au pic R ###
    baseline = np.median(segment)
    r_amp = np.abs(signal[r_idx])
    amp = np.abs(segment[peak_idx] - baseline)

    ### Vérification d'amplitude minimale pour éviter les faux positifs, surtout pour T (qui peut être petit en IC) ###
    if name_peak == "T":
        min_amp_ratio = 0.03 

        if amp < min_amp_ratio * r_amp:
            return None

    return start + peak_idx


def detect_PR_interval(signal, peak_p, peak_q, fs, thresh_fraction = 0.01, max_distance_ms_p = 400, max_distance_ms_q = 400):
    
    ### Définition du Docstring ###

    """
    On détecte les onsets des ondes P et Q via la dérivée du signal ECG.
   
    On cherche le point où la dérivée devient inférieure à un seuil de son maximum local, en partant des pics P et Q respectivement.
    
    """

    deriv = np.gradient(signal)

    max_dist_p = int(max_distance_ms_p / 1000 * fs)
    max_dist_q = int(max_distance_ms_q / 1000 * fs)

    ### Onset P ###
    win_start_p = max(0, peak_p - max_dist_p)
    deriv_p = deriv[win_start_p:peak_p]
    thresh_p = thresh_fraction * np.max(np.abs(deriv_p)) if len(deriv_p) > 0 else 0.01

    onset_p = peak_p
    while onset_p > win_start_p:
        if abs(deriv[onset_p]) <= thresh_p:
            break
        onset_p -= 1

    ### Onset Q ###
    win_start_q = max(0, peak_q - max_dist_q)
    deriv_q = deriv[win_start_q:peak_q]
    thresh_q = thresh_fraction * np.max(np.abs(deriv_q)) if len(deriv_q) > 0 else 0.01

    onset_q = peak_q
    while onset_q > win_start_q:
        if abs(deriv[onset_q]) <= thresh_q:
            break
        onset_q -= 1

    ### Vérification que l'intervalle PR est physiologiquement plausible (80-400 ms) sinon on prend les pics bruts comme onsets ###
    pr_ms = (onset_q - onset_p) / fs * 1000

    if pr_ms < 80 or pr_ms > 400:
        onset_p = peak_p
        onset_q = peak_q

    return onset_p, onset_q



def detect_QT_interval(signal, onset_q, peak_t, fs, thresh_fraction=0.05, max_distance_ms_t=350):
    
    ### Définition du Docstring ###

    """
    On détecte l'offset de l'onde T (fin du QT) via la dérivée du signal ECG.
    
    """
    deriv = np.gradient(signal)

    max_dist_t  = int(max_distance_ms_t / 1000 * fs)
    win_start_t = peak_t
    win_end_t = min(len(signal), peak_t + max_dist_t)
    deriv_t = deriv[win_start_t:win_end_t]

    if len(deriv_t) > 0:
        thresh_t = max(thresh_fraction * np.max(np.abs(deriv_t)), 0.002)
    else:
        thresh_t = 0.002

    min_end = peak_t + int(0.02 * fs)
    if min_end >= win_end_t:
        min_end = peak_t + 1

    ### On cherche le point où la dérivée devient inférieure au seuil, en partant du pic T vers la droite ###
    offset_t = peak_t + 1
    while offset_t < win_end_t - 1:
        if offset_t >= min_end and abs(deriv[offset_t]) <= thresh_t:
            break
        offset_t += 1

    return onset_q, offset_t



def filter_iqr(values, factor=2.0):
    
    ### Définition du Docstring ###

    """
    On filtre les valeurs extrêmes en utilisant l'intervalle interquartile (IQR).
    Masque booléen indiquant les valeurs qui sont dans l'intervalle [Q1 - factor*IQR, Q3 + factor*IQR].
    
    """
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    
    return (values >= low) & (values <= high)



def detect_rpeaks_manual(signal, fs, prominence_factor=3.0):
    
    ### Définition du Docstring ###

    """
    On détecte les pics R manuellement en utilisant find_peaks de scipy, en cherchant les pics négatifs (R inversés) si nécessaire.

    """

    ### On inverse le signal pour trouver les pics négatifs comme des pics positifs, 
    ### en utilisant une distance minimale de 0.4 s entre les pics pour éviter les faux positifs. ###

    neg_signal = -signal
    min_distance = int(0.4 * fs) 
    
    peaks, _ = find_peaks(neg_signal, 
                          distance=min_distance,
                          prominence=np.std(signal) * prominence_factor)
    
    return peaks


def detect_rpeaks_hybrid(signal, fs):
    
    ### Définition du Docstring ###
    
    """
    On détecte les pics R en essayant d'abord biosppy, puis en basculant sur find_peaks de scipy si biosppy échoue ou si les pics R sont négatifs (inversés).
    
    """
    ### biosppy ###
    try:
        out = bsp_ecg.ecg(signal=signal, sampling_rate=fs, show=False)
        r_peaks_bsp = out['rpeaks']
        
        if len(r_peaks_bsp) >= 10:
            
            ### On vérifie la polarité des pics ###
            amplitudes = signal[r_peaks_bsp]
            mean_amp = np.mean(amplitudes)
            
            if mean_amp >= 0:
                ### Pics positifs : biosppy ###
                return r_peaks_bsp, "biosppy"
            else:
                ### Pics négatifs : on bascule sur find_peaks ###
                print("Pics R négatifs détectés, on bascule sur find_peaks (scipy)")
                r_peaks_scipy = detect_rpeaks_manual(signal, fs)
                return r_peaks_scipy, "scipy"
    except Exception as e:
        print(f"Biosppy a échoué ({e}) retour sur find_peaks")
    
    return detect_rpeaks_manual(signal, fs), "scipy"


### Main ###

if __name__ == "__main__":

    INPUT_FOLDER = "processed/Correct_DATA_V2"
    start_time = 0
    window_s = 30

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue
        
        print("===============================================================")
        print(f"Processing {fname} ...")

        data = load_mat_file(path)
        fs = data["fs"]
        ecg_clean = data["ECG_clean"]
        scg_clean = data["SCG_clean"]
        time = data["time"]
        patient= data["patient"]

        ### On Sélectionne le segment ###
        start_idx = int(start_time * fs)
        end_idx = min(start_idx + int(window_s * fs), len(ecg_clean))
        ecg_segment = ecg_clean[start_idx:end_idx]
        scg_segment = scg_clean[start_idx:end_idx]
        t_segment = time[start_idx:end_idx]

        print(f"Segment ecg : {t_segment[0]:.1f}–{t_segment[-1]:.1f} s  "
              f"({len(ecg_segment)} samples @ {fs} Hz)")

        print(f"Segment scg : {t_segment[0]:.1f}–{t_segment[-1]:.1f} s  "
              f"({len(scg_segment)} samples @ {fs} Hz)")
        
       
        ### Détection et suppression des spikes de pacing ###
        spike_indices = detect_pacing_spikes(ecg_segment, fs)

        if len(spike_indices) > 0:
            print(f"\n{len(spike_indices)} spike(s) de pacing détectés, suppression")
            ecg_nospike = remove_pacing_spikes(ecg_segment, spike_indices, fs)
        else:
            print("Aucun spike de pacing détecté")
            ecg_nospike = ecg_segment

        ### Détection des pics R sur le signal sans spikes, avec méthode hybride biosppy + find_peaks ###
        r_peaks, rpeaks_source = detect_rpeaks_hybrid(ecg_nospike, fs)

        if len(r_peaks) < 4:
            print(f"Seulement {len(r_peaks)} R-peaks, segment trop court ou bruité")
            continue

        print(f"R-peaks détectés : {len(r_peaks)} ({rpeaks_source})")

        ### Calcul sur le RR ###
        RR_all = np.diff(r_peaks) / fs
        RR_mean = np.mean(RR_all)
        RR_median  = np.median(RR_all)

        ### Détection des pics (P, Q, S, T) et des instervalles PR et QT ###
        p_peaks = []
        q_peaks = []
        s_peaks= []
        t_peaks = []
        PR_intervals_list = []
        QT_intervals_list = []

        for i, r in enumerate(r_peaks):

            ### RR local (ms) pour adapter les fenêtres ###
            if i > 0:
                rr_local_ms = (r - r_peaks[i - 1]) / fs * 1000
            elif i < len(r_peaks) - 1:
                rr_local_ms = (r_peaks[i + 1] - r) / fs * 1000
            else:
                rr_local_ms = RR_mean * 1000

            ### Fenêtres adaptatives ###
            p_window = min(200, int(rr_local_ms * 0.35))  ### max 35% du RR ###
            t_window = min(500, int(rr_local_ms * 0.50))  #### max 50% du RR ###

            p = detect_peaks_ecg(ecg_nospike, r, fs, window_ms=p_window, offset_ms=80,  name_peak="P")
            q = detect_peaks_ecg(ecg_nospike, r, fs, window_ms=80, offset_ms=10,  name_peak="Q")
            s = detect_peaks_ecg(ecg_nospike, r, fs, window_ms=120, offset_ms=10,  name_peak="S")
            t = detect_peaks_ecg(ecg_nospike, r, fs, window_ms=t_window, offset_ms=150, name_peak="T")
       
            ### Vérification que le pic P ne dépasse pas R, et que P vient avant Q ###
            if p is not None and (p >= r or (q is not None and p >= q)):
                p = None

            ### Vérification que le pic Q ne dépasse pas R, et que Q vient avant R ###   
            if q is not None and q >= r:
                q = None
            
            ### Vérification que le pic S ne dépasse pas R ###
            if s is not None and s <= r:
                s = None
            
            ### intervalle PR ###
            onset_q_pr = None
            if p is not None and q is not None:
                onset_p, onset_q_pr = detect_PR_interval(ecg_nospike, p, q, fs)
                PR_intervals_list.append((onset_p, onset_q_pr))
            elif q is not None:
                # Fallback : si P manque, utiliser Q directement comme onset
                onset_q_pr = q

            ### Intervalle QT, onset_q issu du PR du même battement ###
            if q is not None and t is not None and onset_q_pr is not None:
                onset_q_qt, offset_t = detect_QT_interval(ecg_nospike, onset_q_pr, t, fs)
                QT_intervals_list.append((onset_q_qt, offset_t))
            

            if p is not None: p_peaks.append(p)
            if q is not None: q_peaks.append(q)
            if s is not None: s_peaks.append(s)
            if t is not None: t_peaks.append(t)

        print(f"P={len(p_peaks)}  Q={len(q_peaks)}  S={len(s_peaks)}  T={len(t_peaks)}")

        ### Calcul de la fréquence cardiaque moyenne ###
        FC_mean = np.mean(60 / RR_all)  

        if len(PR_intervals_list) == 0:
            print("Aucun intervalle PR détecté")
        else:
            PR_array = np.array(PR_intervals_list)
            PR_times = t_segment[PR_array]
            PR_durations = PR_times[:, 1] - PR_times[:, 0]

            mask_pr = filter_iqr(PR_durations)
            PR_clean = PR_durations[mask_pr]

            print(f"\nPR intervals : {len(PR_intervals_list)} détectés, "
                  f"{mask_pr.sum()} après filtre IQR")
            #print(f"  PR moyen   : {np.mean(PR_clean)*1000:.1f} ms")
            print(f"PR médiane : {np.median(PR_clean)*1000:.1f} ms")
            #print(f"  PR std     : {np.std(PR_clean)*1000:.1f} ms")

            PR_median_ms = np.median(PR_clean)*1000

            if PR_median_ms < 80:
                print(f"PR impossible (< 80 ms)")
            
            elif PR_median_ms < 100:
                print(f"PR court (< 100 ms)")
            
            elif PR_median_ms > 200:
                print(f"PR prolongé (> 200 ms)")


        if len(QT_intervals_list) == 0:
            print("Aucun intervalle QT détecté")
        else:
            # Valider que tous les indices sont dans les limites
            QT_array = np.array(QT_intervals_list)
            max_idx = len(t_segment)
            
            ### On clippe les indices hors limites ###
            QT_array_clipped = np.clip(QT_array, 0, max_idx - 1)
            
            QT_times = t_segment[QT_array_clipped]
            QT_durations = QT_times[:, 1] - QT_times[:, 0]

            mask_qt = filter_iqr(QT_durations)
            QT_clean = QT_durations[mask_qt]

            ### Fridericia : QTc = QT / RR^(1/3), Formule plus robuste que Bazett en IC ###
            QTC = QT_clean / (RR_median ** (1 / 3))

            ### Médiane pour la fréquence cardiaque ###
            FC_median = np.median(60 / RR_all)

            ### Ecart-type RR ###
            RR_std = np.std(RR_all)
            print(f"\nRR std : {RR_std:.3f} s")
            print(f"\nFC moyenne : {FC_mean:.1f} bpm  |  FC médiane : {FC_median:.1f} bpm")
            print(f"QT intervals : {len(QT_intervals_list)} détectés, "
                  f"{mask_qt.sum()} après filtre IQR")
            print(f"QT brut moyen   : {np.mean(QT_clean)*1000:.1f} ms")
            print(f"QT brut médiane : {np.median(QT_clean)*1000:.1f} ms")
            print(f"QTc (Fridericia) moyen   : {np.mean(QTC)*1000:.1f} ms")
            print(f"QTc (Fridericia) médiane : {np.median(QTC)*1000:.1f} ms")

            ### Médiane pour le QTc ###
            qtc_median_ms = np.median(QTC) * 1000

            if qtc_median_ms > 500:
                print(f" QTc > 500 ms, risque arythmie (torsade de pointes)")
            
            elif qtc_median_ms > 460:
                print(f"QTc prolongé (> 460 ms)")
            
            elif qtc_median_ms < 300:
                print(f"Très suspect (< 300 ms)")
            
            elif qtc_median_ms < 320:
                print(f"Pathologiquement court (< 320 ms)")
            
            elif qtc_median_ms < 330:
                print(f"QTc court (< 330 ms)")


        ### Visualisation ###
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        fig.suptitle(f"ECG + SCG — {fname}  |  patient={patient}  |  FC={FC_mean:.0f} bpm",
                     fontsize=12)

        ### Axe 1 : ECG ###
        ### Signal brut en gris si spikes supprimés, sinon noir direct ###
        if len(spike_indices) > 0:
            ax1.plot(t_segment, ecg_segment, color="lightgray",
                    linewidth=0.8, alpha=0.5, label="ECG brut")
            ax1.plot(t_segment, ecg_nospike, color="black",
                    linewidth=1.0, label="ECG sans spikes")
            ax1.scatter(t_segment[spike_indices], ecg_segment[spike_indices],
                       color="magenta", s=60, zorder=6, marker="v", label="Spikes pacing")
        else:
            ax1.plot(t_segment, ecg_segment, color="black",
                    linewidth=1.0, label="ECG")

        if len(p_peaks) > 0:
            ax1.scatter(t_segment[p_peaks], ecg_nospike[p_peaks],
                       color="green", s=30, zorder=5, label="P")
        if len(q_peaks) > 0:
            ax1.scatter(t_segment[q_peaks], ecg_nospike[q_peaks],
                       color="purple", s=30, zorder=5, label="Q")
        ax1.scatter(t_segment[r_peaks], ecg_nospike[r_peaks],
                   color="red", s=40, zorder=5, label="R")
        if len(s_peaks) > 0:
            ax1.scatter(t_segment[s_peaks], ecg_nospike[s_peaks],
                       color="navy", s=30, zorder=5, label="S")
        if len(t_peaks) > 0:
            ax1.scatter(t_segment[t_peaks], ecg_nospike[t_peaks],
                       color="deepskyblue", s=30, zorder=5, label="T")

        ### Onset P et onset Q (PR) ###
        if len(PR_intervals_list) > 0:
            PR_arr = np.array(PR_intervals_list)
            ax1.scatter(t_segment[PR_arr[:, 0]], ecg_nospike[PR_arr[:, 0]],
                       color="orange", marker="x", s=80, linewidths=1.5, label="PR START")
            ax1.scatter(t_segment[PR_arr[:, 1]], ecg_nospike[PR_arr[:, 1]],
                       color="red", marker="x", s=80, linewidths=1.5, label="PR END")

        ###  Offset T (fin QT) ###
        if len(QT_intervals_list) > 0:
            QT_arr = np.array(QT_intervals_list)
            max_idx = len(t_segment)
            QT_arr_clipped = np.clip(QT_arr, 0, max_idx - 1)
            ax1.scatter(t_segment[QT_arr_clipped[:, 1]], ecg_nospike[QT_arr_clipped[:, 1]],
                       color="brown", marker="x", s=80, linewidths=1.5, label="QT END")

        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8)

        ### Axe 2 : SCG + lignes des pics R ###
        ax2.plot(t_segment, scg_segment, color="steelblue", linewidth=0.9, label="SCG")

        for r in r_peaks:
            ax2.axvline(x=t_segment[r], color="red", alpha=0.3, linewidth=0.8)

        ax2.set_ylabel("SCG (g)")
        ax2.set_xlabel("Temps (s)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()