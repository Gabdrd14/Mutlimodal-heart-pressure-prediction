import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from ecg_segmentation_2 import (
    load_mat_file,
    detect_pacing_spikes,
    remove_pacing_spikes,
    detect_rpeaks_hybrid,
    filter_iqr,
)

def detect_MC_onset(scg, mc_peak_idx, fs, thresh_fraction=0.01, max_lookback_ms=20):
    
        
    ### Définition du Docstring ###

    """
    On détecte l'onset de l'onde MC en remontant depuis le pic MC
    jusqu'à ce que la dérivée repasse sous le seuil adaptatif.
    
    """
    deriv = np.gradient(scg)
    max_lookback = int(max_lookback_ms / 1000 * fs)
    win_start = max(0, mc_peak_idx - max_lookback)

    deriv_window = deriv[win_start:mc_peak_idx]
    if len(deriv_window) == 0:
        return mc_peak_idx

    thresh = thresh_fraction * np.max(np.abs(deriv_window))

    onset = mc_peak_idx
    while onset > win_start:
        if abs(deriv[onset]) <= thresh:
            break
        onset -= 1

    return onset


def detect_AO_onset(scg, ao_peak_idx, fs, thresh_fraction=0.01, max_lookback_ms=30):

    ### Définition du Docstring ###

    """
    On détecte l'onset de l'onde AO en remontant depuis le pic AO
    jusqu'à ce que la dérivée repasse sous le seuil adaptatif.

    """
    deriv = np.gradient(scg)
    max_lookback = int(max_lookback_ms / 1000 * fs)
    win_start = max(0, ao_peak_idx - max_lookback)

    deriv_window = deriv[win_start:ao_peak_idx]
    if len(deriv_window) == 0:
        return ao_peak_idx

    thresh = thresh_fraction * np.max(np.abs(deriv_window))

    onset = ao_peak_idx
    while onset > win_start:
        if abs(deriv[onset]) <= thresh:
            break
        onset -= 1

    return onset


def detect_AC_onset(scg, ac_peak_idx, fs, thresh_fraction=0.01, max_lookback_ms=60):
    
    ### Définition du Docstring ###
    
    """
    On détecte l'onset de l'onde AC en remontant depuis le pic AC
    jusqu'à ce que la dérivée repasse sous le seuil adaptatif.
    
    """
    deriv = np.gradient(scg)
    max_lookback = int(max_lookback_ms / 1000 * fs)
    win_start = max(0, ac_peak_idx - max_lookback)

    deriv_window = deriv[win_start:ac_peak_idx]
    if len(deriv_window) == 0:
        return ac_peak_idx

    thresh = thresh_fraction * np.max(np.abs(deriv_window))

    onset = ac_peak_idx
    while onset > win_start:
        if abs(deriv[onset]) <= thresh:
            break
        onset -= 1

    return onset


def detect_MO_onset(scg, mo_peak_idx, fs, thresh_fraction=0.01, max_lookback_ms=40):
    
    ### Définition du Docstring ###

    """
    On détecte l'onset de l'onde MO en remontant depuis le pic MO
    jusqu'à ce que la dérivée repasse sous le seuil adaptatif.
    
    """
    deriv = np.gradient(scg)
    max_lookback = int(max_lookback_ms / 1000 * fs)
    win_start = max(0, mo_peak_idx - max_lookback)

    deriv_window = deriv[win_start:mo_peak_idx]
    if len(deriv_window) == 0:
        return mo_peak_idx

    thresh = thresh_fraction * np.max(np.abs(deriv_window))

    onset = mo_peak_idx
    while onset > win_start:
        if abs(deriv[onset]) <= thresh:
            break
        onset -= 1

    return onset


def detect_scg_events(scg, r_peaks, fs, min_amp_ratio=0.05):
    
    ### Définition du Docstring ###
    
    """
    On Détecte les événements mécaniques SCG pour chaque beat cardiaque.

    Événements cherchés (délais après R-peak) :
        - MC pic + onset : 10–50 ms
        - AO pic + onset : 60–150 ms
        - AC pic + onset : 250–420 ms
        - MO pic + onset : 380–520 ms

    """

    MC_list = []
    MC_onset_list = []
    AO_list = []
    AO_onset_list = []
    AC_list = []
    AC_onset_list = []
    MO_list = []
    MO_onset_list = []

    for r in r_peaks:
        r_amp = np.abs(scg[r])

        ### MC : pic négatif 10–50 ms après R + onset ###
        mc_idx = None
        mc_onset_idx = None
        mc_start = r + int(0.010 * fs)
        mc_end = r + int(0.050 * fs)
        if mc_end < len(scg) and mc_start < mc_end:
            seg = scg[mc_start:mc_end]
            baseline = np.median(seg)
            peak_idx = np.argmin(seg)
            amp = np.abs(seg[peak_idx] - baseline)
            if amp > min_amp_ratio * r_amp:
                mc_idx = mc_start + peak_idx
                mc_onset_idx = detect_MC_onset(scg, mc_idx, fs)
        MC_list.append(mc_idx)
        MC_onset_list.append(mc_onset_idx)

        ### AO : pic positif 60–150 ms après R + onset ###
        ao_idx = None
        ao_onset_idx = None
        ao_start = r + int(0.060 * fs)
        ao_end = r + int(0.150 * fs)
        if ao_end < len(scg) and ao_start < ao_end:
            seg = scg[ao_start:ao_end]
            baseline = np.median(seg)
            peak_idx = np.argmax(seg)
            amp = np.abs(seg[peak_idx] - baseline)
            if amp > min_amp_ratio * r_amp:
                ao_idx = ao_start + peak_idx
                ao_onset_idx = detect_AO_onset(scg, ao_idx, fs)
        AO_list.append(ao_idx)
        AO_onset_list.append(ao_onset_idx)

        ### AC : pic négatif 250–420 ms après R + onset ###
        ac_idx       = None
        ac_onset_idx = None
        ac_start = r + int(0.250 * fs)
        ac_end = r + int(0.420 * fs)
        if ac_end < len(scg) and ac_start < ac_end:
            seg = scg[ac_start:ac_end]
            baseline = np.median(seg)
            peak_idx = np.argmin(seg)
            amp = np.abs(seg[peak_idx] - baseline)
            if amp > min_amp_ratio * r_amp:
                ac_idx       = ac_start + peak_idx
                ac_onset_idx = detect_AC_onset(scg, ac_idx, fs)
        AC_list.append(ac_idx)
        AC_onset_list.append(ac_onset_idx)

        ### MO : pic positif 380–520 ms après R + onset ###
        mo_idx = None
        mo_onset_idx = None
        mo_start = r + int(0.380 * fs)
        mo_end = r + int(0.520 * fs)
        if mo_end < len(scg) and mo_start < mo_end:
            seg = scg[mo_start:mo_end]
            baseline = np.median(seg)
            peak_idx = np.argmax(seg)
            amp = np.abs(seg[peak_idx] - baseline)
            if amp > min_amp_ratio * r_amp:
                mo_idx       = mo_start + peak_idx
                mo_onset_idx = detect_MO_onset(scg, mo_idx, fs)
        MO_list.append(mo_idx)
        MO_onset_list.append(mo_onset_idx)

        ### On vérifie la séquence temporelle : MC < AO < AC < MO ###
        if mc_idx is not None and ao_idx is not None and not (mc_idx < ao_idx):
            mc_idx = None
            mc_onset_idx = None
        if ao_idx is not None and ac_idx is not None and not (ao_idx < ac_idx):
            ao_idx = None
            ao_onset_idx = None
            ac_idx = None
            ac_onset_idx = None
        if ac_idx is not None and mo_idx is not None and not (ac_idx < mo_idx):
            mo_idx = None
            mo_onset_idx = None

        MC_list[-1] = mc_idx
        MC_onset_list[-1] = mc_onset_idx
        AO_list[-1] = ao_idx
        AO_onset_list[-1] = ao_onset_idx
        AC_list[-1] = ac_idx
        AC_onset_list[-1] = ac_onset_idx
        MO_list[-1] = mo_idx
        MO_onset_list[-1] = mo_onset_idx

    return {
        "MC": MC_list,
        "MC_onset": MC_onset_list,
        "AO": AO_list,
        "AO_onset": AO_onset_list,
        "AC": AC_list,
        "AC_onset": AC_onset_list,
        "MO": MO_list,
        "MO_onset": MO_onset_list,
    }


def compute_scg_intervals(scg_events, r_peaks, fs):
    
    ### Définition du Docstring ###

    """
    On Calcule les intervalles mécaniques à partir des événements SCG.

    Intervalles calculés :
        - PEP  = AO_onset - R          (Pre-Ejection Period)
        - ET   = AC_onset - AO_onset   (Ejection Time)
        - IVCT = AO_onset - MC_onset   (Isovolumic Contraction Time)
        - IVRT = MO_onset - AC_onset   (Isovolumic Relaxation Time)

    """
    PEP_list = []
    ET_list = []
    IVCT_list = []
    IVRT_list = []

    MC_onset_list = scg_events["MC_onset"]
    AO_onset_list = scg_events["AO_onset"]
    AC_onset_list = scg_events["AC_onset"]
    MO_onset_list = scg_events["MO_onset"]

    for i, r in enumerate(r_peaks):
        mc_onset = MC_onset_list[i]
        ao_onset = AO_onset_list[i]
        ac_onset = AC_onset_list[i]
        mo_onset = MO_onset_list[i]

        ### PEP = AO_onset - R ###
        if ao_onset is not None and ao_onset > r:
            pep = (ao_onset - r) / fs * 1000
            if 40 < pep < 200:
                PEP_list.append(pep)

        ### ET = AC_onset - AO_onset ###
        if ao_onset is not None and ac_onset is not None and ac_onset > ao_onset:
            et = (ac_onset - ao_onset) / fs * 1000
            if 150 < et < 450:
                ET_list.append(et)

        ### IVCT = AO_onset - MC_onset ###
        if mc_onset is not None and ao_onset is not None and ao_onset > mc_onset:
            ivct = (ao_onset - mc_onset) / fs * 1000
            if 20 < ivct < 150:
                IVCT_list.append(ivct)

        ### IVRT = MO_onset - AC_onset ###
        if ac_onset is not None and mo_onset is not None and mo_onset > ac_onset:
            ivrt = (mo_onset - ac_onset) / fs * 1000
            if 30 < ivrt < 200:
                IVRT_list.append(ivrt)

    return {
        "PEP":  np.array(PEP_list),
        "ET":   np.array(ET_list),
        "IVCT": np.array(IVCT_list),
        "IVRT": np.array(IVRT_list),
    }

### MAIN ###

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
        ecg_segment = data["ECG_clean"]
        scg_segment = data["SCG_clean"]
        t_segment = data["time"]
        patient = data["patient"]

        ## On sélectionne le segment ###
        start_idx = int(start_time * fs)
        end_idx = min(start_idx + int(window_s * fs), len(ecg_segment))
        ecg_segment = ecg_segment[start_idx:end_idx]
        scg_segment = scg_segment[start_idx:end_idx]
        t_segment = t_segment[start_idx:end_idx]

        ### Détection et suppression des spikes de pacing ###
        print(f"Segment : {t_segment[0]:.1f}–{t_segment[-1]:.1f} s")

        spike_indices = detect_pacing_spikes(ecg_segment, fs)
        
        if len(spike_indices) > 0:
            print(f"{len(spike_indices)} spike(s) de pacing détectés, suppression")
            ecg_nospike = remove_pacing_spikes(ecg_segment, spike_indices, fs)
        else:
            print("Aucun spike de pacing détecté")
            ecg_nospike = ecg_segment

        ### Détection des pics R sur le signal sans spikes, avec méthode hybride biosppy + find_peaks ###
        r_peaks, rpeaks_source = detect_rpeaks_hybrid(ecg_nospike, fs)

        if len(r_peaks) < 4:
            print(f"Seulement {len(r_peaks)} R-peaks, segment trop court ou bruité")
            continue

        RR_all  = np.diff(r_peaks) / fs
        FC_mean = np.mean(60 / RR_all)
        print(f"R-peaks : {len(r_peaks)} ({rpeaks_source})  |  FC={FC_mean:.1f} bpm")

        ### On détecte les événements mécaniques SCG à partir des R-peaks ###
        scg_events = detect_scg_events(scg_segment, r_peaks, fs)

        MC_valid       = [x for x in scg_events["MC"]       if x is not None]
        MC_onset_valid = [x for x in scg_events["MC_onset"] if x is not None]
        AO_valid       = [x for x in scg_events["AO"]       if x is not None]
        AO_onset_valid = [x for x in scg_events["AO_onset"] if x is not None]
        AC_valid       = [x for x in scg_events["AC"]       if x is not None]
        AC_onset_valid = [x for x in scg_events["AC_onset"] if x is not None]
        MO_valid       = [x for x in scg_events["MO"]       if x is not None]
        MO_onset_valid = [x for x in scg_events["MO_onset"] if x is not None]

        ### On calcule les intervalles mécaniques à partir des événements détectés ### 
        intervals = compute_scg_intervals(scg_events, r_peaks, fs)

        print(f"\nIntervalles SCG : ")
        for name, vals in intervals.items():
            if len(vals) > 0:
                vals_clean = vals[filter_iqr(vals)]
                print(f"{name:5s} : {np.median(vals_clean):.1f} ms  "
                      f"(mean={np.mean(vals_clean):.1f}  std={np.std(vals_clean):.1f})")
            else:
                print(f"{name:5s} : non détecté")

        ### Interprétation clinique basée sur les intervalles calculés ###
        if len(intervals["PEP"]) > 0:
            pep_clean = intervals["PEP"][filter_iqr(intervals["PEP"])]
            pep_med = np.median(pep_clean)
            
            if pep_med > 120:
                print(f"\nPEP long ({pep_med:.0f} ms) contractilité réduite")
            
            elif pep_med < 60:
                print(f"\nPEP court ({pep_med:.0f} ms)")

        if len(intervals["ET"]) > 0:
            et_clean = intervals["ET"][filter_iqr(intervals["ET"])]
            et_med = np.median(et_clean)
            
            if et_med < 200:
                print(f"\nET court ({et_med:.0f} ms) éjection réduite")
            
            elif et_med > 350:
                print(f"\nET long ({et_med:.0f} ms)")

        if len(intervals["IVCT"]) > 0:
            ivct_clean = intervals["IVCT"][filter_iqr(intervals["IVCT"])]
            ivct_med   = np.median(ivct_clean)
            
            if ivct_med > 110:
                print(f"\nIVCT long ({ivct_med:.0f} ms) dysfonction systolique / contraction inefficace")
            
            elif ivct_med < 40:
                print(f"\nIVCT court ({ivct_med:.0f} ms) tachycardie possible")

        if len(intervals["IVRT"]) > 0:
            ivrt_clean = intervals["IVRT"][filter_iqr(intervals["IVRT"])]
            ivrt_med   = np.median(ivrt_clean)
            
            if ivrt_med > 120:
                print(f"\nIVRT long ({ivrt_med:.0f} ms) dysfonction diastolique")
            
            elif ivrt_med < 50:
                print(f"\nIVRT court ({ivrt_med:.0f} ms)")

        ### Visualisation du segment ECG + SCG avec annotations des événements détectés ###
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        fig.suptitle(f"ECG + SCG — {fname}  |  patient={patient}  |  FC={FC_mean:.0f} bpm",
                     fontsize=12)

        ### Axe 1 : ECG + R-peaks ###
        ax1.plot(t_segment, ecg_nospike, color="black", linewidth=1.0, label="ECG")
        ax1.scatter(t_segment[r_peaks], ecg_nospike[r_peaks],
                    color="red", s=35, zorder=5, label="R")
        for r in r_peaks:
            ax1.axvline(x=t_segment[r], color="red", alpha=0.15, linewidth=0.7)
        ax1.set_ylabel("ECG (mV)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.3)

        ### Axe 2 : SCG + événements ###
        ax2.plot(t_segment, scg_segment, color="steelblue", linewidth=0.9, label="SCG")

        for r in r_peaks:
            ax2.axvline(x=t_segment[r], color="red", alpha=0.15, linewidth=0.7)

        if len(MC_valid) > 0:
            ax2.scatter(t_segment[MC_valid], scg_segment[MC_valid],
                        color="green", s=40, zorder=5, label="MC pic")
        if len(MC_onset_valid) > 0:
            ax2.scatter(t_segment[MC_onset_valid], scg_segment[MC_onset_valid],
                        color="green", marker="x", s=80, linewidths=2.0,
                        zorder=6, label="MC onset (IVCT start)")

        if len(AO_valid) > 0:
            ax2.scatter(t_segment[AO_valid], scg_segment[AO_valid],
                        color="orange", s=40, zorder=5, label="AO pic")
        if len(AO_onset_valid) > 0:
            ax2.scatter(t_segment[AO_onset_valid], scg_segment[AO_onset_valid],
                        color="orange", marker="x", s=80, linewidths=2.0,
                        zorder=6, label="AO onset (PEP)")

        if len(AC_valid) > 0:
            ax2.scatter(t_segment[AC_valid], scg_segment[AC_valid],
                        color="red", s=40, zorder=5, label="AC pic")
        if len(AC_onset_valid) > 0:
            ax2.scatter(t_segment[AC_onset_valid], scg_segment[AC_onset_valid],
                        color="red", marker="x", s=80, linewidths=2.0,
                        zorder=6, label="AC onset (ET end)")

        if len(MO_valid) > 0:
            ax2.scatter(t_segment[MO_valid], scg_segment[MO_valid],
                        color="purple", s=40, zorder=5, label="MO pic")
        if len(MO_onset_valid) > 0:
            ax2.scatter(t_segment[MO_onset_valid], scg_segment[MO_onset_valid],
                        color="purple", marker="x", s=80, linewidths=2.0,
                        zorder=6, label="MO onset (IVRT start)")

        ax2.set_ylabel("SCG (g)")
        ax2.set_xlabel("Temps (s)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()