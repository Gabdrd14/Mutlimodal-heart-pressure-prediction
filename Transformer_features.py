"""
Features ECG et SCG pour le transformer
-----------------------------------------

Métadonnées
  patient          : identifiant du patient
  fs               : fréquence d'échantillonnage (Hz)
  window_start_s   : début de la fenêtre temporelle (s)
  window_end_s     : fin de la fenêtre temporelle (s)
  n_beats          : nombre de battements détectés (R-peaks)

--- Features scalaires ECG ---

  fc_mean_bpm      : fréquence cardiaque moyenne (bpm)
  fc_median_bpm    : fréquence cardiaque médiane (bpm)

  rr_mean_ms       : intervalle RR moyen (ms)
  rr_std_ms        : écart-type des intervalles RR (ms)

  pr_mean_ms       : intervalle PR moyen (ms)
  pr_median_ms     : intervalle PR médian (ms)
  pr_std_ms        : écart-type de l'intervalle PR (ms)

  qt_median_ms     : intervalle QT médian (ms)

  qtc_mean_ms      : QT corrigé (Fridericia) moyen (ms)
  qtc_median_ms    : QT corrigé (Fridericia) médian (ms)
  qtc_std_ms       : écart-type du QT corrigé (ms)

--- Features scalaires SCG ---

  pep_mean_ms      : période de pré-éjection moyenne (ms)
  pep_median_ms    : période de pré-éjection médiane (ms)
  pep_std_ms       : écart-type de la période de pré-éjection (ms)

  et_mean_ms       : temps d'éjection moyen (ms)
  et_median_ms     : temps d'éjection médian (ms)
  et_std_ms        : écart-type du temps d'éjection (ms)

  ivct_mean_ms     : temps de contraction isovolumique moyen (ms)
  ivct_median_ms   : temps de contraction isovolumique médian (ms)
  ivct_std_ms      : écart-type du temps de contraction isovolumique (ms)

  ivrt_mean_ms     : temps de relaxation isovolumique moyen (ms)
  ivrt_median_ms   : temps de relaxation isovolumique médian (ms)
  ivrt_std_ms      : écart-type du temps de relaxation isovolumique (ms)

--- Séries battement par battement ---
(vecteurs de longueur n_beats, NaN si non détecté)

ECG beat-by-beat

  rr_bb_ms         : intervalle RR par battement (ms)
  pr_bb_ms         : intervalle PR par battement (ms)
  qt_bb_ms         : intervalle QT brut par battement (ms)
  qtc_bb_ms        : QT corrigé (Fridericia) par battement (ms)

SCG beat-by-beat

  pep_bb_ms        : période de pré-éjection par battement (ms)
  et_bb_ms         : temps d'éjection par battement (ms)
  ivct_bb_ms       : temps de contraction isovolumique par battement (ms)
  ivrt_bb_ms       : temps de relaxation isovolumique par battement (ms)
"""

import os
import argparse
import numpy as np
import scipy.io

from ecg_segmentation_2 import (
    detect_pacing_spikes,
    remove_pacing_spikes,
    detect_rpeaks_hybrid,
    detect_peaks_ecg,
    detect_PR_interval,
    detect_QT_interval,
    filter_iqr,
)
from scg_segmentation import (
    detect_scg_events,
)


def _safe_stats(arr):
    
    ### Définition du Docstring ###

    """
    On applique un filtre IQR pour éviter que les outliers ne biaisent les scalaires.
    
    """
    
    if arr is None or len(arr) == 0:
        return np.nan, np.nan, np.nan
    mask = filter_iqr(arr)
    clean = arr[mask]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(clean)), float(np.mean(clean)), float(np.std(clean))


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

    time = np.linspace(
        window_start,
        window_end,
        n_samples
    )

    return {

        "patient": str(mat['patient'].squeeze()),
        "fs": fs,
        "window_start_s": window_start,
        "window_end_s": window_end,


        "time": time,

        ### Channels de base ###
        "ecg": mat['ecg'].squeeze(),
        "scg": mat['scg'].squeeze(),

        "ecg_raw": mat['ecg_raw'].squeeze(),
        "scg_raw": mat['scg_raw'].squeeze(),

        "rhc": mat['rhc'].squeeze(),

        "patch_ACC_lat": mat['patch_ACC_lat'].squeeze(),
        "patch_ACC_hf": mat['patch_ACC_hf'].squeeze(),
        "patch_ACC_dv": mat['patch_ACC_dv'].squeeze(),

        ### Métriques de qualité ###
        "quality_composite": float(mat['quality_composite'].squeeze()),
        "quality_ecg": float(mat['quality_ecg'].squeeze()),
        "quality_scg": float(mat['quality_scg'].squeeze()),
        "quality_rhc": float(mat['quality_rhc'].squeeze()),
    }


### Extraction features ECG battement par battement ###

def extract_ecg_features(ecg_segment, t_segment, r_peaks, fs):
    
    ### Définition du Docstring ##

    """
    On détecte les onsets P/Q/T autour de chaque R-peak, puis on calcule PR, QT et QTc par battement.

    """
    n_beats = len(r_peaks)

    RR_all    = np.diff(r_peaks) / fs * 1000 
    RR_median = np.median(RR_all) / 1000        

    ### lists battement par battement ###
    pr_bb = np.full(n_beats, np.nan)  
    qt_bb = np.full(n_beats, np.nan)   
    qtc_bb = np.full(n_beats, np.nan)   
    rr_bb = np.full(n_beats, np.nan)  
    rr_bb[1:] = RR_all                   
    PR_raw_list = []
    QT_raw_list = []

    for i, r in enumerate(r_peaks):
        rr_local_ms = (
            (r - r_peaks[i - 1]) / fs * 1000 if i > 0
            else (r_peaks[i + 1] - r) / fs * 1000 if i < n_beats - 1
            else np.mean(RR_all)
        )
        p_window = min(200, int(rr_local_ms * 0.35))
        t_window = min(500, int(rr_local_ms * 0.50))

        p = detect_peaks_ecg(ecg_segment, r, fs, window_ms=p_window, offset_ms=80,  name_peak="P")
        q = detect_peaks_ecg(ecg_segment, r, fs, window_ms=80,       offset_ms=10,  name_peak="Q")
        t = detect_peaks_ecg(ecg_segment, r, fs, window_ms=t_window, offset_ms=150, name_peak="T")

        if p is not None and (p >= r or (q is not None and p >= q)):
            p = None
        if q is not None and q >= r:
            q = None

        onset_q_pr = None
        if p is not None and q is not None:
            onset_p, onset_q_pr = detect_PR_interval(ecg_segment, p, q, fs)
            pr_ms = (t_segment[onset_q_pr] - t_segment[onset_p]) * 1000
            
            if 80 <= pr_ms <= 400:
                pr_bb[i] = pr_ms
                PR_raw_list.append(pr_ms)
        elif q is not None:
            onset_q_pr = q

        if q is not None and t is not None and onset_q_pr is not None:
            onset_q_qt, offset_t = detect_QT_interval(ecg_segment, onset_q_pr, t, fs)
            offset_t_clipped = min(offset_t, len(t_segment) - 1)
            qt_ms = (t_segment[offset_t_clipped] - t_segment[onset_q_qt]) * 1000
            
            ### Calcul du QTc avec correction de Fridericia ###
            rr_s = rr_bb[i] / 1000 if not np.isnan(rr_bb[i]) else RR_median
            rr_s = max(rr_s, 0.2)  
            qtc_ms = qt_ms / (rr_s ** (1 / 3))
            
            if 200 <= qt_ms <= 700:
                qt_bb[i]  = qt_ms
                qtc_bb[i] = qtc_ms
                QT_raw_list.append(qt_ms)

    ### Scalaires ###
    pr_arr = pr_bb[~np.isnan(pr_bb)]
    qt_arr = qt_bb[~np.isnan(qt_bb)]
    qtc_arr = qtc_bb[~np.isnan(qtc_bb)]
    rr_arr = rr_bb[~np.isnan(rr_bb)]

    pr_med,  pr_mean,  pr_std  = _safe_stats(pr_arr)
    qt_med,  qt_mean,  qt_std  = _safe_stats(qt_arr)
    qtc_med, qtc_mean, qtc_std = _safe_stats(qtc_arr)

    fc_mean = float(np.mean(60000 / rr_arr)) if len(rr_arr) > 0 else np.nan
    fc_median = float(np.median(60000 / rr_arr)) if len(rr_arr) > 0 else np.nan
    rr_mean = float(np.mean(rr_arr)) if len(rr_arr) > 0 else np.nan
    rr_std = float(np.std(rr_arr)) if len(rr_arr) > 0 else np.nan

    scalars = dict(
        fc_mean_bpm=fc_mean,   fc_median_bpm=fc_median,
        rr_mean_ms=rr_mean,    rr_std_ms=rr_std,
        pr_median_ms=pr_med,   pr_mean_ms=pr_mean,   pr_std_ms=pr_std,
        qt_median_ms=qt_med,   qt_mean_ms=qt_mean,   qt_std_ms=qt_std,
        qtc_median_ms=qtc_med, qtc_mean_ms=qtc_mean, qtc_std_ms=qtc_std,
    )

    
    ### Battement par battement ###

    beat_by_beat = dict(
        rr_bb_ms=rr_bb, pr_bb_ms=pr_bb, qt_bb_ms=qt_bb, qtc_bb_ms=qtc_bb,
    )
    
    return scalars, beat_by_beat



### Extraction features SCG battement par battement ###

def extract_scg_features(scg_segment, r_peaks, fs):
    
    ### Définition du Docstring ###
    
    """
    On détecte les événements SCG (AO, AC, MC, MO) autour de chaque R-peak, puis on calcule PEP, ET, IVCT et IVRT par battement.
    
    """
    
    n_beats = len(r_peaks)
    scg_events = detect_scg_events(scg_segment, r_peaks, fs)

    ### Battement par battement (aligné sur r_peaks) ###
    pep_bb = np.full(n_beats, np.nan)
    et_bb = np.full(n_beats, np.nan)
    ivct_bb = np.full(n_beats, np.nan)
    ivrt_bb = np.full(n_beats, np.nan)

    AO_onset = scg_events["AO_onset"]
    AC_onset = scg_events["AC_onset"]
    MC_onset = scg_events["MC_onset"]
    MO_onset = scg_events["MO_onset"]

    for i, r in enumerate(r_peaks):
        ao = AO_onset[i]
        ac = AC_onset[i]
        mc = MC_onset[i]
        mo = MO_onset[i]

        if ao is not None and ao > r:
            pep = (ao - r) / fs * 1000
            if 40 < pep < 200:
                pep_bb[i] = pep

        if ao is not None and ac is not None and ac > ao:
            et = (ac - ao) / fs * 1000
            if 150 < et < 450:
                et_bb[i] = et

        if mc is not None and ao is not None and ao > mc:
            ivct = (ao - mc) / fs * 1000
            if 20 < ivct < 150:
                ivct_bb[i] = ivct

        if ac is not None and mo is not None and mo > ac:
            ivrt = (mo - ac) / fs * 1000
            if 30 < ivrt < 200:
                ivrt_bb[i] = ivrt

    ### Scalaires ###
    pep_med,  pep_mean,  pep_std  = _safe_stats(pep_bb[~np.isnan(pep_bb)])
    et_med,   et_mean,   et_std   = _safe_stats(et_bb[~np.isnan(et_bb)])
    ivct_med, ivct_mean, ivct_std = _safe_stats(ivct_bb[~np.isnan(ivct_bb)])
    ivrt_med, ivrt_mean, ivrt_std = _safe_stats(ivrt_bb[~np.isnan(ivrt_bb)])

    scalars = dict(
        pep_median_ms=pep_med,   pep_mean_ms=pep_mean,   pep_std_ms=pep_std,
        et_median_ms=et_med,     et_mean_ms=et_mean,     et_std_ms=et_std,
        ivct_median_ms=ivct_med, ivct_mean_ms=ivct_mean, ivct_std_ms=ivct_std,
        ivrt_median_ms=ivrt_med, ivrt_mean_ms=ivrt_mean, ivrt_std_ms=ivrt_std,
    )


    ### Battement par battement ###
    beat_by_beat = dict(
        pep_bb_ms=pep_bb, et_bb_ms=et_bb, ivct_bb_ms=ivct_bb, ivrt_bb_ms=ivrt_bb,
    )
    return scalars, beat_by_beat



def process_patient(path, output_folder, start_time=0, window_s=30):
    
    ### Définition du Docstring ###
    
    """
    On traite un fichier .mat patient et on sauvegarde le .mat avec les features extraites.
    
    """
    fname = os.path.basename(path)
    print(f"\n{'='*60}")
    print(f"Processing {fname} ...")

    data = load_mat_file(path)

    fs = data["fs"]
    patient = data["patient"]
    time_full = data["time"]

    ecg_full = data["ecg"]
    scg_full = data["scg"]

    start_idx = int(start_time * fs)
    end_idx = min(start_idx + int(window_s * fs), len(ecg_full))
    ecg_segment = ecg_full[start_idx:end_idx]
    scg_segment = scg_full[start_idx:end_idx]
    t_segment = time_full[start_idx:end_idx]

    print(f"Segment : {t_segment[0]:.1f}–{t_segment[-1]:.1f} s")

    spike_indices = detect_pacing_spikes(ecg_segment, fs)
    ecg_nospike = (remove_pacing_spikes(ecg_segment, spike_indices, fs)
                     if len(spike_indices) > 0 else ecg_segment)
    
    if len(spike_indices) > 0:
        print(f"{len(spike_indices)} spike(s) de pacing détectés, suppression")

    r_peaks, source = detect_rpeaks_hybrid(ecg_nospike, fs)
    if len(r_peaks) < 4:
        print(f"Seulement {len(r_peaks)} R-peaks, segment trop court ou bruité")
        return None

    n_beats = len(r_peaks)
    print(f"R-peaks : {n_beats} ({source})")

    ### Extraction ECG ###
    ecg_scalars, ecg_bb = extract_ecg_features(ecg_nospike, t_segment, r_peaks, fs)
    print(f"ECG FC={ecg_scalars['fc_mean_bpm']:.1f} bpm  "
          f"PR={ecg_scalars['pr_median_ms']:.0f} ms  "
          f"QTc={ecg_scalars['qtc_median_ms']:.0f} ms")

    ### Extraction SCG ###
    scg_scalars, scg_bb = extract_scg_features(scg_segment, r_peaks, fs)
    print(f"SCG PEP={scg_scalars['pep_median_ms']:.0f} ms  "
          f"ET={scg_scalars['et_median_ms']:.0f} ms  "
          f"IVCT={scg_scalars['ivct_median_ms']:.0f} ms  "
          f"IVRT={scg_scalars['ivrt_median_ms']:.0f} ms")

    
    ### Assemblage du .mat de sortie ###
    mat_out = {
        "patient":        patient,
        "fs":             float(fs),
        "window_start_s": float(t_segment[0]),
        "window_end_s":   float(t_segment[-1]),
        "n_beats":        float(n_beats),
    }

    ### Features ECG / SCG (scalaires) ###
    mat_out.update({k: np.array([[v]]) for k, v in ecg_scalars.items()})
    mat_out.update({k: np.array([[v]]) for k, v in scg_scalars.items()})

    
    ### Features battement par battement ###
    mat_out.update({k: v.reshape(1, -1) for k, v in ecg_bb.items()})
    mat_out.update({k: v.reshape(1, -1) for k, v in scg_bb.items()})


    ### Signaux ### 
    mat_out["ecg"] = ecg_nospike.reshape(1, -1)
    mat_out["scg"] = scg_segment.reshape(1, -1)
    mat_out["time"] = t_segment.reshape(1, -1)

    ### On conserve les channels de base ###
    for key, value in data.items():
        if key in mat_out:
            continue

        if isinstance(value, np.ndarray):
            mat_out[key] = value
        else:
            mat_out[key] = np.array(value)


    ### Sauvegarde ###
    os.makedirs(output_folder, exist_ok=True)
    out_name = fname.replace(".mat", "_features.mat")
    out_path = os.path.join(output_folder, out_name)
    scipy.io.savemat(out_path, mat_out)
    print(f"Sauvegardé : {out_path}")
    return out_path


### Main ###
def main():
    parser = argparse.ArgumentParser(
        description="On génère un .mat de features ECG+SCG par patient pour le transformer RHC."
    )
    parser.add_argument("--input",  default="processed/Corrected_DATA_V2",
                        help="Dossier contenant les .mat patients bruts")
    parser.add_argument("--output", default="processed/features_rhc",
                        help="Dossier de sortie pour les .mat de features")
    parser.add_argument("--window", type=float, default=30,
                        help="Durée du segment analysé en secondes (défaut : 30)")
    parser.add_argument("--start",  type=float, default=0,
                        help="Début du segment en secondes (défaut : 0)")
    args = parser.parse_args()

    mat_files = [
        f for f in os.listdir(args.input)
        if f.lower().endswith(".mat") and os.path.isfile(os.path.join(args.input, f))
    ]
    print(f"Fichiers trouvés : {len(mat_files)}")

    ok, skip = 0, 0
    for fname in sorted(mat_files):
        path = os.path.join(args.input, fname)
        result = process_patient(path, args.output,
                                 start_time=args.start, window_s=args.window)
        if result:
            ok += 1
        else:
            skip += 1

    print("===============================================================")
    print(f"Terminé {ok} patient(s) traités, {skip} sauté(s).")
    print(f"Features sauvegardées dans : {args.output}/")


if __name__ == "__main__":
    main()