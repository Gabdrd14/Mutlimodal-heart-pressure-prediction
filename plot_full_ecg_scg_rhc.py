import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io as sio
from pressure_collector import RHCP_Pipeline

# ==============================
# FUNCTIONS
# ==============================

def load_mat_file(path):
    """
    Charge un fichier .mat et récupère ECG, SCG et RHC automatiquement
    Utilise des signaux de fallback si les principaux ne sont pas disponibles
    """
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print("Keys in .mat:", keys)

    # S'assurer que ce sont des vecteurs 1D
    def squeeze_sig(sig):
        if sig is None:
            return None
        return np.ravel(sig)

    # ========== ECG ==========
    ecg_raw = mat.get('ecg_raw', None)
    ecg_clean = mat.get('ecg_clean', None)
    
    # Fallback: use ECG_lead_II if available
    if ecg_raw is None and 'ECG_lead_II' in mat:
        ecg_raw = mat.get('ECG_lead_II', None)
        print(f"  → Using ECG_lead_II as fallback for ecg_raw")
    
    # If still no ECG, try first available ECG lead
    if ecg_raw is None:
        for lead in ['ECG_lead_I', 'ECG_lead_III', 'aVR', 'aVL', 'aVF', 
                      'ECG_lead_V1', 'ECG_lead_V2', 'ECG_lead_V3', 
                      'ECG_lead_V4', 'ECG_lead_V5', 'ECG_lead_V6']:
            if lead in mat:
                ecg_raw = mat.get(lead, None)
                print(f"  → Using {lead} as fallback for ecg_raw")
                break
    
    # ========== SCG ==========
    scg_raw = mat.get('scg_raw', None)
    scg_clean = mat.get('scg_clean', None)
    
    # Fallback: if no SCG, duplicate ECG (not ideal but better than nothing)
    # if scg_raw is None and ecg_raw is not None:
    #     scg_raw = ecg_raw.copy()
    #     print(f"  → Using ECG as fallback for scg_raw (no accelerometer data)")
    
    # ========== RHC ==========
    rhc_raw = mat.get('rhc_raw', None)
    
    # ========== TIME ==========
    time = mat.get('time', None)

    ecg_raw = squeeze_sig(ecg_raw)
    scg_raw = squeeze_sig(scg_raw)
    ecg_clean = squeeze_sig(ecg_clean) 
    scg_clean = squeeze_sig(scg_clean)
    rhc_raw = squeeze_sig(rhc_raw)
    time = squeeze_sig(time)

    return {
        "ECG": ecg_raw,
        "SCG": scg_raw,
        "ECG_clean": ecg_clean,
        "SCG_clean": scg_clean,
        "RHC": rhc_raw,
        "time": time
    }

def resample_signal(sig, sig_fs, target_fs):
    """Rééchantillonne un signal à la fréquence cible."""
    if sig is None or len(sig) == 0:
        return None
    t_orig = np.arange(len(sig)) / sig_fs
    t_new = np.arange(0, len(sig)/sig_fs, 1/target_fs)
    f = interp1d(t_orig, sig, kind='linear', fill_value="extrapolate")
    return f(t_new)

def plot_full_ecg_scg_rhc(fname, ecg_raw, ecg_clean, scg_raw, scg_clean, rhc_signal,
                          ecg_fs=1000, scg_fs=500, rhc_fs=250, ecg_magnification=1):
    """
    Plot the ENTIRE signal file with ECG, SCG, and RHC overlaid on the same time scale
    """
    
    # Resample SCG -> ECG fs
    scg_raw_rs   = resample_signal(scg_raw, scg_fs, ecg_fs)
    scg_clean_rs = resample_signal(scg_clean, scg_fs, ecg_fs)

    # Resample RHC -> ECG fs
    rhc_rs = resample_signal(rhc_signal, rhc_fs, ecg_fs)

    # Use the full length
    min_len = min(len(ecg_raw), len(scg_raw_rs) if scg_raw_rs is not None else len(ecg_raw),
                  len(rhc_rs) if rhc_rs is not None else len(ecg_raw))
    
    ecg_raw_full   = ecg_raw[:min_len]
    ecg_clean_full = ecg_clean[:min_len] if ecg_clean is not None else None
    scg_raw_full   = scg_raw_rs[:min_len] if scg_raw_rs is not None else None
    scg_clean_full = scg_clean_rs[:min_len] if scg_clean_rs is not None else None
    rhc_full = rhc_rs[:min_len] if rhc_rs is not None else None

    t = np.arange(len(ecg_raw_full)) / ecg_fs

    fig, axs = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # ECG
    axs[0].plot(t, ecg_raw_full * ecg_magnification, label="Raw ECG", alpha=0.6, linewidth=0.5)
    if ecg_clean_full is not None:
        axs[0].plot(t, ecg_clean_full * ecg_magnification, label="Clean ECG", alpha=0.9, linewidth=0.5)
    axs[0].set_title(f"{fname} - ECG (Full Signal)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # SCG
    if scg_raw_full is not None:
        axs[1].plot(t, scg_raw_full, label="Raw SCG", alpha=0.6, linewidth=0.5)
    if scg_clean_full is not None:
        axs[1].plot(t, scg_clean_full, label="Clean SCG", alpha=0.9, linewidth=0.5)
    axs[1].set_title(f"{fname} - SCG (Full Signal)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # RHC
    if rhc_full is not None:
        axs[2].plot(t, rhc_full, label="RHC Pressure", color="darkred", linewidth=0.5)
        axs[2].set_title(f"{fname} - Right Heart Catheter Pressure (Full Signal)")
        axs[2].set_ylabel("Pressure")
        axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[2].set_xlabel("Time (s)")
    
    duration_min = len(ecg_raw_full) / ecg_fs / 60
    fig.suptitle(f"{fname} - Full Signal Plot ({duration_min:.1f} minutes)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    INPUT_FOLDER = "processed"  # dossier contenant les fichiers .mat
    
    DEFAULT_ECG_FS = 500  # WFDB data is 500 Hz
    DEFAULT_SCG_FS = 500
    DEFAULT_RHC_FS = 500

    for fname in sorted(os.listdir(INPUT_FOLDER)):
        path = os.path.join(INPUT_FOLDER, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue

        try:
            print(f"\n{'='*60}")
            print(f"Plotting {fname}")
            print(f"{'='*60}")
            
            # Load all data from .mat file
            data = load_mat_file(path)

            ecg_raw = data["ECG"]
            scg_raw = data["SCG"]
            ecg_clean = data["ECG_clean"]
            scg_clean = data["SCG_clean"]
            value_rhc = data["RHC"]
            
            if ecg_raw is None or value_rhc is None:
                print(f"Signal manquant dans {fname}, skipping.")
                continue

            print(f"✓ Plotting full signal for {fname}...")
            plot_full_ecg_scg_rhc(
                fname,
                ecg_raw,
                ecg_clean,
                scg_raw,
                scg_clean,
                value_rhc,
                ecg_fs=DEFAULT_ECG_FS,
                scg_fs=DEFAULT_SCG_FS,
                rhc_fs=DEFAULT_RHC_FS,
                ecg_magnification=1
            )

        except Exception as e:
            print(f"Erreur sur {fname}: {e}")
            import traceback
            traceback.print_exc()
