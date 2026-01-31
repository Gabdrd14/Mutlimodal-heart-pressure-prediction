import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.io as sio
from  pressure_collector import RHCP_Pipeline
# ==============================
# FUNCTIONS
# ==============================

def load_mat_file(path):
    """
    Charge un fichier .mat et récupère ECG et SCG automatiquement
    même si la structure est simple (variables directes).
    """
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    print("Keys in .mat:", keys)

    # Cherche les signaux connus
    ecg_raw = mat.get('ecg_raw', None)
    scg_raw = mat.get('scg_raw', None)
    ecg_clean = mat.get('ecg_clean', None)
    scg_clean = mat.get('scg_clean', None)
    time = mat.get('time', None)

    # S'assurer que ce sont des vecteurs 1D
    def squeeze_sig(sig):
        if sig is None:
            return None
        return np.ravel(sig)

    ecg_raw = squeeze_sig(ecg_raw)
    scg_raw = squeeze_sig(scg_raw)
    ecg_clean = squeeze_sig(ecg_clean) 
    scg_clean = squeeze_sig(scg_clean) 
    time = squeeze_sig(time)

    return {
        "ECG": ecg_raw,
        "SCG": scg_raw,
        "ECG_clean": ecg_clean * 8,
        "SCG_clean": scg_clean * 8,
        "time": time
    }

def resample_signal(sig, sig_fs, target_fs):
    """Rééchantillonne un signal à la fréquence cible."""
    t_orig = np.arange(len(sig)) / sig_fs
    t_new = np.arange(0, len(sig)/sig_fs, 1/target_fs)
    f = interp1d(t_orig, sig, kind='linear', fill_value="extrapolate")
    return f(t_new)

def plot_ecg_scg(ecg_raw, ecg_clean, scg_raw, scg_clean,
                 ecg_fs=1000, scg_fs=500, start_time=0, window_s=30):
    """Trace ECG et SCG synchronisés."""
    # Rééchantillonne SCG
    scg_raw_rs = resample_signal(scg_raw, scg_fs, ecg_fs)
    scg_clean_rs = resample_signal(scg_clean, scg_fs, ecg_fs)

    start_idx = int(start_time * ecg_fs)
    end_idx = start_idx + int(window_s * ecg_fs)

    ecg_raw_win = ecg_raw[start_idx:end_idx]
    ecg_clean_win = ecg_clean[start_idx:end_idx]
    scg_raw_win = scg_raw_rs[start_idx:end_idx]
    scg_clean_win = scg_clean_rs[start_idx:end_idx]

    t = np.arange(len(ecg_raw_win)) / ecg_fs + start_time

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axs[0].plot(t, ecg_raw_win, label="Raw ECG", alpha=0.6)
    axs[0].plot(t, ecg_clean_win, label="Clean ECG", alpha=0.8)
    axs[0].set_title(f"ECG ({window_s}s window)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, scg_raw_win, label="Raw SCG", alpha=0.6)
    axs[1].plot(t, scg_clean_win, label="Clean SCG", alpha=0.8)
    axs[1].set_title(f"SCG ({window_s}s window)")
    axs[1].legend()
    axs[1].grid(True)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    INPUT_FOLDER = "processed"  # dossier contenant les fichiers
    DAT_FOLDER  = 'dat_signals'
    
    start_time = 820
    window_s = 30
    DEFAULT_ECG_FS = 1000
    DEFAULT_SCG_FS = 500

    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        if not os.path.isfile(path) or not fname.lower().endswith(".mat"):
            continue

        try:
            
            rhc_pip = RHCP_Pipeline(f"{DAT_FOLDER}/{fname.removesuffix('.mat').replace('.','-')}")
            
            a = rhc_pip.run()
            value_rhc_  = a["RHC_pressure"] 
            
            
            
            data = load_mat_file(path)
            ecg_raw = data["ECG"]
            scg_raw = data["SCG"]
            t = data["time"]
            ecg_clean = data['ECG_clean']
            scg_clean = data['SCG_clean']
            
            
            if ecg_raw is None or scg_raw is None:
                print(f"Signal manquant dans {fname}, skipping.")
                continue

            print(f"Plotting {fname} ...")
            plot_ecg_scg(ecg_raw, ecg_clean, scg_raw, scg_clean,
                         ecg_fs=DEFAULT_ECG_FS, scg_fs=DEFAULT_SCG_FS,
                         start_time=start_time, window_s=window_s)

        except Exception as e:
            print(f"Erreur sur {fname}: {e}")
