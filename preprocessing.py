import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.interpolate import interp1d
from scipy.signal import lfilter
import scipy.io 
import matplotlib.pyplot as plt
import pywt



# ==============================
# 1. DATA LOADER
# ==============================

class DataLoader_preprocess_file:

    def __init__(self, record_path):
        self.record_path = record_path

    def load(self):
        record = wfdb.rdrecord(self.record_path)
        signals = record.p_signal
        names = record.sig_name
        return dict(zip(names, signals.T))
    

class Dataloader_raw_file:
    def __init__(self, raw_path):
        self.raw_path = raw_path

    def load(self):
        mat = scipy.io.loadmat(self.raw_path, simplify_cells=True, verify_compressed_data_integrity=False)
        data = mat['data']

        signals_dict = {
            "patch_ECG": np.ravel(data['E_data']),       # ECG
            "patch_ACC_lat": np.ravel(data['A_data_x']), # SCG / acc X
            "patch_ACC_hf": np.ravel(data['A_data_y']),  # SCG / acc Y
            "patch_ACC_dv": np.ravel(data['A_data_z']),  # SCG / acc Z
            "patch_pressure": np.ravel(data['B_pres']),  # Pressure
            "patch_temp": np.ravel(data['B_temp']),      # Temperature
            "patch_humi": np.ravel(data['B_humi'])       # Humidity
        }

        return signals_dict




# ==============================
# 2. ARTIFACT REMOVAL
# ==============================

class ArtifactCleaner:
    def __init__(self, fs=500):
        self.fs = fs

    # ---- Filters ----

    def highpass(self, sig, cutoff=0.5):

        """
        Docstring pour highpass

        filtre passe haut du signal input
        
        :param self: Description
        :param sig: signal
        :param cutoff: filtre du signal
        """

        b, a = butter(2, cutoff/(self.fs/2), btype="high")
        return filtfilt(b, a, sig)

    def bandpass(self, sig, low, high):

        """
        Docstring pour bandpass
        
        filtre passe bas du signal input

        :param self: Description
        :param sig: signal
        :param low: limite basse
        :param high: limite haute
        """


        b, a = butter(4, [low/(self.fs/2), high/(self.fs/2)], btype="band")
        return filtfilt(b, a, sig)

    def swt_filter(self,sig,wavelet = "db4",level = 2 , method = 'soft'):

        """
        Docstring pour swt_filter

        filtre utilisant la methode SWT (Stationary Wavelet Transformation)
        
        :param self: Description
        :param sig: signal 
        :param wavelet: type ondelette
        :param level: niveau de decompostion
        :param method: seuillage des coefficients

        """
        if len(sig) % 2 != 0:
            sig = sig[:-1]  # ou np.append(sig, sig[-1]) pour ne rien perdre

        coeffs= pywt.swt(sig,wavelet,level=level)
        threshold = lambda c , sigma: pywt.threshold(c, sigma * np.sqrt(2*np.log(len(c))), mode= method)
        new_coeffs =  []

        for approx , detail in coeffs :

            sigma = np.median(np.abs(detail)) / 0.6745
            detail_t = threshold(detail ,sigma)
            new_coeffs.append((approx , detail_t))

        clean = pywt.iswt(new_coeffs, wavelet)

        return clean



    def adaptive_filter(self , sig, noise, mu=0.01, order=4):
        
        """
        Docstring pour adaptive_filter
        
        :param self: Description
        :param sig: signal
        :param noise: bruit
        :param mu: 
        :param order: 
        """ 

        clean, _, _ = lfilter([mu] * order, 1, sig, zi=noise)
        return clean
  # ---- Outlier smoothing ----

    def hampel_filter(self, sig, window=15, n_sigmas=3):

        """
        Docstring pour hampel_filter
        
        filtre de Hampel se basant sur la médiane pour del les outliers

        :param self: Description
        :param signal: signal
        :param window: taille de la fenetre
        :param n_sigmas: nb sigma
        """


        clean = sig.copy()
        for i in range(window, len(sig)-window):
            slice = sig[i-window:i+window]
            med = np.median(slice)
            mad = np.median(np.abs(slice - med))
            if mad == 0:
                continue
            if np.abs(sig[i] - med) > n_sigmas * mad:
                clean[i] = med
        return clean

    # ---- Motion artifact SCG / ECG (pacemaker) ----

    def suppress_motion(self, scg):

        """
        Docstring pour suppress_motion
        
        Application d'un filtre médian sur le signal en se basant sur une fenetre de données

        :param self: Description
        :param scg: signal
        """
        return medfilt(scg, kernel_size=5)




# ==============================
# 4. PIPELINE
# ==============================

class CleanPreprocessingPipeline:
    def __init__(self, record_path ,method):

        if method == "raw":
                self.loader = Dataloader_raw_file(record_path)
        elif method == "process":
                self.loader = DataLoader_preprocess_file(record_path)

        else:
            print('methode non ok')
            return 

        # self.loader = DataLoader_preprocess_file(record_path)
        # self.loader = Dataloader_raw_file(record_path)

        self.cleaner = ArtifactCleaner()



    def run(self):

        data = self.loader.load()

        ecg_raw = data["patch_ECG"]  # keep raw
        scg_raw = (
            data["patch_ACC_lat"] +
            data["patch_ACC_hf"] +
            data["patch_ACC_dv"]
        ) / 3

        # # ========== ECG cleaning ==========
        # ecg_clean = self.cleaner.highpass(ecg_raw)
        # ecg_clean = self.cleaner.bandpass(ecg_clean,1,40)
        # ecg_clean = self.cleaner.suppress_motion(ecg_clean)
        # ecg_clean = self.cleaner.hampel_filter(ecg_clean)

        ecg_clean = ecg_raw


        # ========== SCG cleaning ==========

        # scg_clean = self.cleaner.bandpass(scg_raw, 1, 40)
        # scg_clean = self.cleaner.suppress_motion(scg_clean)
        # scg_clean = self.cleaner.hampel_filter(scg_clean)
        # scg_clean = self.cleaner.swt_filter(scg_clean)

        scg_clean = scg_raw

        # ========== Normalize ==========
        ecg_clean = (ecg_clean - np.mean(ecg_clean)) / np.std(ecg_clean)
        scg_clean = (scg_clean - np.mean(scg_clean)) / np.std(scg_clean)


        return {
            "ecg_raw": ecg_raw,      
            "scg_raw": scg_raw,    


            "ecg_clean": ecg_clean,
            "scg_clean": scg_clean
        }





def plot_ecg_scg(ecg_raw, ecg_clean, scg_raw, scg_clean, fs=500, start_time=0):


    start_idx = int(start_time * fs)
    end_idx = start_idx   + 40 * fs  # 20-second window

    # Slice signals
    ecg_raw_win = ecg_raw[start_idx:end_idx]
    ecg_clean_win = ecg_clean[start_idx:end_idx]
    scg_raw_win = scg_raw[start_idx:end_idx]
    scg_clean_win = scg_clean[start_idx:end_idx]

    t = np.arange(len(ecg_raw_win)) / fs + start_time

    fig, axs = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # ECG
    axs[0].plot(t, ecg_raw_win, label="Raw ECG", alpha=0.6)
    axs[0].plot(t, ecg_clean_win, label="Clean ECG", alpha=0.8)
    axs[0].set_title("ECG before and after cleaning (30s window)")
    axs[0].legend()
    axs[0].grid(True)

    # SCG
    axs[1].plot(t, scg_raw_win, label="Raw SCG", alpha=0.6)
    axs[1].plot(t, scg_clean_win, label="Clean SCG", alpha=0.8)
    axs[1].set_title("SCG before and after cleaning (30s window)")
    axs[1].legend()
    axs[1].grid(True)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()




# ==============================
# 5. RUN EXAMPLE
# ==============================


record_path = "1.0.0/processed_data/TRM145-RHC2"
raw_path = "1.0.0/raw_data/wearable_patch/TRM107.RHC1.mat"

# record = wfdb.rdrecord(record_path)
# print(record.sig_name)


#  Attention la methode raw ne marche pas encore ne pas utiliser

# pipeline_raw = CleanPreprocessingPipeline(raw_path,method="raw")
pipeline_process = CleanPreprocessingPipeline(record_path ,method="process")

cleaned = pipeline_process.run()
# cleaned = pipeline_raw.run()

# Extract signals
ecg_raw = cleaned["ecg_raw"]
ecg_clean = cleaned["ecg_clean"]
scg_raw = cleaned["scg_raw"]
scg_clean = cleaned["scg_clean"]

plot_ecg_scg(
    cleaned["ecg_raw"],
    cleaned["ecg_clean"],
    cleaned["scg_raw"],
    cleaned["scg_clean"],
    fs=500,
    start_time=0  # 
)