import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
# from scipy.interpolate import interp1d
# from scipy.signal import lfilter
# from hampel import hampel

import scipy.io as sio
import pywt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================
# DATA LOADERS
# ==============================

class DataLoaderPreprocessFile:

    """
    Docstring pour DataLoaderPreprocessFile

        Prend en argument le dossier parent et recupere les signaux .dat

    """
    def __init__(self, record_path):
        self.record_path = record_path

    def load(self):
        record = wfdb.rdrecord(self.record_path)
        signals = record.p_signal
        names = record.sig_name
        return dict(zip(names, signals.T))


class DataLoaderRawFile:

    """
    Docstring pour DataLoaderRawFile

        Prend en argument le dossier parent et recupere les signaux .mat


    """

    def __init__(self, raw_path):
        self.raw_path = raw_path

    def load(self):
        mat = sio.loadmat(self.raw_path)
        data = mat['data'][0,0]

        ecg = data['E_data'].squeeze()
        t = data['E_time'].squeeze()

        N = min(len(ecg), len(t))
        ecg, t = ecg[:N], t[:N]

        return {
            "patch_ECG": ecg,
            "patch_ACC_lat": data['A_data_x'].squeeze()[:N],
            "patch_ACC_hf": data['A_data_y'].squeeze()[:N],
            "patch_ACC_dv": data['A_data_z'].squeeze()[:N],
            "time_ECG": t
        }


# ==============================
#  ARTIFACT CLEANER
# ==============================

class ArtifactCleaner:

    """
    Docstring pour ArtifactCleaner

        Applique les filtres sur les signaux en question

        filtre  : lowpass , highpass , bandpass( low & high) , median , hampel , swt

    """


    def __init__(self, fs=1000):
        self.fs = fs
        self._filter_cache = {}

    # -------- FILTER CACHE -------- #

    def _get_filter(self, key, builder):
        if key not in self._filter_cache:
            self._filter_cache[key] = builder()
        return self._filter_cache[key]

    # -------- BASIC FILTERS -------- #

    def highpass(self, sig, cutoff):
        key = ("hp", cutoff)
        b, a = self._get_filter(
            key,
            lambda: butter(2, cutoff/(self.fs/2), btype="high")
        )
        return filtfilt(b, a, sig)
    
    def lowpass(self, sig, cutoff, order=4):

        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, sig)


    def bandpass(self, sig, low, high):
        key = ("bp", low, high)
        b, a = self._get_filter(
            key,
            lambda: butter(4, [low/(self.fs/2), high/(self.fs/2)], btype="band")
        )
        return filtfilt(b, a, sig)

    # --------  HAMPEl -------- #

        #  necessite optimisation peut etre avec python -m pip install heartpy. ?
        #  a voir

    def hampel_filter(self, sig, window=31, n_sigmas=3):
        med = medfilt(sig, window)
        diff = np.abs(sig - med)
        mad = medfilt(diff, window)

        threshold = n_sigmas * mad
        out = sig.copy()
        mask = diff > threshold
        out[mask] = med[mask]

        return out

    # -------- MOTION -------- #

    def suppress_motion(self, scg, kernel_size=5):
        return medfilt(scg, kernel_size=kernel_size)


    # --------  SWT -------- #


    def swt_filter(self, sig, wavelet="db4", level=2, method='soft'):
        if len(sig) % 2 != 0:
            sig = sig[:-1]

        coeffs = pywt.swt(sig, wavelet, level=level)
        new_coeffs = []

        for approx, detail in coeffs:
            sigma = np.median(np.abs(detail)) / 0.6745
            thr = sigma * np.sqrt(2*np.log(len(detail)))
            detail_t = pywt.threshold(detail, thr, mode=method)
            new_coeffs.append((approx, detail_t))

        return pywt.iswt(new_coeffs, wavelet)


# ==============================
# BASIC PIPELINE
# ==============================

class CleanPreprocessingPipeline:
    """
    Docstring pour CleanPreprocessingPipeline

        Main pipeline qui lance le filtrage en fonction du type des fichiers ( raw / process )

    """

    def __init__(self, record_path, method="raw", fs=500):
        if method == "raw":
            self.loader = DataLoaderRawFile(record_path)
        elif method == "process":
            self.loader = DataLoaderPreprocessFile(record_path)
        else:
            raise ValueError("method must be raw or process")
        
        self.fs = fs  # Sampling frequency (WFDB data is 500 Hz)
        self.cleaner = ArtifactCleaner(fs=fs)

    def run(self):
        data = self.loader.load()

        ecg_raw = data.get("patch_ECG")
        
        # For process method, extract patch signals and compute SCG
        if "patch_ACC_lat" in data:
            scg_raw = (
                data["patch_ACC_lat"]
                + data["patch_ACC_hf"]
                + data["patch_ACC_dv"]
            ) / 3
        else:
            scg_raw = None

        return {
            "ecg_raw": ecg_raw,
            "scg_raw": scg_raw,
            "time_ECG": data.get("time_ECG"),
            "patch_ACC_lat":    data.get("patch_ACC_lat"),
            "patch_ACC_hf":    data.get("patch_ACC_hf"),
            "patch_ACC_dv":    data.get("patch_ACC_dv"),
        }


# ==============================
# WFDB DATA PROCESSOR
# ==============================

class WFDBDataProcessor:
    """
    Processes WFDB data from dat_signals and saves as .mat files to processed folder
    Extracts ECG, SCG (via bandpass filter), RHC pressure, and other signals
    """
    
    def __init__(self, record_path, fs=500, scg_bandpass=(1, 40)):
        """
        Args:
            record_path: Path to WFDB record (without extension)
            fs: Sampling frequency (Hz)
            scg_bandpass: Bandpass filter range for SCG extraction (default 1-40 Hz)
        """
        self.record_path = record_path
        self.fs = fs
        self.scg_low, self.scg_high = scg_bandpass
        self.cleaner = ArtifactCleaner(fs=fs)
        self.data = None
        
    def load(self):
        """Load WFDB data"""
        try:
            record = wfdb.rdrecord(self.record_path)
            signals = record.p_signal
            names = record.sig_name
            self.signal_dict = dict(zip(names, signals.T))
            logger.info(f"✓ Loaded signals: {list(self.signal_dict.keys())}")
            return self.signal_dict
        except Exception as e:
            logger.error(f"✗ Error loading {self.record_path}: {e}")
            return None
    
    def process(self):
        """Process and clean signals"""
        if self.signal_dict is None:
            return None
        
        processed = {}
        
        # ========== PATCH ECG ==========
        # if "Patch_ECG" in self.signal_dict:
        ecg_raw = self.signal_dict["patch_ECG"].copy()
        # Apply highpass filter to remove baseline drift
        ecg_clean = self.cleaner.highpass(ecg_raw, cutoff=0.5)
        processed["ecg_raw"] = ecg_raw
        processed["ecg_clean"] = ecg_clean
        logger.info("✓ Processed Patch_ECG")
    
        # ========== PATCH ACC -> SCG ==========
        acc_lat = self.signal_dict.get("patch_ACC_lat")
        acc_hf = self.signal_dict.get("patch_ACC_hf")
        acc_dv = self.signal_dict.get("patch_ACC_dv")
        
        # if acc_lat is not None and acc_hf is not None and acc_dv is not None:
            # Combine accelerometer signals
        scg_raw = (acc_lat + acc_hf + acc_dv) / 3.0
        # Extract SCG using bandpass filter (1-40 Hz)
        scg_clean = self.cleaner.bandpass(scg_raw, self.scg_low, self.scg_high)
        processed["scg_raw"] = scg_raw
        processed["scg_clean"] = scg_clean
        processed["patch_ACC_lat"] = acc_lat
        processed["patch_ACC_hf"] = acc_hf
        processed["patch_ACC_dv"] = acc_dv
        logger.info("✓ Processed Patch ACC signals and extracted SCG")
        
        # ========== RHC PRESSURE ==========
        # if "RHC_pressure" in self.signal_dict:
        rhc_raw = self.signal_dict["RHC_pressure"].copy()
        # Apply lowpass filter to smooth RHC signal
        rhc_clean = self.cleaner.lowpass(rhc_raw, cutoff=20)
        processed["rhc_raw"] = rhc_raw
        processed["rhc_clean"] = rhc_clean
        logger.info("✓ Processed RHC_pressure")
        
        # ========== MAC-LAB ECG SIGNALS ==========
        ecg_leads = ["ECG_lead_I", "ECG_lead_II", "ECG_lead_III", 
                     "aVR", "aVL", "aVF", 
                     "ECG_lead_V1", "ECG_lead_V2", "ECG_lead_V3", 
                     "ECG_lead_V4", "ECG_lead_V5", "ECG_lead_V6"]
        
        for lead in ecg_leads:
            if lead in self.signal_dict:
                processed[lead] = self.signal_dict[lead].copy()
        
        if any(lead in processed for lead in ecg_leads):
            logger.info(f"✓ Processed ECG leads: {[l for l in ecg_leads if l in processed]}")
        
        # ========== OTHER MAC-LAB SIGNALS ==========
        other_signals = ["ART", "PLETH", "RESP", "Patch_Hum", "Patch_Pre", "Patch_Temp"]
        for signal_name in other_signals:
            if signal_name in self.signal_dict:
                processed[signal_name] = self.signal_dict[signal_name].copy()
        
        if any(sig in processed for sig in other_signals):
            logger.info(f"✓ Processed other signals: {[s for s in other_signals if s in processed]}")
        
        # ========== TIME ARRAY ==========
        time_array = np.arange(len(processed.get("ecg_raw", processed.get("rhc_raw", 
                                                   processed.get("scg_raw"))))) / self.fs
        processed["time"] = time_array
        
        self.data = processed
        return processed
    
    def save_mat(self, output_path):
        """Save processed data as .mat file"""
        if self.data is None:
            logger.error("✗ No processed data to save. Run process() first.")
            return False
        
        try:
            sio.savemat(output_path, self.data)
            logger.info(f"✓ Saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Error saving {output_path}: {e}")
            return False
    
    def run(self, output_path):
        """Complete pipeline: load -> process -> save"""
        if self.load() is None:
            return False
        self.process()
        return self.save_mat(output_path)
