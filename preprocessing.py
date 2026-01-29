import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import scipy.io as sio
import pywt


# ==============================
# DATA LOADERS
# ==============================

class DataLoaderPreprocessFile:
    def __init__(self, record_path):
        self.record_path = record_path

    def load(self):
        record = wfdb.rdrecord(self.record_path)
        signals = record.p_signal
        names = record.sig_name
        return dict(zip(names, signals.T))


class DataLoaderRawFile:
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

    def bandpass(self, sig, low, high):
        key = ("bp", low, high)
        b, a = self._get_filter(
            key,
            lambda: butter(4, [low/(self.fs/2), high/(self.fs/2)], btype="band")
        )
        return filtfilt(b, a, sig)

    # --------  HAMPel -------- #

    def hampel_filter(self, sig, kernel=31):
        med = medfilt(sig, kernel)
        diff = np.abs(sig - med)
        mad = medfilt(diff, kernel)

        threshold = 3 * mad
        out = sig.copy()
        mask = diff > threshold
        out[mask] = med[mask]

        return out

    # -------- MOTION -------- #

    def suppress_motion(self, scg):
        return medfilt(scg, kernel_size=5)

    # -------- OPTIONAL HEAVY SWT -------- #

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
    def __init__(self, record_path, method="raw"):
        if method == "raw":
            self.loader = DataLoaderRawFile(record_path)
        elif method == "process":
            self.loader = DataLoaderPreprocessFile(record_path)
        else:
            raise ValueError("method must be raw or process")

    def run(self):
        data = self.loader.load()

        ecg_raw = data["patch_ECG"]
        scg_raw = (
            data["patch_ACC_lat"]
            + data["patch_ACC_hf"]
            + data["patch_ACC_dv"]
        ) / 3

        return {
            "ecg_raw": ecg_raw,
            "scg_raw": scg_raw,
            "time_ECG": data.get("time_ECG")
        }
