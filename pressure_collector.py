import wfdb
import matplotlib.pyplot as plt
import pywt
import numpy as np

from preprocessing import DataLoaderPreprocessFile  , DataLoaderRawFile
from scipy.signal import resample


class RHCP_Pipeline:
    def __init__(self, record_path, method="raw"):
        self.loader = DataLoaderPreprocessFile(record_path)
        self.data = None   

    def run(self):
        data = self.loader.load()

        RHC_pressure = data["RHC_pressure"]
        patch_Pre = data["patch_Pre"]
        ECG_lead_II = data["ECG_lead_II"]

        

        # print(data.keys())


        self.data =  {

            "ECG_lead_II" :  ECG_lead_II,
            "RHC_pressure": RHC_pressure,
            "patch_Pre": patch_Pre,
        }

        return self.data
    
    def getValue(self):

        return self.data

INPUT = 'dat_signals/TRM107-RHC1'
RAW = 'raws_signals/TRM107.RHC1'

pipeline = RHCP_Pipeline(INPUT)
ecg_scg = DataLoaderRawFile(RAW)




a = pipeline.run()
value_rhc_  = a["RHC_pressure"]
value_ecg_ = a["ECG_lead_II"]
print(value_rhc_)


b = ecg_scg.load()
value_ecg = b["patch_ECG"]
time_ecg = b["time_ECG"]
print(value_ecg)




fs_ecg = 1000
fs_rhc = 250

n_rhc = len(value_rhc_)

# Resample ECG → même longueur que pression
ecg_resampled = resample(value_ecg_, n_rhc)

# Axe temps (secondes)
t = np.arange(n_rhc) / fs_rhc

# Plot aligné
plt.figure(figsize=(14,6))

plt.plot(t, ecg_resampled, label="ECG lead II", alpha=0.7)
plt.plot(t, value_rhc_, label="RHC pressure", alpha=0.7)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("ECG and RHC pressure")
plt.legend()
plt.grid(True)

plt.show()
