import os
import argparse
import numpy as np
import scipy.io as sio
from datetime import datetime

from preprocessing import CleanPreprocessingPipeline, ArtifactCleaner
import config


# ==============================
# ARGUMENTS
# ==============================

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--raw", action="store_true")
parser.add_argument("--dat", action="store_true")

args = parser.parse_args()

if not (args.raw ^ args.dat):
    parser.error("Choose --raw or --dat")

INPUT_FOLDER = args.input
METHOD = "raw" if args.raw else "process"


# ==============================
# OUTPUT
# ==============================

DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"processed_{DATE}"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
# FILTER ENGINE 
# ==============================

class FilterEngine:
    def __init__(self, cleaner):
        self.c = cleaner

        self.map = {
            "highpass": self.hp,
            "bandpass": self.bp,
            "swt_filter": self.swt,
            "suppress_motion": self.motion,
            "hampel_filter": self.hampel,
        }

    def hp(self, sig, p):
        return self.c.highpass(sig, p["cutoff"])

    def bp(self, sig, p):
        return self.c.bandpass(sig, p["low"], p["high"])

    def swt(self, sig, p):
        return self.c.swt_filter(
            sig,
            wavelet=p.get("wavelet","db4"),
            level=p.get("level",2),
            method=p.get("method","soft")
        )

    def motion(self, sig, p):
        return self.c.suppress_motion(sig)

    def hampel(self, sig, p):
        return self.c.hampel_filter(sig, kernel=p["window"])

    def apply(self, sig, filters):
        out = sig
        for f in filters:
            out = self.map[f["type"]](out, f)
        return out


# ==============================
# MAIN LOOP
# ==============================

DEFAULT_FS = config.ECG_FILTERS[0].get("fs", 1000)

cleaner = ArtifactCleaner(fs=DEFAULT_FS)
engine = FilterEngine(cleaner)

for fname in os.listdir(INPUT_FOLDER):

    ext = fname.lower().split(".")[-1]

    if METHOD == "raw" and ext != "mat":
        continue
    if METHOD == "process" and ext != "dat":
        continue

    print("Processing:", fname)

    path = os.path.join(INPUT_FOLDER, fname)

    pipe = CleanPreprocessingPipeline(path, METHOD)
    data = pipe.run()

    ecg_raw = data["ecg_raw"]
    scg_raw = data["scg_raw"]

    ecg_clean = engine.apply(ecg_raw, config.ECG_FILTERS)
    scg_clean = engine.apply(scg_raw, config.SCG_FILTERS)

    t = data["time_ECG"]
    if t is None:
        t = np.arange(len(ecg_raw)) / DEFAULT_FS

    out = {
        "ecg_raw": ecg_raw,
        "scg_raw": scg_raw,
        "ecg_clean": ecg_clean,
        "scg_clean": scg_clean,
        "time": t
    }

    out_name = fname.replace(".dat", ".mat")
    sio.savemat(os.path.join(OUT_DIR, out_name), out)

    print("Saved:", out_name)

print("Done")
