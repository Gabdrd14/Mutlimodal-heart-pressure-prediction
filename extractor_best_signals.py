import os
import numpy as np
import scipy.io as sio
import logging

from scipy.signal import resample
from pressure_collector import RHCP_Pipeline

# ==============================
# CONFIG
# ==============================

INPUT_FOLDER = "processed"
DAT_FOLDER = "dat_signals"
OUTPUT_FOLDER = "best_segments"

FS = 1000
WINDOW_S = 30
WINDOW_SIZE = WINDOW_S * FS
STEP = WINDOW_SIZE // 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# LOGGING
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# LOAD RHC
# ==============================

def load_rhc_from_dat(mat_filename):

    base = mat_filename.replace(".mat", "").replace(".", "-")

    for f in os.listdir(DAT_FOLDER):
        if f.endswith(".dat") and base in f:
            try:
                record_path = os.path.join(DAT_FOLDER, f.replace(".dat", ""))
                pipeline = RHCP_Pipeline(record_path)
                data = pipeline.run()
                return data["RHC_pressure"].squeeze()
            except:
                return None

    return None

# ==============================
# VALIDITY CHECK 
# ==============================

def is_valid_window(ecg, scg, rhc=None):

    # ECG check
    if ecg is None or np.std(ecg) < 1e-6:
        return False

    # SCG check
    if scg is None or np.std(scg) < 1e-6:
        return False

    # RHC check 
    if rhc is not None:
        if np.isnan(rhc).mean() > 0.2:
            return False
        if np.std(rhc) < 1e-6:
            return False

    return True

# ==============================
# BEST WINDOW 
# ==============================

def extract_best_window(*args, **kwargs):

    best_window = None

    for start in range(0, len(ecg) - WINDOW_SIZE, STEP):

        end = start + WINDOW_SIZE

        ecg_w = ecg[start:end]
        scg_w = scg[start:end]
    
        rhc_w = rhc[start:end] if rhc is not None else None
        
        ecg_raw_w = ecg_raw[start:end]
        scg_raw_w = scg_raw[start:end]
        
        patch_ACC_lat_w = patch_ACC_lat[start:end]
        patch_ACC_hf_w = patch_ACC_hf[start:end]
        patch_ACC_dv_w = patch_ACC_dv[start:end]
        
        
        

        if is_valid_window(ecg_w, scg_w, rhc_w):
            return (ecg_w, scg_w, rhc_w,ecg_raw_w,scg_raw_w,patch_ACC_lat_w,patch_ACC_hf_w,patch_ACC_dv_w)

    return None

# ==============================
# MAIN
# ==============================

logger.info("Starting processing...")

for fname in os.listdir(INPUT_FOLDER):

    if not fname.endswith(".mat"):
        continue

    logger.info(f"Processing {fname}")

    try:
        data = sio.loadmat(os.path.join(INPUT_FOLDER, fname))

        ecg = data["ecg_clean"].squeeze()
        scg = data["scg_clean"].squeeze()
  
        
        ecg_raw = data["ecg_raw"].squeeze()
        scg_raw = data["scg_raw"].squeeze()
        
        
        patch_ACC_lat = data["patch_ACC_lat"].squeeze()
        patch_ACC_hf = data["patch_ACC_hf"].squeeze()
        patch_ACC_dv = data["patch_ACC_dv"].squeeze()


        


        rhc = load_rhc_from_dat(fname)

        if rhc is not None:
            rhc = resample(rhc, len(ecg))

        # best_window = extract_best_window(ecg, scg, rhc)
        
        best_window = extract_best_window(ecg, scg, rhc, ecg_raw, scg_raw, patch_ACC_lat, patch_ACC_hf, patch_ACC_dv)


        if best_window is None:
            logger.warning(f"No valid window for {fname}")
            continue

        ecg_best, scg_best, rhc_best , ecg_raw , scg_raw, patch_ACC_lat, patch_ACC_hf, patch_ACC_dv= best_window

        # patient id
        base_name = fname.replace(".mat", "")
        patient_id = base_name.split(".")[0]

        out_path = os.path.join(OUTPUT_FOLDER, f"{patient_id}_best.mat")


        
        
        sio.savemat(out_path, {
            
            "patient" : patient_id,
            "ecg" : ecg_best,
            "scg": scg_best,
            "ecg_raw" : ecg_raw,
            "scg_raw" :scg_raw,
            "rhc": rhc_best,
            "patch_ACC_lat" : patch_ACC_lat,
            "patch_ACC_hf" : patch_ACC_hf ,
            "patch_ACC_dv" : patch_ACC_dv ,
    

    })

        logger.info(f"Saved → {patient_id}_best.mat")

    except Exception as e:
        logger.error(f"Error with {fname}: {e}")

logger.info("Done.")