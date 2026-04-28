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
# SAFE UTILS (DEBUG VERSION)
# ==============================

def safe_signal(x, name="unknown"):
    if x is None:
        logger.warning(f"[{name}] is None")
        return None

    try:
        x = np.asarray(x).squeeze()
    except Exception as e:
        logger.error(f"[{name}] np.asarray failed: {e}")
        return None

    if x.size == 0:
        logger.warning(f"[{name}] empty array")
        return None

    if x.size < 1000:
        logger.warning(f"[{name}] too small ({x.size})")
        return None

    if not np.all(np.isfinite(x)):
        logger.warning(f"[{name}] contains NaN/Inf")
        return None

    return x


def safe_rhc_load(rhc, ref_len):
    if rhc is None:
        logger.warning("[RHC] None input")
        return None

    try:
        rhc = np.asarray(rhc).squeeze()
    except Exception as e:
        logger.error(f"[RHC] conversion failed: {e}")
        return None

    if rhc.size < 10:
        logger.warning("[RHC] too small")
        return None

    try:
        if len(rhc) > 10:
            rhc = resample(rhc, ref_len)
        else:
            return None
    except Exception as e:
        logger.error(f"[RHC] resample failed: {e}")
        return None

    return rhc


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

                rhc = data.get("RHC_pressure", None)

                if rhc is None:
                    logger.warning(f"[RHC] missing in pipeline for {mat_filename}")
                    return None

                return safe_signal(rhc, "RHC_pressure")

            except Exception as e:
                logger.error(f"[RHC] pipeline failed: {e}")
                return None

    logger.warning(f"[RHC] no file found for {mat_filename}")
    return None


# ==============================
# NORMALIZATION
# ==============================

def normalize_signal(x):
    if x is None:
        return None

    x = np.asarray(x).astype(float)
    x = x - np.median(x)

    scale = np.percentile(np.abs(x), 95)

    if scale < 1e-8:
        return np.zeros_like(x)

    return x / scale


# ==============================
# QUALITY METRIC
# ==============================

def quality_metric(x):
    if x is None or len(x) < 20:
        return -np.inf

    x = normalize_signal(x)

    if x is None:
        return -np.inf

    smooth = np.convolve(x, np.ones(5)/5, mode='same')
    noise = x - smooth

    var_s = np.var(smooth)
    var_n = np.var(noise)

    if var_s <= 0 or var_n <= 0:
        return -np.inf

    snr = var_s / (var_n + 1e-8)

    ac = np.mean(x[:-1] * x[1:]) / (np.var(x) + 1e-8)

    zcr = np.mean((x[1:] * x[:-1]) < 0)

    return snr * ac / (1 + zcr)


# ==============================
# VALIDITY CHECK
# ==============================

def is_valid_window(ecg, scg, rhc=None):

    if ecg is None or scg is None:
        return False

    if len(ecg) < 100 or len(scg) < 100:
        return False

    if np.std(ecg) < 1e-8 or np.std(scg) < 1e-8:
        return False

    if rhc is not None:
        if np.isnan(rhc).mean() > 0.2:
            return False
        if np.std(rhc) < 1e-8:
            return False

    return True


# ==============================
# BEST WINDOW
# ==============================

def extract_best_window(ecg, scg, rhc,
                        ecg_raw, scg_raw,
                        patch_ACC_lat, patch_ACC_hf, patch_ACC_dv):

    if ecg is None or scg is None:
        return None

    n = min(len(ecg), len(scg))

    if n < WINDOW_SIZE:
        logger.warning("Signal too short for windowing")
        return None

    best_value = -np.inf
    best_window = None

    for start in range(0, n - WINDOW_SIZE + 1, STEP):

        end = start + WINDOW_SIZE

        ecg_w = ecg[start:end]
        scg_w = scg[start:end]
        rhc_w = rhc[start:end] if rhc is not None else None

        if ecg_w.size == 0 or scg_w.size == 0:
            logger.warning(f"Empty window at {start}")
            continue

        ecg_raw_w = ecg_raw[start:end]
        scg_raw_w = scg_raw[start:end]

        patch_ACC_lat_w = patch_ACC_lat[start:end]
        patch_ACC_hf_w = patch_ACC_hf[start:end]
        patch_ACC_dv_w = patch_ACC_dv[start:end]

        if not is_valid_window(ecg_w, scg_w, rhc_w):
            continue

        q_ecg = quality_metric(ecg_w)
        q_scg = quality_metric(scg_w)

        if q_ecg == -np.inf or q_scg == -np.inf:
            continue

        sync = np.sum(ecg_w * scg_w)

        value = q_ecg * q_scg * (1 + abs(sync))

        if value > best_value:
            best_value = value
            best_window = (
                ecg_w, scg_w, rhc_w,
                ecg_raw_w, scg_raw_w,
                patch_ACC_lat_w, patch_ACC_hf_w, patch_ACC_dv_w
            )

    return best_window


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

        ecg = safe_signal(data.get("ecg_clean"), "ecg_clean")
        scg = safe_signal(data.get("scg_clean"), "scg_clean")

        ecg_raw = safe_signal(data.get("ecg_raw"), "ecg_raw")
        scg_raw = safe_signal(data.get("scg_raw"), "scg_raw")

        patch_ACC_lat = safe_signal(data.get("patch_ACC_lat"), "patch_ACC_lat")
        patch_ACC_hf = safe_signal(data.get("patch_ACC_hf"), "patch_ACC_hf")
        patch_ACC_dv = safe_signal(data.get("patch_ACC_dv"), "patch_ACC_dv")

        if ecg is None or scg is None:
            logger.warning(f"Skipping {fname}: invalid ECG/SCG")
            continue

        rhc = load_rhc_from_dat(fname)
        rhc = safe_rhc_load(rhc, len(ecg))

        best_window = extract_best_window(
            ecg, scg, rhc,
            ecg_raw, scg_raw,
            patch_ACC_lat, patch_ACC_hf, patch_ACC_dv
        )

        if best_window is None:
            logger.warning(f"No valid window for {fname}")
            continue

        (ecg_best, scg_best, rhc_best,
         ecg_raw_best, scg_raw_best,
         patch_ACC_lat_best, patch_ACC_hf_best, patch_ACC_dv_best) = best_window

        patient_id = fname.replace(".mat", "").split(".")[0]
        out_path = os.path.join(OUTPUT_FOLDER, f"{patient_id}_best.mat")

        sio.savemat(out_path, {
            "patient": patient_id,
            "ecg": ecg_best,
            "scg": scg_best,
            "ecg_raw": ecg_raw_best,
            "scg_raw": scg_raw_best,
            "rhc": rhc_best,
            "patch_ACC_lat": patch_ACC_lat_best,
            "patch_ACC_hf": patch_ACC_hf_best,
            "patch_ACC_dv": patch_ACC_dv_best,
        })

        logger.info(f"Saved → {patient_id}_best.mat")

    except Exception as e:
        logger.error(f"Error with {fname}: {e}")

logger.info("Done.")