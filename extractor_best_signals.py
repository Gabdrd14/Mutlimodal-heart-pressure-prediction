"""
Extraction STRICTE des meilleurs segments 30s par patient.

Stratégie:
1. RHC doit être PARFAIT (pas de sinusoides erratiques, pas de drift, signal stable)
2. ECG doit être de BONNE qualité (rythme régulier)
3. SCG doit être de BONNE qualité (pulsatile, bande cardiaque)
4. Seulement sauvegarder si TOUS les trois critères sont excellents
5. Pas de fallback - rejeter le patient s'il n'y a rien de parfait
"""

import os
import warnings
import numpy as np
import scipy.io as sio
from scipy.signal import resample, resample_poly
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================
# CONFIG 
# ==============================

INPUT_FOLDER  = "processed"
OUTPUT_FOLDER = "segments_30s_strict_5per_patient"

FS          = 500  
WINDOW_S    = 30
WINDOW_SIZE = FS * WINDOW_S

SEARCH_STRIDE_S = 1
SEARCH_STRIDE   = SEARCH_STRIDE_S * FS

TOP_N_SEGMENTS = 5

MIN_RHC_SCORE     = 0.80 # Was 0.80
MIN_ECG_SCORE     = 0.75  # Was 0.80
MIN_SCG_SCORE     = 0.75  # Was 0.80
MIN_QUALITY_SCORE = 0.8  # Was 0.80

MAX_OVERLAP_FRAC = 0.0
MIN_SEP = int(WINDOW_SIZE * (1.0 - MAX_OVERLAP_FRAC))

N_WORKERS = max(1, os.cpu_count() - 1)

CLIP_FRACTION = 0.95

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# FONCTIONS
# ==============================

def _fast_detrend(sig: np.ndarray, window_s: float = 3.0) :
    """On utilise un filtre de moyenne mobile pour estimer la ligne de base, puis on soustrait pour obtenir un signal plus stationnaire"""
    kernel = int(window_s * FS)
    baseline = uniform_filter1d(sig.astype(np.float64), size=kernel, mode='nearest')
    return (sig - baseline).astype(np.float32)


def _flat_check(sig: np.ndarray):
    """Retourne 0.0 si le signal est plat, sinon None """
    return 0.0 if np.var(sig) < 1e-12 else None


def _kurtosis_score(sig: np.ndarray, low=3.0, high=10.0, decay=40.0):
    """score de kurtosis (pic)"""
    std = np.std(sig) + 1e-12
    kurt = float(np.mean(((sig - np.mean(sig)) / std) ** 4))
    if kurt < low:   return kurt / low
    if kurt <= high: return 1.0
    return max(0.0, 1.0 - (kurt - high) / decay)


def _drift_score(sig: np.ndarray):
    """score d'absence de drift basse fréquence"""
    baseline = uniform_filter1d(sig.astype(np.float64), size=int(0.5 * FS), mode='nearest')
    drift_ratio = np.var(baseline) / (np.var(sig) + 1e-12)
    return float(max(0.0, 1.0 - drift_ratio * 2.0))


def _has_null_sections(sig: np.ndarray, null_threshold: float = 1e-6, min_null_fraction: float = 0.01):
    """
    Vérifie si le signal contient des sections significatives de valeurs nulles ou proches de zéro.
    """
    is_null = np.abs(sig) < null_threshold
    null_fraction = float(np.mean(is_null))
    # Reject if >1% of signal is essentially zero
    return null_fraction >= min_null_fraction



# ==============================
# RHC SCORE
# ==============================

def _score_rhc_strict(rhc: np.ndarray):
  
    """ 
    SCORE RHC 
    
    Conditions de rejet immédiat :
    1. Signal plat
    2. Sections nulles (dropout du capteur)
    3. Drift linéaire (R² > 0.50)
    4. Non-stationnarité élevée (ratio de variance moitié > 4)
    5. Flatlines prolongés (>0.5s bloqué)
    6. Absence de pulsatility n'importe où dans la fenêtre
    7. Pression de pouls extrême (< 3 ou > 50 mmHg)
    
    
    
    """
    
    
    
    # Gate 1 - Flat signal
    if (v := _flat_check(rhc)) is not None:
        return v
    
    # Gate 2 - Null/zero sections 
    if _has_null_sections(rhc, null_threshold=0.01, min_null_fraction=0.02):
        return 0.0
    
    # Gate 3 - Linear drift 
    x = np.arange(len(rhc), dtype=np.float64)
    coeffs = np.polyfit(x, rhc.astype(np.float64), 1)
    trend = np.polyval(coeffs, x)
    ss_res = float(np.sum((rhc - trend) ** 2))
    ss_tot = float(np.sum((rhc - np.mean(rhc)) ** 2)) + 1e-12
    r_squared = 1.0 - ss_res / ss_tot
    if r_squared > 0.50:  
        return 0.0
    
    # Gate 4 - Stationarity 
    half = len(rhc) // 2
    var1 = float(np.var(rhc[:half])) + 1e-12
    var2 = float(np.var(rhc[half:])) + 1e-12
    var_ratio = max(var1, var2) / min(var1, var2)
    if var_ratio > 4.0:  # Non-stationary = REJET
        return 0.0
    
    # Gate 5 - Prolonged flatlines 
    diffs = np.abs(np.diff(rhc))
    is_flat = (diffs < 0.005).astype(np.float32)
    kernel_size = int(0.5 * FS)  # 0.5 second window
    if kernel_size > len(is_flat):
        return 0.0
    rolling_flats = uniform_filter1d(is_flat, size=kernel_size, mode='constant', cval=0.0)
    if np.any(rolling_flats > 0.90):  # 90% flat = REJET
        return 0.0
    
    # Gate 6 - Pulsatility everywhere (bloc)
    chunk_len = int(4.0 * FS)  # bloc de 4 secondes
    n_chunks = len(rhc) // chunk_len
    for i in range(n_chunks):
        chunk = rhc[i * chunk_len : (i + 1) * chunk_len]
        pp = float(np.percentile(chunk, 95) - np.percentile(chunk, 5))
        if pp < 2.0:   # Si on trouve un bloc de 4s sans au moins 2 mmHg de pulsatility, REJET
            return 0.0
    
    # Gate 7 - Pulse pressure in physiological range
    pp = float(np.percentile(rhc, 95) - np.percentile(rhc, 5))
    if pp < 3.0 or pp > 50.0:
        return 0.0
    
    
    rhc_dt = _fast_detrend(rhc)
    if (v := _flat_check(rhc_dt)) is not None:
        return v
    
    # Score 1: Regularité du pouls (autocorrélation normalisée)
    try:
        x = rhc_dt - np.mean(rhc_dt)
        norm = float(np.dot(x, x))
        if norm < 1e-12:
            pulse_reg_sc = 0.0
        else:
            n = len(x)
            nfft = 1 << (2 * n - 1).bit_length()
            X = np.fft.rfft(x, n=nfft)
            ac = np.fft.irfft(X * np.conj(X), n=nfft)[:n]
            ac /= (norm + 1e-12)
            lag_min = int(0.33 * FS)
            lag_max = min(int(1.50 * FS), n - 1)
            if lag_min >= lag_max:
                pulse_reg_sc = 0.0
            else:
                pulse_reg_sc = float(np.clip(np.max(ac[lag_min:lag_max]), 0.0, 1.0))
    except:
        pulse_reg_sc = 0.0
    
    # Score 2: Stationnarité (ratio de variance moitié, transformé pour donner un score entre 0 et 1)
    station_sc = float(np.clip(1.0 - (var_ratio - 1.5) / 2.5, 0.0, 1.0))
    
    # Score 3: Morphologie (variabilité des pics RHC, plus les pics sont similaires en hauteur, meilleur le score)
    try:
        peaks = np.where((rhc_dt[1:-1] > rhc_dt[:-2]) & (rhc_dt[1:-1] > rhc_dt[2:]))[0] + 1
        if len(peaks) < 3:
            morph_sc = 0.2
        else:
            peak_heights = rhc_dt[peaks]
            cv = float(np.std(peak_heights)) / (float(np.mean(peak_heights)) + 1e-9)
            morph_sc = float(np.clip(1.0 - cv * 1.5, 0.0, 1.0))
    except:
        morph_sc = 0.0
    
    # Composite: 50% régularité du pouls, 30% stationnarité, 20% morphologie
    final_score = 0.50 * pulse_reg_sc + 0.30 * station_sc + 0.20 * morph_sc
    return float(np.clip(final_score, 0.0, 1.0))


# ==============================
# ECG SCORE
# ==============================

def _score_ecg_strict(ecg: np.ndarray):
    """
    Score ECG baser sur la régularité du rythme 
    """
    if (v := _flat_check(ecg)) is not None:
        return v
    
    # Rejet si sections nulles 
    if _has_null_sections(ecg, null_threshold=1e-5, min_null_fraction=0.02):
        return 0.0
    
    try:
        # Detection R-peaks
        sq = ecg ** 2
        threshold = 0.3 * np.max(sq)  
        above = (sq > threshold).astype(np.int8)
        edges = np.where(np.diff(above) == 1)[0]
        
        if len(edges) < 3:  
            return 0.2  
        
        rr = np.diff(edges)
        rr = rr[(rr > 0.3 * FS) & (rr < 2.0 * FS)]  
        
        if len(rr) < 2:  
            return 0.3
        
        # Coefficient de variation des intervalles RR (rythme régulier = CV faible)
        cv = np.std(rr) / (np.mean(rr) + 1e-9)
        rr_regularity = float(max(0.0, 1.0 - cv * 2.5))  
        
        # Métriques de forme : kurtosis (pic) et drift (baseline stable)
        kurtosis_sc = _kurtosis_score(ecg, 2.0, 15.0, 60.0)  
        drift_sc = _drift_score(ecg)
        
        return float(np.clip(0.45*rr_regularity + 0.35*kurtosis_sc + 0.20*drift_sc, 0.0, 1.0))
        
    except Exception:
        return 0.3


# ==============================
# SCG SCORE
# ==============================

def _score_scg_strict(scg: np.ndarray):
    
    """
    Score SCG basé sur la puissance dans la bande cardiaque (1-40 Hz) et la pulsatility (peak-to-peak).
    """
    if (v := _flat_check(scg)) is not None:
        return v
    
    # Rejet si sections nulles (dropout du capteur)
    if _has_null_sections(scg, null_threshold=1e-6, min_null_fraction=0.02):
        return 0.0
    
    try:
        # Fréquence cardiaque typique : 1-40 Hz, on regarde la puissance relative dans cette bande
        mag = np.abs(np.fft.rfft(scg))
        freqs = np.fft.rfftfreq(len(scg), d=1.0 / FS)
        total_pwr = float(np.dot(mag, mag)) + 1e-12
        
        # Calculer la puissance dans la bande cardiaque
        mask = (freqs >= 5.0) & (freqs <= 40.0)
        band_pwr = float(np.dot(mag[mask], mag[mask]))
        band_ratio = band_pwr / total_pwr
        
        
        band_sc = float(np.clip((band_ratio - 0.35) / 0.40, 0.0, 1.0))
        
        # Pulsation : on regarde le peak-to-peak du signal, doit être d'au moins 0.01 pour être considéré comme valide
        pp = float(np.percentile(scg, 95) - np.percentile(scg, 5))
        if pp < 0.01:  # Trop faible  = REJET
            return 0.0
        
        pulsatility_sc = float(np.clip((pp - 0.01) / 0.1, 0.0, 1.0))
        
        # constante de kurtosis pour récompenser les formes de signal plus "pic" (caractéristique des SCG de bonne qualité)
        kurtosis_sc = _kurtosis_score(scg, 2.0, 8.0, 30.0)
        
        return float(np.clip(0.50*band_sc + 0.35*pulsatility_sc + 0.15*kurtosis_sc, 0.0, 1.0))
        
    except Exception:
        return 0.0


# ==============================
# RHC LOADER 
# ==============================

def load_rhc_from_mat(mat_data: dict):
    """Load RHC signal from .mat data dictionary"""
    
    for key in ['rhc_raw', 'rhc_clean', 'RHC_pressure']:
        if key in mat_data:
            rhc = np.asarray(mat_data[key]).squeeze().astype(np.float32)
            if rhc.ndim == 0 or len(rhc) == 0 or np.all(np.isnan(rhc)):
                continue
            
            # Handle NaN values
            if np.any(np.isnan(rhc)):
                idx = np.arange(len(rhc))
                valid = ~np.isnan(rhc)
                if valid.sum() < 2:
                    continue
                rhc[~valid] = np.interp(idx[~valid], idx[valid], rhc[valid])
            
            return rhc
    
    return None


def _resample_to(sig: np.ndarray, target_len: int) :
    
    """On resample le signal RHC pour qu'il ait la même longueur que ECG/SCG,
       en utilisant une méthode de resampling adaptée à la taille du signal pour éviter les artefacts"""
    if len(sig) == target_len:
        return sig
    g = gcd(len(sig), target_len)
    up, down = target_len // g, len(sig) // g
    if max(up, down) <= 500:
        return resample_poly(sig, up, down).astype(np.float32)
    return resample(sig, target_len).astype(np.float32)


# ==============================
# WINDOW SEARCH
# ==============================

def find_best_windows(ecg: np.ndarray, scg: np.ndarray, rhc: np.ndarray, top_n: int = TOP_N_SEGMENTS):
    
    """On cherche les meilleurs segments de 30s en utilisant une approche en trois phases :
    1. On scanne le signal RHC avec une fenêtre glissante de 30s et on ne garde que les fenêtres qui passent un seuil strict de qualité RHC.
    2. Pour les fenêtres RHC gagnantes, on calcule les scores ECG et SCG. Seules les fenêtres où ECG ET SCG sont tous les deux excellents sont retenues.
    3. Parmi ces candidats, on calcule un score  (0.35*ECG + 0.40*SCG + 0.25*RHC) et on sélectionne les meilleurs segments non chevauchants qui dépassent le seuil de qualite."""
  
    n = len(ecg)
    
    # Phase 1:
    rhc_candidates = []
    for start in range(0, n - WINDOW_SIZE + 1, SEARCH_STRIDE):
        rhc_sc = _score_rhc_strict(rhc[start:start + WINDOW_SIZE])
        if rhc_sc >= MIN_RHC_SCORE:
            rhc_candidates.append((rhc_sc, start))
    
    if not rhc_candidates:
        print(f"  [SEARCH] No RHC windows passed (MIN: {MIN_RHC_SCORE:.2f})")
        return []
    
    rhc_candidates.sort(reverse=True)
    print(f"  [SEARCH] {len(rhc_candidates)} RHC-excellent window(s) out of {(n - WINDOW_SIZE) // SEARCH_STRIDE + 1}")
    
    # Phase 2: 
    scored = []
    for rhc_sc, start in rhc_candidates:
        e = start + WINDOW_SIZE
        ecg_sc = _score_ecg_strict(ecg[start:e])
        scg_sc = _score_scg_strict(scg[start:e])
        
        # Si ECG ou SCG ne sont pas excellents, on rejette la fenêtre même si RHC est parfait
        if ecg_sc >= MIN_ECG_SCORE and scg_sc >= MIN_SCG_SCORE:
            comp = 0.35 * ecg_sc + 0.40 * scg_sc + 0.25 * rhc_sc
            scored.append((comp, rhc_sc, ecg_sc, scg_sc, start))
    
    if not scored:
        print(f"  [SEARCH] RHC-excellent mais ECG/SCG mauvais. "
              f"Min ECG: {MIN_ECG_SCORE:.2f}, Min SCG: {MIN_SCG_SCORE:.2f}")
        return []
    
    scored.sort(reverse=True)
    
    # Phase 3:
    selected = []
    for comp, rhc_sc, ecg_sc, scg_sc, start in scored:
        if comp < MIN_QUALITY_SCORE:
            continue
        if any(abs(start - s) <= MIN_SEP for s, *_ in selected):

            continue
        selected.append((start, comp, rhc_sc, ecg_sc, scg_sc))
        if len(selected) == top_n:
            break
    
    if not selected:
        print(f"  [SEARCH] aucune fenetre valable ({MIN_QUALITY_SCORE:.2f})")
        return []
    
    return selected


# ==============================
# FILE PROCESSING
# ==============================

def process_file(fname: str) -> list[dict]:
    results = []
    try:
        data = sio.loadmat(os.path.join(INPUT_FOLDER, fname))
        
        def _load(key, fallback_keys=None):
            if key in data:
                sig = np.asarray(data[key]).squeeze().astype(np.float32)
                if sig.ndim > 0 and len(sig) > 0:
                    return sig
            
            if fallback_keys:
                for fb_key in fallback_keys:
                    if fb_key in data:
                        sig = np.asarray(data[fb_key]).squeeze().astype(np.float32)
                        if sig.ndim > 0 and len(sig) > 0:
                            print(f"    → Using {fb_key} for {key}")
                            return sig
            
            return None
        
        # Load ECG 
        ecg_clean = _load("ecg_clean", ["ecg_raw", "ECG_lead_II", "ECG_lead_I"])
        ecg_raw = _load("ecg_raw", ["ecg_clean", "ECG_lead_II", "ECG_lead_I"])
        
        # Load SCG with fallbacks
        scg_clean = _load("scg_clean", ["scg_raw"])
        scg_raw = _load("scg_raw", ["scg_clean"])
        
        # Load accelerometer 
        patch_ACC_lat = _load("patch_ACC_lat")
        patch_ACC_hf = _load("patch_ACC_hf")
        patch_ACC_dv = _load("patch_ACC_dv")
        
  
        
        if ecg_raw is None or ecg_clean is None or scg_raw is None or scg_clean is None:
            missing = []
            if ecg_raw is None: missing.append("ecg_raw")
            if ecg_clean is None: missing.append("ecg_clean")
            if scg_raw is None: missing.append("scg_raw")
            if scg_clean is None: missing.append("scg_clean")
            print(f"  ✗ Missing signals: {', '.join(missing)}")
            return [{"file": fname, "statuts": "missing_signals", "missing": missing}]
        
        n = min(len(ecg_clean), len(ecg_raw), len(scg_clean), len(scg_raw))
        if patch_ACC_lat is not None:
            n = min(n, len(patch_ACC_lat))
        if patch_ACC_hf is not None:
            n = min(n, len(patch_ACC_hf))
        if patch_ACC_dv is not None:
            n = min(n, len(patch_ACC_dv))
        
        ecg_clean = ecg_clean[:n]
        ecg_raw = ecg_raw[:n]
        scg_clean = scg_clean[:n]
        scg_raw = scg_raw[:n]
        
        if patch_ACC_lat is not None:
            patch_ACC_lat = patch_ACC_lat[:n]
        if patch_ACC_hf is not None:
            patch_ACC_hf = patch_ACC_hf[:n]
        if patch_ACC_dv is not None:
            patch_ACC_dv = patch_ACC_dv[:n]
        
        if n < WINDOW_SIZE:
            print(f"  Signal too short: {n} samples < {WINDOW_SIZE} required")
            return [{"file": fname, "status": "too_short"}]
        
        # On charge le signal RHC
        rhc = load_rhc_from_mat(data)
        if rhc is None:
            print(f"  ✗ No RHC signal found")
            return [{"file": fname, "statuts": "no_rhc"}]
        
        if len(rhc) != n:
            rhc = _resample_to(rhc, n)
        
        print(f"  charger: ECG={n}, SCG={n}, RHC={len(rhc)}")
        windows = find_best_windows(ecg_clean, scg_clean, rhc, top_n=5)
        
        if not windows:
            return [{"file": fname, "statuts": "rejected"}]
        
        patient_id = fname.replace(".mat", "").split(".")[0]
        
        for rank, (start, comp, rhc_sc, ecg_sc, scg_sc) in enumerate(windows, 1):
            s, e = start, start + WINDOW_SIZE
            out_path = os.path.join(OUTPUT_FOLDER, f"{patient_id}_segment_{rank:02d}.mat")
            
            out_dict = {
                "patient": patient_id,
                "ecg": ecg_clean[s:e],
                "scg": scg_clean[s:e],
                "ecg_raw": ecg_raw[s:e],
                "scg_raw": scg_raw[s:e],
                "rhc": rhc[s:e],
                "quality_composite": np.array([comp]),
                "quality_ecg": np.array([ecg_sc]),
                "quality_scg": np.array([scg_sc]),
                "quality_rhc": np.array([rhc_sc]),
                "window_start_s": np.array([s / FS]),
                "window_end_s": np.array([e / FS]),
                "fs": np.array([FS]),
            }
            
            # Add accelerometer data if available
            if patch_ACC_lat is not None:
                out_dict["patch_ACC_lat"] = patch_ACC_lat[s:e]
            if patch_ACC_hf is not None:
                out_dict["patch_ACC_hf"] = patch_ACC_hf[s:e]
            if patch_ACC_dv is not None:
                out_dict["patch_ACC_dv"] = patch_ACC_dv[s:e]
            
            sio.savemat(out_path, out_dict)
            
            print(f" [{fname}] {s/FS:.0f}–{e/FS:.0f}s | "
                  f"RHC={rhc_sc:.3f} ECG={ecg_sc:.3f} SCG={scg_sc:.3f} "
                  f"COMP={comp:.3f}")
            
            results.append({
                "fichiers": fname, "statuts": "saved",
                "rhc": rhc_sc, "ecg": ecg_sc, "scg": scg_sc, "comp": comp,
                "rank": rank
            })
    
    except Exception as exc:
        print(f"  ✗ [{fname}] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        results.append({"fichiers": fname, "statuts": "error", "error": str(exc)})
    
    return results


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    import time
    
    fnames = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mat")])
    print(f"\n{'='*70}")
    print(f"SEGMENTS EXTRACTION")
    print(f"Files: {len(fnames)} | RHC≥{MIN_RHC_SCORE:.2f} ECG≥{MIN_ECG_SCORE:.2f} SCG≥{MIN_SCG_SCORE:.2f}")
    print(f"{'='*70}\n")
    
    t0 = time.perf_counter()
    summary = []
    
    if N_WORKERS == 1:
        for fname in fnames:
            print(f"{fname}")
            summary.extend(process_file(fname))
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(process_file, f): f for f in fnames}
            for fut in as_completed(futures):
                summary.extend(fut.result())
    
    elapsed = time.perf_counter() - t0
    saved = [r for r in summary if r["statuts"] == "saved"]
    rejected = [r for r in summary if r["statuts"] != "saved"]
    patients_saved = len({r["fichiers"] for r in saved})
    
    print(f"\n{'='*70}")
    print(f"RESULTATS: {elapsed:.1f}s")
    print(f"  Sauvegardés: {len(saved)} segment(s) from {patients_saved}/{len(fnames)} patients")
    print(f"  Rejetés: {len(rejected)} patient(s)")
    
    if saved:
        scores = {label: [r[key] for r in saved] for label, key in [("RHC", "rhc"), ("ECG", "ecg"), ("SCG", "scg"), ("Composite", "comp")]}
        for label, vals in scores.items():
            print(f"  {label:11s}: min={min(vals):.3f}  mean={np.mean(vals):.3f}  max={max(vals):.3f}")
    
    print(f"{'='*70}\n")


