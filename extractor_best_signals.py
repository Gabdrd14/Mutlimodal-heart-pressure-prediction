import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample
from pressure_collector import RHCP_Pipeline

# ==============================
# CONFIG
# ==============================

INPUT_FOLDER  = "processed"
DAT_FOLDER    = "dat_signals"
OUTPUT_FOLDER = "segments_30s"

FS          = 1000
WINDOW_S    = 30
WINDOW_SIZE = FS * WINDOW_S

# How many seconds to skip between candidate windows (increase for speed)
SEARCH_STRIDE_S = 5
SEARCH_STRIDE   = SEARCH_STRIDE_S * FS

# Minimum acceptable quality score (0–1). Files below this threshold are skipped.
MIN_QUALITY_SCORE = 0.3

# Clipping threshold: samples whose absolute value exceeds this fraction of the
# signal's max are considered clipped.
CLIP_FRACTION = 0.98

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# QUALITY SCORING
# ==============================

# RHC physiological pressure range (mmHg). Segments whose mean falls outside
# this range are likely artefacts or disconnected catheters.
RHC_PRESSURE_MIN =  0.0    # mmHg
RHC_PRESSURE_MAX = 50.0    # mmHg  (right-heart systolic rarely exceeds 50)


def signal_quality_score(ecg: np.ndarray, scg: np.ndarray,
                          rhc: np.ndarray | None = None) -> float:
    """
    Composite quality score in [0, 1] for a 30-second window.

    Weights:
      ECG  40 %  –  most critical for cardiac timing
      SCG  35 %  –  mechanical signal, different artefact profile from ECG
      RHC  25 %  –  pressure reference; absent RHC → neutral 0.5 contribution
    """
    ecg_score = _score_ecg(ecg)
    scg_score = _score_scg(scg)
    rhc_score = _score_rhc(rhc) if rhc is not None else 0.5

    return 0.40 * ecg_score + 0.35 * scg_score + 0.25 * rhc_score


# ── Shared helpers ───────────────────────────────────────────────────────────

def _flat_check(sig: np.ndarray) -> float | None:
    """Return 0.0 immediately if the signal is essentially flat."""
    if np.var(sig) < 1e-12:
        return 0.0
    return None


def _kurtosis_score(sig: np.ndarray, low: float = 3.0,
                    high: float = 10.0, decay: float = 40.0) -> float:
    """Score ≈ 1 when kurtosis ∈ [low, high]; penalises outliers."""
    mean = np.mean(sig)
    std  = np.std(sig) + 1e-12
    kurt = np.mean(((sig - mean) / std) ** 4)
    if kurt < low:
        return kurt / low
    if kurt <= high:
        return 1.0
    return max(0.0, 1.0 - (kurt - high) / decay)


def _clipping_score(sig: np.ndarray) -> float:
    """Penalises samples near the signal's absolute maximum (sensor saturation)."""
    clip_thr   = CLIP_FRACTION * np.max(np.abs(sig))
    clip_ratio = np.mean(np.abs(sig) >= clip_thr)
    return max(0.0, 1.0 - clip_ratio * 20.0)   # 5 % clipped → 0


def _drift_score(sig: np.ndarray) -> float:
    """Penalises slow baseline drift via a moving-mean high-pass proxy."""
    kernel = int(0.5 * FS)
    if len(sig) <= kernel:
        return 0.5
    baseline    = np.convolve(sig, np.ones(kernel) / kernel, mode='same')
    drift_ratio = np.var(baseline) / (np.var(sig) + 1e-12)
    return max(0.0, 1.0 - drift_ratio * 2.0)


# ── ECG scorer ───────────────────────────────────────────────────────────────

def _score_ecg(ecg: np.ndarray) -> float:
    """
    ECG-specific quality score.

    Sub-scores & weights:
      Flat check   (hard gate)
      Variance     20 %  – energy present
      Kurtosis     20 %  – impulsive artefacts
      Clipping     20 %  – sensor saturation
      Drift        15 %  – baseline wander
      R-peak reg.  25 %  – cardiac rhythm present and regular
    """
    if (v := _flat_check(ecg)) is not None:
        return v

    var_score   = min(np.var(ecg) / (np.median(np.abs(ecg)) + 1e-9), 1.0)
    kurt_score  = _kurtosis_score(ecg, low=3.0, high=10.0, decay=40.0)
    clip_score  = _clipping_score(ecg)
    drift_score = _drift_score(ecg)
    rpeak_score = _rpeak_regularity(ecg)

    return min(
        0.20 * var_score +
        0.20 * kurt_score +
        0.20 * clip_score +
        0.15 * drift_score +
        0.25 * rpeak_score,
        1.0,
    )


def _rpeak_regularity(ecg: np.ndarray) -> float:
    """
    Detect R-peaks on the squared signal and score the regularity of RR intervals.
    Score = 1 for a perfectly regular rhythm; 0 for no detected beats.
    """
    try:
        sq    = ecg ** 2
        thr   = 0.5 * np.max(sq)
        above = (sq > thr).astype(int)
        edges = np.where(np.diff(above) == 1)[0]

        if len(edges) < 3:
            return 0.0

        rr = np.diff(edges)
        rr = rr[(rr > 0.4 * FS) & (rr < 1.5 * FS)]   # plausible 40–150 bpm

        if len(rr) < 2:
            return 0.0

        cv = np.std(rr) / (np.mean(rr) + 1e-9)        # coefficient of variation
        return max(0.0, 1.0 - cv * 3.0)
    except Exception:
        return 0.0


# ── SCG scorer ───────────────────────────────────────────────────────────────

def _score_scg(scg: np.ndarray) -> float:
    """
    SCG-specific quality score.

    SCG has no sharp R-peaks but carries cardiac mechanical vibrations
    (10–40 Hz band).  Key failure modes:
      • Flat / disconnected sensor
      • Motion artefacts → very high kurtosis
      • Clipping
      • Baseline drift
      • Loss of cardiac frequency content (silent segment within the window)

    Sub-scores & weights:
      Flat check        (hard gate)
      Variance          20 %
      Kurtosis          25 %  – motion artefacts dominate SCG failures
      Clipping          15 %
      Drift             15 %
      Cardiac-band SNR  25 %  – energy in 10–40 Hz vs total
    """
    if (v := _flat_check(scg)) is not None:
        return v

    var_score    = min(np.var(scg) / (np.median(np.abs(scg)) + 1e-9), 1.0)
    # SCG is smoother than ECG; ideal kurtosis closer to 3–6
    kurt_score   = _kurtosis_score(scg, low=2.0, high=6.0, decay=20.0)
    clip_score   = _clipping_score(scg)
    drift_score  = _drift_score(scg)
    band_score   = _cardiac_band_score(scg)

    return min(
        0.20 * var_score +
        0.25 * kurt_score +
        0.15 * clip_score +
        0.15 * drift_score +
        0.25 * band_score,
        1.0,
    )


def _cardiac_band_score(scg: np.ndarray,
                         low_hz: float = 10.0, high_hz: float = 40.0) -> float:
    """
    Estimate the fraction of signal energy in the cardiac mechanical band
    (10–40 Hz) relative to total energy.  A good SCG segment should have most
    of its power there; motion artefacts push energy into lower frequencies.
    """
    try:
        fft_mag   = np.abs(np.fft.rfft(scg))
        freqs     = np.fft.rfftfreq(len(scg), d=1.0 / FS)
        total_pwr = np.sum(fft_mag ** 2) + 1e-12
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        band_pwr  = np.sum(fft_mag[band_mask] ** 2)
        ratio     = band_pwr / total_pwr
        # A ratio > 0.25 is good; below 0.05 suggests artefact-dominated segment
        return float(np.clip((ratio - 0.05) / 0.20, 0.0, 1.0))
    except Exception:
        return 0.5


# ── RHC scorer ───────────────────────────────────────────────────────────────

def _score_rhc(rhc: np.ndarray) -> float:
    """
    RHC pressure signal quality score.

    Key failure modes:
      • Flat line (catheter disconnected or wedged)
      • Out-of-range mean (physiologically implausible values)
      • Very low pulsatility (damped / over-wedged waveform)
      • High-frequency noise / transducer ringing

    Sub-scores & weights:
      Flat check         (hard gate)
      Physiological range 30 %  – mean pressure must be plausible
      Pulsatility         30 %  – pressure should oscillate with heartbeat
      Noise / ringing     20 %  – kurtosis-based, looser than ECG/SCG
      Clipping            20 %  – transducer saturation
    """
    if (v := _flat_check(rhc)) is not None:
        return v

    # ── 1. Physiological range ───────────────────────────────────────────────
    mean_p = np.mean(rhc)
    if mean_p < RHC_PRESSURE_MIN or mean_p > RHC_PRESSURE_MAX:
        # Clearly out of range → hard reject
        return 0.0
    # Soft score: penalise values near the boundaries
    center    = (RHC_PRESSURE_MIN + RHC_PRESSURE_MAX) / 2.0
    half_span = (RHC_PRESSURE_MAX - RHC_PRESSURE_MIN) / 2.0
    range_score = 1.0 - abs(mean_p - center) / half_span

    # ── 2. Pulsatility (peak-to-peak amplitude relative to mean) ────────────
    # Expect pulse pressure of at least 3 mmHg; over-damped → low score
    pp        = np.percentile(rhc, 95) - np.percentile(rhc, 5)
    pulse_score = float(np.clip((pp - 3.0) / 15.0, 0.0, 1.0))

    # ── 3. Noise / ringing  ─────────────────────────────────────────────────
    # RHC should be smoother than ECG; very high kurtosis → ringing artefact
    kurt_score = _kurtosis_score(rhc, low=2.0, high=8.0, decay=30.0)

    # ── 4. Clipping ──────────────────────────────────────────────────────────
    clip_score = _clipping_score(rhc)

    return min(
        0.30 * range_score +
        0.30 * pulse_score +
        0.20 * kurt_score  +
        0.20 * clip_score,
        1.0,
    )


# ==============================
# BEST-WINDOW SEARCH
# ==============================

def find_best_window(ecg: np.ndarray, scg: np.ndarray,
                      rhc: np.ndarray | None = None) -> tuple[int, float]:
    """
    Slide a WINDOW_SIZE window over ecg / scg (and rhc when available) with
    step SEARCH_STRIDE.  Return (best_start_index, best_score).
    Early-exit if a near-perfect window (score ≥ 0.9) is found.
    """
    n          = len(ecg)
    best_start = -1
    best_score = -1.0

    for start in range(0, n - WINDOW_SIZE + 1, SEARCH_STRIDE):
        end      = start + WINDOW_SIZE
        rhc_win  = rhc[start:end] if rhc is not None else None
        score    = signal_quality_score(ecg[start:end], scg[start:end], rhc_win)

        if score > best_score:
            best_score = score
            best_start = start

        if best_score >= 0.9:           # good enough → stop early
            break

    return best_start, best_score


# ==============================
# LOAD RHC
# ==============================

# Maximum tolerated NaN fraction before a signal is considered unrecoverable.
MAX_RHC_NAN_RATIO = 0.10   # 10 %


def _interpolate_nans(sig: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN runs using valid neighbours."""
    out   = sig.copy()
    nans  = np.isnan(out)
    idx   = np.arange(len(out))
    valid = ~nans
    if valid.sum() < 2:
        return out
    out[nans] = np.interp(idx[nans], idx[valid], out[valid])
    return out


def _edge_fill_nans(sig: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill edge NaNs unreachable by interpolation."""
    out = sig.copy()
    last = None
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last = out[i]
        elif last is not None:
            out[i] = last
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if not np.isnan(out[i]):
            nxt = out[i]
        elif nxt is not None:
            out[i] = nxt
    return out


def load_rhc_from_dat(mat_filename: str) -> np.ndarray | None:
    """
    Load and clean the RHC pressure signal.
    Returns a fully valid (NaN-free) float32 array, or None if the signal
    cannot be recovered. Callers must skip the file when None is returned.
    """
    base = mat_filename.replace(".mat", "").replace(".", "-")

    for f in os.listdir(DAT_FOLDER):
        if not (f.endswith(".dat") and base in f):
            continue
        try:
            record_path = os.path.join(DAT_FOLDER, f.replace(".dat", ""))
            pipeline    = RHCP_Pipeline(record_path)
            data        = pipeline.run()

            rhc = np.asarray(data.get("RHC_pressure")).squeeze().astype(np.float32)

            if rhc.ndim == 0 or len(rhc) == 0:
                print("  [RHC] empty array from pipeline — skipping file")
                return None

            nan_ratio = float(np.isnan(rhc).mean())

            if nan_ratio == 1.0:
                print("  [RHC] entirely NaN — skipping file")
                return None

            if nan_ratio > MAX_RHC_NAN_RATIO:
                print(f"  [RHC] {nan_ratio*100:.1f}% NaNs — too many to repair, skipping file")
                return None

            # Repair isolated NaN runs
            if nan_ratio > 0.0:
                rhc = _interpolate_nans(rhc)
                if np.isnan(rhc).any():
                    rhc = _edge_fill_nans(rhc)
                # If NaNs still remain after both repair passes, discard
                if np.isnan(rhc).any():
                    print("  [RHC] NaNs unrepairable — skipping file")
                    return None
                print(f"  [RHC] repaired NaN samples ({nan_ratio*100:.1f}% affected)")

            return rhc

        except Exception as e:
            print(f"  [RHC] pipeline failed: {e} — skipping file")
            return None

    print(f"  [RHC] no .dat file found for {mat_filename} — skipping file")
    return None


# ==============================
# MAIN
# ==============================

print("Starting processing...")
summary = []   # collect per-patient results for a final report

for fname in sorted(os.listdir(INPUT_FOLDER)):

    if not fname.endswith(".mat"):
        continue

    print(f"\nProcessing {fname}")

    try:
        data = sio.loadmat(os.path.join(INPUT_FOLDER, fname))

        # ── Load all signals ────────────────────────────────────────────────
        ecg          = np.asarray(data["ecg_clean"]).squeeze()
        scg          = np.asarray(data["scg_clean"]).squeeze()
        ecg_raw      = np.asarray(data["ecg_raw"]).squeeze()
        scg_raw      = np.asarray(data["scg_raw"]).squeeze()
        patch_ACC_lat = np.asarray(data["patch_ACC_lat"]).squeeze()
        patch_ACC_hf  = np.asarray(data["patch_ACC_hf"]).squeeze()
        patch_ACC_dv  = np.asarray(data["patch_ACC_dv"]).squeeze()

        # ── Validate that all signals have the same length ──────────────────
        signal_lengths = {
            "ecg":          len(ecg),
            "scg":          len(scg),
            "ecg_raw":      len(ecg_raw),
            "scg_raw":      len(scg_raw),
            "patch_ACC_lat": len(patch_ACC_lat),
            "patch_ACC_hf":  len(patch_ACC_hf),
            "patch_ACC_dv":  len(patch_ACC_dv),
        }
        n = len(ecg)
        mismatched = {k: v for k, v in signal_lengths.items() if v != n}
        if mismatched:
            print(f"  [WARN] length mismatch: {mismatched} — truncating to shortest")
            n = min(signal_lengths.values())
            ecg           = ecg[:n]
            scg           = scg[:n]
            ecg_raw       = ecg_raw[:n]
            scg_raw       = scg_raw[:n]
            patch_ACC_lat = patch_ACC_lat[:n]
            patch_ACC_hf  = patch_ACC_hf[:n]
            patch_ACC_dv  = patch_ACC_dv[:n]

        if n < WINDOW_SIZE:
            print(f"  Skipping {fname} (recording too short: {n/FS:.1f}s)")
            summary.append({"file": fname, "status": "too_short"})
            continue

        # ── Load & resample RHC — mandatory: skip file if unavailable ────────
        rhc = load_rhc_from_dat(fname)
        if rhc is None:
            summary.append({"file": fname, "status": "no_rhc"})
            continue

        if len(rhc) != n:
            print(f"  [RHC] resampling from {len(rhc)} to {n} samples")
            rhc = resample(rhc, n).astype(np.float32)
            # scipy.resample can introduce edge NaNs via spectral ringing — repair
            if np.isnan(rhc).any():
                rhc = _interpolate_nans(rhc)
            if np.isnan(rhc).any():
                rhc = _edge_fill_nans(rhc)
            if np.isnan(rhc).any():
                print("  [RHC] NaNs after resampling unrepairable — skipping file")
                summary.append({"file": fname, "status": "no_rhc"})
                continue

        # ── Find best 30-second window (ECG + SCG + RHC all scored) ──────────
        best_start, best_score = find_best_window(ecg, scg, rhc)

        s, e = best_start, best_start + WINDOW_SIZE

        # Detailed per-signal breakdown for logging
        ecg_sc  = _score_ecg(ecg[s:e])
        scg_sc  = _score_scg(scg[s:e])
        rhc_sc  = _score_rhc(rhc[s:e])

        print(f"  Best window : {s/FS:.1f}s – {e/FS:.1f}s  "
              f"(composite={best_score:.3f} | "
              f"ecg={ecg_sc:.3f}  scg={scg_sc:.3f}  rhc={rhc_sc:.3f})")

        if best_score < MIN_QUALITY_SCORE:
            print(f"  Skipping {fname} (quality too low: {best_score:.3f} < {MIN_QUALITY_SCORE})")
            summary.append({"file": fname, "status": "low_quality",
                             "score": best_score,
                             "ecg_score": ecg_sc, "scg_score": scg_sc, "rhc_score": rhc_sc})
            continue

        # ── Extract segments ──────────────────────────────────────────────────
        ecg_seg           = ecg[s:e]
        scg_seg           = scg[s:e]
        ecg_raw_seg       = ecg_raw[s:e]
        scg_raw_seg       = scg_raw[s:e]
        patch_ACC_lat_seg = patch_ACC_lat[s:e]
        patch_ACC_hf_seg  = patch_ACC_hf[s:e]
        patch_ACC_dv_seg  = patch_ACC_dv[s:e]
        rhc_seg           = rhc[s:e]

        # Paranoia check — should never trigger given all guards above
        if np.isnan(rhc_seg).any():
            print(f"  [ERROR] unexpected NaN in rhc_seg for {fname} — skipping")
            summary.append({"file": fname, "status": "no_rhc"})
            continue

        # ── Save ──────────────────────────────────────────────────────────────
        patient_id = fname.replace(".mat", "").split(".")[0]
        out_path   = os.path.join(OUTPUT_FOLDER, f"{patient_id}_segment.mat")

        sio.savemat(out_path, {
            "patient":         patient_id,
            "ecg":             ecg_seg,
            "scg":             scg_seg,
            "ecg_raw":         ecg_raw_seg,
            "scg_raw":         scg_raw_seg,
            "rhc":             rhc_seg,
            "patch_ACC_lat":   patch_ACC_lat_seg,
            "patch_ACC_hf":    patch_ACC_hf_seg,
            "patch_ACC_dv":    patch_ACC_dv_seg,
            # traceability metadata
            "quality_score":   np.array([best_score]),
            "quality_ecg":     np.array([ecg_sc]),
            "quality_scg":     np.array([scg_sc]),
            "quality_rhc":     np.array([rhc_sc]),
            "window_start_s":  np.array([s / FS]),
            "window_end_s":    np.array([e / FS]),
            "fs":              np.array([FS]),
        })

        print(f"  Saved → {out_path}")
        summary.append({"file": fname, "status": "ok", "score": best_score,
                         "ecg_score": ecg_sc, "scg_score": scg_sc,
                         "rhc_score": rhc_sc, "start_s": s / FS})

    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        summary.append({"file": fname, "status": "error", "error": str(e)})

# ── Print summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"DONE — {len(summary)} file(s) processed")
ok      = [r for r in summary if r["status"] == "ok"]
no_rhc  = [r for r in summary if r["status"] == "no_rhc"]
other   = [r for r in summary if r["status"] not in ("ok", "no_rhc")]
print(f"  Saved        : {len(ok)}")
print(f"  Skipped (no valid RHC) : {len(no_rhc)}")
print(f"  Skipped (other)        : {len(other)}")
if ok:
    for label, key in [("Composite", "score"), ("ECG", "ecg_score"),
                        ("SCG", "scg_score"), ("RHC", "rhc_score")]:
        vals = [r[key] for r in ok if not np.isnan(r.get(key, float("nan")))]
        if vals:
            print(f"  {label:10s}: min={min(vals):.3f}  "
                  f"mean={np.mean(vals):.3f}  max={max(vals):.3f}")
for r in no_rhc:
    print(f"  ✗ [no RHC]  {r['file']}")
for r in other:
    print(f"  ✗ [{r['status']}]  {r['file']}")
print("=" * 60)