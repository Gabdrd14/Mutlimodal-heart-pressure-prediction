"""
RHC Segment Predictor — V2
===========================
Loads a single segment .mat file and predicts RHC pressure per second using
a trained RHCLightningModuleV2 checkpoint (3-branch architecture).

Usage:
  python predict_segment_v2.py --segment path/to/segment.mat
  python predict_segment_v2.py --segment FILE --checkpoint CHECKPOINT_PATH
  python predict_segment_v2.py --segment FILE --plot --save_fig output.png
"""

import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

# Import model components from V2
sys.path.insert(0, str(Path(__file__).parent / "processed"))
from rhc_predictor_V2 import (
    RHCLightningModuleV2, RHCSequencePredictorV2,
    N_SIGNALS, N_SCALAR, N_BB, BB_LEN,
    D_MODEL, D_SCALAR, D_BB,
    CNN_DROPOUT, TRANS_DROPOUT,
    SIGNAL_KEYS, SCALAR_KEYS, BB_KEYS,
    FS, WINDOW_SIZE, N_SECONDS,
    LR, WEIGHT_DECAY,
    load_scalar, load_bb,
)

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTOR CLASS
# ──────────────────────────────────────────────────────────────────────────────

class SegmentPredictor:
    """
    Load V2 checkpoint and make predictions on single segments.
    """

    def __init__(self, checkpoint_path: str | None = None):
        """
        Args:
            checkpoint_path: Path to .ckpt file. If None, finds the best checkpoint
                            in checkpoints_v2/ folder.
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()

        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint provided and none found in checkpoints_v2/ folder"
            )

        self.checkpoint_path = Path(checkpoint_path)
        print(f"[Predictor] Loading checkpoint: {self.checkpoint_path}")

        # Load the V2 Lightning module
        self.model_lightning = RHCLightningModuleV2.load_from_checkpoint(
            str(self.checkpoint_path),
            n_signals=N_SIGNALS,
            n_scalar=N_SCALAR,
            n_bb=N_BB,
            d_model=D_MODEL,
            d_scalar=D_SCALAR,
            d_bb=D_BB,
            cnn_dropout=CNN_DROPOUT,
            trans_dropout=TRANS_DROPOUT,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.model_lightning.eval()
        self.device = torch.device("cpu")
        self.model_lightning.to(self.device)

        # Retrieve normalisation stats saved in hparams
        self.rhc_mean = float(self.model_lightning.hparams.rhc_mean)
        self.rhc_std  = float(self.model_lightning.hparams.rhc_std)

        # scalar_mean / scalar_std may or may not be saved in hparams depending
        # on training — we fall back to zeros/ones if absent (inference still
        # works, just less well-normalised).
        hp = self.model_lightning.hparams
        if hasattr(hp, "scalar_mean") and hp.scalar_mean is not None:
            self.scalar_mean = np.asarray(hp.scalar_mean, dtype=np.float32)
            self.scalar_std  = np.asarray(hp.scalar_std,  dtype=np.float32)
        else:
            print("[Predictor] Warning: scalar normalisation stats not found in "
                  "checkpoint hparams — using mean=0, std=1 as fallback.")
            self.scalar_mean = np.zeros(N_SCALAR, dtype=np.float32)
            self.scalar_std  = np.ones(N_SCALAR,  dtype=np.float32)

    @staticmethod
    def _find_best_checkpoint() -> Path | None:
        """Find the latest checkpoint in checkpoints_v2/ folder."""
        for folder in ["processed/checkpoints_v2", "checkpoints_v2", "processed/checkpoints"]:
            checkpoint_dir = Path(folder)
            if checkpoint_dir.exists():
                ckpts = sorted(checkpoint_dir.glob("**/*.ckpt"))
                if ckpts:
                    return ckpts[-1]
        return None

    def predict(self, segment_path: str) -> dict:
        """
        Predict RHC for a segment using the 3-branch V2 model.

        Args:
            segment_path: Path to segment .mat file

        Returns:
            dict with keys:
              - pred:    (N_SECONDS,) float — predicted per-second RHC in mmHg
              - target:  (N_SECONDS,) float or None — ground truth per-second RHC
              - signals: dict of normalised input time-series
              - scalars: dict of raw scalar values
              - stats:   dict with quality metrics
        """
        segment_path = Path(segment_path)
        if not segment_path.exists():
            raise FileNotFoundError(f"Segment not found: {segment_path}")

        print(f"\n[Segment] Loading: {segment_path.name}")
        mat_data = sio.loadmat(str(segment_path))

        # ── Branch 1 : temporal signals (N_SIGNALS, 15000) ──────────────────
        channels = []
        for key in SIGNAL_KEYS:
            if key not in mat_data:
                raise KeyError(f"Missing signal key: '{key}'")
            sig = np.asarray(mat_data[key]).squeeze().astype(np.float32)
            if len(sig) != WINDOW_SIZE:
                raise ValueError(
                    f"Signal '{key}' has wrong length: {len(sig)} != {WINDOW_SIZE}"
                )
            sig_norm = (sig - sig.mean()) / (sig.std() + 1e-8)
            channels.append(sig_norm)
        x_sig = torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0)  # (1, N_SIGNALS, 15000)

        # ── Branch 2 : scalar features (N_SCALAR,) ──────────────────────────
        scalars_raw = np.array(
            [load_scalar(mat_data, k) for k in SCALAR_KEYS], dtype=np.float32
        )
        scalars_norm = scalars_raw.copy()
        nan_mask = np.isnan(scalars_norm)
        if nan_mask.any():
            print(f"[Segment] Warning: {nan_mask.sum()} scalar(s) missing — "
                  "replaced by train mean.")
            scalars_norm[nan_mask] = self.scalar_mean[nan_mask]
        scalars_norm = (scalars_norm - self.scalar_mean) / self.scalar_std
        x_scalar = torch.from_numpy(scalars_norm).unsqueeze(0)  # (1, N_SCALAR)

        # ── Branch 3 : beat-by-beat series (N_BB, BB_LEN) ───────────────────
        bb_channels = []
        for key in BB_KEYS:
            arr = load_bb(mat_data, key, target_len=BB_LEN)
            arr_norm = (arr - arr.mean()) / (arr.std() + 1e-8)
            bb_channels.append(arr_norm)
        x_bb = torch.from_numpy(np.stack(bb_channels, axis=0)).unsqueeze(0)  # (1, N_BB, BB_LEN)

        # ── Ground truth (optional) ──────────────────────────────────────────
        target = None
        if "rhc" in mat_data:
            rhc_raw = np.asarray(mat_data["rhc"]).squeeze().astype(np.float32)
            target = np.array(
                [np.mean(rhc_raw[s * FS:(s + 1) * FS]) for s in range(N_SECONDS)],
                dtype=np.float32,
            )

        # ── Inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            x_sig    = x_sig.to(self.device)
            x_scalar = x_scalar.to(self.device)
            x_bb     = x_bb.to(self.device)

            pred_norm = self.model_lightning.model(x_sig, x_scalar, x_bb)  # (1, N_SECONDS)
            pred = (pred_norm * self.rhc_std + self.rhc_mean).cpu().numpy().squeeze()  # (N_SECONDS,)

        stats = self._compute_stats(pred, target)

        return {
            "pred":       pred,
            "target":     target,
            "signals":    {key: channels[i] for i, key in enumerate(SIGNAL_KEYS)},
            "scalars":    {key: float(scalars_raw[i]) for i, key in enumerate(SCALAR_KEYS)},
            "stats":      stats,
            "checkpoint": str(self.checkpoint_path),
        }

    @staticmethod
    def _compute_stats(pred: np.ndarray, target: np.ndarray | None) -> dict:
        stats = {"pred_mean": float(np.mean(pred)), "pred_std": float(np.std(pred))}
        if target is not None:
            mae  = float(np.mean(np.abs(pred - target)))
            rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
            stats["mae_mmhg"]    = mae
            stats["rmse_mmhg"]   = rmse
            stats["target_mean"] = float(np.mean(target))
            stats["target_std"]  = float(np.std(target))
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_prediction(result: dict, segment_name: str, save_path: str | None = None):
    """
    Plot predicted vs target RHC and input signals (ECG & SCG).
    """
    fig = plt.figure(figsize=(14, 10))
    time_s = np.arange(N_SECONDS)
    pred   = result["pred"]
    target = result["target"]

    # ── RHC plot ─────────────────────────────────────────────────────────────
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(time_s, pred, "b-o", label="Predicted RHC",    linewidth=2, markersize=4)
    if target is not None:
        ax1.plot(time_s, target, "r-s", label="Ground Truth RHC", linewidth=2, markersize=4)
        mae  = result["stats"].get("mae_mmhg",  0)
        rmse = result["stats"].get("rmse_mmhg", 0)
        ax1.text(
            0.02, 0.98,
            f"MAE: {mae:.2f} mmHg | RMSE: {rmse:.2f} mmHg",
            transform=ax1.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("RHC Pressure (mmHg)")
    ax1.set_title(f"{segment_name} — RHC Prediction (V2 model)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, N_SECONDS - 1)

    # ── Input signals (ECG & SCG) ─────────────────────────────────────────────
    ax2 = plt.subplot(2, 1, 2)
    signals = result["signals"]
    ecg = signals.get("ecg")
    scg = signals.get("scg")

    if ecg is not None:
        ax2.plot(np.linspace(0, N_SECONDS, len(ecg)), ecg,
                 "b-", alpha=0.7, label="ECG", linewidth=1)
        ax2.set_ylabel("ECG (Z-scored)", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

    if scg is not None:
        ax2_right = ax2.twinx()
        ax2_right.plot(np.linspace(0, N_SECONDS, len(scg)), scg,
                       "r-", alpha=0.7, label="SCG", linewidth=1)
        ax2_right.set_ylabel("SCG (Z-scored)", color="r")
        ax2_right.tick_params(axis="y", labelcolor="r")

    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Input Signals (ECG & SCG)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    for s in range(1, N_SECONDS):
        ax2.axvline(s, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved to: {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# PRINTING
# ──────────────────────────────────────────────────────────────────────────────

def print_results(result: dict, segment_name: str):
    """Pretty-print prediction results."""
    print("\n" + "=" * 80)
    print(f"PREDICTION RESULTS (V2): {segment_name}")
    print("=" * 80)

    stats  = result["stats"]
    pred   = result["pred"]
    target = result["target"]

    print(f"\nPredicted RHC (per-second, mmHg):")
    print(f"  Mean:  {stats['pred_mean']:6.2f} mmHg")
    print(f"  Std:   {stats['pred_std']:6.2f} mmHg")
    print(f"  Min:   {np.min(pred):6.2f} mmHg")
    print(f"  Max:   {np.max(pred):6.2f} mmHg")
    print(f"  Range: {np.max(pred) - np.min(pred):6.2f} mmHg")

    if target is not None:
        print(f"\nGround Truth RHC (per-second, mmHg):")
        print(f"  Mean:  {stats['target_mean']:6.2f} mmHg")
        print(f"  Std:   {stats['target_std']:6.2f} mmHg")
        print(f"  Min:   {np.min(target):6.2f} mmHg")
        print(f"  Max:   {np.max(target):6.2f} mmHg")
        print(f"  Range: {np.max(target) - np.min(target):6.2f} mmHg")

        errors = np.abs(pred - target)
        print(f"\nError Metrics:")
        print(f"  MAE:          {stats['mae_mmhg']:6.2f} mmHg")
        print(f"  RMSE:         {stats['rmse_mmhg']:6.2f} mmHg")
        print(f"  Max error:    {np.max(errors):5.2f} mmHg (second {np.argmax(errors)})")

    # Show a summary of scalar features
    #print(f"\nScalar features loaded ({N_SCALAR} total):")
    #for key, val in result["scalars"].items():
     #   flag = "  ⚠ NaN" if np.isnan(val) else ""
        #print(f"  {key:<22}: {val:8.2f}{flag}")

    print(f"\nModel checkpoint:")
    print(f"  {result['checkpoint']}")
    print("\n" + "=" * 80)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict RHC for a single segment using V2 trained model"
    )
    parser.add_argument(
        "--segment", type=str, required=True,
        help="Path to segment .mat file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .ckpt checkpoint (auto-finds best if not specified)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Display matplotlib plot",
    )
    parser.add_argument(
        "--save_fig", type=str, default=None,
        help="Save plot to file (PNG/PDF)",
    )
    parser.add_argument(
        "--no_print", action="store_true",
        help="Skip printing results",
    )

    args = parser.parse_args()

    try:
        predictor = SegmentPredictor(checkpoint_path=args.checkpoint)
        result     = predictor.predict(args.segment)

        if not args.no_print:
            print_results(result, Path(args.segment).stem)

        if args.plot or args.save_fig:
            plot_prediction(result, Path(args.segment).stem, save_path=args.save_fig)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)