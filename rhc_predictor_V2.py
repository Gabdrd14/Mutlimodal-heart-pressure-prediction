"""
Prédicteur RHC v2 — Architecture Hybride à 3 Branches
======================================================
Prédit la pression RHC à 1 Hz (30 valeurs / fenêtre 30s)
à partir de trois types de données :

  Branche 1 — Signaux temporels (7 × 15000) :
    CNN + Transformer inter-signaux (identique à v1)

  Branche 2 — Scalaires par segment (N_SCALAR,) :
    MLP → vecteur répété 30 fois pour alignement temporel

  Branche 3 — Séries beat-by-beat (N_BB × 30) :
    CNN 1D léger → 30 tokens temporels

  Fusion : concat des 3 branches + projection → décodeur → (B, 30)

Nouveaux channels utilisés :
  Scalaires  : fc_mean, fc_median, rr_mean, rr_std, pr_median, pr_mean,
               qt_median, qt_mean, qtc_median, qtc_mean,
               pep_median, pep_mean, et_median, et_mean,
               ivct_median, ivct_mean, ivrt_median, ivrt_mean
  Beat-by-beat (ECG) : rr_bb, pr_bb, qt_bb, qtc_bb
  Beat-by-beat (SCG) : pep_bb, et_bb, ivct_bb, ivrt_bb

Perte : 0.7 * MSE + 0.3 * MAE + SMOOTH_LAMBDA * smoothness

Utilisation
-----------
  python rhc_predictor_v2.py --segments_folder segments_30s_strict
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

FS          = 500
WINDOW_S    = 30
WINDOW_SIZE = FS * WINDOW_S   # 15 000
N_SECONDS   = WINDOW_S        # 30 tokens de sortie

# --- Branche 1 : signaux temporels ---
SIGNAL_KEYS = [
    "ecg", "scg",
    "patch_ACC_lat", "patch_ACC_hf", "patch_ACC_dv",
]
N_SIGNALS = len(SIGNAL_KEYS)  # 5

# --- Branche 2 : scalaires par segment ---
SCALAR_KEYS = [
    "fc_mean_bpm", "fc_median_bpm",
    "rr_mean_ms",  "rr_std_ms",
    "pr_median_ms", "pr_mean_ms",
    "qt_median_ms", "qt_mean_ms",
    "qtc_median_ms", "qtc_mean_ms",
    "pep_median_ms", "pep_mean_ms",
    "et_median_ms",  "et_mean_ms",
    "ivct_median_ms", "ivct_mean_ms",
    "ivrt_median_ms", "ivrt_mean_ms",
]
N_SCALAR = len(SCALAR_KEYS)   # 18

# --- Branche 3 : séries beat-by-beat ---
# ECG-derived : rr, pr, qt, qtc
# SCG-derived : pep, et, ivct, ivrt
BB_ECG_KEYS = ["rr_bb_ms", "pr_bb_ms", "qt_bb_ms", "qtc_bb_ms"]
BB_SCG_KEYS = ["pep_bb_ms", "et_bb_ms", "ivct_bb_ms", "ivrt_bb_ms"]
BB_KEYS     = BB_ECG_KEYS + BB_SCG_KEYS
N_BB        = len(BB_KEYS)            # 8
BB_LEN      = N_SECONDS               # on resample à 30 points

# --- Architecture ---
CNN_CHANNELS  = [1, 32, 64, 128]
CNN_KERNELS   = [15, 9, 5]
CNN_STRIDES   = [2,  2, 2]
CNN_DROPOUT   = 0.2

D_MODEL       = 128
D_SCALAR      = D_MODEL // 4   # 32
D_BB          = D_MODEL // 4   # 32
D_FUSED       = D_MODEL + D_SCALAR + D_BB  # 192 → projeté vers D_MODEL

N_HEADS       = 4
FF_DIM        = 256
N_LAYERS      = 2
TRANS_DROPOUT = 0.1

SMOOTH_LAMBDA = 0.05

# Training
BATCH_SIZE   = 8
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 25


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_scalar(mat: dict, key: str) -> float:
    """Charge un scalaire depuis le mat, retourne NaN si absent."""
    if key not in mat:
        return float("nan")
    v = np.asarray(mat[key]).squeeze()
    return float(v) if v.ndim == 0 else float(v.flat[0])


def load_bb(mat: dict, key: str, target_len: int = BB_LEN) -> np.ndarray:
    """
    Charge une série beat-by-beat, remplace NaN par interpolation linéaire,
    et resample à target_len points.
    Retourne un array float32 de shape (target_len,).
    """
    if key not in mat:
        return np.zeros(target_len, dtype=np.float32)

    arr = np.asarray(mat[key]).squeeze().astype(np.float32)

    # Remplacer NaN par interpolation linéaire
    nans = np.isnan(arr)
    if nans.all():
        return np.zeros(target_len, dtype=np.float32)
    if nans.any():
        x   = np.arange(len(arr))
        arr = np.interp(x, x[~nans], arr[~nans]).astype(np.float32)

    # Resample vers target_len via interpolation
    x_src = np.linspace(0, 1, len(arr))
    x_dst = np.linspace(0, 1, target_len)
    arr   = np.interp(x_dst, x_src, arr).astype(np.float32)

    return arr


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class RHCDataset(Dataset):
    """
    Retourne :
      x_sig    : (N_SIGNALS, 15000) float32  — signaux temporels normalisés
      x_scalar : (N_SCALAR,)        float32  — scalaires normalisés (global)
      x_bb     : (N_BB, BB_LEN)     float32  — beat-by-beat normalisés (global)
      y        : (N_SECONDS,)       float32  — RHC par seconde normalisé
    """

    def __init__(self, segments_folder: str, augment: bool = False,
                 rhc_mean: float = None, rhc_std: float = None,
                 scalar_mean: np.ndarray = None, scalar_std: np.ndarray = None,
                 bb_mean: np.ndarray = None, bb_std: np.ndarray = None):
        self.folder  = Path(segments_folder)
        self.augment = augment
        self.files   = sorted(self.folder.glob("*_seg*.mat")) or \
                       sorted(self.folder.glob("*_segment*.mat"))

        if not self.files:
            raise FileNotFoundError(f"No .mat files in '{segments_folder}'.")

        # Filtrage fichiers invalides
        valid, skipped = [], []
        for f in self.files:
            try:
                mat = sio.loadmat(f)
                rhc = np.asarray(mat["rhc"]).squeeze()
                if rhc.ndim == 0 or len(rhc) < WINDOW_SIZE or np.any(np.isnan(rhc)):
                    skipped.append(f.name)
                else:
                    valid.append(f)
            except Exception:
                skipped.append(f.name)

        self.files = valid
        if skipped:
            print(f"[Dataset] Dropped {len(skipped)} file(s) with invalid RHC.")

        # Normalisation RHC
        if rhc_mean is None or rhc_std is None:
            self.rhc_mean, self.rhc_std = self._compute_rhc_stats()
        else:
            self.rhc_mean, self.rhc_std = rhc_mean, rhc_std

        # Normalisation scalaires
        if scalar_mean is None or scalar_std is None:
            self.scalar_mean, self.scalar_std = self._compute_scalar_stats()
        else:
            self.scalar_mean = scalar_mean
            self.scalar_std  = scalar_std

        # Normalisation beat-by-beat (globale)
        if bb_mean is None or bb_std is None:
            self.bb_mean, self.bb_std = self._compute_bb_stats()
        else:
            self.bb_mean = bb_mean
            self.bb_std  = bb_std

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _compute_rhc_stats(self):

        """
        On calcule la moyenne et l'écart-type globaux de la pression RHC par seconde
        sur tout le dataset (train + val + test) pour la normalisation.
        
        """
        all_vals = []
        for f in self.files:
            mat = sio.loadmat(f)
            rhc = np.asarray(mat["rhc"]).squeeze().astype(np.float32)
            for s in range(N_SECONDS):
                block = rhc[s * FS:(s + 1) * FS]
                if len(block) == FS:
                    all_vals.append(float(np.mean(block)))
        return float(np.mean(all_vals)), float(np.std(all_vals)) + 1e-8

    def _compute_scalar_stats(self):

        """
        On calcule la moyenne et l'écart-type globaux de chaque scalaire sur tout le dataset.
        On empile tous les scalaires → (N_files, N_SCALAR) puis stats
        
        """

        all_scalars = []
        for f in self.files:
            mat = sio.loadmat(f)
            row = [load_scalar(mat, k) for k in SCALAR_KEYS]
            all_scalars.append(row)
        arr = np.array(all_scalars, dtype=np.float32)
        for j in range(arr.shape[1]):
            col    = arr[:, j]
            median = np.nanmedian(col)
            arr[np.isnan(arr[:, j]), j] = median
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr,  axis=0) + 1e-8
        return mean, std

    def _compute_bb_stats(self):
        """
        Calcule mean/std globales de chaque canal BB sur tout le dataset.
        On empile tous les segments → (N_files * BB_LEN, N_BB) puis stats.
        """
        all_bb = []
        for f in self.files:
            mat = sio.loadmat(f)
            row = [load_bb(mat, k, target_len=BB_LEN) for k in BB_KEYS]
            all_bb.append(np.stack(row, axis=0))   # (N_BB, BB_LEN)

        arr = np.stack(all_bb, axis=0)             # (N_files, N_BB, BB_LEN)
        # Reshape : (N_files * BB_LEN, N_BB) pour avoir les stats par canal
        arr  = arr.transpose(0, 2, 1).reshape(-1, N_BB)
        mean = np.nanmean(arr, axis=0)             # (N_BB,)
        std  = np.nanstd(arr,  axis=0) + 1e-8
        return mean, std

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        mat = sio.loadmat(self.files[idx])

        # --- y : RHC par seconde normalisé ---
        rhc = np.asarray(mat["rhc"]).squeeze().astype(np.float32)
        y   = np.array(
            [np.mean(rhc[s * FS:(s + 1) * FS]) for s in range(N_SECONDS)],
            dtype=np.float32,
        )
        y = (y - self.rhc_mean) / self.rhc_std
        y = torch.from_numpy(y)

        # --- x_sig : signaux temporels (N_SIGNALS, 15000) — normalisation locale ---
        channels = []
        for key in SIGNAL_KEYS:
            sig = np.asarray(mat[key]).squeeze().astype(np.float32)
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            channels.append(sig)
        x_sig = torch.from_numpy(np.stack(channels, axis=0))   # (N_SIGNALS, 15000)

        # --- x_scalar : scalaires — normalisation globale (N_SCALAR,) ---
        scalars  = np.array([load_scalar(mat, k) for k in SCALAR_KEYS],
                            dtype=np.float32)
        nan_mask = np.isnan(scalars)
        scalars[nan_mask] = self.scalar_mean[nan_mask]
        scalars  = (scalars - self.scalar_mean) / self.scalar_std
        x_scalar = torch.from_numpy(scalars)                    # (N_SCALAR,)

        # --- x_bb : beat-by-beat — normalisation globale (N_BB, BB_LEN) ---
        bb_channels = []
        for i, key in enumerate(BB_KEYS):
            arr = load_bb(mat, key, target_len=BB_LEN)          # (BB_LEN,)
            arr = (arr - self.bb_mean[i]) / self.bb_std[i]
            bb_channels.append(arr)
        x_bb = torch.from_numpy(np.stack(bb_channels, axis=0))  # (N_BB, BB_LEN)

        if self.augment:
            x_sig, x_scalar, x_bb = self._augment(x_sig, x_scalar, x_bb)

        return x_sig, x_scalar, x_bb, y

    @staticmethod
    def _augment(x_sig, x_scalar, x_bb):
        # Bruit amplitude sur les signaux temporels
        x_sig = x_sig * (1.0 + 0.10 * (torch.rand(x_sig.shape[0], 1) * 2 - 1))
        x_sig = x_sig + 0.02 * torch.randn_like(x_sig)
        # Flip axes accéléromètre
        for i in range(2, 5):
            if torch.rand(1).item() > 0.5:
                x_sig[i] = -x_sig[i]
        # Bruit léger sur scalaires et bb
        x_scalar = x_scalar + 0.01 * torch.randn_like(x_scalar)
        x_bb     = x_bb     + 0.01 * torch.randn_like(x_bb)
        return x_sig, x_scalar, x_bb


# ──────────────────────────────────────────────────────────────────────────────
# BRANCHE 1 : SignalEncoder
# ──────────────────────────────────────────────────────────────────────────────

class SignalEncoder(nn.Module):
    """CNN 1D : (B, 1, 15000) → (B, D_MODEL, 30)"""

    def __init__(self, d_model: int = D_MODEL, dropout: float = CNN_DROPOUT):
        super().__init__()
        layers = []
        in_ch  = CNN_CHANNELS[0]
        for out_ch, k, s in zip(CNN_CHANNELS[1:], CNN_KERNELS, CNN_STRIDES):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers += [
            nn.Conv1d(in_ch, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=62, stride=62),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, D_MODEL, N_SECONDS)


class CrossSignalTemporalTransformer(nn.Module):
    """Transformer inter-signaux : (B, N_SIGNALS, D, T) → (B, T, D)"""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
                 n_layers=N_LAYERS, dropout=TRANS_DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.signal_emb  = nn.Parameter(torch.zeros(1, N_SIGNALS, d_model))

    def forward(self, x: torch.Tensor):
        B, N, D, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * T, N, D)
        x = x + self.signal_emb
        x = self.transformer(x)
        x = x.mean(dim=1)
        return x.reshape(B, T, D)                # (B, N_SECONDS, D_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
# BRANCHE 2 : ScalarEncoder
# ──────────────────────────────────────────────────────────────────────────────

class ScalarEncoder(nn.Module):
    """
    MLP sur les scalaires segment → vecteur répété T fois.
    (B, N_SCALAR) → (B, T, D_SCALAR)
    """

    def __init__(self, n_scalar=N_SCALAR, d_out=D_SCALAR, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_scalar, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, d_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, t: int = N_SECONDS) -> torch.Tensor:
        h = self.mlp(x)                                    # (B, D_SCALAR)
        return h.unsqueeze(1).expand(-1, t, -1)            # (B, T, D_SCALAR)


# ──────────────────────────────────────────────────────────────────────────────
# BRANCHE 3 : BeatByBeatEncoder
# ──────────────────────────────────────────────────────────────────────────────

class BeatByBeatEncoder(nn.Module):
    """
    CNN 1D léger sur les séries beat-by-beat (normalisées globalement).
    (B, N_BB, BB_LEN=30) → (B, T=30, D_BB)
    """

    def __init__(self, n_bb=N_BB, d_out=D_BB, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_bb, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, d_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)           # (B, D_BB, BB_LEN=30)
        return h.transpose(1, 2)   # (B, T=30, D_BB)


# ──────────────────────────────────────────────────────────────────────────────
# DÉCODEUR
# ──────────────────────────────────────────────────────────────────────────────

class TemporalDecoder(nn.Module):
    """MLP partagé + conv causale : (B, T, D_MODEL) → (B, T)"""

    def __init__(self, d_model=D_MODEL, dropout=0.2):
        super().__init__()
        self.causal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=2,
            groups=d_model, bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.causal_conv(x.transpose(1, 2))
        h = h[:, :, :x.shape[1]]
        h = self.norm(h.transpose(1, 2))
        x = x + h
        return self.mlp(x).squeeze(-1)    # (B, T)


# ──────────────────────────────────────────────────────────────────────────────
# MODÈLE COMPLET v2
# ──────────────────────────────────────────────────────────────────────────────

class RHCSequencePredictorV2(nn.Module):
    """
    Architecture hybride 3 branches :
      Branche 1 : SignalEncoder × N_SIGNALS + CrossSignalTransformer → (B, T, D_MODEL)
      Branche 2 : ScalarEncoder                                       → (B, T, D_SCALAR)
      Branche 3 : BeatByBeatEncoder                                   → (B, T, D_BB)
      Fusion    : concat + LayerNorm + projection                     → (B, T, D_MODEL)
      Décodeur  : TemporalDecoder                                     → (B, T)
    """

    def __init__(self, n_signals=N_SIGNALS, n_scalar=N_SCALAR, n_bb=N_BB,
                 d_model=D_MODEL, d_scalar=D_SCALAR, d_bb=D_BB,
                 cnn_dropout=CNN_DROPOUT, trans_dropout=TRANS_DROPOUT):
        super().__init__()

        # Branche 1
        self.encoders    = nn.ModuleList([
            SignalEncoder(d_model=d_model, dropout=cnn_dropout)
            for _ in range(n_signals)
        ])
        self.transformer = CrossSignalTemporalTransformer(
            d_model=d_model, dropout=trans_dropout
        )

        # Branche 2
        self.scalar_enc  = ScalarEncoder(n_scalar=n_scalar, d_out=d_scalar)

        # Branche 3
        self.bb_enc      = BeatByBeatEncoder(n_bb=n_bb, d_out=d_bb)

        # Fusion : concat → projection vers D_MODEL
        d_fused          = d_model + d_scalar + d_bb
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(d_fused),
            nn.Linear(d_fused, d_model),
            nn.GELU(),
        )

        # Décodeur
        self.decoder     = TemporalDecoder(d_model=d_model)

    def forward(self, x_sig: torch.Tensor,
                x_scalar: torch.Tensor,
                x_bb: torch.Tensor) -> torch.Tensor:

        # Branche 1 : (B, N_SIGNALS, 15000) → (B, T, D_MODEL)
        feats  = torch.stack(
            [enc(x_sig[:, i:i+1, :]) for i, enc in enumerate(self.encoders)],
            dim=1,
        )                                        # (B, N, D, T)
        h_sig  = self.transformer(feats)         # (B, T, D_MODEL)

        # Branche 2 : (B, N_SCALAR) → (B, T, D_SCALAR)
        h_scalar = self.scalar_enc(x_scalar, t=h_sig.shape[1])

        # Branche 3 : (B, N_BB, BB_LEN) → (B, T, D_BB)
        h_bb     = self.bb_enc(x_bb)

        # Fusion
        h_fused  = torch.cat([h_sig, h_scalar, h_bb], dim=-1)  # (B, T, D_FUSED)
        h_fused  = self.fusion_proj(h_fused)                    # (B, T, D_MODEL)

        return self.decoder(h_fused)                            # (B, T)


# ──────────────────────────────────────────────────────────────────────────────
# LIGHTNING MODULE
# ──────────────────────────────────────────────────────────────────────────────

class RHCLightningModuleV2(L.LightningModule):
    """
    Loss = 0.7 * MSE + 0.3 * MAE + SMOOTH_LAMBDA * smoothness
    Métriques : {stage}/loss, {stage}/mae, {stage}/mae_mmhg
    """

    def __init__(self, n_signals=N_SIGNALS, n_scalar=N_SCALAR, n_bb=N_BB,
                 d_model=D_MODEL, d_scalar=D_SCALAR, d_bb=D_BB,
                 cnn_dropout=CNN_DROPOUT, trans_dropout=TRANS_DROPOUT,
                 lr=LR, weight_decay=WEIGHT_DECAY,
                 rhc_mean: float = 0.0, rhc_std: float = 1.0,
                 scalar_mean=None, scalar_std=None,
                 bb_mean=None, bb_std=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = RHCSequencePredictorV2(
            n_signals=n_signals, n_scalar=n_scalar, n_bb=n_bb,
            d_model=d_model, d_scalar=d_scalar, d_bb=d_bb,
            cnn_dropout=cnn_dropout, trans_dropout=trans_dropout,
        )

    def _loss(self, pred, target):
        mse    = F.mse_loss(pred, target)
        mae    = F.l1_loss(pred, target)
        smooth = F.mse_loss(pred[:, 1:], pred[:, :-1])
        return 0.7 * mse + 0.3 * mae + SMOOTH_LAMBDA * smooth

    def _shared_step(self, batch, stage: str):
        x_sig, x_scalar, x_bb, y = batch
        pred     = self.model(x_sig, x_scalar, x_bb)
        loss     = self._loss(pred, y)
        mae      = F.l1_loss(pred, y)
        mae_mmhg = F.l1_loss(
            pred * self.hparams.rhc_std + self.hparams.rhc_mean,
            y    * self.hparams.rhc_std + self.hparams.rhc_mean,
        )
        self.log(f"{stage}/loss",     loss,     on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mae",      mae,      on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mae_mmhg", mae_mmhg, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_step(batch, "test")

    def predict_step(self, batch, _):
        x_sig, x_scalar, x_bb, _ = batch
        pred = self.model(x_sig, x_scalar, x_bb)
        return pred * self.hparams.rhc_std + self.hparams.rhc_mean

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=MAX_EPOCHS, eta_min=1e-6,
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}


# ──────────────────────────────────────────────────────────────────────────────
# DATA MODULE
# ──────────────────────────────────────────────────────────────────────────────

class RHCDataModuleV2(L.LightningDataModule):

    def __init__(self, segments_folder: str, batch_size=BATCH_SIZE,
                 num_workers=0, val_split=0.15, test_split=0.15, seed=42):
        super().__init__()
        self.folder      = segments_folder
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.val_split   = val_split
        self.test_split  = test_split
        self.seed        = seed
        self.rhc_mean    = 0.0
        self.rhc_std     = 1.0
        self.scalar_mean = np.zeros(N_SCALAR, dtype=np.float32)
        self.scalar_std  = np.ones(N_SCALAR,  dtype=np.float32)
        self.bb_mean     = np.zeros(N_BB,     dtype=np.float32)
        self.bb_std      = np.ones(N_BB,      dtype=np.float32)

    def setup(self, stage: Optional[str] = None):
        rng  = torch.Generator().manual_seed(self.seed)
        full = RHCDataset(self.folder)
        n    = len(full)
        n_test  = max(1, int(n * self.test_split))
        n_val   = max(1, int(n * self.val_split))
        n_train = n - n_val - n_test

        train_idx, val_idx, test_idx = random_split(
            range(n), [n_train, n_val, n_test], generator=rng
        )

        train_files = [full.files[i] for i in train_idx]

        # --- Stats RHC (sur train uniquement) ---
        all_rhc = []
        for f in train_files:
            mat = sio.loadmat(f)
            rhc = np.asarray(mat["rhc"]).squeeze().astype(np.float32)
            for s in range(N_SECONDS):
                block = rhc[s * FS:(s + 1) * FS]
                if len(block) == FS:
                    all_rhc.append(float(np.mean(block)))
        self.rhc_mean = float(np.mean(all_rhc))
        self.rhc_std  = float(np.std(all_rhc)) + 1e-8

        # --- Stats scalaires (sur train uniquement) ---
        all_scalars = []
        for f in train_files:
            mat = sio.loadmat(f)
            row = [load_scalar(mat, k) for k in SCALAR_KEYS]
            all_scalars.append(row)
        arr = np.array(all_scalars, dtype=np.float32)
        for j in range(arr.shape[1]):
            col    = arr[:, j]
            median = float(np.nanmedian(col))
            arr[np.isnan(arr[:, j]), j] = median
        self.scalar_mean = np.nanmean(arr, axis=0)
        self.scalar_std  = np.nanstd(arr,  axis=0) + 1e-8

        # --- Stats beat-by-beat (sur train uniquement) ---
        all_bb = []
        for f in train_files:
            mat = sio.loadmat(f)
            row = [load_bb(mat, k, target_len=BB_LEN) for k in BB_KEYS]
            all_bb.append(np.stack(row, axis=0))          # (N_BB, BB_LEN)

        arr_bb         = np.stack(all_bb, axis=0)         # (N_train, N_BB, BB_LEN)
        arr_bb         = arr_bb.transpose(0, 2, 1).reshape(-1, N_BB)  # (N_train*BB_LEN, N_BB)
        self.bb_mean   = np.nanmean(arr_bb, axis=0)       # (N_BB,)
        self.bb_std    = np.nanstd(arr_bb,  axis=0) + 1e-8

        print(f"[DataModule] train={n_train}  val={n_val}  test={n_test}")
        print(f"[DataModule] RHC  mean={self.rhc_mean:.2f} mmHg  std={self.rhc_std:.2f} mmHg")
        print(f"[DataModule] BB   mean={self.bb_mean.round(1)}  std={self.bb_std.round(1)}")

        def _make(augment):
            return RHCDataset(
                self.folder, augment=augment,
                rhc_mean=self.rhc_mean,     rhc_std=self.rhc_std,
                scalar_mean=self.scalar_mean, scalar_std=self.scalar_std,
                bb_mean=self.bb_mean,       bb_std=self.bb_std,
            )

        self.train_ds = Subset(_make(augment=True),  list(train_idx))
        self.val_ds   = Subset(_make(augment=False), list(val_idx))
        self.test_ds  = Subset(_make(augment=False), list(test_idx))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=False, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=False, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=False, persistent_workers=False)


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────────────────────

def train(segments_folder="processed/features_rhc", output_dir="checkpoints_v2",
          fast_dev_run=False):
    L.seed_everything(42)

    dm = RHCDataModuleV2(segments_folder)
    dm.setup()

    model = RHCLightningModuleV2(
        rhc_mean=dm.rhc_mean,
        rhc_std=dm.rhc_std,
        scalar_mean=dm.scalar_mean.tolist(),
        scalar_std=dm.scalar_std.tolist(),
        bb_mean=dm.bb_mean.tolist(),
        bb_std=dm.bb_std.tolist(),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="rhc_v2-{epoch:02d}-val_mae{val/mae:.3f}",
            monitor="val/mae", mode="min", save_top_k=3,
        ),
        EarlyStopping(monitor="val/loss", patience=PATIENCE, mode="min"),
    ]

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=TensorBoardLogger("tb_logs", name="rhc_v2"),
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)

    print(f"\nBest checkpoint: {callbacks[0].best_model_path}")
    return callbacks[0].best_model_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments_folder", default="processed/features_rhc")
    parser.add_argument("--output_dir",      default="checkpoints_v2")
    parser.add_argument("--fast_dev_run",    action="store_true")
    args = parser.parse_args()

    train(args.segments_folder, args.output_dir, args.fast_dev_run)