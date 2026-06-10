"""
Prédicteur RHC par Seconde — PyTorch Lightning
==============================================
Prédit la pression RHC à une résolution de 1 Hz (30 valeurs pour une fenêtre de 30 secondes)
à partir de 7 signaux d’entrée échantillonnés à 500 Hz.

Cible y : (30,) float32  — pression RHC moyenne par seconde [mmHg, normalisée]
Entrée x : (7, 15000) float32 — normalisation Z-score par segment
           [ECG, SCG, ECG_raw, SCG_raw, ACC_lat, ACC_hf, ACC_dv]

Architecture
------------
  1. Encodeur CNN par signal     (7 × 500 Hz → 7 × d_model × 30 tokens)
       Trois couches Conv1d avec stride réduisent 15 000 → 1 875 → 234 frames,
       puis un pooling moyen adaptatif regroupe chaque bloc d’une seconde
       en 30 tokens.

  2. Transformer inter-signaux   (7 signaux × 30 tokens → 30 × d_model)
       Pour chacun des 30 tokens temporels, un Transformer fusionne
       simultanément les informations provenant des 7 signaux.

  3. Décodeur temporel           (30 × d_model → 30 scalaires)
       Un MLP partagé transforme chaque token fusionné
       en une valeur prédite de pression.

Perte : 0.7 × MSE + 0.3 × MAE sur la séquence cible normalisée.
         Une régularisation de lissage temporel
         (norme L2 sur les différences premières) est ajoutée
         avec un faible poids afin d’éviter des variations
         irréalistes d’une seconde à l’autre.

Utilisation
------------
  python rhc_predictor.py --segments_folder segments_30s_strict
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
WINDOW_SIZE = FS * WINDOW_S       # 15 000 points (500 Hz × 30 s)
N_SECONDS   = WINDOW_S            # 30 sorties par segment, une par seconde

SIGNAL_KEYS = [
    "ecg",
    "scg",
    "ecg_raw",
    "scg_raw",
    "patch_ACC_lat",
    "patch_ACC_hf",
    "patch_ACC_dv",
]
N_SIGNALS = len(SIGNAL_KEYS)      # 7


CNN_CHANNELS = [1, 32, 64, 128]
CNN_KERNELS  = [15, 9, 5]
CNN_STRIDES  = [2,  2, 2]
CNN_DROPOUT  = 0.2

D_MODEL      = 128    # must match CNN_CHANNELS[-1]
N_HEADS      = 4
FF_DIM       = 256
N_LAYERS     = 2
TRANS_DROPOUT= 0.1

# Temporal smoothness regularisation weight
SMOOTH_LAMBDA = 0.05

# Training
BATCH_SIZE   = 8      
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 15


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class RHCDataset(Dataset):
    """
    Dataset PyTorch pour les segments de 30 secondes.
    Chaque segment est chargé depuis un fichier .mat contenant les 7 signaux d’entrée et la séquence cible de pression RHC.
    Returns:
      x : (N_SIGNALS, 15000) float32 
      y : (N_SECONDS,)       float32  
    """

    def __init__(self, segments_folder: str, augment: bool = False,
                 rhc_mean: float | None = None, rhc_std: float | None = None):
        self.folder  = Path(segments_folder)
        self.augment = augment
        self.files   = sorted(self.folder.glob("*_seg*.mat")) or \
                       sorted(self.folder.glob("*_segment.mat"))

        if not self.files:
            raise FileNotFoundError(
                f"No segment .mat files found in '{segments_folder}'."
            )

        # Pre-filtrage : supprimer les fichiers avec RHC manquant / NaN / de longueur incorrecte
        valid, skipped = [], []
        for f in self.files:
            try:
                mat = sio.loadmat(f)
                rhc = np.asarray(mat["rhc"]).squeeze()
                if rhc.ndim == 0 or len(rhc) == 0 or np.any(np.isnan(rhc)):
                    skipped.append(f.name)
                elif len(rhc) < WINDOW_SIZE:
                    skipped.append(f.name)
                else:
                    valid.append(f)
            except Exception:
                skipped.append(f.name)

        self.files = valid
        if skipped:
            print(f"[Dataset] Dropped {len(skipped)} file(s) with invalid RHC:")
            for n in skipped:
                print(f"{n}")

   
        if rhc_mean is None or rhc_std is None:
            self.rhc_mean, self.rhc_std = self._compute_stats()
        else:
            self.rhc_mean = rhc_mean
            self.rhc_std  = rhc_std

    def _compute_stats(self):
        """Calculer la moyenne et l’écart-type sur toutes les valeurs par seconde à travers tous les fichiers."""
        all_vals = []
        for f in self.files:
            mat = sio.loadmat(f)
            rhc = np.asarray(mat["rhc"]).squeeze().astype(np.float32)
            for s in range(N_SECONDS):
                block = rhc[s * FS:(s + 1) * FS]
                if len(block) == FS:
                    all_vals.append(float(np.mean(block)))
        mu  = float(np.mean(all_vals))
        std = float(np.std(all_vals)) + 1e-8
        return mu, std

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        mat = sio.loadmat(self.files[idx])

        rhc = np.asarray(mat["rhc"]).squeeze().astype(np.float32)
        y   = np.array(
            [np.mean(rhc[s * FS:(s + 1) * FS]) for s in range(N_SECONDS)],
            dtype=np.float32,
        )
        y = (y - self.rhc_mean) / self.rhc_std
        y = torch.from_numpy(y)                # (30,)

        channels = []
        for key in SIGNAL_KEYS:
            sig = np.asarray(mat[key]).squeeze().astype(np.float32)
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            channels.append(sig)

        x = torch.from_numpy(np.stack(channels, axis=0))   # (7, 15000)

        if self.augment:
            x = self._augment(x)

        return x, y

    @staticmethod
    def _augment(x: torch.Tensor):
        x = x * (1.0 + 0.10 * (torch.rand(x.shape[0], 1) * 2 - 1))
        x = x + 0.02 * torch.randn_like(x)
        for i in range(4, 7):
            if torch.rand(1).item() > 0.5:
                x[i] = -x[i]
        return x


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

class SignalEncoder(nn.Module):

    """
    
    1-D CNN par signal : (B, 1, 15000) → (B, D_MODEL, 30)
    Trois couches Conv1d à stride réduisent progressivement la résolution temporelle,
    puis un AdaptiveAvgPool1d(N_SECONDS) produit exactement un token par seconde.
    Chaque token porte le contexte temporel local d’environ 1 seconde de signal.
    """

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
            nn.AvgPool1d(kernel_size=62, stride=62),   # 1875 → 30 tokens
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, D_MODEL, N_SECONDS)


class CrossSignalTemporalTransformer(nn.Module):
    
    """
    Pour chaque position temporelle (token) parmi les 30 secondes, ce module fusionne les informations provenant des 7 signaux d
    ’entrée à l’aide d’un Transformer.
    
    L’entrée est de forme (B, N_SIGNALS, D_MODEL, N_SECONDS) et la sortie est de forme (B, N_SECONDS, D_MODEL).

    La dimension temporelle est gérée en reshaping : nous traitons tous les B × N_SECONDS positions 
    en une seule passe avant avec la dimension des signaux comme séquence.
    """
    

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
                 n_layers=N_LAYERS, dropout=TRANS_DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        self.signal_emb = nn.Parameter(torch.zeros(1, N_SIGNALS, d_model))

    def forward(self, x: torch.Tensor):
        # x: (B, N_SIGNALS, D_MODEL, N_SECONDS)
        B, N, D, T = x.shape

        x = x.permute(0, 3, 1, 2).reshape(B * T, N, D)   # (B*T, N, D)
        x = x + self.signal_emb                            # add signal embeddings

        x = self.transformer(x)        # (B*T, N, D)
        x = x.mean(dim=1)              # pool over signals → (B*T, D)
        x = x.reshape(B, T, D)        # (B, N_SECONDS, D)
        return x


class TemporalDecoder(nn.Module):
    
    """
    MLP partagé appliqué indépendamment à chacun des N_SECONDS tokens.
    Ajoute un lissage temporel causal via une conv1d depthwise avant la projection
    finale — cela permet aux secondes adjacentes de partager du contexte sans fuir d’informations futures à l’inférence.
    Input:  (B, N_SECONDS, D_MODEL)
    Output: (B, N_SECONDS)
    
    """
    def __init__(self, d_model=D_MODEL, dropout=0.2):
        super().__init__()
        # Causal temporal context (padding = kernel-1 on the left only)
        self.causal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=2,
            groups=d_model, bias=False,   # depthwise
        )
        self.norm = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.causal_conv(x.transpose(1, 2))   # (B, D, T+2)
        h = h[:, :, :x.shape[1]]                  # (B, D, T) — causal trim
        h = self.norm(h.transpose(1, 2))           # (B, T, D)
        x = x + h                                  # residual

        out = self.mlp(x)                          # (B, T, 1)
        return out.squeeze(-1)                     # (B, T=N_SECONDS)


class RHCSequencePredictor(nn.Module):
    """
      model:
      SignalEncoder × N_SIGNALS  →  (B, N, D, T)
      CrossSignalTemporalTransformer  →  (B, T, D)
      TemporalDecoder  →  (B, T)   where T = N_SECONDS = 30
    """

    def __init__(self, n_signals=N_SIGNALS, d_model=D_MODEL,
                 cnn_dropout=CNN_DROPOUT, trans_dropout=TRANS_DROPOUT):
        super().__init__()
        self.encoders    = nn.ModuleList([
            SignalEncoder(d_model=d_model, dropout=cnn_dropout)
            for _ in range(n_signals)
        ])
        self.transformer = CrossSignalTemporalTransformer(
            d_model=d_model, dropout=trans_dropout
        )
        self.decoder     = TemporalDecoder(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_SIGNALS, 30000)
        feats = torch.stack(
            [enc(x[:, i:i+1, :]) for i, enc in enumerate(self.encoders)],
            dim=1,
        )                                    # (B, N_SIGNALS, D_MODEL, N_SECONDS)
        fused = self.transformer(feats)      # (B, N_SECONDS, D_MODEL)
        return self.decoder(fused)           # (B, N_SECONDS)


# ──────────────────────────────────────────────────────────────────────────────
# LIGHTNING MODULE
# ──────────────────────────────────────────────────────────────────────────────

class RHCLightningModule(L.LightningModule):
    """
    Loss = 0.7 * MSE + 0.3 * MAE  +  SMOOTH_LAMBDA * smoothness

    Smoothness term: mean squared first-difference of predictions,
    penalising implausible second-to-second pressure jumps.

    Metriques:
      {stage}/loss       — total loss
      {stage}/mae        — MAE normalisé 
      {stage}/mae_mmhg   — MAE en mmHg )
    """

    def __init__(self, n_signals=N_SIGNALS, d_model=D_MODEL,
                 cnn_dropout=CNN_DROPOUT, trans_dropout=TRANS_DROPOUT,
                 lr=LR, weight_decay=WEIGHT_DECAY,
                 rhc_mean: float = 0.0, rhc_std: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = RHCSequencePredictor(
            n_signals=n_signals, d_model=d_model,
            cnn_dropout=cnn_dropout, trans_dropout=trans_dropout,
        )

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse    = F.mse_loss(pred, target)
        mae    = F.l1_loss(pred, target)
        smooth = F.mse_loss(pred[:, 1:], pred[:, :-1])
        return 0.7 * mse + 0.3 * mae + SMOOTH_LAMBDA * smooth

    def _shared_step(self, batch, stage: str):
        x, y   = batch                          # x:(B,7,15000)  y:(B,30)
        pred   = self.model(x)                  # (B, 30)
        loss   = self._loss(pred, y)
        mae    = F.l1_loss(pred, y)

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
        """Retourne des prédictions par seconde en mmHg. Shape : (B, 30)."""
        x, _ = batch
        pred  = self.model(x)
        return pred * self.hparams.rhc_std + self.hparams.rhc_mean

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
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

class RHCDataModule(L.LightningDataModule):

    def __init__(self, segments_folder: str, batch_size=BATCH_SIZE,
                 num_workers=4, val_split=0.15, test_split=0.15, seed=42):
        super().__init__()
        self.folder      = segments_folder
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.val_split   = val_split
        self.test_split  = test_split
        self.seed        = seed
        self.rhc_mean    = 0.0
        self.rhc_std     = 1.0

    def setup(self, stage: Optional[str] = None):
        rng    = torch.Generator().manual_seed(self.seed)
        full   = RHCDataset(self.folder)
        n      = len(full)
        n_test  = max(1, int(n * self.test_split))
        n_val   = max(1, int(n * self.val_split))
        n_train = n - n_val - n_test

        train_idx, val_idx, test_idx = random_split(
            range(n), [n_train, n_val, n_test], generator=rng
        )

        train_files = [full.files[i] for i in train_idx]
        all_vals    = []
        for f in train_files:
            mat = sio.loadmat(f)
            rhc = np.asarray(mat["rhc"]).squeeze().astype(np.float32)
            for s in range(N_SECONDS):
                block = rhc[s * FS:(s + 1) * FS]
                if len(block) == FS:
                    all_vals.append(float(np.mean(block)))
        self.rhc_mean = float(np.mean(all_vals))
        self.rhc_std  = float(np.std(all_vals)) + 1e-8

        print(f"[DataModule] train={n_train}  val={n_val}  test={n_test}")
        print(f"[DataModule] RHC per-second — "
              f"mean={self.rhc_mean:.2f} mmHg  std={self.rhc_std:.2f} mmHg")

        # Build three dataset instances so augment applies to train only
        def _make(augment):
            return RHCDataset(self.folder, augment=augment,
                              rhc_mean=self.rhc_mean, rhc_std=self.rhc_std)

        self.train_ds = Subset(_make(augment=True),  list(train_idx))
        self.val_ds   = Subset(_make(augment=False), list(val_idx))
        self.test_ds  = Subset(_make(augment=False), list(test_idx))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=0,
                          pin_memory=False, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0,
                          pin_memory=False, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0,
                          pin_memory=False, persistent_workers=False)



def train(segments_folder="segments_30s", output_dir="checkpoints",
          fast_dev_run=False):
    L.seed_everything(42)

    dm    = RHCDataModule(segments_folder)
    dm.setup()
    model = RHCLightningModule(rhc_mean=dm.rhc_mean, rhc_std=dm.rhc_std)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="rhc_seq-{epoch:02d}-val_mae{val/mae:.3f}",
            monitor="val/mae", mode="min", save_top_k=3,
        ),
        EarlyStopping(monitor="val/loss", patience=PATIENCE, mode="min"),
    ]

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=TensorBoardLogger("tb_logs", name="rhc_seq"),
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        accelerator="cpu",
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
    parser.add_argument("--segments_folder", default="../segments_30s_strict")
    parser.add_argument("--output_dir",      default="checkpoints")
    parser.add_argument("--fast_dev_run",    action="store_true")
    args = parser.parse_args()


    train(args.segments_folder, args.output_dir, args.fast_dev_run)