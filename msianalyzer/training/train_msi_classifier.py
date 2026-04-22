# Standard library imports
import argparse
import csv
import json
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

# Third-party imports
import matplotlib.pyplot as plt
import m2aia as m2
import numpy as np
import pandas as pd
import seaborn as sns
import timm
from timm.scheduler import CosineLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch._inductor.config as inductor_config
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

# Local imports
from msianalyzer.config import (
    LABELING_CSV_DIR,
    LABELS,
    MODELS_DIR,
    PROCESSED_DIR,
    TARGET_HW,
    ioncache_dir_for_imzml,
)

from msianalyzer.utils import msi_utils
from msianalyzer.utils.ion_cache import IonImageCache
from msianalyzer.training.augmentation_gpu import apply_gpu_augmentations, GPU_SUPPORTED_MODES
from msianalyzer.training.augmentations import augmentation_strength

# Set precision to high for better performance
torch.set_float32_matmul_precision("high")

# turn off the small-GPU-unfriendly path
inductor_config.max_autotune_gemm = False
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

# ====================== Hyperparameters ======================
COARSE_MODE     = "full"        # "2class", "3class", "full" determines which coarse labels to use
PRETRAINED      = True          # whether to use pretrained weights
CLASSIFIER_MODEL    = "convnextv2_tiny"    # ResNet model to use, resnet50,tf_efficientnet_b3, etc.

# Scheduler
SCHEDULER       = "cosine"      # "cosine" (or "step")
WARMUP_EPOCHS   = 8             # epochs to warmup
MIN_LR          = 1e-6          # minimum learning rate

BATCH_SIZE      = 16            # batch size
EPOCHS          = 100           # number of epochs
PATIENCE        = 10             # epochs to wait after last improvement

VAL_FRACTION    = 0.1           # fraction of datasets to use for validation
TEST_FRACTION   = 0.1           # fraction of datasets to use for testing (only if no fixed TEST_FILES/args are given)

LR              = 3e-4          # learning rate, default 10e-4
WEIGHT_DECAY    = 1e-4          # weight decay
SEED            = 42            # random seed
NUM_WORKERS     = 4             # number of workers for data loading (cuda only, 2 otherwise)
LABEL_SMOOTHING = 0.0           # label smoothing factor for hard-label loss
 
def seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return np.random.default_rng(seed)

# Information scores (keep >0 to preserve gradients)
LABEL_CONFIDENCE = {
    "structured": 1.0,
    "weak_structured": 1.0,
    "localized": 1.0,
    "fragmented": 1.0,
    "unstructured": 1.0,
    "negative": 1.0,
}

# Full smoothing matrix (rows keyed by true class, cols sum to ~1 before scaling)
SOFT_MIXING = {
    "structured":        {"structured": 0.80, "weak_structured": 0.10, "localized": 0.05, "fragmented": 0.00, "unstructured": 0.00, "negative": 0.05},
    "weak_structured":   {"structured": 0.10, "weak_structured": 0.75, "localized": 0.08, "fragmented": 0.02, "unstructured": 0.00, "negative": 0.05},
    "localized":         {"structured": 0.02, "weak_structured": 0.08, "localized": 0.80, "fragmented": 0.10, "unstructured": 0.00, "negative": 0.00},
    "fragmented":        {"structured": 0.00, "weak_structured": 0.02, "localized": 0.08, "fragmented": 0.80, "unstructured": 0.10, "negative": 0.00},
    "unstructured":      {"structured": 0.00, "weak_structured": 0.00, "localized": 0.00, "fragmented": 0.05, "unstructured": 0.95, "negative": 0.00},
    "negative":          {"structured": 0.15, "weak_structured": 0.05, "localized": 0.00, "fragmented": 0.00, "unstructured": 0.00, "negative": 0.80},
}

EPS_SOFT = 1e-6  # avoid zero rows
# =============================================================

# --------------------- Coarse labels ---------------------
if COARSE_MODE == "2class":
    MERGE_MAP = {
        "structured":"positive", "weak_structured":"positive",
        "localized":"positive", "fragmented":"positive",
        "unstructured":"non_positive", "negative":"positive",
    }
    CLASSES = ["non_positive","positive"]
elif COARSE_MODE == "2class-non-frag":
    MERGE_MAP = {
        "structured":"positive", "weak_structured":"positive",
        "localized":"positive", "fragmented":"non_positive",
        "unstructured":"non_positive", "negative":"positive",
    }
    CLASSES = ["non_positive","positive"]
elif COARSE_MODE == "2class-big-structure":
    MERGE_MAP = {
        "structured":"positive", "weak_structured":"positive",
        "localized":"non_positive", "fragmented":"non_positive",
        "unstructured":"non_positive", "negative":"positive",
    }
    CLASSES = ["non_positive","positive"]
elif COARSE_MODE == "3class":
    MERGE_MAP = {
        "structured":"structured", "weak_structured":"structured",
        "localized":"partial_structured", "fragmented":"non_structured",
        "unstructured":"non_structured", "negative":"non_structured",
    }
    CLASSES = ["non_structured","partial_structured","structured"]
else: # default all 6 classes (no merging)
    COARSE_MODE = "full"
    MERGE_MAP = {name:name for name in LABELS}
    CLASSES = LABELS

def to_coarse(y_str: str) -> str:
    return MERGE_MAP.get(y_str, "non_structured")

cls_to_idx = {c:i for i,c in enumerate(CLASSES)}

# --------------------- Device ---------------------
device   = "cuda" if torch.cuda.is_available() else "cpu"

def _gpu_base_geom(imgs: torch.Tensor) -> torch.Tensor:
    """Apply simple flip/rotation augmentations on GPU."""
    if torch.rand(1, device=imgs.device) < 0.5:
        imgs = imgs.flip(-1)
    if torch.rand(1, device=imgs.device) < 0.5:
        imgs = imgs.flip(-2)
    k = torch.randint(0, 4, (1,), device=imgs.device).item()
    return torch.rot90(imgs, k, dims=(-2, -1))

# --------------------- Label mapping ---------------------
label2id = {name: i for i, name in enumerate(LABELS)}
id2label = {i: name for name, i in label2id.items()}

# --------------------- Utils ---------------------
@dataclass
class EarlyStopping:
    patience: int = PATIENCE        # epochs to wait after last improvement
    min_delta: float = 1e-4   # required improvement in val_loss
    best_loss: float = float("inf")
    best_state: Optional[Dict] = None
    wait: int = 0
    stopped: bool = False
    stop_epoch: Optional[int] = None

    def step(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        improved = (self.best_loss - val_loss) > self.min_delta
        if improved:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                self.stop_epoch = epoch
        return self.stopped

def get_mz_bounds(reader) -> Tuple[float, float]:
    if hasattr(reader, "GetMassRange"):
        lo, hi = reader.GetMassRange()
        return float(lo), float(hi)
    # Fallback: X axis
    if hasattr(reader, "GetXAxis"):
        x = reader.GetXAxis()
    else:
        x, _ = reader.GetSpectrum(0)
    return float(x[0]), float(x[-1])

def window_intersects_bounds(mz: float, ppm: float, mz_min: float, mz_max: float) -> bool:
    delta = abs(mz) * abs(ppm) * 1e-6
    lo, hi = mz - delta, mz + delta
    return not (hi < mz_min or lo > mz_max)

# Try to pull ppm from METASPACE cache; if missing, attempt API fetch via msi_utils,
# but fall back safely if the API is unavailable.
def get_ppm_with_cache_refresh(imzml_path: Path, default_ppm: float = 3.0) -> float:
    dataset_id = msi_utils.get_metaspace_id_from_imzml(imzml_path) or imzml_path.stem
    try:
        data, _from_cache = msi_utils.get_metaspace_data(dataset_id, use_cache=True)
    except Exception as e:
        print(f"⚠️  METASPACE unavailable for {dataset_id}: {e}. Using default ppm={default_ppm}")
        return float(default_ppm)

    if not data:
        print(f"⚠️  No METASPACE data for {dataset_id}. Using default ppm={default_ppm}")
        return float(default_ppm)

    config = data.get("config") or {}
    ppm = None
    if isinstance(config, dict):
        img_gen = config.get("image_generation") or {}
        if isinstance(img_gen, dict):
            ppm = img_gen.get("ppm")
        if ppm is None:
            ppm = config.get("ppm")

    if ppm is None:
        print(f"⚠️  ppm missing in METASPACE config for {dataset_id}. Using default ppm={default_ppm}")
        return float(default_ppm)
    return float(ppm)

# Safe extractor (uses your msi_utils.extract_ion_image if present; else fallback)
def extract_ion_image_safe(reader, mz_value: float, ppm: float,
                           mz_bounds: Optional[Tuple[float, float]] = None):
    try:
        # prefer your project’s implementation if present
        return msi_utils.extract_ion_image(reader, mz_value, ppm, mz_bounds)
    except Exception:
        pass

    # Fallback pure implementation
    if mz_bounds is None:
        mz_min, mz_max = get_mz_bounds(reader)
    else:
        mz_min, mz_max = mz_bounds

    if not np.isfinite(mz_value): return None
    ppm = abs(float(ppm))
    delta = float(mz_value) * ppm * 1e-6
    lo = float(mz_value) - delta
    hi = float(mz_value) + delta
    if hi < mz_min or lo > mz_max:  # fully out of range
        return None
    clo, chi = max(lo, mz_min), min(hi, mz_max)
    if chi <= clo: return None
    eps = max(1e-9, (clo + chi) * 1e-12)
    center = float(0.5 * (clo + chi))
    half   = float(max(0.5 * (chi - clo) - eps, eps))
    try:
        arr = reader.GetArray(center, half)
    except Exception:
        return None
    return np.asarray(arr).squeeze()

# Exponential Moving Average (EMA)
@torch.no_grad()
def ema_update(student, teacher, m: float = 0.996):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

def clone_as_ema(model):
    import copy
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


class WeightDrop(nn.Module):
    """DropConnect wrapper that applies dropout to module weights during training."""

    def __init__(self, module: nn.Module, weight_p: float):
        super().__init__()
        self.module = module
        self.weight_p = float(weight_p)

    def _drop_weight(self) -> torch.Tensor:
        weight = self.module.weight
        return F.dropout(weight, p=self.weight_p, training=self.training)

    def forward(self, *args, **kwargs):
        if isinstance(self.module, nn.Linear):
            input_tensor = args[0]
            weight = self._drop_weight()
            return F.linear(input_tensor, weight, self.module.bias)
        if isinstance(self.module, nn.Conv2d):
            input_tensor = args[0]
            weight = self._drop_weight()
            return F.conv2d(
                input_tensor,
                weight,
                self.module.bias,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
                groups=self.module.groups,
            )
        raise TypeError(f"WeightDrop does not support module type {type(self.module)}")

    def train(self, mode: bool = True):
        self.module.train(mode)
        return super().train(mode)


def apply_weight_dropout(module: nn.Module, weight_p: float,
                         module_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d)) -> nn.Module:
    if weight_p <= 0.0:
        return module

    for name, child in list(module.named_children()):
        if isinstance(child, WeightDrop):
            apply_weight_dropout(child.module, weight_p, module_types)
            continue
        if isinstance(child, module_types):
            setattr(module, name, WeightDrop(child, weight_p))
        else:
            apply_weight_dropout(child, weight_p, module_types)
    return module

# Binary cross-entropy loss
bce_logits = torch.nn.functional.binary_cross_entropy_with_logits

def unsup_loss_binary(student, teacher, x_w, x_s, tau: float):
    with torch.no_grad():
        p = torch.sigmoid(teacher(x_w).squeeze(1))  # [B]
    mask = (p >= tau) | (p <= 1.0 - tau)
    if mask.sum() == 0:
        return x_w.new_tensor(0.0)
    targets = (p > 0.5).float()
    logits_s = student(x_s[mask]).squeeze(1)
    return bce_logits(logits_s, targets[mask])

# NEW: open any compatible existing cache in ioncache/ by reading its meta first.
def open_existing_cache(imzml_path: Path, ppm: float):
    cache_dir = ioncache_dir_for_imzml(imzml_path)
    metas = sorted(cache_dir.glob("mz-*.json"))
    if not metas:
        raise FileNotFoundError(f"No caches found in {cache_dir}")

    for meta_path in metas:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if float(meta.get("ppm", -1)) != float(ppm):
            continue
        data_path = meta_path.with_suffix(".dat")
        if not data_path.exists():
            continue

        cached_mz = meta.get("mz", [])
        ekw = meta.get("extract_kwargs", {})
        cache = IonImageCache(
            imzml_path=imzml_path,
            reader=None,
            mz_list=cached_mz,
            ppm=ppm,
            extract_kwargs=ekw,
        )
        cache._loaded_meta = meta
        print(f"✓ Using cache: {meta_path.name} (ppm={ppm}, n_mz={len(cached_mz)})")
        return cache

    raise FileNotFoundError(f"No cache with ppm={ppm} in {cache_dir}")
# --------------------- Dataset ---------------------
class IonImageClassificationDataset(Dataset):
    """
    Cache-backed: pulls ion images from IonImageCache (hotspot-scaled once).
    Outputs tensors [1,H,W] with p99 scaling kept (no double-scaling if your
    extract_ion_image already returns [0,1]; keep here for safety).
    """
    def __init__(self, ion_cache: IonImageCache, labels_idx: np.ndarray, mz_list: np.ndarray):
        assert len(labels_idx) == len(mz_list)
        self.cache      = ion_cache
        self.labels_idx = labels_idx.astype(np.int64)
        self.mz_list    = mz_list.astype(float)
        self.enable_augment = False

    def __len__(self): return len(self.mz_list)

    def __getitem__(self, idx):
        mz = float(self.mz_list[idx])
        img = self.cache.get_by_mz(mz)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        # --- labels ---
        hard_idx = int(self.labels_idx[idx])  # index in CURRENT class space
        label_t = torch.tensor(hard_idx, dtype=torch.long)

        # p99 scaling + simple augs + resize (as you had)
        p99 = torch.quantile(img_t.flatten(), 0.99) if img_t.numel() else torch.tensor(1.0)
        img_t = (img_t / p99.clamp_min(1e-6)).clamp_(0, 1)
        if self.enable_augment:
            if torch.rand(1) < 0.5:
                img_t = torch.flip(img_t, dims=[-1])
            if torch.rand(1) < 0.5:
                img_t = torch.flip(img_t, dims=[-2])
            k = torch.randint(0, 4, (1,)).item()
            img_t = torch.rot90(img_t, k, dims=[-2, -1])
        img_t = F.interpolate(img_t.unsqueeze(0), size=TARGET_HW, mode="bilinear",
                              align_corners=False).squeeze(0)

        # NEW: meta for disagreement export
        meta = {
            "imzml": str(self.cache.imzml_path),
            "dataset": Path(self.cache.imzml_path).name,
            "mz": mz,
            "uid": f"{Path(self.cache.imzml_path).name}:{mz:.9f}",
        }
        if (globals().get("args", None) is not None) and (args.training_type == "soft"):
            hard_name = id2label[hard_idx] if COARSE_MODE == "full" else CLASSES[hard_idx]
            row = SOFT_MIXING.get(hard_name, {hard_name: 1.0})
            C = len(CLASSES)
            vec = torch.zeros(C, dtype=torch.float32)
            for j, cls in enumerate(CLASSES):
                vec[j] = float(row.get(cls, 0.0))
            vec = vec + EPS_SOFT
            vec = vec / vec.sum().clamp_min(EPS_SOFT)
            info = float(LABEL_CONFIDENCE.get(hard_name, 1.0))
            soft_target = info * vec
            meta["soft_target"] = soft_target
        return img_t, label_t, meta

# --------------------- Load & wrap datasets ---------------------
def load_pair(imzml_path: Path, csv_path: Path, ppm: float):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"⚠️  Skip {imzml_path.name}: missing or empty CSV {csv_path.name}")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Skip {imzml_path.name}: error reading CSV {csv_path.name}: {e}")
        return None

    if "mz" not in df.columns or "label" not in df.columns:
        print(f"⚠️  Skip {imzml_path.name}: invalid CSV format {csv_path.name}")
        return None

    # labels / mz (your existing logic)
    if COARSE_MODE == 'full':
        labels_all = df["label"].map(label2id).to_numpy(dtype=np.int64)
    else:
        labels_all = df["label"].apply(lambda x: cls_to_idx[to_coarse(x)]).to_numpy(dtype=np.int64)
    mz_all = df["mz"].astype(float).to_numpy()

    # ---------- FAST PATH: open existing cache, no m2aia ----------
    try:
        cache = open_existing_cache(imzml_path, ppm)   # see helper below

        # intersect label rows with cache m/z (round to 9 decimals to match meta precision)
        cached_mz = np.array([float(m) for m in cache.mz_list])
        cache_set = {round(m, 9) for m in cached_mz}
        mask = np.array([round(float(m), 9) in cache_set for m in mz_all], dtype=bool)

        if not mask.any():
            print(f"⚠️ {imzml_path.name}: no label m/z present in existing cache (ppm={ppm})")
            return None

        mz_list    = mz_all[mask]
        labels_idx = labels_all[mask]        # <-- use precomputed mapping

        return IonImageClassificationDataset(cache, labels_idx, mz_list)

    except FileNotFoundError as e:
        print(f"ℹ️ No compatible cache for {imzml_path.name}: {e}")
    except Exception as e:
        # Don’t hide real errors as “cache not found”
        import traceback; traceback.print_exc()
        print(f"❌ Cache load failed for {imzml_path.name}; falling back to build.")

    # ---------- SLOW PATH: build once with m2aia, then reuse ----------
    r = m2.ImzMLReader(str(imzml_path))
    mz_min, mz_max = get_mz_bounds(r)
    mask = np.array([window_intersects_bounds(mz, ppm, mz_min, mz_max) for mz in mz_all], dtype=bool)
    if not mask.any():
        print(f"⚠️  Skip {imzml_path.name}: all m/z out of range (bounds [{mz_min:.6f}, {mz_max:.6f}], ppm={ppm})")
        return None
    dropped = int((~mask).sum())
    if dropped:
        print(f"ℹ️  {imzml_path.name}: dropped {dropped}/{len(mask)} m/z outside bounds [{mz_min:.6f}, {mz_max:.6f}] at {ppm} ppm")

    mz_list    = mz_all[mask]
    labels_idx = labels_all[mask]

    cache = IonImageCache(imzml_path=imzml_path, reader=r, mz_list=mz_list, ppm=ppm, extract_kwargs={})
    return IonImageClassificationDataset(cache, labels_idx, mz_list)


# --------- Visualize Disagreements with Ground Truth and Teacher Predictions ---------
@torch.no_grad()
def visualize_disagreements(disagreements_csv, num_samples=10, save_dir=None, epoch_idx=None, phase=None):
    """
    Create visualizations of disagreement samples showing ground truth vs teacher predictions.
    
    Args:
        disagreements_csv: Path to the disagreements CSV file
        num_samples: Number of top disagreements to visualize
        save_dir: Directory to save visualizations (optional)
        epoch_idx: Epoch index for saving visualizations
        phase: Phase of training ("train" or "val")
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    if epoch_idx is None:
        print("⚠️  Epoch index not provided, setting unknown")
        epoch_idx = "unknown"

    if phase is None:
        print("⚠️  Phase not provided, setting unknown")
        phase = "unknown"
        

    if not os.path.exists(disagreements_csv):
        print(f"⚠️  Disagreements CSV not found: {disagreements_csv}")
        return
    
    # Load disagreements
    df = pd.read_csv(disagreements_csv)
    df = df.nlargest(num_samples, 'avg_loss')
    
    print(f"📊 Generating visualizations for {len(df)} disagreements...")
    
    # Create figure for all visualizations
    fig, axes = plt.subplots(nrows=len(df), ncols=4, figsize=(20, 5 * len(df)))
    if len(df) == 1:
        axes = axes.reshape(1, -1)  # Handle single row case
    
    # Set models to eval mode
    model.eval()
    if teacher is not None:
        teacher.eval()
    
    for idx, (i, row) in enumerate(df.iterrows()):
        try:
            # Extract info from row
            mz = float(row['mz'])
            dataset_name = row['dataset']
            path = row['path']
            gt_label = row['label_name']
            avg_loss = row['avg_loss']
            
            # Get prediction info from CSV (already calculated)
            student_pred = row.get('student_pred', 'N/A')
            student_conf = row.get('student_conf', 0.0)
            student_label = row.get('student_label', 'N/A')
            teacher_pred = row.get('teacher_pred', 'N/A')
            teacher_conf = row.get('teacher_conf', 0.0)
            teacher_label = row.get('teacher_label', 'N/A')
            
            # Find the corresponding dataset/cache
            cache = None
            for ds_meta in wrapped_meta:
                if ds_meta["name"] == dataset_name:
                    # Find the dataset index
                    ds_idx = next((i for i, meta in enumerate(wrapped_meta) if meta["name"] == dataset_name), None)
                    if ds_idx is not None:
                        cache = wrapped_datasets[ds_idx].cache
                    break
            
            if cache is None:
                print(f"⚠️  Could not find cache for {dataset_name}")
                continue
            
            # Load ion image
            img = cache.get_by_mz(mz)
            if img is None:
                print(f"⚠️  Could not load image for {mz} in {dataset_name}")
                continue
            
            # Convert to tensor and normalize (handle different input shapes)
            img_t = torch.as_tensor(img, dtype=torch.float32)
            
            # Add batch dimension if needed
            if img_t.dim() == 2:  # (H, W)
                img_t = img_t.unsqueeze(0)  # (1, H, W)
            elif img_t.dim() == 3:  # (H, W, C) or (C, H, W)
                if img_t.shape[0] == 1:  # Already has channel dim
                    pass
                else:  # (H, W, C) -> (C, H, W)
                    img_t = img_t.permute(2, 0, 1)
            
            # Now img_t should be (C, H, W), add batch dimension
            img_t = img_t.unsqueeze(0)  # (1, C, H, W)
            
            p99 = torch.quantile(img_t.flatten(), 0.99)
            img_normalized = (img_t / p99.clamp_min(1e-6)).clamp_(0, 1)
            
            # Create visualization using CSV data instead of recalculating predictions
            # 1. Original image
            axes[idx, 0].imshow(img, cmap='viridis')
            axes[idx, 0].set_title(f'Ion Image\n{dataset_name}:{mz:.6f}')
            axes[idx, 0].axis('off')
            
            # 2. Ground truth
            axes[idx, 1].text(0.5, 0.5, f'Ground Truth:\n{gt_label}', 
                            ha='center', va='center', fontsize=12, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[idx, 1].set_title(f'Loss: {avg_loss:.4f}')
            axes[idx, 1].axis('off')
            
            # 3. Student prediction
            axes[idx, 2].text(0.5, 0.5, f'Student:\n{student_label}\nConf: {student_conf:.3f}', 
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[idx, 2].set_title(f'Student Prediction')
            axes[idx, 2].axis('off')
            
            # 4. Teacher prediction
            if teacher is not None and teacher_label != 'N/A':
                axes[idx, 3].text(0.5, 0.5, f'Teacher:\n{teacher_label}\nConf: {teacher_conf:.3f}', 
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
                axes[idx, 3].set_title(f'Teacher Prediction')
            else:
                axes[idx, 3].text(0.5, 0.5, 'Teacher: N/A', 
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[idx, 3].set_title('Teacher Not Available')
            axes[idx, 3].axis('off')
            
        except Exception as e:
            print(f"⚠️  Error processing row {i}: {e}")
            # Fill with error message
            for col in range(4):
                axes[idx, col].text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                                  ha='center', va='center', fontsize=10)
                axes[idx, col].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        vis_path = Path(save_dir) / f"disagreements_{phase}_epoch{epoch_idx:03d}_vis.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"Saved disagreement visualizations to {vis_path}")
    else:
        plt.show(block=False)
        plt.pause(2.0)
    
    plt.close()

# --------------------- Parse arguments ---------------------

#currently to resume training
parser = argparse.ArgumentParser()
parser.add_argument("--timm_model", dest="timm_model", type=str, default=str(CLASSIFIER_MODEL),
                    help="Choose timm model (e.g., resnet50, tf_efficientnet_b3).")
parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from (default: best_model.pt if it exists)")
parser.add_argument("--eval-only", action="store_true",
                    help="Load checkpoint, run validation once, and exit")
parser.add_argument("--use_ema", action="store_true", default=False, help="Enable EMA teacher.")
parser.add_argument("--ema_m", type=float, default=0.996, help="EMA decay (m).")
parser.add_argument("--pseudo_tau", type=float, default=0.9, help="Confidence threshold for pseudo-labels.")
parser.add_argument("--lambda_u", type=float, default=1.0, help="Weight for unsupervised (consistency) loss.")
parser.add_argument("--export_disagreements", type=int, default=0, help="Top-K highest-loss samples to export for relabel.")
parser.add_argument("--visualize_disagreements", action="store_true", default=False, help="Visualize disagreements.")
parser.add_argument("--weight_dropout", type=float, default=0.05,
                    help="Probability for DropConnect-style weight dropout applied to linear/conv layers.")
parser.add_argument("--drop_path_rate", type=float, default=0.05,
                    help="Stochastic depth rate (0.0 = no drop, 1.0 = drop all).")
parser.add_argument("--eval-split", type=str, default="val", choices=["val", "test"],
                    help="Which split to evaluate in --eval-only mode (default: val)")
# not implemented yet
_AUGMENTATION_CHOICES = sorted(set(GPU_SUPPORTED_MODES) | {"default", "advanced"})
parser.add_argument("--augmentations", type=str, default="default", choices=_AUGMENTATION_CHOICES,
                    help="Type of augmentations to use")
parser.add_argument("--augmentation_device", type=str, default="cpu", choices=["cpu", "gpu"],
                    help="Where to execute augmentations (cpu = dataset, gpu = batch on CUDA).")
parser.add_argument("--augmentation_repeat_factor", type=int, default=1,
                    help="Repeat the training datasets this many times per epoch (>=1).")
parser.add_argument("--training_type", type=str, default="hard", choices=["hard", "soft"],
                    help="Type of training: hard (integer labels) or soft (information scores 0 to 1)")
parser.add_argument("--test_files", type=str,
                    help="Comma-separated list or newline-delimited file of dataset filenames to pin to the test split")
parser.add_argument("--val_files", type=str,
    help="Comma-separated list or newline-delimited file of dataset filenames to pin to the val split")
args = parser.parse_args()

CLASSIFIER_MODEL = args.timm_model
SEED = args.seed
rng = seed_everything(SEED)

USE_GPU_AUG = args.augmentation_device == "gpu"
CAN_USE_GPU_AUG = USE_GPU_AUG and device == "cuda"
if USE_GPU_AUG and device != "cuda":
    print("⚠️  GPU augmentations requested but CUDA is not available. Falling back to CPU.")
if not CAN_USE_GPU_AUG and args.augmentations not in {"default", "advanced"}:
    print("⚠️  Requested augmentation requires GPU; falling back to 'default' on CPU.")
    args.augmentations = "default"
args.augmentation_repeat_factor = max(1, int(args.augmentation_repeat_factor))

# Run directory under data/models/<timestamp>

RUN_TS  = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{RUN_TS}_{CLASSIFIER_MODEL}_{COARSE_MODE}_{args.training_type}_dpr-{args.drop_path_rate}_wd-{args.weight_dropout}_aug-{args.augmentations}_augdev-{args.augmentation_device}_{SCHEDULER}_pretrained-{int(PRETRAINED)}_{device}"
RUN_DIR = MODELS_DIR / RUN_NAME
PER_SAMPLE_DIR = RUN_DIR / "disagreements"
VIS_DIR = PER_SAMPLE_DIR / "visualizations"
CONF_HIST_DIR = RUN_DIR / "confidence_histograms"

RUN_DIR.mkdir(parents=True, exist_ok=True)
PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
CONF_HIST_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Saving run artifacts to: {RUN_DIR}")

def parse_fixed_list(spec: Optional[str]) -> Optional[List[str]]:
    """Return dataset filenames from a CLI hint (file path or comma-separated)."""
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return []
    candidate = Path(spec)
    if candidate.exists():
        lines = candidate.read_text().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return [part.strip() for part in spec.split(",") if part.strip()]

# --------------------- Load & wrap datasets ---------------------
wrapped_datasets: List[Dataset] = []
wrapped_meta = []  # keep info for splitting (training, validation and test) & reporting

for imzml_file in sorted(Path(PROCESSED_DIR).rglob("*.imzML")):
    csv_file = Path(LABELING_CSV_DIR) / f"{imzml_file.stem}.csv"

    ppm = get_ppm_with_cache_refresh(imzml_file)
    # DEBUG:
    print(f"Loading {imzml_file.name} with ppm {ppm}")

    ds = load_pair(imzml_file, csv_file, ppm)
    if ds is None:
        continue

    # collect label histogram per dataset (for optional balancing)
    counts = np.bincount(ds.labels_idx, minlength=len(LABELS))
    wrapped_datasets.append(ds)
    wrapped_meta.append({
        "name": imzml_file.name,
        "n_items": len(ds),
        "label_counts": counts,
    })

if not wrapped_datasets:
    raise RuntimeError("No valid datasets found.")

name_to_idx = {meta["name"]: idx for idx, meta in enumerate(wrapped_meta)}
fixed_test_files = parse_fixed_list(args.test_files)
fixed_test_indices: Set[int] = set()
if fixed_test_files:
    missing = [fname for fname in fixed_test_files if fname not in name_to_idx]
    if missing:
        raise ValueError(f"Requested test files not found among datasets: {missing}")
    fixed_test_indices = {name_to_idx[fname] for fname in fixed_test_files}
fixed_val_files = parse_fixed_list(args.val_files)
fixed_val_indices: Set[int] = set()
if fixed_val_files:
    missing_v = [fname for fname in fixed_val_files if fname not in name_to_idx]
    if missing_v:
        raise ValueError(f"Requested val files not found among datasets: {missing_v}")
    fixed_val_indices = {name_to_idx[fname] for fname in fixed_val_files}


full_ds = ConcatDataset(wrapped_datasets)
print(f"✅ Total ion images: {len(full_ds)} from {len(wrapped_datasets)} dataset(s)")

# --------------------- Model & training ---------------------
NUM_CLASSES = len(CLASSES)

model = timm.create_model(
    CLASSIFIER_MODEL,
    pretrained=PRETRAINED,
    in_chans=1,
    num_classes=NUM_CLASSES,
    drop_path_rate=args.drop_path_rate,
).to(device)
# Optional DropConnect regularisation on weights
if args.weight_dropout > 0:
    print(f"Applying weight dropout p={args.weight_dropout:.3f} to convolutional and linear layers")
    model = apply_weight_dropout(model, args.weight_dropout)
# model = torch.compile(model, mode="reduce-overhead") # compile so that it is faster

teacher = clone_as_ema(model).to(device) if args.use_ema else None
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
if SCHEDULER == "cosine":
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=EPOCHS,         # total epochs
        lr_min=MIN_LR,            # min LR at the end of schedule
        warmup_t=WARMUP_EPOCHS,   # warmup epochs
        warmup_lr_init=LR * 0.1,  # warmup starting LR
        t_in_epochs=True,         # step by epoch (simplest)
    )

# graceful exit
def _save_checkpoint(epoch: int, is_best: bool = False):
    """Save model checkpoint.
    
    Args:
        epoch: Current epoch number
        is_best: If True, saves as best model
    """
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'labels': LABELS,
        'input_hw': TARGET_HW,
        'best_val_loss': best[0],
        'best_val_acc': best[1],
        'hyperparams': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'val_fraction': VAL_FRACTION,
            'test_fraction': TEST_FRACTION,
            'seed': SEED,
            'arch': CLASSIFIER_MODEL,
            'in_chans': 1
        }
    }
    
    # Always save last checkpoint
    last_path = RUN_DIR / "last.pt"
    torch.save(checkpoint, last_path)
    
    # Save best checkpoint if applicable
    if is_best:
        best_path = RUN_DIR / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"✓ Saved E{ep} best model to {best_path}")
        
    return last_path

def _sig_handler(sig, frame):
    print(f"\nCaught {sig}. Saving last checkpoint and exiting…")
    try:
        _save_checkpoint(locals().get('ep', 0))
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)

# --------------------- Split datasets ---------------------

n_groups = len(wrapped_datasets)
n_test_groups = max(1, int(round(TEST_FRACTION * n_groups)))

#rng = np.random.default_rng(SEED)
perm = list(rng.permutation(n_groups))
available_for_split = [idx for idx in perm if idx not in fixed_test_indices and idx not in fixed_val_indices]

if fixed_val_indices:
    val_ids = set(fixed_val_indices)
else:
    n_val_groups = max(1, int(round(VAL_FRACTION * n_groups)))
    val_ids = set(available_for_split[:n_val_groups])
    if len(available_for_split) < n_val_groups:
        raise ValueError("Not enough datasets available for the validation split after reserving fixed test files.")

if fixed_test_indices:
    test_ids = set(fixed_test_indices)
else:
    remaining_for_test = available_for_split[n_val_groups:]
    if len(remaining_for_test) < n_test_groups:
        raise ValueError("Not enough datasets remaining to satisfy the requested test fraction.")
    test_ids = set(remaining_for_test[:n_test_groups])

train_ids = {i for i in range(n_groups) if i not in val_ids and i not in test_ids}

if not train_ids:
    raise ValueError("No datasets left for training after applying validation/test constraints.")

train_wrapped = [wrapped_datasets[i] for i in sorted(train_ids)]
val_wrapped   = [wrapped_datasets[i] for i in sorted(val_ids)]
test_wrapped  = [wrapped_datasets[i] for i in sorted(test_ids)]

for ds in train_wrapped:
    if hasattr(ds, "enable_augment"):
        ds.enable_augment = not CAN_USE_GPU_AUG
for ds in val_wrapped:
    if hasattr(ds, "enable_augment"):
        ds.enable_augment = False
for ds in test_wrapped:
    if hasattr(ds, "enable_augment"):
        ds.enable_augment = False

train_repeat = args.augmentation_repeat_factor
train_wrapped_repeated = train_wrapped * train_repeat
train_ds = ConcatDataset(train_wrapped_repeated)
val_ds   = ConcatDataset(val_wrapped)
test_ds  = ConcatDataset(test_wrapped)

per_item = defaultdict(lambda: {"loss": 0.0, "n": 0, "label": None, "dataset": None, "mz": None, "path": None, "student_pred": None, "student_conf": None, "student_label": None, "teacher_pred": None, "teacher_conf": None, "teacher_label": None})

print(f"Train datasets: {len(train_wrapped)} | Val datasets: {len(val_wrapped)} | Test datasets: {len(test_wrapped)}")
val_print_label = "VAL files (fixed)" if fixed_val_indices else "VAL files"
print(f"{val_print_label}:", [wrapped_meta[i]["name"] for i in sorted(val_ids)])
test_print_label = "TEST files (fixed)" if fixed_test_indices else "TEST files"
print(f"{test_print_label}:", [wrapped_meta[i]["name"] for i in sorted(test_ids)])
print(f"Train items: {len(train_ds)} | Val items: {len(val_ds)} | Test items: {len(test_ds)}")

# --------------------- Label Balancing ---------------------

def bincount_from_wrapped(wrapped_list):
    
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for ds in wrapped_list:
        # each ds is IonImageClassificationDataset (has labels_idx)
        c = np.bincount(ds.labels_idx, minlength=NUM_CLASSES)
        counts[:len(c)] += c
    return counts

train_counts = bincount_from_wrapped(train_wrapped) 
val_counts   = bincount_from_wrapped(val_wrapped)
test_counts  = bincount_from_wrapped(test_wrapped)

print("Train label counts:", {CLASSES[i]: int(n) for i, n in enumerate(train_counts)})
print("Val label counts:", {CLASSES[i]: int(n) for i, n in enumerate(val_counts)})
print("Test label counts:", {CLASSES[i]: int(n) for i, n in enumerate(test_counts)})

# Class weights: inverse frequency (avoid div-by-zero)
class_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
for i, n in enumerate(train_counts):
    class_weights[i] = 0.0 if n == 0 else 1.0 / float(n)

"""
# Using "normal shuffling+ weight loss" instead for now
# Per-sample weights for the *train* ConcatDataset
sample_weights = []
for ds in train_wrapped:
    sample_weights.extend(class_weights[ds.labels_idx])

sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
"""

print("Class weights used:", {CLASSES[i]: float(w) for i, w in enumerate(class_weights)})

# --------------------- Loss & scaler ---------------------

cw = torch.tensor(class_weights[:NUM_CLASSES], dtype=torch.float32, device=device)
criterion_hard = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
scaler = torch.amp.GradScaler('cuda', enabled=(device=="cuda"))

# --------------------- Loaders ---------------------
num_workers = NUM_WORKERS if device == "cuda" else 2
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    #sampler=sampler,           # balanced sampling
    num_workers=num_workers,
    pin_memory=(device=="cuda"),
    timeout=30,
    drop_last=False,
)
val_loader   = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,             # keep val deterministic
    num_workers=num_workers,
    pin_memory=(device=="cuda"),
    timeout=30,
    drop_last=False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,             # keep test deterministic
    num_workers=num_workers,
    pin_memory=(device=="cuda"),
    timeout=30,
    drop_last=False,
)

history = {"train_loss": [], "train_acc": [],
           "val_loss": [], "val_acc": [],
           "test_loss": [], "test_acc": [],
           "train_bal_acc": [], "val_bal_acc": [], "test_bal_acc": []}
early = EarlyStopping(patience=PATIENCE, min_delta=1e-4)

# --------------------- Resume & initial comparison ---------------------
start_epoch = 1
best = (1e9, 0.0)  # (best_val_loss, best_val_acc)

# Default to best_model.pt if user didn’t pass --resume but file exists
resume_path = args.resume
if resume_path is None and os.path.exists("best_model.pt"):
    resume_path = "best_model.pt"

if resume_path is not None and os.path.exists(resume_path):
    print(f"🔄 Resuming from {resume_path}...")
    try:
        ckpt = load_ckpt(resume_path, model, optimizer)
        # Initialize best metrics from checkpoint if available
        best_loss = float(ckpt.get("best_val_loss", 1e9))
        best_acc = float(ckpt.get("best_val_acc", 0.0))
        best = (best_loss, best_acc)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"✓ Loaded checkpoint from epoch {start_epoch-1}")
        print(f"  Best validation loss: {best_loss:.4f}, accuracy: {best_acc:.4f}")
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        print("   Starting training from scratch...")
        start_epoch = 1
        best = (1e9, 0.0)
else:
    start_epoch = 1
    best = (1e9, 0.0)
    if resume_path and not os.path.exists(resume_path):
        print(f"⚠️  Checkpoint not found at {resume_path}, starting from scratch")

# Define quick_validate function for initial validation
def quick_validate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    def _run(pass_loader):
        nonlocal total, correct, loss_sum, all_preds, all_labels
        with torch.no_grad():
            for imgs, labels, meta in pass_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                if args.training_type == "soft" and isinstance(meta, dict) and ("soft_target" in meta):
                    T = meta["soft_target"]
                    if not torch.is_tensor(T):
                        T = torch.stack(T)
                    T = T.to(device=logits.device, dtype=logits.dtype)
                    if T.dim() == 1:
                        T = T.unsqueeze(0)
                    loss, _ = soft_ce_loss(logits, T, cw)
                else:
                    loss = criterion_hard(logits, labels)
                bs = imgs.size(0)
                loss_sum += float(loss.item() if isinstance(loss, torch.Tensor) else loss) * bs
                preds = logits.argmax(1)
                correct  += (preds == labels).sum().item()
                total    += bs
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    try:
        _run(loader)
    except RuntimeError as e:
        if "timed out" not in str(e).lower():
            raise
        # Fallback: disable multiprocessing & timeout
        safe_loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            timeout=0,
            drop_last=False,
        )
        _run(safe_loader)
    acc = correct / max(total, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return loss_sum / max(total, 1), acc, bal_acc


# If eval_only flag is set, run validation and exit
# If eval_only flag is set, run validation and exit
if args.eval_only and resume_path and os.path.exists(resume_path):
    print("\n--- Running evaluation only ---")
    target_loader = val_loader if args.eval_split == "val" else test_loader
    split_name = args.eval_split.upper()
    loss, acc, bal_acc = quick_validate(target_loader)
    print(f"\n📊 {split_name} evaluation results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.2%}")
    print(f"  Balanced accuracy: {bal_acc:.2%}")
    exit(0)
elif args.eval_only:
    print("⚠️  No checkpoint found for evaluation. Exiting...")
    exit(1)

def soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None):
    """Cross-entropy with non-negative targets encoding information scores."""
    logp = F.log_softmax(logits, dim=1)
    weighted_targets = targets
    if class_weights is not None:
        weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        weighted_targets = weighted_targets * weights.unsqueeze(0)
    denom = weighted_targets.sum(dim=1).clamp_min(1e-6)
    per_sample = -(weighted_targets * logp).sum(dim=1) / denom
    return per_sample.mean(), per_sample


def save_confidence_histogram(probs: np.ndarray, phase: str, epoch_idx: int, out_dir: Path, bins: int = 10):
    """Persist histogram of max prediction confidences for a dataset split."""
    if probs.size == 0:
        return
    confidences = probs.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(confidences, bins=edges)
    df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts.astype(int),
    })
    csv_path = out_dir / f"{phase}_epoch{epoch_idx:03d}.csv"
    df.to_csv(csv_path, index=False)


def run_epoch(loader, train: bool, epoch_idx: int, phase: str):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels, all_probs = [], [], []

    # per-sample loss accumulator (for disagreement export)
    per_item.clear()

    bar = tqdm(loader, unit="batch", total=len(loader), desc=f"{phase} E{epoch_idx}")

    for b_idx, batch in enumerate(bar):
        t0 = perf_counter()
        imgs, labels, meta = batch   # <- meta is our per-item id for now
        imgs, labels = imgs.to(device), labels.to(device)
        if train and CAN_USE_GPU_AUG and device == "cuda":
            strength = float(augmentation_strength(args.augmentations))
            keep_prob = max(0.0, min(1.0, 1.0 - strength))
            if keep_prob < 1.0:
                mask = torch.rand(imgs.size(0), device=imgs.device) < keep_prob
                if mask.any():
                    aug_imgs = _gpu_base_geom(imgs[mask])
                    if args.augmentations != "default":
                        aug_imgs = apply_gpu_augmentations(aug_imgs, mode=args.augmentations)
                    imgs = imgs.clone()
                    imgs[mask] = aug_imgs
            else:
                imgs = _gpu_base_geom(imgs)
                if args.augmentations != "default":
                    imgs = apply_gpu_augmentations(imgs, mode=args.augmentations)

        # Build weak/strong views ON THE FLY (no dataset refactor needed)
        x_w = imgs
        x_s = imgs

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device=="cuda")):
            # supervised on strong view
            logits_s = model(x_s)
            if args.training_type == "soft" and isinstance(meta, dict) and ("soft_target" in meta):
                T = meta["soft_target"]
                if not torch.is_tensor(T):
                    T = torch.stack(T)
                T = T.to(device=logits_s.device, dtype=logits_s.dtype)
                if T.dim() == 1:
                    T = T.unsqueeze(0)
                loss_sup, per_sample_losses = soft_ce_loss(logits_s, T, cw)
            else:
                loss_sup = criterion_hard(logits_s, labels)
                weight = cw.to(device=logits_s.device, dtype=logits_s.dtype)
                per_sample_losses = F.cross_entropy(
                    logits_s.detach(),
                    labels,
                    weight=weight,
                    reduction="none",
                    label_smoothing=LABEL_SMOOTHING,
                )

            # unsupervised via teacher (multiclass CE with hard pseudo labels)
            loss_u = 0.0
            if teacher is not None and args.lambda_u > 0:
                with torch.no_grad():
                    t_prob = F.softmax(teacher(x_w), dim=1)   # [B,C]
                    conf, pseudo = t_prob.max(dim=1)          # hard pseudo labels
                mask = conf >= args.pseudo_tau
                if mask.any():
                    s_logits_masked = logits_s[mask]
                    pseudo_masked = pseudo[mask]
                    loss_u = F.cross_entropy(s_logits_masked, pseudo_masked, reduction="mean")
            # total loss
            loss = loss_sup + args.lambda_u * (loss_u if isinstance(loss_u, torch.Tensor) else 0.0)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                ema_update(model, teacher, m=args.ema_m)

        # -------- per-sample supervised loss (for disagreements) --------
        with torch.no_grad():
            per_sample_ce = per_sample_losses.detach()
            # meta is a dict of lists after collation
            ds_list = meta["dataset"]
            imzml_list = meta["imzml"]
            mz_list = meta["mz"]
            for i in range(labels.size(0)):
                uid = f"{ds_list[i]}:{float(mz_list[i]):.9f}"
                v = per_item[uid]
                v["loss"] += float(per_sample_ce[i].item())
                v["n"]    += 1
                v["label"] = int(labels[i])
                v["dataset"] = ds_list[i]
                v["mz"] = float(mz_list[i])
                v["path"] = imzml_list[i]

                # NEW: capture predictions for disagreements
                # Student prediction
                probs_s = F.softmax(logits_s[i].detach(), dim=0)
                student_pred = probs_s.argmax().item()
                student_conf = probs_s.max().item()
                v["student_pred"] = student_pred
                v["student_conf"] = student_conf
                v["student_label"] = CLASSES[student_pred] if 0 <= student_pred < len(CLASSES) else f"unknown_{student_pred}"

                # Teacher prediction (if available)
                if teacher is not None:
                    with torch.no_grad():
                        logits_t = teacher(x_w[i].unsqueeze(0))
                        probs_t = F.softmax(logits_t.squeeze(0), dim=0)
                        teacher_pred = probs_t.argmax().item()
                        teacher_conf = probs_t.max().item()
                        v["teacher_pred"] = teacher_pred
                        v["teacher_conf"] = teacher_conf
                        v["teacher_label"] = CLASSES[teacher_pred] if 0 <= teacher_pred < len(CLASSES) else f"unknown_{teacher_pred}"
                else:
                    v["teacher_pred"] = -1
                    v["teacher_conf"] = 0.0
                    v["teacher_label"] = "N/A"

        # -------- metrics --------
        bs = imgs.size(0)
        probs = F.softmax(logits_s.detach(), dim=1)
        preds = probs.argmax(1)
        batch_acc = (preds == labels).float().mean().item()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        loss_sum += float(loss.item()) * bs
        correct  += int((preds == labels).sum().item())
        total    += bs

        # Optional: show sup/unsup breakdown
        if teacher is not None and isinstance(loss_u, torch.Tensor):
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            sup=f"{loss_sup.item():.4f}",
                            unsup=f"{loss_u.item():.4f}",
                            acc=f"{batch_acc*100:.1f}%",
                            step_s=f"{perf_counter()-t0:.2f}")
        else:
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            acc=f"{batch_acc*100:.1f}%",
                            step_s=f"{perf_counter()-t0:.2f}")

    # -------- export disagreements AFTER the epoch loop --------
    rows = []
    for uid, v in per_item.items():
        rows.append({
            "uid": uid,
            "avg_loss": v["loss"] / max(1, v["n"]),
            "label_idx": v["label"],
            "label_name": CLASSES[v["label"]] if 0 <= v["label"] < len(CLASSES) else str(v["label"]),
            "dataset": v["dataset"],
            "mz": f"{v['mz']:.9f}" if v["mz"] is not None else "",
            "path": v["path"] or "",
            "student_pred": v["student_pred"],
            "student_conf": v["student_conf"],
            "student_label": v["student_label"],
            "teacher_pred": v["teacher_pred"],
            "teacher_conf": v["teacher_conf"],
            "teacher_label": v["teacher_label"],
        })
    rows.sort(key=lambda r: r["avg_loss"], reverse=True)
    rows = rows[:args.export_disagreements]

    if rows:
        out_csv = os.path.join(PER_SAMPLE_DIR, f"disagreements_{phase}_epoch{epoch_idx:03d}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Exported top-{len(rows)} disagreements to {out_csv}")
        
        # Generate visualizations for disagreements
        if args.visualize_disagreements:
            visualize_disagreements(out_csv, num_samples=min(10, len(rows)), save_dir=VIS_DIR, epoch_idx=epoch_idx, phase=phase)
        
    per_item.clear()

    # Log loss
    if teacher is not None:
        log_str = f"sup={loss_sup.item():.4f} | unsup={loss_u if isinstance(loss_u,float) else loss_u.item():.4f}"
    else:
        log_str = f"sup={loss_sup.item():.4f}"

    avg_loss = loss_sum / max(total, 1)
    avg_acc  = correct / max(total, 1)
    all_probs = np.concatenate(all_probs) if all_probs else np.array([])
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels), all_probs

# --------------------- Training Setup ---------------------
 
best = (1e9, 0.0)
best_epoch: Optional[int] = None
best_val_preds = None
best_val_labels = None
best_val_probs = None
final_test_loss = None
final_test_acc = None
final_test_bal_acc = None

try:
    for ep in range(start_epoch, EPOCHS + 1):
        # Training
        train_loss, train_acc, train_preds, train_labels, train_probs = run_epoch(train_loader, True, ep, "train")
        train_bal_acc = balanced_accuracy_score(train_labels, train_preds) if train_labels.size else 0.0

        # Validation
        with torch.no_grad():
            val_loss, val_acc, val_preds, val_labels, val_probs = run_epoch(val_loader, False, ep, "val")
            val_bal_acc = balanced_accuracy_score(val_labels, val_preds) if val_labels.size else 0.0

        # Test
        with torch.no_grad():
            test_loss, test_acc, test_preds, test_labels, test_probs = run_epoch(test_loader, False, ep, "test")
            test_bal_acc = balanced_accuracy_score(test_labels, test_preds) if test_labels.size else 0.0

        # Save confidence histograms per split
        save_confidence_histogram(train_probs, "train", ep, CONF_HIST_DIR)
        save_confidence_histogram(val_probs, "val", ep, CONF_HIST_DIR)
        save_confidence_histogram(test_probs, "test", ep, CONF_HIST_DIR)

        # Update best metrics and save predictions if this is the best model
        is_best = (val_loss, -val_acc) < (best[0], -best[1])
        if is_best:
            best = (val_loss, val_acc)
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_val_probs = val_probs
            best_epoch = ep
        
        # Save checkpoint
        _save_checkpoint(ep, is_best=is_best)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["train_bal_acc"].append(train_bal_acc)
        history["val_bal_acc"].append(val_bal_acc)
        history["test_bal_acc"].append(test_bal_acc)

        print(f"[{ep}/{EPOCHS}] train {train_loss:.4f}/{train_acc:.3f}/{train_bal_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}/{val_bal_acc:.3f}")
        
        # Advance LR schedule (epoch-based)
        if scheduler is not None:
            scheduler.step(ep)

        # log current LR
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"LR now: {curr_lr:.3e}")

        # Early stopping check
        if early.step(val_loss, model, epoch=ep):
            print(f"⏹ Early stopping at E{ep} (best val_loss={early.best_loss:.4f})")
            break
except Exception as e:
    print(f"Error during training: {e}")

# Restore best weights if early stopped
if early.best_state is not None:
    model.load_state_dict(early.best_state)
    print("↩︎ Restored best model weights.")
    print("\n--- Final evaluation on TEST with best VAL weights ---")
    final_test_loss, final_test_acc, final_test_bal_acc = quick_validate(test_loader)
    print(f"TEST: loss={final_test_loss:.4f}, acc={final_test_acc:.3%}, bal_acc={final_test_bal_acc:.3%}")
    with open(RUN_DIR / "final_test_summary.json", "w") as f:
        json.dump({"loss": final_test_loss, "accuracy": final_test_acc, "balanced_accuracy": final_test_bal_acc}, f, indent=2)


# --------- Simple per-class metrics ---------

epochs_ran = range(1, len(history["train_loss"]) + 1)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")

# Save history as CSV
epochs_ran = list(range(1, len(history["train_loss"]) + 1))
hist_df = pd.DataFrame({
    "epoch": epochs_ran,
    "train_loss": history["train_loss"],
    "train_acc":  history["train_acc"],
    "val_loss":   history["val_loss"],
    "val_acc":    history["val_acc"],
    "test_loss":  history["test_loss"],
    "test_acc":   history["test_acc"],
    "train_bal_acc": history["train_bal_acc"],
    "val_bal_acc": history["val_bal_acc"],
    "test_bal_acc": history["test_bal_acc"],
})
hist_df.to_csv(RUN_DIR / f"history_{ts}.csv", index=False)

# --------- Persist training history ---------
hist_json = RUN_DIR / "history.json"
hist_csv = RUN_DIR / "history.csv"

# Save history as JSON and CSV
with open(hist_json, "w") as f:
    json.dump(history, f, indent=2)

history_df = pd.DataFrame(history).assign(epoch=list(range(1, len(history["train_loss"]) + 1)))
history_df.to_csv(hist_csv, index=False)

# --------- Plot training curves ---------
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs_ran, history["train_loss"], 'b-', label='Train')
plt.plot(epochs_ran, history["val_loss"], 'r-', label='Validation')
plt.plot(epochs_ran, history["test_loss"], 'g-', label='Test')
plt.axvline(best_epoch, color='k', linestyle='--', label='Best Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training, Validation and Test Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_ran, history["train_acc"], 'b-', label='Train')
plt.plot(epochs_ran, history["val_acc"], 'r-', label='Validation')
plt.plot(epochs_ran, history["test_acc"], 'g-', label='Test')
plt.axvline(best_epoch, color='k', linestyle='--', label='Best Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0)
plt.title('Training, Validation and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(RUN_DIR / "training_curves.png", dpi=150, bbox_inches='tight')
plt.close()

# --------- Confusion Matrix Calculation ---------
# Use the correct labels based on COARSE_MODE
    
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)

# Set model to evaluation mode
model.eval()

# Get predictions on validation set
with torch.no_grad():
    for images, labels, _ in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Update confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

# Convert to numpy array for easier handling
cm = confusion_matrix.astype(int)

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{label}" for label in CLASSES],
    columns=[f"Pred_{label}" for label in CLASSES]
)
cm_df.to_csv(RUN_DIR / "confusion_matrix.csv")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    square=True,
    cbar=True,
    annot_kws={"size": 10}
)
plt.title('Confusion Matrix (rows=true, cols=pred)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(RUN_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# Calculate accuracy per file
print("\nAccuracy per file:")
per_file = {}
model.eval()
with torch.no_grad():
    for ds, meta in zip(val_wrapped, [m for i,m in enumerate(wrapped_meta) if i in val_ids]):
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        tot=correct=0
        for x,y,_ in loader:
            p = model(x.to(device)).argmax(1).cpu()
            correct += (p==y).sum().item(); tot += y.numel()
        accuracy = correct/tot
        per_file[meta["name"]] = accuracy
        
        # Add emoji based on accuracy
        if accuracy < 0.5:
            emoji = "🔴" 
        elif accuracy < 0.8:
            emoji = "⚠️"
        else:   
            emoji = "✅"
            
        print(f"{emoji} {meta['name']}: {accuracy:.3f}")

print("\nAccuracy per TEST file:")
per_file_test = {}
model.eval()
with torch.no_grad():
    for ds, meta in zip(test_wrapped, [m for i,m in enumerate(wrapped_meta) if i in test_ids]):
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        tot=correct=0
        for x,y,_ in loader:
            p = model(x.to(device)).argmax(1).cpu()
            correct += (p==y).sum().item(); tot += y.numel()
        accuracy = correct / max(tot,1)
        per_file_test[meta["name"]] = accuracy
        emoji = "🔴" if accuracy < 0.5 else ("⚠️" if accuracy < 0.8 else "✅")
        print(f"{emoji} {meta['name']}: {accuracy:.3f}")


# --------- Confusion Matrix per File ---------

from sklearn.metrics import confusion_matrix as sk_cm

RUN_CONF_VAL_DIR = RUN_DIR / "confusion_matrix_per_file"
RUN_CONF_VAL_DIR.mkdir(parents=True, exist_ok=True)

for ds, meta in zip(val_wrapped, [m for i, m in enumerate(wrapped_meta) if i in val_ids]):
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(device))
            pred = logits.argmax(1).cpu().numpy()
            all_p.append(pred)
            all_t.append(y.numpy())
    y_true = np.concatenate(all_t) if all_t else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])
    cm_file = sk_cm(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # save CSV + plot
    cm_df = pd.DataFrame(cm_file,
        index=[f"True_{c}" for c in CLASSES],
        columns=[f"Pred_{c}" for c in CLASSES])
    cm_df.to_csv(RUN_CONF_VAL_DIR / f"confusion_matrix_{meta['name']}.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_file, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, square=True, cbar=True)
    plt.title(f'Confusion Matrix - {meta["name"]} (rows=true, cols=pred)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(RUN_CONF_VAL_DIR / f"confusion_matrix_{meta['name']}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Confusion Matrix (TEST)
cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
model.eval()
with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device); labels = labels.to(device)
        preds = model(images).argmax(1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm_test[t.long(), p.long()] += 1

cm_test_df = pd.DataFrame(
    cm_test,
    index=[f"True_{label}" for label in CLASSES],
    columns=[f"Pred_{label}" for label in CLASSES]
)
cm_test_df.to_csv(RUN_DIR / "confusion_matrix_test.csv")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, square=True, cbar=True)
plt.title('Confusion Matrix TEST (rows=true, cols=pred)')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig(RUN_DIR / "confusion_matrix_test.png", dpi=150, bbox_inches='tight'); plt.close()

RUN_CONF_TEST_DIR = RUN_DIR / "confusion_matrix_per_file_test"
RUN_CONF_TEST_DIR.mkdir(parents=True, exist_ok=True)

for ds, meta in zip(test_wrapped, [m for i, m in enumerate(wrapped_meta) if i in test_ids]):
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    all_t, all_p = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            all_p.append(pred); all_t.append(y.numpy())
    y_true = np.concatenate(all_t) if all_t else np.array([])
    y_pred = np.concatenate(all_p) if all_p else np.array([])

    cm_file = sk_cm(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    pd.DataFrame(cm_file,
        index=[f"True_{c}" for c in CLASSES],
        columns=[f"Pred_{c}" for c in CLASSES]
    ).to_csv(RUN_CONF_TEST_DIR / f"confusion_matrix_{meta['name']}.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_file, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, square=True, cbar=True)
    plt.title(f'Confusion Matrix TEST - {meta["name"]} (rows=true, cols=pred)')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.xticks(rotation=45, ha='right')
    plt.tight_layout(); plt.savefig(RUN_CONF_TEST_DIR / f"confusion_matrix_{meta['name']}.png", dpi=150, bbox_inches='tight'); plt.close()

# --------- Save run metadata ---------
meta = {
    "Timestamp": RUN_TS,
    "Device": device,
    "Labels": MERGE_MAP,
    "Training type": f"{args.training_type} labels",
    "Soft Label Confidence": LABEL_CONFIDENCE if args.training_type == "soft" else None,
    "Soft Mixing Information": SOFT_MIXING if args.training_type == "soft" else None,
    "Arguments": {
        "Resume": args.resume,
        "Eval only": args.eval_only,
        "Use EMA": args.use_ema,
        "EMA m": args.ema_m,
        "Pseudo tau": args.pseudo_tau,
        "Lambda u": args.lambda_u,
        "Export disagreements": args.export_disagreements,
        "Weight dropout": args.weight_dropout,
        "Path Dropout Rate": args.drop_path_rate,
    },
    "Augmentations": {
        "Mode": args.augmentations,
        "Device": args.augmentation_device,
        "Strength": float(augmentation_strength(args.augmentations)),
        "Keep probability (1-strength)": float(1.0 - augmentation_strength(args.augmentations)),
        "Repeat factor": args.augmentation_repeat_factor,
    },
    "Arguments provided": vars(args),
    "Hyperparameters": {
        "Pretrained": PRETRAINED,
        "Resnet model": CLASSIFIER_MODEL,
        "Coarse mode": COARSE_MODE,
        "Scheduler Information":{
            "Scheduler": SCHEDULER,
            "Warmup epochs": WARMUP_EPOCHS,
            "Minimum learning rate": MIN_LR,
            },
        "Batch size": BATCH_SIZE,
        "Maximum epochs": EPOCHS,
        "Patience": PATIENCE,
        "Validation fraction": VAL_FRACTION,
        "Learning rate": LR,
        "Weight decay": WEIGHT_DECAY,
        "Random seed": SEED,
        "Number of workers": NUM_WORKERS,
    },
    "Input hw": TARGET_HW,
    "Number of train items": len(train_ds),
    "Number of val items": len(val_ds),
    "Number of test items": len(test_ds),
    "Best epoch": best_epoch,
    "Accuracy of Validation Files": per_file,
    "Accuracy of Test Files": per_file_test,
    "Train files": [wrapped_meta[i]["name"] for i in range(len(wrapped_meta)) if i not in val_ids and i not in test_ids],
    "Validation files": [wrapped_meta[i]["name"] for i in sorted(val_ids)],
    "Test files": [wrapped_meta[i]["name"] for i in sorted(test_ids)],
    "Requested test files": fixed_test_files if fixed_test_files else None,
    "Final Test Summary": {"loss": final_test_loss, "accuracy": final_test_acc, "balanced_accuracy": final_test_bal_acc},
    "Best Epoch": best_epoch,
    "Run directory": str(RUN_DIR)
}
with open(RUN_DIR / "run_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
