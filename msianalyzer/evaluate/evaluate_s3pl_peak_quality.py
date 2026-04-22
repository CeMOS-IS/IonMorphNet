#!/usr/bin/env python3
"""
Evaluate MSI classifier peak rankings using the S3PL correlation-based metrics.
"""

import argparse
import sys
import json
import itertools
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from pyimzml.ImzMLParser import ImzMLParser, getionimage

from msianalyzer.config import MODELS_DIR, TARGET_HW, LABELS
from msianalyzer.utils import msi_utils
from msianalyzer.external.s3pl import PeakEvaluation, PeakEvaluationMultipleClasses, check_for_labels
from msianalyzer.external.s3pl import data_configs
from msianalyzer.external.s3pl.create_pearson_labels import resolve_mask_path
from msianalyzer.external.s3pl import sota_comparison
from msianalyzer.utils.spectral_cube import SpectralCube
from msianalyzer.utils.targeted_cube import TargetedSpectralCube


REGRESSION_TARGETS = {
    "structured": (1.0, 1.0),
    "weak_structured": (0.85, 0.8),
    "localized": (0.65, 0.7),
    "fragmented": (0.4, 0.45),
    "unstructured": (0.2, 0.25),
    "negative": (0.95, 0.1),
}
REGRESSION_TARGET_DIM = len(next(iter(REGRESSION_TARGETS.values())))


def regression_logits_to_probs(logits: torch.Tensor, classes: Sequence[str]) -> torch.Tensor:
    matrix = torch.tensor([REGRESSION_TARGETS[name] for name in classes], device=logits.device, dtype=logits.dtype)
    diff = logits.unsqueeze(1) - matrix.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)
    sims = torch.exp(-dist_sq)
    probs = sims / sims.sum(dim=1, keepdim=True)
    return probs


def resolve_training_type(meta: Dict) -> str:
    raw = meta.get("Training type", "").lower()
    if "regression" in raw:
        return "regression"
    if "soft" in raw:
        return "soft"
    return "hard"


class WeightDropLegacy(torch.nn.Module):
    """Legacy wrapper that stores weight_raw parameters."""

    def __init__(self, module: nn.Module, weight_p: float):
        super().__init__()
        self.module = module
        self.weight_p = weight_p
        for name, param in list(self.module.named_parameters(recurse=False)):
            if "weight" in name:
                self.register_parameter(f"{name}_raw", nn.Parameter(param.data))
                self.module._parameters.pop(name, None)

    def _setweights(self):
        for name, raw in list(self.named_parameters(recurse=False)):
            if not name.endswith("_raw"):
                continue
            w = torch.nn.functional.dropout(raw, p=self.weight_p, training=self.training)
            setattr(self.module, name[:-4], torch.nn.Parameter(w))

    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return super().train(mode)


class WeightDrop(torch.nn.Module):
    """Training-time wrapper used by train_msi_classifier_fast.py."""

    def __init__(self, module: nn.Module, weight_p: float):
        super().__init__()
        self.module = module
        self.weight_p = float(weight_p)

    def _drop_weight(self) -> torch.Tensor:
        weight = self.module.weight
        return torch.nn.functional.dropout(weight, p=self.weight_p, training=self.training)

    def forward(self, *args, **kwargs):
        if isinstance(self.module, nn.Linear):
            input_tensor = args[0]
            weight = self._drop_weight()
            return torch.nn.functional.linear(input_tensor, weight, self.module.bias)
        if isinstance(self.module, nn.Conv2d):
            input_tensor = args[0]
            weight = self._drop_weight()
            return torch.nn.functional.conv2d(
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


def apply_weight_dropout(
    module: nn.Module,
    weight_p: float,
    module_types: Tuple[type, ...] = (nn.Linear, nn.Conv2d),
    weightdrop_cls: type = WeightDrop,
) -> nn.Module:
    if weight_p <= 0.0:
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, (WeightDrop, WeightDropLegacy)):
            apply_weight_dropout(child.module, weight_p, module_types, weightdrop_cls=weightdrop_cls)
            continue
        if isinstance(child, module_types):
            setattr(module, name, weightdrop_cls(child, weight_p))
        else:
            apply_weight_dropout(child, weight_p, module_types, weightdrop_cls=weightdrop_cls)
    return module


def model_has_weightdrop(module: nn.Module) -> bool:
    return any(isinstance(child, (WeightDrop, WeightDropLegacy)) for child in module.modules())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S3PL-style evaluation for classifier peak rankings.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory (or name under data/models).")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename inside run directory.")
    parser.add_argument("--data-root", type=Path, default="data/S3PL_Evaluation_Datasets", help="Root folder containing S3PL evaluation datasets.")
    parser.add_argument("--groups", type=str, nargs="*", default=None, help="Subset of subfolders (e.g. gbm cac). Defaults to all.")
    parser.add_argument("--top-peaks", type=int, default=None, help="Number of peaks to keep per dataset. Defaults to S3PL config.")
    parser.add_argument("--informative-classes", type=str, default=None, help="Comma-separated class names considered informative.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6], help="Pearson thresholds for evaluation.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers for inference (0 disables multiprocessing).")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu or cuda).")
    parser.add_argument("--resize-policy", type=str, default="letterbox", choices=["stretch", "letterbox"],
                        help="Resize strategy before model input: stretch (bilinear) or letterbox (pad preserve aspect).")
    parser.add_argument("--letterbox-min-side", type=int, default=64,
                        help="Minimum long-side size after letterbox scaling (0 disables).")
    parser.add_argument("--letterbox-max-scale", type=float, default=4.0,
                        help="Maximum upscaling factor for letterbox (0 disables).")
    parser.add_argument("--letterbox-pad-value", type=float, default=0.0,
                        help="Pad value for letterbox resize.")
    parser.add_argument("--resize-debug-count", type=int, default=0,
                        help="Print resize diagnostics for the first N samples per dataset.")
    parser.add_argument("--legacy-preprocess", action="store_true", default=False,
                        help="Disable mean/std normalization to match the old baseline preprocessing.")
    parser.add_argument("--legacy-baseline", action="store_true", default=True,
                        help="Bundle baseline-compatible settings (stretch resize, no spectral cubes, no mean/std).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for results directory.")
    parser.add_argument("--sweep-informative", action="store_true", default=False,
                        help="Evaluate all unique splits of classes into informative vs non-informative groups.")
    parser.add_argument("--structure-threshold", type=float, default=None,
                        help="Minimum structure score (from regression targets) required to treat a class as informative.")
    parser.add_argument("--informativeness-threshold", type=float, default=None,
                        help="Minimum informativeness score required to treat a class as informative.")
    parser.add_argument("--cube-mode", type=str, default="auto",
                        choices=["auto", "dense", "targeted", "imzml"],
                        help="Cube source: auto uses cached cubes if available; dense builds/uses SpectralCube; "
                             "targeted builds/uses TargetedSpectralCube; imzml reads directly (no cube building).")
    parser.add_argument("--run-sota", action="store_true", default=True,
                        help="After evaluation, generate SOTA comparison plot using evaluation_results.csv.")
    parser.add_argument("--regression-score-mode", type=str, default="raw", choices=["prob", "raw"],
                        help="Scoring mode for regression checkpoints: 'prob' (distance softmax) or 'raw' (use regression outputs dim0/1).")
    parser.add_argument("--rank-by", type=str, default="informative",
                        choices=["informative", "structure"],
                        help="Which score to use for ranking peaks (informative or structure).")
    parser.add_argument("--export-regression-preds", action="store_true", default=False,
                        help="When using regression checkpoints, export per-mz regression outputs for each dataset.")
    parser.add_argument("--export-regression-summary", action="store_true", default=False,
                        help="Export per-dataset summary of regression outputs (per-class mean and label histogram).")
    parser.add_argument("--default-ppm", type=float, default=20.0,
                        help="Fallback ppm window when no cached ppm is available.")
    return parser.parse_args()


def resolve_run_dir(spec: Path) -> Path:
    if spec.exists():
        return spec.resolve()
    candidate = MODELS_DIR / spec
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Run directory not found: {spec}")


def load_meta(run_dir: Path) -> Dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run_meta.json not found in {run_dir}")
    with open(meta_path, "r") as f:
        return json.load(f)


def derive_classes(coarse_mode: str) -> List[str]:
    if coarse_mode == "2class":
        return ["non_positive", "positive"]
    if coarse_mode == "2class-non-frag":
        return ["non_positive", "positive"]
    if coarse_mode == "2class-big-structure":
        return ["non_positive", "positive"]
    if coarse_mode == "3class":
        return ["non_structured", "partial_structured", "structured"]
    return list(LABELS)


def instantiate_model(meta: Dict, num_classes: int, device: torch.device, training_type: str) -> nn.Module:
    hyper = meta["Hyperparameters"]
    arch = hyper.get("Resnet model", "resnet50")
    pretrained = bool(hyper.get("Pretrained", True))
    args_block = meta.get("Arguments", {})
    drop_path = float(args_block.get("Path Dropout Rate", 0.0))
    weight_dropout = float(args_block.get("Weight dropout", 0.0))
    output_dim = REGRESSION_TARGET_DIM if training_type == "regression" else num_classes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        model = timm.create_model(arch, pretrained=pretrained, in_chans=1, num_classes=output_dim, drop_path_rate=drop_path)
    model.to(device)
    return model, weight_dropout


def load_checkpoint(run_dir: Path, checkpoint_name: str, device: torch.device) -> Tuple[Dict, bool]:
    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"Unexpected checkpoint format at {ckpt_path}")
    has_weight_raw = any(k.endswith("weight_raw") for k in checkpoint["model"].keys())
    return checkpoint, has_weight_raw


class S3PLInferenceDataset(Dataset):
    def __init__(
        self,
        cube,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        resize_policy: str = "letterbox",
        letterbox_min_side: int = 64,
        letterbox_max_scale: float = 4.0,
        letterbox_pad_value: float = 0.0,
        resize_debug_count: int = 0,
    ):
        self.cube = cube
        self.mz_values = cube.mz_values
        self.mean = mean
        self.std = std
        self.resize_policy = resize_policy
        self.letterbox_min_side = int(max(0, letterbox_min_side))
        self.letterbox_max_scale = float(letterbox_max_scale)
        self.letterbox_pad_value = float(letterbox_pad_value)
        self.resize_debug_remaining = int(max(0, resize_debug_count))

    def __len__(self) -> int:
        return len(self.mz_values)

    def __getitem__(self, idx: int):
        if hasattr(self.cube, "get_image_by_index"):
            img = self.cube.get_image_by_index(idx)
        else:
            img = self.cube.get_image(self.mz_values[idx])
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        p99 = torch.quantile(img_t.flatten(), 0.99) if img_t.numel() else torch.tensor(1.0)
        img_t = (img_t / p99.clamp_min(1e-6)).clamp_(0, 1)
        orig_h, orig_w = img_t.shape[-2], img_t.shape[-1]
        resized_inner = (orig_h, orig_w)
        if self.resize_policy == "letterbox":
            img_t, resized_inner = _letterbox_resize(
                img_t,
                TARGET_HW,
                min_side=self.letterbox_min_side,
                max_scale=self.letterbox_max_scale,
                pad_value=self.letterbox_pad_value,
            )
        else:
            img_t = F.interpolate(img_t.unsqueeze(0), size=TARGET_HW, mode="bilinear", align_corners=False).squeeze(0)
        if self.resize_debug_remaining > 0:
            inner_h, inner_w = resized_inner
            print(f"🧪 s3pl_resize {orig_h}x{orig_w} -> {inner_h}x{inner_w} (target {TARGET_HW[0]}x{TARGET_HW[1]})")
            self.resize_debug_remaining -= 1
        if (self.mean is not None) and (self.std is not None):
            img_t = (img_t - self.mean) / self.std
        mz = float(self.mz_values[idx])
        return img_t, torch.tensor(mz, dtype=torch.float32)


class DirectImzMLCube:
    """Load ion images directly from imzML without building cube caches."""

    def __init__(self, imzml_path: Path, ppm: float) -> None:
        self.imzml_path = Path(imzml_path)
        self.ppm = float(ppm)
        self.parser = ImzMLParser(str(self.imzml_path))
        self.mz_values, _ = self.parser.getspectrum(0)

    def get_image_by_index(self, idx: int) -> np.ndarray:
        mz = float(self.mz_values[idx])
        tol = mz * self.ppm * 1e-6
        return getionimage(self.parser, mz, tol=tol)

    def close(self) -> None:
        self.parser = None


def _letterbox_resize(
    img: torch.Tensor,
    target_hw: Tuple[int, int],
    min_side: int = 0,
    max_scale: float = 0.0,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize with preserved aspect ratio, then center-pad to target."""
    _, h, w = img.shape
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if h <= 0 or w <= 0 or target_h <= 0 or target_w <= 0:
        return img, (h, w)
    scale = min(target_h / h, target_w / w)
    if h < target_h and w < target_w:
        scale = 1.0
    elif min_side > 0:
        scale = max(scale, float(min_side) / max(h, w))
    scale = min(scale, target_h / h, target_w / w)
    if max_scale and max_scale > 0:
        scale = min(scale, float(max_scale))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    new_h = min(new_h, target_h)
    new_w = min(new_w, target_w)
    if new_h != h or new_w != w:
        mode = "nearest" if scale >= 1.0 else "area"
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode=mode).squeeze(0)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    if pad_h <= 0 and pad_w <= 0:
        return img, (new_h, new_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=float(pad_value)), (new_h, new_w)


def load_mz_values_from_cache(cache_dir: Optional[Path], stem: str) -> Optional[np.ndarray]:
    if cache_dir is None:
        return None
    mz_path = cache_dir / f"{stem}_mz.npy"
    if mz_path.exists():
        try:
            return np.load(mz_path)
        except Exception:
            return None
    return None


def load_mz_values_from_imzml(imzml_path: Path) -> Optional[np.ndarray]:
    try:
        parser = ImzMLParser(str(imzml_path))
        mz_vals, _ = parser.getspectrum(0)
        return mz_vals
    except Exception:
        return None


def collect_imzml_files(root: Path, groups: Optional[Sequence[str]]) -> List[Path]:
    roots = []
    if groups:
        for g in groups:
            sub = root / g
            if not sub.exists():
                raise FileNotFoundError(f"Group folder not found: {sub}")
            roots.append(sub)
    else:
        roots = [p for p in root.iterdir() if p.is_dir()]
    imzml_files: List[Path] = []
    for folder in roots:
        imzml_files.extend(sorted(folder.glob("*.imzML")))
    if not imzml_files:
        raise RuntimeError(f"No imzML files found under {root}")
    return imzml_files


@dataclass
class CubeSpec:
    imzml_path: Path
    cache_dir: Path
    mz_values: Optional[np.ndarray] = None


def collect_cube_specs(root: Path, groups: Optional[Sequence[str]]) -> List[CubeSpec]:
    roots: List[Path] = []
    if groups:
        for g in groups:
            sub = root / g
            if not sub.exists():
                raise FileNotFoundError(f"Group folder not found: {sub}")
            roots.append(sub)
    else:
        roots = [p for p in root.iterdir() if p.is_dir()]

    specs: List[CubeSpec] = []
    for folder in roots:
        cube_dir = folder / "spectral_cubes"
        if not cube_dir.exists():
            continue
        for meta_path in sorted(cube_dir.glob("*_cube.json")):
            stem = meta_path.stem.replace("_cube", "")
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                imzml_path = Path(meta.get("imzml_path", folder / f"{stem}.imzML"))
                mz_values = np.load(meta_path.with_name(f"{stem}_mz.npy"))
            except Exception:
                imzml_path = folder / f"{stem}.imzML"
                mz_values = None
            specs.append(CubeSpec(imzml_path=imzml_path, cache_dir=cube_dir, mz_values=mz_values))
    if not specs:
        raise RuntimeError(f"No spectral cubes found under {root}")
    return specs


def parse_informative(raw: Optional[str], classes: Sequence[str]) -> Tuple[List[str], List[str]]:
    if raw is None or not raw.strip():
        informative = [c for c in classes if "non" not in c and "unstructured" not in c]
        if not informative:
            informative = list(classes)
        non_inf = [c for c in classes if c not in informative]
        return informative, non_inf
    tokens = [c.strip() for c in raw.split(",") if c.strip()]
    missing = [c for c in tokens if c not in classes]
    if missing:
        raise ValueError(f"Unknown class names in --informative-classes: {missing}")
    informative = tokens
    non_inf = [c for c in classes if c not in informative]
    return informative, non_inf


def informative_from_thresholds(struct_thresh: Optional[float],
                                info_thresh: Optional[float],
                                classes: Sequence[str]) -> Tuple[List[str], List[str]]:
    s_thr = float("-inf") if struct_thresh is None else float(struct_thresh)
    i_thr = float("-inf") if info_thresh is None else float(info_thresh)
    informative, non_inf = [], []
    for name in classes:
        vec = REGRESSION_TARGETS.get(name)
        if vec is None:
            non_inf.append(name)
            continue
        struct_score, info_score = vec
        if struct_score >= s_thr and info_score >= i_thr:
            informative.append(name)
        else:
            non_inf.append(name)
    return informative, non_inf


def max_f1_given_peak_count(num_picked: int, num_true: int) -> float:
    if num_picked <= 0 or num_true <= 0:
        return 0.0
    if num_picked < num_true:
        max_recall = num_picked / num_true
        return 2 * max_recall / (1 + max_recall)
    if num_picked > num_true:
        max_precision = num_true / num_picked
        return 2 * max_precision / (1 + max_precision)
    return 1.0


def mixed_metrics_from_groundtruth(peak_list: List[float],
                                   all_mz: np.ndarray,
                                   true_peaks: Set[float],
                                   false_peaks: Set[float]) -> Dict[str, float]:
    peak_set = set(peak_list)
    all_set = set(all_mz.tolist())
    non_peaks = all_set - peak_set
    tp = len(peak_set & true_peaks)
    fp = len(peak_set & false_peaks)
    fn = len(non_peaks & true_peaks)
    if tp == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    return {"recall": recall, "precision": precision, "F1": f1}


def adapt_peak_list(peak_list: List[float], all_mz: np.ndarray) -> List[float]:
    all_set = set(all_mz.tolist())
    adapted = []
    for mz in peak_list:
        if mz in all_set:
            adapted.append(mz)
            continue
        idx = int(np.argmin(np.abs(all_mz - float(mz))))
        adapted.append(float(all_mz[idx]))
    return adapted


def class_metrics_from_groundtruth(peak_list: List[float],
                                   all_mz: np.ndarray,
                                   class_groundtruth: Dict[int, Tuple[Set[float], Set[float]]]) -> Dict[str, Dict[str, float]]:
    peak_set = set(peak_list)
    all_set = set(all_mz.tolist())
    non_peaks = all_set - peak_set
    metrics: Dict[str, Dict[str, float]] = {}
    for class_id, (true_set, false_set) in class_groundtruth.items():
        tp = len(peak_set & true_set)
        fp = len(peak_set & false_set)
        fn = len(non_peaks & true_set)
        if tp == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
        metrics[f"class {class_id}"] = {"recall": recall, "precision": precision, "F1": f1}
    return metrics


def generate_class_splits(classes: Sequence[str]) -> List[Tuple[List[str], List[str]]]:
    """Return all informative/non-informative splits of the class list."""
    total = set(classes)
    splits: List[Tuple[List[str], List[str]]] = []
    n = len(classes)
    for r in range(1, n):
        for combo in itertools.combinations(classes, r):
            informative = list(combo)
            non_inf = [c for c in classes if c not in informative]
            splits.append((informative, non_inf))
    return splits


def run_inference(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device,
                  classes: Sequence[str], training_type: str, keep_regression: bool = False,
                  num_workers: int = 0) -> Dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=max(0, int(num_workers)), drop_last=False)
    all_probs: List[np.ndarray] = []
    all_mz: List[np.ndarray] = []
    all_reg: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for imgs, mz_values in loader:
            logits = model(imgs.to(device))
            if training_type == "regression":
                probs = regression_logits_to_probs(logits, classes)
                if keep_regression:
                    all_reg.append(logits.cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)
            probs = probs.cpu().numpy()
            all_probs.append(probs)
            all_mz.append(mz_values.numpy())
    if not all_probs:
        raise RuntimeError("No predictions generated for dataset.")
    probs = np.concatenate(all_probs, axis=0)
    mz = np.concatenate(all_mz, axis=0)
    reg = np.concatenate(all_reg, axis=0) if all_reg else None
    return {
        "probs": probs,
        "mz": mz,
        "regression": reg,
    }


def compute_scores_from_outputs(outputs: Dict[str, np.ndarray],
                                classes: Sequence[str],
                                informative: Sequence[str],
                                training_type: str,
                                score_mode: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Optional[np.ndarray], np.ndarray]:
    probs = outputs["probs"]
    mz_values = outputs["mz"]
    regression_vals = outputs.get("regression")

    if training_type == "regression" and score_mode == "raw":
        if regression_vals is None:
            raise RuntimeError("Regression outputs missing despite score_mode='raw'.")
        structure_scores = np.clip(regression_vals[:, 0], 0.0, 1.0)
        informative_scores = regression_vals[:, 1] if regression_vals.shape[1] > 1 else regression_vals[:, 0]
        informative_scores = np.clip(informative_scores, 0.0, 1.0)
    else:
        informative_idx = [classes.index(c) for c in informative]
        informative_scores = probs[:, informative_idx].sum(axis=1)
        structure_col = classes.index("structured") if "structured" in classes else informative_idx[0]
        structure_scores = probs[:, structure_col]

    pred_indices = probs.argmax(axis=1)
    pred_labels = [classes[i] for i in pred_indices]
    return informative_scores, structure_scores, pred_labels, pred_indices, regression_vals, mz_values


def generate_sota_plot(metrics_csv: Path, out_dir: Path) -> Optional[Path]:
    """Generate and save SOTA comparison scatter plots using our mSCF1 scores."""
    try:
        ours_map, max_map = sota_comparison.load_ours_scores_with_max(metrics_csv)
        sota_comparison.update_with_ours(ours_map)
    except Exception as e:
        print("SOTA comparison plot will only be created when GBM, RCC and CAC datasets are used for evaluation.")
        return None

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(1, 4)

    ax1 = fig.add_subplot(gs[0, 1])
    for result, symbol, method in zip(sota_comparison.results_gbm, sota_comparison.markers, sota_comparison.labels):
        ax1.scatter(np.arange(len(sota_comparison.datasets_gbm)), result, marker=symbol, label=method)
    ax1.set_title('GBM')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("mSCF1")
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.yaxis.grid()
    ax1.set_axisbelow(True)
    ax1.set_xticks(np.arange(len(sota_comparison.datasets_gbm)))
    ax1.set_xticklabels(sota_comparison.datasets_gbm, rotation=45, ha='right')

    ax2 = fig.add_subplot(gs[0, 2])
    for result, symbol, method in zip(sota_comparison.results_rcc, sota_comparison.markers, sota_comparison.labels):
        ax2.scatter(np.arange(len(sota_comparison.datasets_rcc)), result, marker=symbol)
    ax2.set_title('RCC')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.yaxis.grid()
    ax2.set_axisbelow(True)
    ax2.set_xticks(np.arange(len(sota_comparison.datasets_rcc)))
    ax2.set_xticklabels(sota_comparison.datasets_rcc, rotation=45, ha='right')

    ax3 = fig.add_subplot(gs[0, 3])
    for result, symbol, method in zip(sota_comparison.results_adeno, sota_comparison.markers, sota_comparison.labels):
        ax3.scatter(np.arange(len(sota_comparison.datasets_adeno)), result, marker=symbol)
    ax3.set_title('CAC')
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax3.yaxis.grid()
    ax3.set_axisbelow(True)
    ax3.set_xticks(np.arange(len(sota_comparison.datasets_adeno)))
    ax3.set_xticklabels(sota_comparison.datasets_adeno, rotation=45, ha='right')

    axlegend = fig.add_subplot(gs[0, 0])
    axlegend.axis('off')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.02, 0., 0.2, 0.85))
    fig.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.0)

    out_path = out_dir / "sota_comparison.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"SOTA comparison plot saved to {out_path}")
    return out_path


def load_s3pl_baseline(table_path: Path) -> Dict[str, Optional[float]]:
    if not table_path.exists():
        return {}
    df = pd.read_csv(table_path)
    row = df[df["Method"] == "S3PL"]
    if row.empty:
        return {}
    s3pl = row.iloc[0]
    return {
        "gbm": float(s3pl.get("GBM_mSCF1")) if pd.notna(s3pl.get("GBM_mSCF1")) else None,
        "renal_cell_carcinoma": float(s3pl.get("RCC_mSCF1")) if pd.notna(s3pl.get("RCC_mSCF1")) else None,
        "cac": float(s3pl.get("CAC_mSCF1")) if pd.notna(s3pl.get("CAC_mSCF1")) else None,
    }


def evaluate_dataset(
    dataname: str,
    cube: SpectralCube,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    classes: Sequence[str],
    informative: Sequence[str],
    top_k: Optional[int],
    thresholds: Sequence[float],
    training_type: str,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    score_mode: str,
    num_workers: int = 0,
    rank_by: str = "informative",
    resize_policy: str = "letterbox",
    letterbox_min_side: int = 64,
    letterbox_max_scale: float = 4.0,
    letterbox_pad_value: float = 0.0,
    resize_debug_count: int = 0,
    structure_threshold: Optional[float] = None,
    informativeness_threshold: Optional[float] = None,
    outputs: Optional[Dict[str, np.ndarray]] = None,
    parser: Optional[ImzMLParser] = None,
    all_mz: Optional[np.ndarray] = None,
    groundtruth_by_threshold: Optional[Dict[str, Tuple[Set[float], Set[float]]]] = None,
    class_groundtruth_by_threshold: Optional[Dict[str, Dict[int, Tuple[Set[float], Set[float]]]]] = None,
) -> Tuple[Dict, List[Tuple[float, float, float, str]], List[Tuple[float, float, float, str]], Dict[str, Set[float]]]:
    if outputs is None:
        dataset = S3PLInferenceDataset(
            cube,
            mean=mean,
            std=std,
            resize_policy=resize_policy,
            letterbox_min_side=letterbox_min_side,
            letterbox_max_scale=letterbox_max_scale,
            letterbox_pad_value=letterbox_pad_value,
            resize_debug_count=resize_debug_count,
        )
        outputs = run_inference(
            model,
            dataset,
            batch_size,
            device,
            classes,
            training_type,
            keep_regression=(training_type == "regression"),
            num_workers=num_workers,
        )

    informative_scores, structure_scores, pred_labels, pred_indices, regression_vals, mz_values = compute_scores_from_outputs(
        outputs,
        classes,
        informative,
        training_type,
        score_mode,
    )

    if rank_by == "structure":
        order = np.argsort(-structure_scores)
    else:
        order = np.argsort(-informative_scores)

    mask = np.ones_like(informative_scores, dtype=bool)
    if structure_threshold is not None:
        mask &= structure_scores >= float(structure_threshold)
    if informativeness_threshold is not None:
        mask &= informative_scores >= float(informativeness_threshold)
    filtered_order = order[mask[order]]
    sorted_scores = informative_scores[order]
    sorted_struct = structure_scores[order]
    sorted_mz = mz_values[order]
    sorted_labels = [pred_labels[i] for i in order]
    if top_k is None:
        default_k = data_configs.num_GT_peaks.get(dataname)
        top_k = default_k if default_k is not None else len(sorted_mz)
    if top_k <= 0:
        top_k = len(sorted_mz)
    top_k = min(top_k, len(filtered_order))
    peak_list = mz_values[filtered_order[:top_k]].tolist()

    folderpath = str(Path(cube.imzml_path).parent) + "/"

    if parser is None or all_mz is None:
        parser = ImzMLParser(folderpath + dataname + ".imzML")
        all_mz, _ = parser.getspectrum(0)
    check_for_labels(folderpath, dataname)
    mask_path = Path(resolve_mask_path(folderpath, dataname))
    mask_arr = np.load(mask_path)
    num_classes = int(np.unique(mask_arr).size)
    class_range = list(range(num_classes))

    per_threshold = {}
    true_peaks_by_threshold: Dict[str, Set[float]] = {}
    mixed_f1 = []
    max_f1 = []
    adapted_peak_list = peak_list
    if all_mz is not None and groundtruth_by_threshold is not None:
        adapted_peak_list = adapt_peak_list(peak_list, all_mz)

    for t in thresholds:
        class_cache = class_groundtruth_by_threshold.get(str(t)) if class_groundtruth_by_threshold else None
        if class_cache and all_mz is not None:
            class_metrics = class_metrics_from_groundtruth(adapted_peak_list, all_mz, class_cache)
        else:
            evaluator_cls = PeakEvaluationMultipleClasses(
                dataname, class_range, t, folderpath, peak_list, parser=parser, all_mz=all_mz
            )
            class_metrics = evaluator_cls.calculate_metrics()

        cache_entry = groundtruth_by_threshold.get(str(t)) if groundtruth_by_threshold else None
        if cache_entry and all_mz is not None:
            true_set, false_set = cache_entry
            mixed = mixed_metrics_from_groundtruth(adapted_peak_list, all_mz, true_set, false_set)
            max_f1_value = max_f1_given_peak_count(len(peak_list), len(true_set))
            true_peaks_by_threshold[str(t)] = {round(float(mz), 9) for mz in true_set}
            mixed["max_F1"] = max_f1_value
            f1 = mixed["F1"]
        else:
            evaluator = PeakEvaluation(
                dataname, class_range, t, folderpath, peak_list, parser=parser, all_mz=all_mz
            )
            recall, precision, f1 = evaluator.calculate_metrics()
            max_f1_value = max_f1_given_peak_count(len(peak_list), len(evaluator.true_peaks))
            true_peaks_by_threshold[str(t)] = {round(float(mz), 9) for mz in evaluator.true_peaks}
            mixed = {
                "recall": recall,
                "precision": precision,
                "F1": f1,
                "max_F1": max_f1_value,
            }
        mixed_f1.append(f1)
        max_f1.append(max_f1_value)

        per_threshold[str(t)] = {
            "class_metrics": class_metrics,
            "mixed_metrics": mixed,
        }

    top_indices = filtered_order[:top_k]
    top_pairs = list(zip(
        peak_list,
        informative_scores[top_indices].tolist(),
        structure_scores[top_indices].tolist(),
        [pred_labels[i] for i in top_indices],
    ))
    full_pairs = list(zip(sorted_mz.tolist(), sorted_scores.tolist(), sorted_struct.tolist(), sorted_labels))

    best_mixed_f1 = None
    for metrics in per_threshold.values():
        f1 = metrics["mixed_metrics"].get("F1")
        if f1 is None:
            continue
        if (best_mixed_f1 is None) or (f1 > best_mixed_f1):
            best_mixed_f1 = f1

    summary = {
        "top_k": top_k,
        "threshold_metrics": per_threshold,
        "mSCF1": float(np.mean(mixed_f1)) if mixed_f1 else None,
        "max_mSCF1": float(np.mean(max_f1)) if max_f1 else None,
        "best_mixed_F1": best_mixed_f1,
        "regression_outputs": regression_vals.tolist() if (regression_vals is not None) else None,
        "mz_values": mz_values.tolist(),
        "pred_labels": pred_labels,
        "label_hist": {cls: int((np.array(pred_indices) == idx).sum()) for idx, cls in enumerate(classes)},
        "num_candidates": int(mask.sum()),
        "top_k_effective": int(top_k),
    }
    return summary, top_pairs, full_pairs, true_peaks_by_threshold


def main():
    args = parse_args()
    if args.legacy_baseline:
        args.resize_policy = "stretch"
        args.legacy_preprocess = True
        args.cube_mode = "cube"
    run_dir = resolve_run_dir(args.run_dir)
    meta = load_meta(run_dir)
    coarse_mode = meta.get("Hyperparameters", {}).get("Coarse mode", "full")
    classes = derive_classes(coarse_mode)
    training_type = resolve_training_type(meta)
    if training_type == "hard":
        #if "--resize-policy" not in sys.argv:
        #    args.resize_policy = "stretch"
        if "--rank-by" not in sys.argv:
            args.rank_by = "informative"
    threshold_mode = (args.structure_threshold is not None) or (args.informativeness_threshold is not None)
    if threshold_mode:
        if training_type != "regression":
            raise ValueError("--structure-threshold/--informativeness-threshold require a regression checkpoint.")
        if args.sweep_informative:
            raise ValueError("Cannot sweep informative splits when using threshold-based informative selection.")
        informative, non_informative = informative_from_thresholds(
            args.structure_threshold,
            args.informativeness_threshold,
            classes,
        )
        if not informative:
            raise ValueError("No classes satisfied the provided thresholds.")
        splits = [(informative, non_informative)]
    elif args.sweep_informative:
        splits = generate_class_splits(classes)
        if not splits:
            raise ValueError("No informative/non-informative splits generated for sweep.")
    else:
        informative, non_informative = parse_informative(args.informative_classes, classes)
        splits = [(informative, non_informative)]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if training_type == "regression" and coarse_mode != "full":
        raise ValueError("Regression checkpoints require full label space for evaluation.")
    model, weight_dropout = instantiate_model(meta, len(classes), device, training_type)
    data_cfg = resolve_model_data_config(model)
    mean = torch.tensor(data_cfg.get("mean", [0.0]), dtype=torch.float32).view(-1)
    std = torch.tensor(data_cfg.get("std", [1.0]), dtype=torch.float32).view(-1)
    mean = mean[:1].view(1, 1, 1)
    std = std[:1].view(1, 1, 1).clamp_min(1e-6)
    if args.legacy_preprocess:
        mean = None
        std = None
    checkpoint, ckpt_has_weight_raw = load_checkpoint(run_dir, args.checkpoint, device)
    if ckpt_has_weight_raw and not model_has_weightdrop(model):
        # Apply legacy wrappers to load weight_raw params.
        model = apply_weight_dropout(model, weight_dropout, weightdrop_cls=WeightDropLegacy)
    elif (not ckpt_has_weight_raw) and (weight_dropout > 0) and not model_has_weightdrop(model):
        model = apply_weight_dropout(model, weight_dropout, weightdrop_cls=WeightDrop)
    missing = model.load_state_dict(checkpoint["model"], strict=False)
    if missing.missing_keys:
        print(f"⚠️ Missing keys when loading checkpoint: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"⚠️ Unexpected keys when loading checkpoint: {missing.unexpected_keys}")

    cube_mode = args.cube_mode
    use_cubes = cube_mode != "imzml"
    cube_specs: Optional[List[CubeSpec]] = None
    imzml_files: List[Path] = []

    msianalyzer_dir = Path(__file__).resolve().parent.parent.parent
    if use_cubes:
        try:
            cube_specs = collect_cube_specs(msianalyzer_dir / args.data_root, args.groups)
        except Exception as e:
            print(f"⚠️  Falling back to imzML files (could not load spectral cubes): {e}")
            use_cubes = False
            if cube_mode == "auto":
                cube_mode = "dense"
    if not use_cubes:
        imzml_files = collect_imzml_files(msianalyzer_dir / args.data_root, args.groups)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or (run_dir / "evaluation_s3pl" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    cubes = {}
    if use_cubes and cube_specs:
        imzml_files = [spec.imzml_path for spec in cube_specs]
        for spec in cube_specs:
            ppm = msi_utils.get_ppm_from_cache_only(spec.imzml_path.stem) or args.default_ppm
            if ppm == 3:  # metaspace cache missing; fall back to provided default
                ppm = args.default_ppm
            mz_vals = spec.mz_values
            if mz_vals is None:
                mz_vals = load_mz_values_from_cache(spec.cache_dir, spec.imzml_path.stem)
            use_targeted = (cube_mode in ("auto", "targeted")) and (mz_vals is not None)
            if cube_mode == "dense":
                use_targeted = False
            if use_targeted:
                cubes[spec.imzml_path] = TargetedSpectralCube(
                    spec.imzml_path, mz_list=mz_vals, ppm=ppm, build_if_missing=True
                )
            else:
                cubes[spec.imzml_path] = SpectralCube(spec.imzml_path, cache_dir=spec.cache_dir)
    else:
        for imzml_path in imzml_files:
            ppm = msi_utils.get_ppm_from_cache_only(imzml_path.stem) or args.default_ppm
            if ppm == 3:
                ppm = args.default_ppm
            if cube_mode == "imzml":
                cubes[imzml_path] = DirectImzMLCube(imzml_path, ppm=ppm)
                continue
            mz_vals = None
            if cube_mode in ("auto", "targeted"):
                mz_vals = load_mz_values_from_imzml(imzml_path)
            use_targeted = (cube_mode in ("auto", "targeted")) and (mz_vals is not None)
            if cube_mode == "dense":
                use_targeted = False
            if use_targeted:
                cubes[imzml_path] = TargetedSpectralCube(imzml_path, mz_list=mz_vals, ppm=ppm, build_if_missing=True)
                continue
            cubes[imzml_path] = SpectralCube(imzml_path)
    outputs_cache: Dict[str, Dict[str, np.ndarray]] = {}
    parser_cache: Dict[str, Tuple[ImzMLParser, np.ndarray]] = {}
    groundtruth_cache: Dict[str, Dict[str, Tuple[Set[float], Set[float]]]] = {}
    class_groundtruth_cache: Dict[str, Dict[str, Dict[int, Tuple[Set[float], Set[float]]]]] = {}
    if args.sweep_informative:
        print("Running inference once per dataset for informative sweep...")
        for imzml_path, cube in cubes.items():
            t0 = time.time()
            dataset = S3PLInferenceDataset(
                cube,
                mean=mean,
                std=std,
                resize_policy=args.resize_policy,
                letterbox_min_side=args.letterbox_min_side,
                letterbox_max_scale=args.letterbox_max_scale,
                letterbox_pad_value=args.letterbox_pad_value,
                resize_debug_count=args.resize_debug_count,
            )
            outputs_cache[imzml_path.stem] = run_inference(
                model,
                dataset,
                args.batch_size,
                device,
                classes,
                training_type,
                keep_regression=(training_type == "regression"),
                num_workers=args.num_workers,
            )
            t1 = time.time()
            parser = ImzMLParser(str(imzml_path))
            all_mz, _ = parser.getspectrum(0)
            parser_cache[imzml_path.stem] = (parser, all_mz)
            folderpath = str(imzml_path.parent) + "/"
            dataname = imzml_path.stem
            check_for_labels(folderpath, dataname)
            mask_path = Path(resolve_mask_path(folderpath, dataname))
            mask_arr = np.load(mask_path)
            num_classes = int(np.unique(mask_arr).size)
            class_range = list(range(num_classes))
            gt_per_threshold: Dict[str, Tuple[Set[float], Set[float]]] = {}
            class_gt_per_threshold: Dict[str, Dict[int, Tuple[Set[float], Set[float]]]] = {
                str(t): {} for t in args.thresholds
            }
            class_rankings: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for class_id in class_range:
                idx_ranking = np.load(folderpath + "labels/" + dataname + "_class" + str(class_id) + "_ranking.npy")
                pearson_ranking = np.load(
                    folderpath + "labels/" + dataname + "_class" + str(class_id) + "_pearson_ranking.npy"
                )
                class_rankings[class_id] = (idx_ranking, pearson_ranking)
            for t in args.thresholds:
                evaluator = PeakEvaluation(
                    dataname, class_range, t, folderpath, [], parser=parser, all_mz=all_mz
                )
                evaluator.get_groundtruth()
                gt_per_threshold[str(t)] = (set(evaluator.true_peaks), set(evaluator.false_peaks))
                for class_id in class_range:
                    idx_ranking, pearson_ranking = class_rankings[class_id]
                    num_best_peaks = int(np.argmin(abs(pearson_ranking - t)))
                    true_idx = idx_ranking[:num_best_peaks]
                    false_idx = idx_ranking[num_best_peaks:]
                    true_set = {float(all_mz[idx]) for idx in true_idx}
                    false_set = {float(all_mz[idx]) for idx in false_idx}
                    class_gt_per_threshold[str(t)][class_id] = (true_set, false_set)
            groundtruth_cache[dataname] = gt_per_threshold
            class_groundtruth_cache[dataname] = class_gt_per_threshold
            t2 = time.time()
            print(f"  Precompute {imzml_path.stem}: inference={t1 - t0:.1f}s, groundtruth={t2 - t1:.1f}s")
    sweep_entries = []
    try:
        for split_idx, (informative_split, non_informative_split) in enumerate(splits, start=1):
            if threshold_mode:
                print(f"[Split {split_idx}/{len(splits)}] regression thresholds: structure ≥ {args.structure_threshold}, informativeness ≥ {args.informativeness_threshold}; informative={informative_split}, non_informative={non_informative_split}")
            else:
                print(f"[Split {split_idx}/{len(splits)}] informative={informative_split}, non_informative={non_informative_split}")
            if args.sweep_informative:
                slug = "informative_" + ("-".join(informative_split) if informative_split else "none")
                split_dir = out_dir / slug
                split_dir.mkdir(parents=True, exist_ok=True)
            else:
                split_dir = out_dir

            metrics_summary = {}
            peaks_dir = split_dir / "peaks"
            peaks_dir.mkdir(parents=True, exist_ok=True)
            dataset_rows = []

            for imzml_path, cube in cubes.items():
                dataname = imzml_path.stem
                group_name = imzml_path.parent.name
                print(f"  Evaluating {dataname}...")
                t0 = time.time()
                cached = parser_cache.get(dataname) if args.sweep_informative else None
                parser = cached[0] if cached else None
                all_mz = cached[1] if cached else None
                summary, top_pairs, full_pairs, true_peaks_by_threshold = evaluate_dataset(
                    dataname=dataname,
                    cube=cube,
                    model=model,
                    batch_size=args.batch_size,
                    device=device,
                    classes=classes,
                    informative=informative_split,
                    top_k=args.top_peaks,
                    thresholds=args.thresholds,
                    training_type=training_type,
                    mean=mean,
                    std=std,
                    score_mode=args.regression_score_mode,
                    num_workers=args.num_workers,
                    rank_by=args.rank_by,
                    resize_policy=args.resize_policy,
                    letterbox_min_side=args.letterbox_min_side,
                    letterbox_max_scale=args.letterbox_max_scale,
                    letterbox_pad_value=args.letterbox_pad_value,
                    resize_debug_count=args.resize_debug_count,
                    structure_threshold=args.structure_threshold,
                    informativeness_threshold=args.informativeness_threshold,
                    outputs=outputs_cache.get(dataname) if args.sweep_informative else None,
                    parser=parser,
                    all_mz=all_mz,
                    groundtruth_by_threshold=groundtruth_cache.get(dataname) if args.sweep_informative else None,
                    class_groundtruth_by_threshold=class_groundtruth_cache.get(dataname) if args.sweep_informative else None,
                )
                t1 = time.time()
                print(f"    Evaluation {dataname} done in {t1 - t0:.1f}s")
                metrics_summary[dataname] = summary

                peak_csv = peaks_dir / f"{dataname}_top_peaks.csv"
                threshold_keys = [str(t) for t in args.thresholds]
                correct_headers = [f"correct_t{t}" for t in threshold_keys]
                with open(peak_csv, "w") as f:
                    f.write("mz,informative_score,structure_score,predicted_label," + ",".join(correct_headers) + "\n")
                    for mz, info_score, struct_score, label in top_pairs:
                        mz_key = round(float(mz), 9)
                        flags = [
                            "1" if mz_key in true_peaks_by_threshold.get(t, set()) else "0"
                            for t in threshold_keys
                        ]
                        f.write(f"{mz},{info_score},{struct_score},{label}," + ",".join(flags) + "\n")
                ranking_csv = peaks_dir / f"{dataname}_all_scores.csv"
                with open(ranking_csv, "w") as f:
                    f.write("mz,informative_score,structure_score,predicted_label," + ",".join(correct_headers) + "\n")
                    for mz, info_score, struct_score, label in full_pairs:
                        mz_key = round(float(mz), 9)
                        flags = [
                            "1" if mz_key in true_peaks_by_threshold.get(t, set()) else "0"
                            for t in threshold_keys
                        ]
                        f.write(f"{mz},{info_score},{struct_score},{label}," + ",".join(flags) + "\n")
                if args.export_regression_preds and training_type == "regression" and summary.get("regression_outputs") is not None:
                    reg_csv = peaks_dir / f"{dataname}_regression_outputs.csv"
                    with open(reg_csv, "w") as f:
                        f.write("mz,pred_structure,pred_informativeness,top_label\n")
                        reg_vals = summary["regression_outputs"]
                        for mz_val, reg_vec, lbl in zip(summary["mz_values"], reg_vals, summary["pred_labels"]):
                            s_val = reg_vec[0] if len(reg_vec) > 0 else ""
                            i_val = reg_vec[1] if len(reg_vec) > 1 else ""
                            f.write(f"{mz_val},{s_val},{i_val},{lbl}\n")
                if args.export_regression_summary and training_type == "regression" and summary.get("regression_outputs") is not None:
                    # per-class mean regression outputs + predicted label hist
                    reg_arr = np.asarray(summary["regression_outputs"])
                    pred_arr = np.asarray(summary["pred_labels"])
                    class_means = {}
                    for cls in classes:
                        mask = pred_arr == cls
                        if mask.any():
                            cls_mean = reg_arr[mask].mean(axis=0)
                            class_means[cls] = cls_mean.tolist()
                        else:
                            class_means[cls] = None
                    summary_csv = peaks_dir / f"{dataname}_regression_summary.csv"
                    with open(summary_csv, "w") as f:
                        f.write("class,count,mean_structure,mean_informativeness\n")
                        for cls in classes:
                            cnt = summary["label_hist"].get(cls, 0)
                            mean_vec = class_means.get(cls)
                            ms, mi = ("", "") if mean_vec is None else (mean_vec[0], mean_vec[1] if len(mean_vec) > 1 else "")
                            f.write(f"{cls},{cnt},{ms},{mi}\n")
                dataset_rows.append({
                    "dataset": dataname,
                    "group": group_name,
                    "mSCF1": summary.get("mSCF1"),
                    "max_mSCF1": summary.get("max_mSCF1"),
                    "best_mixed_F1": summary.get("best_mixed_F1"),
                })

            dataset_metrics_csv = split_dir / "evaluation_results.csv"
            s3pl_baseline = load_s3pl_baseline(Path("data/S3PL_Evaluation_Weigand/evaluation_data/mSCF1_method_table.csv"))
            baseline_all = None
            if s3pl_baseline:
                baseline_vals = [v for v in s3pl_baseline.values() if v is not None]
                if baseline_vals:
                    baseline_all = float(np.mean(baseline_vals))
            with open(dataset_metrics_csv, "w") as f:
                f.write("filename,dataset,mSCF1\n")
                for row in dataset_rows:
                    mscf1 = row["mSCF1"]
                    max_mscf1 = row.get("max_mSCF1")
                    best_mixed = row["best_mixed_F1"]
                    mscf1_str = "" if mscf1 is None else f"{mscf1:.6f}"
                    max_mscf1_str = "" if max_mscf1 is None else f"{max_mscf1:.6f}"
                    best_mixed_str = "" if best_mixed is None else f"{best_mixed:.6f}"
                    f.write(f"{row['dataset']},{row['group']},{mscf1_str},\n")
                if dataset_rows:
                    valid_mscf1 = [row["mSCF1"] for row in dataset_rows if row["mSCF1"] is not None]
                    valid_best = [row["best_mixed_F1"] for row in dataset_rows if row["best_mixed_F1"] is not None]
                    valid_max = [row["max_mSCF1"] for row in dataset_rows if row.get("max_mSCF1") is not None]
                    mean_mscf1 = float(np.mean(valid_mscf1)) if valid_mscf1 else None
                    mean_best = float(np.mean(valid_best)) if valid_best else None
                    mean_max = float(np.mean(valid_max)) if valid_max else None
                    mean_mscf1_str = "" if mean_mscf1 is None else f"{mean_mscf1:.6f}"
                    mean_best_str = "" if mean_best is None else f"{mean_best:.6f}"
                    mean_max_str = "" if mean_max is None else f"{mean_max:.6f}"
                    delta_all = "" if (mean_mscf1 is None or baseline_all is None) else f"{mean_mscf1 - baseline_all:.6f}"
                    f.write(f"mean,all,{mean_mscf1_str},\n")
                    groups = sorted({row["group"] for row in dataset_rows})
                    for group in groups:
                        group_rows = [r for r in dataset_rows if r["group"] == group]
                        group_mscf1 = [r["mSCF1"] for r in group_rows if r["mSCF1"] is not None]
                        group_best = [r["best_mixed_F1"] for r in group_rows if r["best_mixed_F1"] is not None]
                        group_max = [r["max_mSCF1"] for r in group_rows if r.get("max_mSCF1") is not None]
                        g_mean_mscf1 = float(np.mean(group_mscf1)) if group_mscf1 else None
                        g_mean_best = float(np.mean(group_best)) if group_best else None
                        g_mean_max = float(np.mean(group_max)) if group_max else None
                        g_mscf1_str = "" if g_mean_mscf1 is None else f"{g_mean_mscf1:.6f}"
                        g_best_str = "" if g_mean_best is None else f"{g_mean_best:.6f}"
                        g_max_str = "" if g_mean_max is None else f"{g_mean_max:.6f}"
                        baseline = s3pl_baseline.get(group) if s3pl_baseline else None
                        delta = "" if (g_mean_mscf1 is None or baseline is None) else f"{g_mean_mscf1 - baseline:.6f}"
                        f.write(f"mean,{group},{g_mscf1_str},\n")
            if args.run_sota:
                generate_sota_plot(dataset_metrics_csv, split_dir)

            group_means = {}
            for group in sorted({row["group"] for row in dataset_rows}):
                group_rows = [r for r in dataset_rows if r["group"] == group]
                group_mscf1 = [r["mSCF1"] for r in group_rows if r["mSCF1"] is not None]
                group_best = [r["best_mixed_F1"] for r in group_rows if r["best_mixed_F1"] is not None]
                group_max = [r["max_mSCF1"] for r in group_rows if r.get("max_mSCF1") is not None]
                group_means[group] = {
                    "mean_mSCF1": float(np.mean(group_mscf1)) if group_mscf1 else None,
                    "mean_max_mSCF1": float(np.mean(group_max)) if group_max else None,
                    "mean_best_mixed_F1": float(np.mean(group_best)) if group_best else None,
                }
            s3pl_baseline = load_s3pl_baseline(Path("data/S3PL_Evaluation_Weigand/evaluation_data/mSCF1_method_table.csv"))
            baseline_all = None
            if s3pl_baseline:
                baseline_vals = [v for v in s3pl_baseline.values() if v is not None]
                if baseline_vals:
                    baseline_all = float(np.mean(baseline_vals))
            delta_vs_s3pl = {}
            if mean_mscf1 is not None and baseline_all is not None:
                delta_vs_s3pl["all"] = mean_mscf1 - baseline_all
            for group, stats in group_means.items():
                baseline = s3pl_baseline.get(group) if s3pl_baseline else None
                g_mscf1 = stats.get("mean_mSCF1")
                if g_mscf1 is not None and baseline is not None:
                    delta_vs_s3pl[group] = g_mscf1 - baseline
            report = {
                "run_dir": str(run_dir),
                "checkpoint": args.checkpoint,
                "device": str(device),
                "informative_classes": informative_split,
                "non_informative_classes": non_informative_split,
                "thresholds": args.thresholds,
                "top_peaks": args.top_peaks,
                "results": metrics_summary,
                "mean_mSCF1": mean_mscf1,
                "mean_max_mSCF1": mean_max,
                "mean_best_mixed_F1": mean_best,
                "group_means": group_means,
                "s3pl_baseline_mSCF1": s3pl_baseline,
                "delta_vs_s3pl_mSCF1": delta_vs_s3pl,
            }
            report_path = split_dir / "metrics.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            if args.sweep_informative:
                mscf1_values = [summary.get("mSCF1") for summary in metrics_summary.values() if summary.get("mSCF1") is not None]
                mean_mscf1 = float(np.mean(mscf1_values)) if mscf1_values else None
                max_mscf1_values = [summary.get("max_mSCF1") for summary in metrics_summary.values() if summary.get("max_mSCF1") is not None]
                mean_max = float(np.mean(max_mscf1_values)) if max_mscf1_values else None
                best_mixed = []
                for summary in metrics_summary.values():
                    best = max(thres["mixed_metrics"]["F1"] for thres in summary["threshold_metrics"].values())
                    best_mixed.append(best)
                mean_best = float(np.mean(best_mixed)) if best_mixed else None
                sweep_entries.append({
                    "informative": informative_split,
                    "non_informative": non_informative_split,
                    "mean_mSCF1": mean_mscf1,
                    "mean_max_mSCF1": mean_max,
                    "mean_best_mixed_F1": mean_best,
                    "metrics_path": str(report_path),
                })

        if args.sweep_informative:
            summary_path = out_dir / "sweep_summary.csv"
            with open(summary_path, "w") as f:
                f.write("informative_classes,non_informative_classes,mean_mSCF1,mean_max_mSCF1,mean_best_mixed_F1,metrics_path\n")
                for entry in sorted(sweep_entries, key=lambda e: (e["mean_best_mixed_F1"] or -1), reverse=True):
                    inf_str = "+".join(entry["informative"]) if entry["informative"] else "none"
                    non_str = "+".join(entry["non_informative"]) if entry["non_informative"] else "none"
                    mscf1_str = "" if entry["mean_mSCF1"] is None else f"{entry['mean_mSCF1']:.6f}"
                    max_mscf1_str = "" if entry.get("mean_max_mSCF1") is None else f"{entry['mean_max_mSCF1']:.6f}"
                    mean_best_str = "" if entry["mean_best_mixed_F1"] is None else f"{entry['mean_best_mixed_F1']:.6f}"
                    f.write(f"{inf_str},{non_str},{mscf1_str},{max_mscf1_str},{mean_best_str},{entry['metrics_path']}\n")
            if sweep_entries:
                best = max(sweep_entries, key=lambda e: e["mean_best_mixed_F1"] or -1)
                best_info = "+".join(best["informative"]) or "none"
                best_val = f"{best['mean_best_mixed_F1']:.4f}" if best["mean_best_mixed_F1"] is not None else "n/a"
                print(f"Sweep complete. Best split by mean mixed F1: {best_info} -> {best_val}")
            print(f"Sweep summary saved to {summary_path}")
        else:
            print(f"S3PL evaluation written to {out_dir}")
    finally:
        for cube in cubes.values():
            if hasattr(cube, "close"):
                cube.close()


if __name__ == "__main__":
    main()

"""
Run example:

python scripts/evaluate_s3pl_peak_quality.py \
  --run-dir data/models/20251014-142231_resnet50_full_hard_dpr-0.0_wd-0.0_aug-default_cosine_pretrained-1_cuda \
  --data-root data/S3PL_Evaluation_Weigand \
  --groups gbm renal_cell_carcinoma \
  --informative-classes structured,weak_structured,localized,negative \
  --top-peaks 500 \
  --device cuda

python scripts/evaluate_s3pl_peak_quality.py \
  --run-dir data/models/20251014-142231_resnet50_full_hard_dpr-0.0_wd-0.0_aug-default_cosine_pretrained-1_cuda \
  --data-root data/S3PL_Evaluation_Weigand \
  --groups gbm \
  --top-peaks 500 \
  --sweep-informative

python msianalyzer/evaluate/evaluate_s3pl_peak_quality.py \
  --run-dir data/models/.../your_regression_run \
  --structure-threshold 0.7 \
  --informativeness-threshold 0.5 \
  --data-root <S3PL_data_root>

"""
def model_has_weightdrop(module: nn.Module) -> bool:
    return any(isinstance(child, WeightDrop) for child in module.modules())
