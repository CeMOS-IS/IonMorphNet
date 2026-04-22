#!/usr/bin/env python3
"""
Classify each ion image of an imzML+ibd file into one of the six morphology classes.
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
    parser = argparse.ArgumentParser(description="Classify the morphology of each ion image of an imzML and ibd file.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory (or name under data/models).")
    parser.add_argument("--imzml-folderpath", type=Path, required=True, help="Absolute filepath to the folder where imzML files to be analyzed are located (e.g.: /home/user/experiment/).")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt", help="Checkpoint filename inside run directory.")
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
                        help="After evaluation, generate SOTA comparison plot using dataset_metrics.csv.")
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


def collect_imzml_files(root: Path) -> List[Path]:
    imzml_files: List[Path] = []
    imzml_files.extend(sorted(root.glob("*.imzML")))
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
                                training_type: str,
                                score_mode: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Optional[np.ndarray], np.ndarray]:
    probs = outputs["probs"]
    mz_values = outputs["mz"]

    pred_indices = probs.argmax(axis=1)
    pred_labels = [classes[i] for i in pred_indices]
    return pred_labels, pred_indices, mz_values


def evaluate_dataset(
    dataname: str,
    cube: SpectralCube,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    classes: Sequence[str],
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

    morphology_predictions, pred_indices, mz_values = compute_scores_from_outputs(
        outputs,
        classes,
        training_type,
        score_mode,
    )

    return morphology_predictions, mz_values


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
            cube_specs = collect_cube_specs(msianalyzer_dir / args.imzml_folderpath)
        except Exception as e:
            print(f"⚠️  Falling back to imzML files (could not load spectral cubes): {e}")
            use_cubes = False
            if cube_mode == "auto":
                cube_mode = "dense"
    if not use_cubes:
        imzml_files = collect_imzml_files(msianalyzer_dir / args.imzml_folderpath)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_dir or (run_dir / "morphology_predictions" / timestamp)
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
            morphology_predictions, mz_values = evaluate_dataset(
                dataname=dataname,
                cube=cube,
                model=model,
                batch_size=args.batch_size,
                device=device,
                classes=classes,
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

            prediction_csv = peaks_dir / f"{dataname}_all_predictions.csv"
            df = pd.DataFrame({"m/z": mz_values, "morphology class": morphology_predictions})
            df.to_csv(prediction_csv, index=False)

            print(f"Morphology predictions for {dataname} written to {prediction_csv}")

    finally:
        for cube in cubes.values():
            if hasattr(cube, "close"):
                cube.close()


if __name__ == "__main__":
    main()

def model_has_weightdrop(module: nn.Module) -> bool:
    return any(isinstance(child, WeightDrop) for child in module.modules())
