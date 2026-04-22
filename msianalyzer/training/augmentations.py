import inspect
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import albumentations as A

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cv2 = None

AUGMENTATION_SPECS: Dict[str, Dict[str, Any]] = {
    "default": {"label": "Baseline", "color": "#000000", "strength": 0.0},
    "advanced": {"label": "Advanced", "color": "#1f77b4", "strength": 0.4},
    "auto_strength": {"label": "Auto (strength-weighted)", "color": "#444444", "strength": 0.2},
    "grid_mask": {"label": "Grid Masking", "color": "#D45500", "strength": 0.6},
    "random_mask": {"label": "Random Masking", "color": "#FF4500", "strength": 0.6},
    "cutmix": {"label": "CutMix-like", "color": "#1E88E5", "strength": 0.8}, #copy
    "cutout": {"label": "Cutout", "color": "#FFD700", "strength": 0.5},
    "jitter": {"label": "Jittering", "color": "#2CA02C", "strength": 0.2},
    "scaling": {"label": "Scaling", "color": "#FF69B4", "strength": 0.3},
    "max_mask": {"label": "MAX Masking", "color": "#8A2BE2", "strength": 0.7},
    "mz_shift": {"label": "m/z shift", "color": "#17BECF", "strength": 0.3},
    "gaussian_blur": {"label": "Gauss Blur", "color": "#8B4513", "strength": 0.3},
    "ldm": {"label": "LDM-inspired", "color": "#A93226", "strength": 0.9},
    "elastic_transform": {"label": "Elastic Transform", "color": "#FF1493", "strength": 0.6},
    "grid_distortion": {"label": "Grid Distortion", "color": "#00CED1", "strength": 0.6},
    "thin_plate_spline": {"label": "Thin Plate Spline", "color": "#32CD32", "strength": 0.5},
    "smooth_displacement": {"label": "Smooth Displacement", "color": "#6495ED", "strength": 0.7},
    "histogram_grf": {"label": "Histogram GRF", "color": "#7B68EE", "strength": 0.5},
    "local_permutation": {"label": "Local Permutation", "color": "#FF8C00", "strength": 0.6},
    "fourier_surrogate": {"label": "Fourier Surrogate", "color": "#20B2AA", "strength": 0.7},
    "smod": {"label": "SMOD PCA", "color": "#FF69A0", "strength": 0.8},
}


AUGMENTATION_ALIASES: Dict[str, str] = {
    "baseline": "default",
    "none": "default",
    "no_aug": "default",
    "auto": "auto_strength",
    "gaus_blur": "gaussian_blur",
    "gauss_blur": "gaussian_blur",
    "ldm_aug": "ldm",
    "elastic": "elastic_transform",
    "grid_distort": "grid_distortion",
    "tps": "thin_plate_spline",
    "smooth_disp": "smooth_displacement",
    "hist_grf": "histogram_grf",
    "grf_hist": "histogram_grf",
    "grf": "histogram_grf",
    "local_perm": "local_permutation",
    "fourier_phase": "fourier_surrogate",
    "phase_surrogate": "fourier_surrogate",
    "smod_aug": "smod",
}

@contextlib.contextmanager
def _seed_context(seed: Optional[int], device: torch.device) -> None:
    if seed is None:
        yield
        return
    device_names: List[str] = []
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        device_names = [f"cuda:{idx}"]
    with torch.random.fork_rng(devices=device_names):
        torch.manual_seed(int(seed))
        yield


def resolve_augmentation_mode(mode: Optional[str]) -> str:
    key = (mode or "default").strip().lower().replace(" ", "_")
    key = AUGMENTATION_ALIASES.get(key, key)
    if key not in AUGMENTATION_SPECS:
        raise ValueError(f"Unknown augmentation mode '{mode}'. Available modes: {sorted(AUGMENTATION_SPECS)}")
    return key


def augmentation_display_label(mode: str) -> str:
    spec = AUGMENTATION_SPECS.get(mode, {})
    return spec.get("label", mode.replace("_", " ").title())


def augmentation_color(mode: str) -> str:
    spec = AUGMENTATION_SPECS.get(mode, {})
    return spec.get("color", "#333333")


def augmentation_strength(mode: str) -> float:
    """Heuristic strength in [0,1]; higher means heavier / more destructive."""
    spec = AUGMENTATION_SPECS.get(mode, {})
    return float(spec.get("strength", 0.5))


def _sample_augmentation_by_strength(
    candidates: Optional[Iterable[str]] = None,
    original_probability: float = 0.5,
    bias: float = 1.5,
) -> str:
    """
    Sample an augmentation mode, preferring lighter transforms.
    - candidates: optional iterable of modes to consider (resolve_augmentation_mode is applied).
    - original_probability: extra weight for leaving the image unchanged (baseline geom only).
    - bias: exponent applied to (1 - strength); higher bias emphasizes light augs.
    """
    if candidates:
        pool = [resolve_augmentation_mode(c) for c in candidates]
    else:
        pool = [
            k for k in AUGMENTATION_SPECS.keys()
            if k not in {"default", "auto_strength"}
        ]
    pool = [p for p in pool if p not in {"default", "auto_strength"}]
    if not pool:
        pool = ["default"]

    weights: List[float] = []
    for mode in pool:
        s = augmentation_strength(mode)
        # Invert strength so lighter augs are sampled more; bias sharpens the preference.
        w = max((1.0 - s) ** bias, 1e-4)
        weights.append(w)

    # Add weight for leaving the sample mostly untouched (only base geom).
    pool.append("default")
    weights.append(max(original_probability, 0.0))

    probs = torch.tensor(weights, dtype=torch.float32)
    probs = probs / probs.sum().clamp_min(1e-6)
    idx = torch.multinomial(probs, num_samples=1).item()
    return pool[idx]


_SMOD_MODELS: Dict[Tuple[int, int, Tuple[Any, ...]], "SMODDeformationModel"] = {}


def apply_train_augmentations(
    img: torch.Tensor,
    mode: str = "default",
    override_params: Optional[Dict[str, Any]] = None,
    keep_probability: float = 1.0,
) -> torch.Tensor:
    """Apply configured augmentations to a single image tensor [C,H,W]."""
    canonical = resolve_augmentation_mode(mode)
    seed = None
    if override_params:
        seed = override_params.get("seed")
    with _seed_context(seed, img.device):
        if canonical == "default":
            return img
        aug = _augment_base_geom(img)
        if canonical != "default" and keep_probability < 1.0:
            if torch.rand(1).item() > keep_probability:
                return aug
        if canonical == "auto_strength":
            params = override_params or {}
            sampled = _sample_augmentation_by_strength(
                candidates=params.get("candidates"),
                original_probability=float(params.get("original_probability", 0.5)),
                bias=float(params.get("bias", 1.5)),
            )
            if sampled == "default":
                return aug  # only base geom
            # Dispatch to the sampled mode using same params; keep_probability already applied.
            return apply_train_augmentations(
                aug,
                mode=sampled,
                override_params=params,
                keep_probability=1.0,
            )
        if canonical == "advanced":
            aug = _random_resized_crop_tensor(aug, scale_range=(0.85, 1.0))
            aug = _gamma_jitter(aug, gamma_range=(0.9, 1.1))
            aug = _gaussian_blur(aug, sigma_range=(0.0, 0.8))
            if torch.rand(1) < 0.3:
                aug = _tiny_elastic_deform(aug, alpha=1.0, sigma=4.0)
            return aug
        if canonical == "grid_mask":
            return _apply_grid_mask_tensor(aug)
        if canonical == "random_mask":
            return _apply_random_rect_mask(aug, fill_value=0.0)
        if canonical == "cutmix":
            return _apply_cutmix_like(aug)
        if canonical == "cutout":
            return _apply_random_rect_mask(aug, fill_value=0.0, size_range=(0.1, 0.3))
        if canonical == "jitter":
            return _apply_gaussian_noise(aug, sigma_range=(0.01, 0.08))
        if canonical == "scaling":
            return _random_resized_crop_tensor(aug, scale_range=(0.6, 1.1))
        if canonical == "max_mask":
            return _apply_top_intensity_mask(aug, quantile=0.90)
        if canonical == "mz_shift":
            return _apply_shift(aug, max_shift=5)
        if canonical == "gaussian_blur":
            return _gaussian_blur(aug, sigma_range=(0.6, 1.6))
        if canonical == "ldm":
            return _apply_ldm_like(aug)
        if canonical == "elastic_transform":
            return _apply_elastic_transform(aug)
        if canonical == "grid_distortion":
            return _apply_grid_distortion(aug)
        if canonical == "thin_plate_spline":
            return _apply_thin_plate_spline(aug)
        if canonical == "smooth_displacement":
            params = override_params or {}
            return _apply_smooth_displacement(
                aug,
                sigma=params.get("sigma"),
                amplitude=params.get("amplitude"),
                seed=params.get("seed"),
            )
        if canonical == "histogram_grf":
            params = override_params or {}
            return _apply_histogram_grf(
                aug,
                seed=params.get("seed"),
                magnitude_blur_sigma=params.get("magnitude_blur_sigma"),
                spectral_exponent=params.get("spectral_exponent"),
            )
        if canonical == "local_permutation":
            params = override_params or {}
            return _apply_local_permutation(
                aug,
                radius=params.get("radius"),
                swaps=params.get("swaps"),
                patch_size=params.get("patch_size"),
                seed=params.get("seed"),
            )
        if canonical == "fourier_surrogate":
            params = override_params or {}
            return _apply_fourier_surrogate(
                aug,
                phase_mix=params.get("phase_mix"),
                magnitude_blur_sigma=params.get("magnitude_blur_sigma"),
                iaaft_iterations=params.get("iaaft_iterations"),
                seed=params.get("seed"),
            )
        if canonical == "smod":
            params = _normalize_smod_config(override_params or {})
            if not params["enabled"]:
                return aug
            model = _get_smod_model(
                aug.shape[-2],
                aug.shape[-1],
                params,
                device=aug.device,
                dtype=aug.dtype,
            )
            warped, _ = apply_smod(
                aug,
                None,
                model,
                p=params["p"],
                seed=params.get("seed"),
                scale_override=params.get("scale_override"),
                smooth_sigma=params["smooth_sigma"],
            )
            return warped
        return aug


def _random_resized_crop_tensor(img: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.0)) -> torch.Tensor:
    """Random cropped zoom that mimics torchvision's RandomResizedCrop but keeps aspect."""
    c, h, w = img.shape
    min_scale, max_scale = scale_range
    if not (0.0 < min_scale <= max_scale <= 1.0):
        return img
    scale = float(torch.empty(1).uniform_(min_scale, max_scale))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if new_h >= h and new_w >= w:
        return img
    top = torch.randint(0, h - new_h + 1, (1,)).item()
    left = torch.randint(0, w - new_w + 1, (1,)).item()
    crop = img[:, top:top + new_h, left:left + new_w]
    return F.interpolate(crop.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def _gamma_jitter(img: torch.Tensor, gamma_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    lo, hi = gamma_range
    if lo <= 0 or hi <= 0:
        return img
    gamma = float(torch.empty(1).uniform_(lo, hi))
    return img.clamp_min(1e-6).pow(gamma).clamp(0.0, 1.0)


def _gaussian_blur(img: torch.Tensor, sigma_range: Tuple[float, float] = (0.0, 0.8)) -> torch.Tensor:
    lo, hi = sigma_range
    if hi <= 0:
        return img
    sigma = float(torch.empty(1).uniform_(lo, hi))
    if sigma <= 0.0:
        return img
    kernel = max(3, int(2 * math.ceil(3 * sigma) + 1))
    if kernel % 2 == 0:
        kernel += 1
    return TF.gaussian_blur(img, kernel_size=kernel, sigma=sigma)


def _tiny_elastic_deform(img: torch.Tensor, alpha: float = 1.0, sigma: float = 4.0) -> torch.Tensor:
    if alpha <= 0 or sigma <= 0:
        return img
    device, dtype = img.device, img.dtype
    c, h, w = img.shape
    coords_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    coords_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij") if hasattr(torch, "meshgrid") else torch.meshgrid(coords_y, coords_x)

    disp_x = TF.gaussian_blur(torch.randn(1, h, w, device=device, dtype=dtype), kernel_size=max(3, int(2 * math.ceil(3 * sigma) + 1)), sigma=sigma)
    disp_y = TF.gaussian_blur(torch.randn(1, h, w, device=device, dtype=dtype), kernel_size=max(3, int(2 * math.ceil(3 * sigma) + 1)), sigma=sigma)

    disp_x = disp_x[0] * (alpha / max(w, 1))
    disp_y = disp_y[0] * (alpha / max(h, 1))
    grid = torch.stack((grid_x + disp_x, grid_y + disp_y), dim=-1)
    warped = F.grid_sample(
        img.unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return warped.squeeze(0)


def _augment_base_geom(img: torch.Tensor) -> torch.Tensor:
    if torch.rand(1) < 0.5:
        img = torch.flip(img, dims=[-1])
    if torch.rand(1) < 0.5:
        img = torch.flip(img, dims=[-2])
    k = torch.randint(0, 4, (1,), device=img.device).item()
    img = torch.rot90(img, k, dims=[-2, -1])
    return img


def _apply_grid_mask_tensor(img: torch.Tensor, keep_ratio: float = 0.6,
                            cell_range: Tuple[int, int] = (16, 48)) -> torch.Tensor:
    c, h, w = img.shape
    mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
    cell_h = torch.randint(cell_range[0], cell_range[1] + 1, (1,), device=img.device).item()
    cell_w = torch.randint(cell_range[0], cell_range[1] + 1, (1,), device=img.device).item()
    band_h = max(1, int(cell_h * (1.0 - keep_ratio)))
    band_w = max(1, int(cell_w * (1.0 - keep_ratio)))
    offset_h = torch.randint(0, cell_h, (1,), device=img.device).item()
    offset_w = torch.randint(0, cell_w, (1,), device=img.device).item()
    for y in range(offset_h, h, cell_h):
        mask[y:y + band_h, :] = 0
    for x in range(offset_w, w, cell_w):
        mask[:, x:x + band_w] = 0
    return img * mask.unsqueeze(0)


def _apply_random_rect_mask(img: torch.Tensor, fill_value: Optional[float] = None,
                            size_range: Tuple[float, float] = (0.2, 0.5)) -> torch.Tensor:
    out = img.clone()
    _, h, w = out.shape
    rh = int(h * float(torch.empty(1).uniform_(size_range[0], size_range[1])))
    rw = int(w * float(torch.empty(1).uniform_(size_range[0], size_range[1])))
    rh = max(1, min(h, rh))
    rw = max(1, min(w, rw))
    top = torch.randint(0, max(1, h - rh + 1), (1,), device=img.device).item()
    left = torch.randint(0, max(1, w - rw + 1), (1,), device=img.device).item()
    if fill_value is None:
        fill_value = float(out.mean())
    out[:, top:top + rh, left:left + rw] = fill_value
    return out


def _apply_cutmix_like(img: torch.Tensor) -> torch.Tensor:
    base = img.clone()
    other = torch.rot90(base, 1, dims=[-2, -1])
    if other.shape[-2:] != base.shape[-2:]:
        other = F.interpolate(
            other.unsqueeze(0),
            size=base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    lam = float(torch.empty(1).uniform_(0.3, 0.7))
    _, h, w = base.shape
    rh = int(h * float(torch.empty(1).uniform_(0.3, 0.6)))
    rw = int(w * float(torch.empty(1).uniform_(0.3, 0.6)))
    rh = max(1, min(h, rh))
    rw = max(1, min(w, rw))
    top = torch.randint(0, max(1, h - rh + 1), (1,), device=img.device).item()
    left = torch.randint(0, max(1, w - rw + 1), (1,), device=img.device).item()
    patch = lam * base[:, top:top + rh, left:left + rw] + (1.0 - lam) * other[:, top:top + rh, left:left + rw]
    base[:, top:top + rh, left:left + rw] = patch
    return base


def _apply_gaussian_noise(img: torch.Tensor, sigma_range: Tuple[float, float] = (0.01, 0.05)) -> torch.Tensor:
    sigma = float(torch.empty(1).uniform_(sigma_range[0], sigma_range[1]))
    noise = torch.randn_like(img) * sigma
    return (img + noise).clamp_(0.0, 1.0)


def _apply_top_intensity_mask(img: torch.Tensor, quantile: float = 0.95) -> torch.Tensor:
    out = img.clone()
    thresh = torch.quantile(out.flatten(), quantile)
    out = torch.where(out >= thresh, out.new_zeros(()).expand_as(out), out)
    return out


def _apply_shift(img: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    if max_shift <= 0:
        return img
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=img.device).item())
    if shift == 0:
        return img
    return torch.roll(img, shifts=shift, dims=-1)


def _apply_ldm_like(img: torch.Tensor) -> torch.Tensor:
    out = _apply_gaussian_noise(img, sigma_range=(0.02, 0.06))
    out = _random_resized_crop_tensor(out, scale_range=(0.85, 1.0))
    out = _gaussian_blur(out, sigma_range=(0.2, 0.9))
    return out


def _apply_elastic_transform(img: torch.Tensor) -> torch.Tensor:
    transform = _instantiate_albumentations_transform(
        A.ElasticTransform,
        alpha=40.0,
        sigma=6.0,
        alpha_affine=10.0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        approximate=True,
        p=1.0,
    )
    return _apply_albumentations_transform(img, transform)


def _apply_grid_distortion(img: torch.Tensor) -> torch.Tensor:
    transform = _instantiate_albumentations_transform(
        A.GridDistortion,
        num_steps=5,
        distort_limit=0.3,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    )
    return _apply_albumentations_transform(img, transform)


def _apply_thin_plate_spline(img: torch.Tensor) -> torch.Tensor:
    transform = _instantiate_albumentations_transform(
        A.ThinPlateSpline,
        scale_limit=0.1,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    )
    return _apply_albumentations_transform(img, transform)


def _apply_smooth_displacement(
    img: torch.Tensor,
    sigma: Optional[float] = None,
    amplitude: Optional[float] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    device, dtype = img.device, img.dtype
    _, h, w = img.shape
    if h < 2 or w < 2:
        return img

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    if sigma is None:
        rng = torch.rand(1, generator=generator)
        sigma = float((rng * (8.0 - 2.0) + 2.0).item())
    if amplitude is None:
        rng = torch.rand(1, generator=generator)
        amplitude = float((rng * (6.0 - 1.0) + 1.0).item())

    disp_x = _generate_smooth_field(h, w, sigma, device, dtype, generator=generator) * amplitude
    disp_y = _generate_smooth_field(h, w, sigma, device, dtype, generator=generator) * amplitude

    coords_y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    coords_x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    base_grid_y, base_grid_x = torch.meshgrid(
        coords_y,
        coords_x,
        indexing="ij",
    ) if hasattr(torch, "meshgrid") else torch.meshgrid(coords_y, coords_x)

    disp_x_norm = (disp_x / max(w - 1, 1)) * 2.0
    disp_y_norm = (disp_y / max(h - 1, 1)) * 2.0

    grid = torch.stack((base_grid_x + disp_x_norm, base_grid_y + disp_y_norm), dim=-1)
    warped = F.grid_sample(
        img.unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )
    return warped.squeeze(0).clamp_(0.0, 1.0)


def _apply_albumentations_transform(img: torch.Tensor, transform: A.BasicTransform) -> torch.Tensor:
    device = img.device
    dtype = img.dtype
    np_img = (
        img.detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    augmented = transform(image=np_img)["image"]
    if augmented.ndim == 2:
        augmented = augmented[..., np.newaxis]
    augmented = np.asarray(augmented, dtype=np.float32)
    tensor = torch.from_numpy(augmented).permute(2, 0, 1)
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor.clamp(0.0, 1.0)


def _instantiate_albumentations_transform(transform_cls, **kwargs):
    sig = inspect.signature(transform_cls.__init__)
    allowed = {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {key: value for key, value in kwargs.items() if key in allowed}
    return transform_cls(**filtered)


def _apply_local_permutation(
    img: torch.Tensor,
    radius: Optional[float] = None,
    swaps: Optional[int] = None,
    patch_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Applies local permutation to an image.

    Args:
        img (torch.Tensor): Input image of shape (C, H, W).
        radius (Optional[float]): Radius of the local permutation. Defaults to 0.05 * min(H, W).
        swaps (Optional[int]): Number of swaps to perform. Defaults to 0.02 * H * W.
        patch_size (Optional[int]): Size of the patch to swap. Defaults to 1.
        seed (Optional[int]): Seed for the random number generator. Defaults to None.

    Returns:
        torch.Tensor: Output image of shape (C, H, W).
    """
    out = img.clone()
    _, h, w = out.shape
    if h == 0 or w == 0:
        return out

    radius = int(round(radius if radius is not None else max(1.0, min(h, w) * 0.05)))
    radius = max(1, radius)
    default_swaps = int(0.02 * h * w)
    swaps = int(swaps) if swaps is not None else default_swaps
    swaps = max(1, swaps)
    patch_size = int(patch_size) if patch_size is not None else 1
    patch_size = max(1, min(patch_size, min(h, w)))

    device = out.device
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    def rand_int(low: int, high: int) -> int:
        if generator is not None:
            return int(torch.randint(low, high, (1,), generator=generator).item())
        return int(torch.randint(low, high, (1,)).item())

    def _bounds(center: int, size: int, limit: int) -> Tuple[int, int]:
        size = min(size, limit)
        if size <= 0:
            return center, center
        start = center - size // 2
        end = start + size
        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit
        start = max(0, min(start, limit - size))
        end = start + size
        return start, end

    radius_sq = radius * radius
    for _ in range(swaps):
        y = rand_int(0, h)
        x = rand_int(0, w)

        target_found = False
        for _ in range(8):
            dy = rand_int(0, 2 * radius + 1) - radius
            dx = rand_int(0, 2 * radius + 1) - radius
            if dy * dy + dx * dx > radius_sq:
                continue
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w and not (yy == y and xx == x):
                target_found = True
                break
        if not target_found:
            continue

        if patch_size <= 1:
            tmp = out[:, y, x].clone()
            out[:, y, x] = out[:, yy, xx]
            out[:, yy, xx] = tmp
            continue

        y1, y2 = _bounds(y, patch_size, h)
        x1, x2 = _bounds(x, patch_size, w)
        yy1, yy2 = _bounds(yy, patch_size, h)
        xx1, xx2 = _bounds(xx, patch_size, w)
        if (y2 - y1) != (yy2 - yy1) or (x2 - x1) != (xx2 - xx1):
            continue
        if y1 == yy1 and x1 == xx1:
            continue

        patch_a = out[:, y1:y2, x1:x2].clone()
        patch_b = out[:, yy1:yy2, xx1:xx2].clone()
        out[:, y1:y2, x1:x2] = patch_b
        out[:, yy1:yy2, xx1:xx2] = patch_a

    return out


def _apply_histogram_grf(
    img: torch.Tensor,
    seed: Optional[int] = None,
    magnitude_blur_sigma: Optional[float] = None,
    spectral_exponent: Optional[float] = None,
) -> torch.Tensor:
    """
    Applies a histogram-based gradient refinement field (GRF) to a given image tensor.

    Args:
        img: The input image tensor [C,H,W].
        seed: Optional random seed for the histogram matching process.
        magnitude_blur_sigma: Optional standard deviation for the Gaussian blur kernel
            to be applied to the magnitude of the frequency domain image.
        spectral_exponent: Optional exponent for the spectral gradient refinement.

    Returns:
        A tensor with the same shape as `img`, containing the histogram-based GRF field.
    """
    out = torch.empty_like(img)
    device, dtype = img.device, img.dtype
    h, w = img.shape[-2], img.shape[-1]
    if h == 0 or w == 0:
        return img.clone()

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    two_pi = 2.0 * math.pi
    for c in range(img.shape[0]):
        base = img[c]
        freq = torch.fft.fft2(base)
        magnitude = freq.abs()
        if magnitude_blur_sigma is not None and magnitude_blur_sigma > 0.0:
            kernel = max(3, int(2 * math.ceil(3 * magnitude_blur_sigma) + 1))
            if kernel % 2 == 0:
                kernel += 1
            mag4d = magnitude.unsqueeze(0).unsqueeze(0)
            magnitude = TF.gaussian_blur(
                mag4d,
                kernel_size=kernel,
                sigma=float(magnitude_blur_sigma),
            ).squeeze(0).squeeze(0)
        if spectral_exponent is not None and spectral_exponent != 0.0:
            fy = torch.fft.fftfreq(h, device=device, dtype=dtype).reshape(-1, 1)
            fx = torch.fft.fftfreq(w, device=device, dtype=dtype).reshape(1, -1)
            radius = torch.sqrt(fx * fx + fy * fy) + 1e-6
            magnitude = magnitude * torch.pow(radius, float(spectral_exponent))
        magnitude = magnitude.clamp_min(1e-6)

        phase = torch.rand((h, w), generator=generator, dtype=dtype, device=device)
        phase = phase.to(device=device) * two_pi
        complex_freq = magnitude * torch.complex(torch.cos(phase), torch.sin(phase))
        complex_freq = _enforce_hermitian(complex_freq)
        dc_val = freq[0, 0]
        complex_freq[0, 0] = torch.complex(dc_val.real, torch.zeros_like(dc_val.real))
        if h % 2 == 0:
            nyquist_row = complex_freq[h // 2, 0]
            complex_freq[h // 2, 0] = torch.complex(nyquist_row.real, torch.zeros_like(nyquist_row.real))
        if w % 2 == 0:
            nyquist_col = complex_freq[0, w // 2]
            complex_freq[0, w // 2] = torch.complex(nyquist_col.real, torch.zeros_like(nyquist_col.real))
            if h % 2 == 0:
                corner = complex_freq[h // 2, w // 2]
                complex_freq[h // 2, w // 2] = torch.complex(corner.real, torch.zeros_like(corner.real))

        random_field = torch.fft.ifft2(complex_freq).real
        random_field = (random_field - random_field.mean()) / random_field.std().clamp_min(1e-6)
        matched = _histogram_match(random_field, base)
        out[c] = matched
    return out.clamp(0.0, 1.0)


def _apply_fourier_surrogate(
    img: torch.Tensor,
    phase_mix: Optional[float] = None,
    magnitude_blur_sigma: Optional[float] = None,
    iaaft_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Applies a Fourier surrogate method to a given image tensor.

    Parameters:
        img (torch.Tensor): Input image tensor [C,H,W].
        phase_mix (Optional[float], default=None): Phase mixing parameter. If 0.0, the original phase is used. If 1.0, a random phase is used. If between 0.0 and 1.0, a linear combination of the two is used.
        magnitude_blur_sigma (Optional[float], default=None): Standard deviation of the Gaussian kernel used for blurring the magnitude.
        iaaft_iterations (Optional[int], default=None): Number of iterations of the iterative amplitude adjustment Fourier transform (IAAFT) algorithm.
        seed (Optional[int], default=None): Random seed used for generating the random phase and for the IAAFT algorithm.

    Returns:
        torch.Tensor: Output image tensor [C,H,W] with the same shape as the input image tensor.
    """
    out = torch.empty_like(img)
    h, w = img.shape[-2], img.shape[-1]
    if h == 0 or w == 0:
        return img.clone()

    phase_mix = float(phase_mix) if phase_mix is not None else 0.5
    phase_mix = float(max(0.0, min(1.0, phase_mix)))
    magnitude_blur_sigma = float(magnitude_blur_sigma) if magnitude_blur_sigma is not None else 0.0
    iaaft_iterations = int(iaaft_iterations) if iaaft_iterations is not None else 0
    iaaft_iterations = max(0, iaaft_iterations)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    for c in range(img.shape[0]):
        base = img[c]
        spectrum = torch.fft.fft2(base)
        amplitude = spectrum.abs()
        if magnitude_blur_sigma > 0.0:
            kernel = max(3, int(2 * math.ceil(3 * magnitude_blur_sigma) + 1))
            if kernel % 2 == 0:
                kernel += 1
            amp4d = amplitude.unsqueeze(0).unsqueeze(0)
            amplitude = TF.gaussian_blur(amp4d, kernel_size=kernel, sigma=magnitude_blur_sigma).squeeze(0).squeeze(0)
            amplitude = amplitude.clamp_min(1e-6)

        phase_orig = torch.angle(spectrum)
        rand_phase = torch.rand((h, w), generator=generator, dtype=phase_orig.dtype) * (2.0 * math.pi)

        if phase_mix <= 0.0:
            phase_combined = phase_orig
        elif phase_mix >= 1.0:
            phase_combined = rand_phase
        else:
            cos_comb = (1.0 - phase_mix) * torch.cos(phase_orig) + phase_mix * torch.cos(rand_phase)
            sin_comb = (1.0 - phase_mix) * torch.sin(phase_orig) + phase_mix * torch.sin(rand_phase)
            phase_combined = torch.atan2(sin_comb, cos_comb)

        surrogate_spec = amplitude * torch.exp(1j * phase_combined)
        surrogate_spec = _enforce_hermitian(surrogate_spec)
        surrogate_spec[0, 0] = spectrum[0, 0]
        field = torch.fft.ifft2(surrogate_spec).real

        if iaaft_iterations > 0:
            target = base
            for _ in range(iaaft_iterations):
                field = _histogram_match(field, target)
                field_spec = torch.fft.fft2(field)
                new_phase = torch.angle(field_spec)
                surrogate_spec = amplitude * torch.exp(1j * new_phase)
                surrogate_spec = _enforce_hermitian(surrogate_spec)
                surrogate_spec[0, 0] = spectrum[0, 0]
                field = torch.fft.ifft2(surrogate_spec).real
            field = _histogram_match(field, target)
        else:
            field = _histogram_match(field, base)

        out[c] = field
    return out.clamp(0.0, 1.0)


def _enforce_hermitian(spectrum: torch.Tensor) -> torch.Tensor:
    spectrum = spectrum.clone()
    h, w = spectrum.shape
    for y in range(h):
        for x in range(w):
            y_sym = (-y) % h
            x_sym = (-x) % w
            if y > y_sym or (y == y_sym and x > x_sym):
                spectrum[y_sym, x_sym] = spectrum[y, x].conj()
    dc = spectrum[0, 0]
    spectrum[0, 0] = torch.complex(dc.real, torch.zeros_like(dc.real))
    if h % 2 == 0:
        nyquist_row = spectrum[h // 2, 0]
        spectrum[h // 2, 0] = torch.complex(nyquist_row.real, torch.zeros_like(nyquist_row.real))
    if w % 2 == 0:
        nyquist_col = spectrum[0, w // 2]
        spectrum[0, w // 2] = torch.complex(nyquist_col.real, torch.zeros_like(nyquist_col.real))
        if h % 2 == 0:
            corner = spectrum[h // 2, w // 2]
            spectrum[h // 2, w // 2] = torch.complex(corner.real, torch.zeros_like(corner.real))
    return spectrum


class SMODDeformationModel:
    """Principal-component deformation model for SMOD augmentations."""

    def __init__(
        self,
        num_modes: int = 16,
        scale: float = 1.0,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_svs: bool = False,
        svs_steps: int = 6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_modes = max(1, int(num_modes))
        self.scale = float(scale)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_svs = enable_svs
        self.svs_steps = max(0, int(svs_steps))
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.mean: Optional[torch.Tensor] = None
        self.components: Optional[torch.Tensor] = None
        self.sdevs: Optional[torch.Tensor] = None
        self.spatial_shape: Optional[Tuple[int, int]] = None

    # ---------------------- fitting ----------------------
    def fit_from_fields(self, fields: torch.Tensor) -> None:
        fields = fields.to(device=self.device, dtype=self.dtype)
        if fields.dim() != 4 or fields.size(1) != 2:
            raise ValueError("Expected fields with shape [N, 2, H, W].")
        n, _, h, w = fields.shape
        if n < 1:
            raise ValueError("At least one deformation field is required.")
        flat = fields.reshape(n, -1)
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean
        max_modes = min(self.num_modes, min(centered.shape[0], centered.shape[1]) - 1)
        if max_modes <= 0:
            comps = torch.zeros((0, centered.shape[1]), device=self.device, dtype=self.dtype)
            svals = torch.zeros(0, device=self.device, dtype=self.dtype)
        else:
            _, svals, v = torch.pca_lowrank(centered, q=max_modes)
            comps = v[:, :max_modes].T
        if comps.size(0) < self.num_modes:
            pad = self.num_modes - comps.size(0)
            comps = torch.cat(
                [comps, torch.zeros(pad, centered.size(1), device=self.device, dtype=self.dtype)],
                dim=0,
            )
            svals = torch.cat([svals, torch.zeros(pad, device=self.device, dtype=self.dtype)], dim=0)
        self.mean = mean.reshape(2, h, w)
        self.components = comps.reshape(self.num_modes, 2, h, w)
        denom = math.sqrt(max(n - 1, 1))
        self.sdevs = svals / denom
        self.spatial_shape = (h, w)

    def fit_from_dir(self, path: Union[str, Path]) -> None:
        directory = Path(path)
        if not directory.exists():
            raise FileNotFoundError(f"SMOD fields directory '{directory}' not found.")
        fields: List[torch.Tensor] = []
        for file in sorted(directory.iterdir()):
            if file.suffix.lower() not in {".pt", ".npy"}:
                continue
            if file.suffix.lower() == ".pt":
                tensor = torch.load(file, map_location=self.device)
            else:
                tensor = torch.from_numpy(np.load(file)).to(self.device)
            if tensor.dim() == 3 and tensor.size(0) == 2:
                field = tensor
            elif tensor.dim() == 3 and tensor.size(-1) == 2:
                field = tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Field '{file}' must have shape [2,H,W] or [H,W,2].")
            field = field.to(self.dtype)
            fields.append(field)
        if not fields:
            raise ValueError(f"No deformation fields found in '{directory}'.")
        stacked = torch.stack(fields, dim=0)
        self.fit_from_fields(stacked)

    def try_load_cache(self, height: int, width: int) -> bool:
        if self.cache_dir is None:
            return False
        cache_path = self._cache_path(height, width)
        if not cache_path.exists():
            return False
        payload = torch.load(cache_path, map_location=self.device)
        if payload.get("num_modes") != self.num_modes:
            return False
        self.mean = payload["mean"].to(self.device, self.dtype)
        self.components = payload["components"].to(self.device, self.dtype)
        self.sdevs = payload["sdevs"].to(self.device, self.dtype)
        self.spatial_shape = tuple(payload["shape"])  # type: ignore[assignment]
        self.scale = float(payload.get("scale", self.scale))
        self.enable_svs = bool(payload.get("enable_svs", self.enable_svs))
        self.svs_steps = int(payload.get("svs_steps", self.svs_steps))
        return True

    def save_cache(self) -> None:
        if self.cache_dir is None or self.mean is None or self.components is None or self.sdevs is None:
            return
        height, width = self.spatial_shape or self.mean.shape[-2:]
        cache_path = self._cache_path(height, width)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": self.mean.detach().cpu(),
            "components": self.components.detach().cpu(),
            "sdevs": self.sdevs.detach().cpu(),
            "shape": (height, width),
            "num_modes": self.num_modes,
            "scale": self.scale,
            "enable_svs": self.enable_svs,
            "svs_steps": self.svs_steps,
        }
        torch.save(payload, cache_path)

    @torch.no_grad()
    def sample_field(
        self,
        seed: Optional[int] = None,
        scale_override: Optional[float] = None,
        smooth_sigma: float = 1.0,
    ) -> torch.Tensor:
        if self.mean is None or self.components is None or self.sdevs is None:
            raise RuntimeError("SMOD model must be fitted before sampling.")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        z = torch.randn(self.num_modes, generator=generator, device=self.device, dtype=self.dtype)
        coeffs = z * self.sdevs
        field = self.mean + (coeffs.view(self.num_modes, 1, 1, 1) * self.components).sum(dim=0)
        scale = float(self.scale if scale_override is None else scale_override)
        field = field * scale
        if smooth_sigma > 0.0:
            kernel = max(3, int(2 * math.ceil(3 * smooth_sigma) + 1))
            if kernel % 2 == 0:
                kernel += 1
            field = TF.gaussian_blur(field.unsqueeze(0), kernel_size=kernel, sigma=smooth_sigma).squeeze(0)
        field = torch.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
        if self.enable_svs and self.svs_steps > 0:
            field = _stationary_velocity_to_displacement(field, self.svs_steps)
        return field

    def _cache_path(self, height: int, width: int) -> Path:
        assert self.cache_dir is not None
        name = f"smod_pca_{height}x{width}_{self.num_modes}modes.pt"
        return self.cache_dir / name


def _create_identity_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def warp_img_and_mask(
    img: torch.Tensor,
    mask: Optional[torch.Tensor],
    field: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if img.dim() != 3:
        raise ValueError("Image must have shape [C,H,W].")
    _, height, width = img.shape
    device = img.device
    dtype = img.dtype
    grid = _create_identity_grid(height, width, device=device, dtype=dtype)
    disp = field.to(device=device, dtype=dtype)
    norm_y = (disp[0] * 2.0) / max(height - 1, 1)
    norm_x = (disp[1] * 2.0) / max(width - 1, 1)
    grid = grid.clone()
    grid[..., 0] = grid[..., 0] + norm_x
    grid[..., 1] = grid[..., 1] + norm_y
    grid = grid.unsqueeze(0)
    warped_img = F.grid_sample(
        img.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)
    warped_mask = None
    if mask is not None:
        if mask.dim() == 2:
            mask_in = mask.unsqueeze(0)
        elif mask.dim() == 3 and mask.size(0) == 1:
            mask_in = mask
        else:
            raise ValueError("Mask must have shape [H,W] or [1,H,W].")
        warped_mask = F.grid_sample(
            mask_in.unsqueeze(0).to(device=device, dtype=dtype),
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        warped_mask = warped_mask if mask.dim() == 3 else warped_mask.squeeze(0)
    return warped_img, warped_mask


def _warp_field(field: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    _, height, width = field.shape
    identity = _create_identity_grid(height, width, device=field.device, dtype=field.dtype)
    norm_y = (displacement[0] * 2.0) / max(height - 1, 1)
    norm_x = (displacement[1] * 2.0) / max(width - 1, 1)
    grid = identity.clone()
    grid[..., 0] = grid[..., 0] + norm_x
    grid[..., 1] = grid[..., 1] + norm_y
    grid = grid.unsqueeze(0)
    warped = F.grid_sample(
        field.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)
    return warped


def _compose_fields(base: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    return base + _warp_field(update, base)


def _stationary_velocity_to_displacement(velocity: torch.Tensor, steps: int) -> torch.Tensor:
    displacement = velocity / (2 ** steps)
    for _ in range(steps):
        displacement = _compose_fields(displacement, displacement)
    return displacement


def _generate_smod_fields(
    count: int,
    height: int,
    width: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    samples: List[torch.Tensor] = []
    for _ in range(count):
        sigma = float(torch.rand(1, generator=generator, device=device).item() * 6.0 + 2.0)
        amplitude = float(torch.rand(1, generator=generator, device=device).item() * 4.0 + 2.0)
        dy = _generate_smooth_field(height, width, sigma, device, dtype, generator=generator) * amplitude
        dx = _generate_smooth_field(height, width, sigma, device, dtype, generator=generator) * amplitude
        samples.append(torch.stack([dy, dx], dim=0))
    return torch.stack(samples, dim=0)


def _load_displacement_fields(
    root: Path,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fields: List[torch.Tensor] = []
    for file in sorted(root.iterdir()):
        if file.suffix.lower() not in {".pt", ".npy"}:
            continue
        if file.suffix.lower() == ".pt":
            tensor = torch.load(file, map_location=device)
        else:
            tensor = torch.from_numpy(np.load(file)).to(device)
        if tensor.dim() == 3 and tensor.size(0) == 2:
            field = tensor
        elif tensor.dim() == 3 and tensor.size(-1) == 2:
            field = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Field '{file}' must have shape [2,H,W] or [H,W,2].")
        field = field.to(dtype)
        if field.shape[-2:] != (height, width):
            field = F.interpolate(field.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=True).squeeze(0)
        fields.append(field)
    if not fields:
        raise ValueError(f"No displacement fields found in '{root}'.")
    return torch.stack(fields, dim=0)


def apply_smod(
    img: torch.Tensor,
    mask: Optional[torch.Tensor],
    model: SMODDeformationModel,
    p: float = 0.5,
    seed: Optional[int] = None,
    scale_override: Optional[float] = None,
    smooth_sigma: float = 1.0,
    return_field: bool = False,
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=img.device)
        generator.manual_seed(int(seed))
    apply = True
    if p < 1.0:
        rnd = torch.rand(1, generator=generator, device=img.device).item()
        apply = rnd < p
    field = model.sample_field(seed=seed, scale_override=scale_override, smooth_sigma=smooth_sigma)
    if not apply:
        if return_field:
            return img, mask, field
        return img, mask
    warped_img, warped_mask = warp_img_and_mask(img, mask, field)
    if return_field:
        return warped_img, warped_mask, field
    return warped_img, warped_mask


def _normalize_smod_config(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = params.get("smod", params)
    cache_dir = cfg.get("cache_dir")
    cache_dir = str(cache_dir) if cache_dir is not None else None
    fields_dir = cfg.get("fields_dir")
    fields_dir = str(fields_dir) if fields_dir is not None else None
    scale_override = cfg.get("scale_override")
    scale_override = float(scale_override) if scale_override is not None else None
    normalized = {
        "enabled": bool(cfg.get("enabled", True)),
        "p": float(cfg.get("p", 0.5)),
        "num_modes": int(cfg.get("num_modes", 16)),
        "scale": float(cfg.get("scale", 1.0)),
        "cache_dir": cache_dir,
        "fields_dir": fields_dir,
        "fit_subset": int(cfg.get("fit_subset", 64)),
        "enable_svs": bool(cfg.get("enable_svs", False)),
        "svs_steps": int(cfg.get("svs_steps", 6)),
        "smooth_sigma": float(cfg.get("smooth_sigma", 1.0)),
        "seed": cfg.get("seed"),
        "scale_override": scale_override,
    }
    return normalized


def _smod_key(height: int, width: int, cfg: Dict[str, Any]) -> Tuple[int, int, Tuple[Any, ...]]:
    key_items = (
        cfg["num_modes"],
        round(cfg["scale"], 6),
        cfg.get("cache_dir"),
        cfg.get("fields_dir"),
        cfg["enable_svs"],
        cfg["svs_steps"],
        round(cfg["smooth_sigma"], 6),
    )
    return height, width, key_items


def _get_smod_model(
    height: int,
    width: int,
    cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> SMODDeformationModel:
    key = _smod_key(height, width, cfg)
    model = _SMOD_MODELS.get(key)
    if model is None:
        model = SMODDeformationModel(
            num_modes=cfg["num_modes"],
            scale=cfg["scale"],
            cache_dir=cfg.get("cache_dir"),
            enable_svs=cfg["enable_svs"],
            svs_steps=cfg["svs_steps"],
            device=device,
            dtype=dtype,
        )
        _SMOD_MODELS[key] = model

    if model.mean is None:
        fields_dir = cfg.get("fields_dir")
        if fields_dir:
            fields = _load_displacement_fields(Path(fields_dir), height, width, device, dtype)
            model.fit_from_fields(fields)
        elif model.try_load_cache(height, width):
            pass
        else:
            generator = torch.Generator(device=device)
            seed = cfg.get("seed")
            if seed is not None:
                generator.manual_seed(int(seed))
            fields = _generate_smod_fields(
                max(2 * cfg["num_modes"], cfg["fit_subset"]),
                height,
                width,
                generator,
                device,
                dtype,
            )
            model.fit_from_fields(fields)
            model.save_cache()
    return model


def _histogram_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    src_flat = source.flatten()
    ref_flat = reference.flatten()
    ref_sorted, _ = torch.sort(ref_flat)
    src_sorted, src_indices = torch.sort(src_flat)
    matched = torch.empty_like(src_flat)
    matched[src_indices] = ref_sorted
    return matched.view_as(source)


def _generate_smooth_field(
    h: int,
    w: int,
    sigma: float,
    device,
    dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    noise = torch.randn(1, 1, h, w, generator=generator, dtype=dtype)
    noise = noise.to(device=device)
    kernel = max(3, int(2 * math.ceil(3 * sigma) + 1))
    if kernel % 2 == 0:
        kernel += 1
    blurred = TF.gaussian_blur(noise, kernel_size=kernel, sigma=sigma)
    field = blurred.squeeze(0).squeeze(0)
    field = field - field.mean()
    std = field.std().clamp_min(1e-6)
    return field / std


__all__ = [
    "AUGMENTATION_SPECS",
    "AUGMENTATION_ALIASES",
    "resolve_augmentation_mode",
    "augmentation_display_label",
    "augmentation_color",
    "apply_train_augmentations",
    "_seed_context",
    "warp_img_and_mask",
    "SMODDeformationModel",
    "apply_smod",
]
