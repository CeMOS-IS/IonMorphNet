"""GPU-friendly augmentation utilities.

These augmentations operate directly on CUDA tensors to reduce CPU load.
Unsupported modes fall back to the existing CPU pipeline so callers can
decide when to stick with the original implementation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Sequence
import math

import torch
import torch.nn.functional as F

from msianalyzer.training.augmentations import (
    resolve_augmentation_mode,
    _apply_histogram_grf,
    _seed_context,
    _sample_augmentation_by_strength,
)

GPU_SUPPORTED_MODES = {
    "default",
    "advanced",
    "auto_strength",
    "grid_mask",
    "random_mask",
    "cutmix", #RENAME to COPY
    "jitter",
    "scaling",
    "gaussian_blur",
    "cutout",
    "max_mask",
    "mz_shift",
    "ldm",
    "smooth_displacement",
    "local_permutation",
    "fourier_surrogate",
    "histogram_grf",
    "elastic_transform",
    "grid_distortion",
    "thin_plate_spline",
}


def apply_gpu_augmentations(
    imgs: torch.Tensor,
    mode: str,
    *,
    keep_probability: float = 1.0,
    override_params: Optional[Dict] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the requested augmentation mode to an entire batch on GPU."""
    if imgs.ndim != 4:
        return imgs
    seed = None
    if override_params:
        seed = override_params.get("seed")
    if generator is None and seed is not None:
        if imgs.device.type == "cuda":
            generator = torch.Generator(device=imgs.device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed))
    original = None
    mask = None
    with _seed_context(seed, imgs.device):
        if keep_probability < 1.0:
            original = imgs
            probs = torch.rand(imgs.shape[0], device=imgs.device, generator=generator)
            mask = probs < keep_probability
            if not mask.any():
                return imgs
            mask = mask.view(-1, 1, 1, 1)
            imgs = imgs.clone()
        else:
            imgs = imgs.clone()

        canonical = resolve_augmentation_mode(mode)
        if canonical not in GPU_SUPPORTED_MODES:
            return imgs

        if canonical == "default":
            return imgs

        if canonical == "auto_strength":
            params = override_params or {}
            sampled = _sample_augmentation_by_strength(
                candidates=params.get("candidates"),
                original_probability=float(params.get("original_probability", 0.5)),
                bias=float(params.get("bias", 1.5)),
            )
            if sampled == "default":
                return imgs
            return apply_gpu_augmentations(
                imgs,
                sampled,
                keep_probability=1.0,
                override_params=params,
                generator=generator,
            )

        if canonical == "advanced":
            imgs = _random_resized_crop(imgs, (0.85, 1.0), generator)
            imgs = _gamma_jitter(imgs, (0.9, 1.1), generator)
            imgs = _gaussian_blur(imgs, (0.0, 0.8), generator)
            return imgs

    if canonical == "grid_mask":
        params = override_params or {}
        keep_ratio = float(params.get("keep_ratio", 0.6))
        cell_range = _pair(params.get("cell_range"), (16, 48))
        return _grid_mask_batch(imgs, keep_ratio=keep_ratio, cell_range=cell_range, generator=generator)

    if canonical == "random_mask":
        params = override_params or {}
        size_range = _pair(params.get("size_range"), (0.2, 0.5))
        fill_value = params.get("fill_value")
        return _random_rect_mask_batch(imgs, size_range=size_range, fill_value=fill_value, generator=generator)

    if canonical == "cutmix":
        params = override_params or {}
        lam_range = _pair(params.get("lambda_range") or params.get("lam_range"), (0.3, 0.7))
        size_range = _pair(params.get("size_range"), (0.3, 0.6))
        return _cutmix_batch(imgs, lam_range=lam_range, size_range=size_range, generator=generator)

    if canonical == "jitter":
        return _apply_gaussian_noise(imgs, (0.01, 0.08), generator)

    if canonical == "gaussian_blur":
        return _gaussian_blur(imgs, (0.6, 1.6), generator)

    if canonical == "cutout":
        return _apply_cutout(imgs, size_range=(0.1, 0.3), generator=generator)

    if canonical == "scaling":
        return _random_resized_crop(imgs, (0.6, 1.05), generator)

    if canonical == "max_mask":
        params = override_params or {}
        quantile = float(params.get("quantile", 0.9))
        return _max_mask_batch(imgs, quantile=quantile)

    if canonical == "mz_shift":
        params = override_params or {}
        max_shift = int(params.get("max_shift", 5))
        return _mz_shift_batch(imgs, max_shift=max_shift, generator=generator)

    if canonical == "ldm":
        params = override_params or {}
        noise_range = _pair(params.get("noise_sigma_range"), (0.02, 0.06))
        crop_range = _pair(params.get("scale_range"), (0.85, 1.0))
        blur_range = _pair(params.get("blur_sigma_range"), (0.2, 0.9))
        return _ldm_like_batch(imgs, noise_range=noise_range, crop_range=crop_range, blur_range=blur_range, generator=generator)

    if canonical == "smooth_displacement":
        params = override_params or {}
        return _smooth_displacement(imgs, sigma=params.get("sigma", 4.0), amplitude=params.get("amplitude", 2.0), generator=generator)

    if canonical == "elastic_transform":
        params = override_params or {}
        return _elastic_transform_batch(
            imgs,
            alpha=float(params.get("alpha", 40.0)),
            sigma=float(params.get("sigma", 6.0)),
            generator=generator,
        )

    if canonical == "grid_distortion":
        params = override_params or {}
        return _grid_distortion_batch(
            imgs,
            num_steps=int(params.get("num_steps", 5)),
            distort_limit=float(params.get("distort_limit", 0.3)),
            generator=generator,
        )

    if canonical == "thin_plate_spline":
        params = override_params or {}
        return _thin_plate_spline_batch(
            imgs,
            num_ctrl_pts=int(params.get("num_ctrl_pts", 45)),
            strength=float(params.get("strength", 0.2)),
            generator=generator,
        )

    if canonical == "local_permutation":
        params = override_params or {}
        return _local_permutation(imgs, radius=params.get("radius", 5), swaps=params.get("swaps", 200), generator=generator)

    if canonical == "fourier_surrogate":
        params = override_params or {}
        return _fourier_surrogate(
            imgs,
            phase_mix=params.get("phase_mix", 0.7),
            magnitude_blur_sigma=params.get("magnitude_blur_sigma", 1.5),
            generator=generator,
        )

    if canonical == "histogram_grf":
        params = override_params or {}
        return _histogram_grf_batch(
            imgs,
            seed=params.get("seed"),
            magnitude_blur_sigma=params.get("magnitude_blur_sigma"),
            spectral_exponent=params.get("spectral_exponent"),
        )

        augmented = _base_geom(imgs, generator=generator)
        if mask is None:
            return augmented
        return torch.where(mask, augmented, original)


def _base_geom(imgs: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    imgs = _random_flip(imgs, generator)
    imgs = _random_rotation(imgs, generator)
    return imgs


def _random_flip(imgs: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if torch.rand(1, device=imgs.device, generator=generator) < 0.5:
        imgs = imgs.flip(-1)
    if torch.rand(1, device=imgs.device, generator=generator) < 0.5:
        imgs = imgs.flip(-2)
    return imgs


def _random_rotation(imgs: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    k = torch.randint(0, 4, (1,), device=imgs.device, generator=generator).item()
    return torch.rot90(imgs, k, dims=(-2, -1))


def _random_resized_crop(imgs: torch.Tensor, scale_range: Tuple[float, float], generator: Optional[torch.Generator]) -> torch.Tensor:
    b, c, h, w = imgs.shape
    min_scale, max_scale = scale_range
    scale = float(_rand_uniform(1, min_scale, max_scale, imgs.device, generator)[0])
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if new_h >= h or new_w >= w:
        return imgs
    top = torch.randint(0, h - new_h + 1, (1,), device=imgs.device, generator=generator).item()
    left = torch.randint(0, w - new_w + 1, (1,), device=imgs.device, generator=generator).item()
    cropped = imgs[:, :, top:top + new_h, left:left + new_w]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def _gaussian_blur(imgs: torch.Tensor, sigma_range: Tuple[float, float], generator: Optional[torch.Generator]) -> torch.Tensor:
    sigma = float(_rand_uniform(1, sigma_range[0], sigma_range[1], imgs.device, generator)[0])
    if sigma <= 0.0:
        return imgs
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=imgs.device, dtype=imgs.dtype)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    ch = imgs.shape[1]
    kernel_h = kernel.unsqueeze(3).repeat(ch, 1, 1, 1)
    kernel_v = kernel.unsqueeze(2).repeat(ch, 1, 1, 1)
    imgs = F.conv2d(imgs, kernel_h, padding=(radius, 0), groups=ch)
    imgs = F.conv2d(imgs, kernel_v, padding=(0, radius), groups=ch)
    return imgs


def _gamma_jitter(imgs: torch.Tensor, gamma_range: Tuple[float, float], generator: Optional[torch.Generator]) -> torch.Tensor:
    gamma = float(_rand_uniform(1, gamma_range[0], gamma_range[1], imgs.device, generator)[0])
    return imgs.clamp_min(1e-6).pow(gamma)


def _apply_gaussian_noise(imgs: torch.Tensor, sigma_range: Tuple[float, float], generator: Optional[torch.Generator]) -> torch.Tensor:
    sigma = float(_rand_uniform(1, sigma_range[0], sigma_range[1], imgs.device, generator)[0])
    noise = torch.randn(
        imgs.shape,
        device=imgs.device,
        dtype=imgs.dtype,
        generator=generator,
    ) * sigma
    return (imgs + noise).clamp(0.0, 1.0)


def _apply_cutout(imgs: torch.Tensor, size_range: Tuple[float, float], generator: Optional[torch.Generator]) -> torch.Tensor:
    b, c, h, w = imgs.shape
    min_frac, max_frac = size_range
    frac = float(_rand_uniform(1, min_frac, max_frac, imgs.device, generator)[0])
    cut_h = max(1, int(h * frac))
    cut_w = max(1, int(w * frac))
    top = torch.randint(0, h - cut_h + 1, (1,), device=imgs.device, generator=generator).item()
    left = torch.randint(0, w - cut_w + 1, (1,), device=imgs.device, generator=generator).item()
    imgs = imgs.clone()
    imgs[:, :, top:top + cut_h, left:left + cut_w] = 0.0
    return imgs


def _smooth_displacement(imgs: torch.Tensor, sigma: float, amplitude: float, generator: Optional[torch.Generator]) -> torch.Tensor:
    b, c, h, w = imgs.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=imgs.device),
        torch.linspace(-1, 1, w, device=imgs.device),
        indexing="ij",
    )
    flow = torch.randn((b, 2, h, w), device=imgs.device, generator=generator)
    flow = _gaussian_blur(flow, (sigma, sigma), generator)
    flow = flow * (amplitude / (flow.abs().max().clamp_min(1e-6)))
    grid = torch.stack((grid_x, grid_y), dim=-1) + flow.permute(0, 2, 3, 1)
    return F.grid_sample(imgs, grid.clamp(-1, 1), mode="bilinear", padding_mode="reflection", align_corners=True)


def _local_permutation(imgs: torch.Tensor, radius: int, swaps: int, generator: Optional[torch.Generator]) -> torch.Tensor:
    b, c, h, w = imgs.shape
    imgs = imgs.clone()
    for _ in range(int(swaps)):
        y = torch.randint(radius, h - radius, (b,), device=imgs.device, generator=generator)
        x = torch.randint(radius, w - radius, (b,), device=imgs.device, generator=generator)
        dy = torch.randint(-radius, radius + 1, (b,), device=imgs.device, generator=generator)
        dx = torch.randint(-radius, radius + 1, (b,), device=imgs.device, generator=generator)
        src = imgs[torch.arange(b), :, y, x]
        dst = imgs[torch.arange(b), :, (y + dy).clamp(0, h - 1), (x + dx).clamp(0, w - 1)]
        imgs[torch.arange(b), :, y, x] = dst
        imgs[torch.arange(b), :, (y + dy).clamp(0, h - 1), (x + dx).clamp(0, w - 1)] = src
    return imgs


def _fourier_surrogate(imgs: torch.Tensor, phase_mix: float, magnitude_blur_sigma: float, generator: Optional[torch.Generator]) -> torch.Tensor:
    fft = torch.fft.fftn(imgs, dim=(-2, -1))
    magnitude = torch.abs(fft)
    phase = torch.angle(fft)
    random_phase = torch.rand(
        phase.shape,
        device=phase.device,
        dtype=phase.dtype,
        generator=generator,
    ) * 2 * torch.pi - torch.pi
    new_phase = (1 - phase_mix) * phase + phase_mix * random_phase
    if magnitude_blur_sigma > 0:
        magnitude = _gaussian_blur(magnitude, (magnitude_blur_sigma, magnitude_blur_sigma), generator)
    fft_new = magnitude * torch.exp(1j * new_phase)
    surrogate = torch.fft.ifftn(fft_new, dim=(-2, -1)).real
    return surrogate.clamp(0.0, 1.0)

_BASE_GRID_CACHE = {}


def _get_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (h, w, device, dtype)
    grid = _BASE_GRID_CACHE.get(key)
    if grid is None:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        _BASE_GRID_CACHE[key] = grid
    return grid


def _elastic_transform_batch(imgs: torch.Tensor, alpha: float = 40.0, sigma: float = 6.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    b, c, h, w = imgs.shape
    grid = _get_base_grid(h, w, imgs.device, imgs.dtype).unsqueeze(0).repeat(b, 1, 1, 1)
    disp = torch.randn(b, 2, h, w, device=imgs.device, generator=generator)
    disp = _gaussian_blur(disp, (sigma, sigma), generator)
    flat = disp.view(b, 2, -1)
    max_abs = flat.abs().max(dim=-1, keepdim=True)[0].view(b, 2, 1, 1).clamp_min(1e-6)
    disp = (disp / max_abs) * alpha
    disp = disp.permute(0, 2, 3, 1)
    norm = torch.tensor([max(w - 1, 1), max(h - 1, 1)], device=imgs.device, dtype=imgs.dtype).view(1, 1, 1, 2)
    disp_norm = (disp / norm) * 2.0
    warped_grid = (grid + disp_norm).clamp(-1, 1)
    return F.grid_sample(imgs, warped_grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def _grid_distortion_batch(imgs: torch.Tensor, num_steps: int = 5, distort_limit: float = 0.3, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    b, c, h, w = imgs.shape
    grid = _get_base_grid(h, w, imgs.device, imgs.dtype).permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
    coarse = _rand_uniform(
        (b, 2, num_steps + 1, num_steps + 1),
        -distort_limit,
        distort_limit,
        imgs.device,
        generator,
        dtype=imgs.dtype,
    )
    offsets = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=True)
    warped_grid = (grid + offsets).permute(0, 2, 3, 1).clamp(-1, 1)
    return F.grid_sample(imgs, warped_grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def _thin_plate_spline_batch(imgs: torch.Tensor, num_ctrl_pts: int = 16, strength: float = 0.08, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    b, c, h, w = imgs.shape
    ctrl_pts = torch.rand(b, num_ctrl_pts, 2, device=imgs.device, generator=generator) * 2 - 1
    offsets = torch.empty_like(ctrl_pts).uniform_(-strength, strength)
    base_grid = _get_base_grid(h, w, imgs.device, imgs.dtype)
    grid = base_grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
    # approximate TPS via radial basis interpolation
    pts = torch.stack(torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=imgs.device),
        torch.linspace(-1.0, 1.0, w, device=imgs.device),
        indexing="ij",
    ), dim=-1).view(1, h * w, 2).repeat(b, 1, 1)
    diffs = pts.unsqueeze(2) - ctrl_pts.unsqueeze(1)
    dists_sq = (diffs ** 2).sum(-1).clamp_min(1e-6)
    weights = dists_sq * dists_sq.log()
    weights = weights / weights.max(dim=1, keepdim=True)[0].clamp_min(1e-6)
    disp = torch.matmul(weights, offsets)
    disp = disp.view(b, h, w, 2)
    warped_grid = (grid + disp).clamp(-1, 1)
    return F.grid_sample(imgs, warped_grid, mode="bilinear", padding_mode="reflection", align_corners=True)

def _grid_mask_batch(imgs: torch.Tensor, keep_ratio: float, cell_range: Tuple[int, int], generator: Optional[torch.Generator]) -> torch.Tensor:
    out = imgs.clone()
    b, c, h, w = imgs.shape
    for i in range(b):
        mask = torch.ones((h, w), dtype=imgs.dtype, device=imgs.device)
        cell_h = torch.randint(int(cell_range[0]), int(cell_range[1]) + 1, (1,), device=imgs.device, generator=generator).item()
        cell_w = torch.randint(int(cell_range[0]), int(cell_range[1]) + 1, (1,), device=imgs.device, generator=generator).item()
        band_h = max(1, int(cell_h * (1.0 - keep_ratio)))
        band_w = max(1, int(cell_w * (1.0 - keep_ratio)))
        offset_h = torch.randint(0, cell_h, (1,), device=imgs.device, generator=generator).item()
        offset_w = torch.randint(0, cell_w, (1,), device=imgs.device, generator=generator).item()
        for y in range(offset_h, h, cell_h):
            mask[y:y + band_h, :] = 0
        for x in range(offset_w, w, cell_w):
            mask[:, x:x + band_w] = 0
        out[i] = out[i] * mask.unsqueeze(0)
    return out


def _random_rect_mask_batch(imgs: torch.Tensor, size_range: Tuple[float, float], fill_value: Optional[float], generator: Optional[torch.Generator]) -> torch.Tensor:
    out = imgs.clone()
    b, c, h, w = imgs.shape
    for i in range(b):
        rh = int(h * float(_rand_uniform(1, size_range[0], size_range[1], imgs.device, generator)[0]))
        rw = int(w * float(_rand_uniform(1, size_range[0], size_range[1], imgs.device, generator)[0]))
        rh = max(1, min(h, rh))
        rw = max(1, min(w, rw))
        top = torch.randint(0, max(1, h - rh + 1), (1,), device=imgs.device, generator=generator).item()
        left = torch.randint(0, max(1, w - rw + 1), (1,), device=imgs.device, generator=generator).item()
        fill = float(fill_value) if fill_value is not None else float(out[i].mean())
        out[i, :, top:top + rh, left:left + rw] = fill
    return out


def _cutmix_batch(imgs: torch.Tensor, lam_range: Tuple[float, float], size_range: Tuple[float, float], generator: Optional[torch.Generator]) -> torch.Tensor:
    out = imgs.clone()
    rolled = torch.roll(out, shifts=1, dims=0)
    b, c, h, w = imgs.shape
    for i in range(b):
        lam = float(_rand_uniform(1, lam_range[0], lam_range[1], imgs.device, generator)[0])
        rh = int(h * float(_rand_uniform(1, size_range[0], size_range[1], imgs.device, generator)[0]))
        rw = int(w * float(_rand_uniform(1, size_range[0], size_range[1], imgs.device, generator)[0]))
        rh = max(1, min(h, rh))
        rw = max(1, min(w, rw))
        top = torch.randint(0, max(1, h - rh + 1), (1,), device=imgs.device, generator=generator).item()
        left = torch.randint(0, max(1, w - rw + 1), (1,), device=imgs.device, generator=generator).item()
        patch = lam * out[i, :, top:top + rh, left:left + rw] + (1.0 - lam) * rolled[i, :, top:top + rh, left:left + rw]
        out[i, :, top:top + rh, left:left + rw] = patch
    return out


def _max_mask_batch(imgs: torch.Tensor, quantile: float) -> torch.Tensor:
    flat = imgs.view(imgs.shape[0], -1)
    thresh = torch.quantile(flat, quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    return torch.where(imgs >= thresh, imgs.new_zeros(()), imgs)


def _mz_shift_batch(imgs: torch.Tensor, max_shift: int, generator: Optional[torch.Generator]) -> torch.Tensor:
    shifts = torch.randint(-max_shift, max_shift + 1, (imgs.shape[0],), device=imgs.device, generator=generator)
    out = []
    for img, shift in zip(imgs, shifts):
        out.append(torch.roll(img, shifts=int(shift.item()), dims=-1))
    return torch.stack(out, dim=0)


def _ldm_like_batch(
    imgs: torch.Tensor,
    noise_range: Tuple[float, float],
    crop_range: Tuple[float, float],
    blur_range: Tuple[float, float],
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    imgs = _apply_gaussian_noise(imgs, noise_range, generator)
    imgs = _random_resized_crop(imgs, crop_range, generator)
    imgs = _gaussian_blur(imgs, blur_range, generator)
    return imgs


def _histogram_grf_batch(
    imgs: torch.Tensor,
    seed: Optional[int],
    magnitude_blur_sigma: Optional[float],
    spectral_exponent: Optional[float],
) -> torch.Tensor:
    out = imgs.clone()
    for i in range(imgs.shape[0]):
        out[i] = _apply_histogram_grf(
            out[i],
            seed=seed,
            magnitude_blur_sigma=magnitude_blur_sigma,
            spectral_exponent=spectral_exponent,
        )
    return out


__all__ = ["apply_gpu_augmentations", "GPU_SUPPORTED_MODES"]
def _rand_uniform(shape, low, high, device, generator, dtype=torch.float32):
    tensor = torch.rand(shape, device=device, generator=generator, dtype=dtype)
    return tensor * (high - low) + low


def _pair(value: Optional[Sequence[float]], default: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"Expected sequence of length 2, got {value}")
