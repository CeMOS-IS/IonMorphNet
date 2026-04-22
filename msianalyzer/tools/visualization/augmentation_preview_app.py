"""
Interactive augmentation preview app.

Launch with:
    streamlit run msianalyzer/tools/visualization/augmentation_preview_app.py

Dependencies:
    pip install streamlit pillow
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from msianalyzer.config import PROCESSED_DIR, TARGET_HW
from msianalyzer.training.augmentations import (
    AUGMENTATION_SPECS,
    SMODDeformationModel,
    _generate_smod_fields,
    apply_smod,
    apply_train_augmentations,
    resolve_augmentation_mode,
)


def _list_sample_images() -> List[Path]:
    sample_root = Path("all_images")
    if not sample_root.exists():
        return []
    return sorted(
        [p for p in sample_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    )


def _load_image_from_path(path: Path) -> np.ndarray:
    img = Image.open(path).convert("F")
    arr = np.array(img, dtype=np.float32)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def _load_image_from_upload(upload) -> np.ndarray:
    suffix = Path(upload.name).suffix.lower()
    if suffix == ".npy":
        arr = np.load(upload)
        arr = np.array(arr, dtype=np.float32)
    else:
        arr = Image.open(upload).convert("F")
        arr = np.array(arr, dtype=np.float32)
        if arr.max() > 0:
            arr /= arr.max()
    return arr


def _prepare_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    p99 = torch.quantile(tensor.flatten(), 0.99) if tensor.numel() else torch.tensor(1.0)
    return (tensor / p99.clamp_min(1e-6)).clamp(0.0, 1.0)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    array = np.clip(array, 0.0, 1.0)
    array = (array * 255).astype(np.uint8)
    return array


def _resize_to_target(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-2:] == TARGET_HW:
        return tensor
    resized = F.interpolate(tensor.unsqueeze(0), size=TARGET_HW, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def _smod_model_key(height: int, width: int, cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        int(height),
        int(width),
        int(cfg["num_modes"]),
        round(float(cfg["scale"]), 6),
        cfg.get("fields_dir"),
        cfg.get("cache_dir"),
        bool(cfg["enable_svs"]),
        int(cfg["svs_steps"]),
        int(cfg["fit_subset"]),
        round(float(cfg["smooth_sigma"]), 6),
        cfg.get("seed"),
    )


def _drop_smod_model(key: Tuple[Any, ...]) -> None:
    models: Dict[Tuple[Any, ...], SMODDeformationModel] = st.session_state.get("smod_models", {})
    if key in models:
        del models[key]


def _ensure_smod_model(
    height: int,
    width: int,
    cfg: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[SMODDeformationModel]:
    models: Dict[Tuple[Any, ...], SMODDeformationModel] = st.session_state.setdefault("smod_models", {})
    key = _smod_model_key(height, width, cfg)
    if key in models:
        return models[key]

    model = SMODDeformationModel(
        num_modes=int(cfg["num_modes"]),
        scale=float(cfg["scale"]),
        cache_dir=cfg.get("cache_dir"),
        enable_svs=bool(cfg["enable_svs"]),
        svs_steps=int(cfg["svs_steps"]),
        device=device,
        dtype=dtype,
    )
    try:
        fields_dir = cfg.get("fields_dir")
        if fields_dir:
            model.fit_from_dir(fields_dir)
        else:
            cache_loaded = False
            if cfg.get("cache_dir"):
                cache_loaded = model.try_load_cache(height, width)
            if not cache_loaded:
                gen_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
                generator = torch.Generator(device=gen_device)
                seed = cfg.get("seed")
                if seed is not None:
                    generator.manual_seed(int(seed))
                field_count = max(int(cfg["fit_subset"]), 2 * int(cfg["num_modes"]))
                fields = _generate_smod_fields(
                    field_count,
                    height,
                    width,
                    generator,
                    device,
                    dtype,
                )
                model.fit_from_fields(fields)
                model.save_cache()
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"Failed to prepare SMOD model: {exc}")
        return None

    models[key] = model
    return model


def _field_to_heatmap(field: torch.Tensor) -> np.ndarray:
    if field.dim() != 3 or field.size(0) != 2:
        raise ValueError("Expected displacement field with shape [2,H,W].")
    magnitude = torch.linalg.norm(field, dim=0)
    max_val = float(magnitude.max().item())
    if max_val <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    norm_mag = (magnitude / max_val).clamp(0.0, 1.0).cpu().numpy()
    heatmap = (norm_mag * 255).astype(np.uint8)
    heatmap_rgb = np.stack([heatmap] * 3, axis=-1)
    return heatmap_rgb


def main() -> None:
    st.set_page_config(page_title="Augmentation Preview", layout="wide")
    st.title("MSI Augmentation Preview")

    st.sidebar.header("Image source")
    source_mode = st.sidebar.radio(
        "Select sample source",
        options=["Upload image", "Sample from all_images directory"],
    )

    image_array: Optional[np.ndarray] = None
    if source_mode == "Upload image":
        upload = st.sidebar.file_uploader("Upload PNG/JPG/TIFF or NumPy (.npy) tensor", type=["png", "jpg", "jpeg", "tif", "tiff", "npy"])
        if upload is not None:
            try:
                image_array = _load_image_from_upload(upload)
            except Exception as exc:
                st.error(f"Failed to load uploaded file: {exc}")
    else:
        sample_paths = _list_sample_images()
        if not sample_paths:
            st.sidebar.warning("No sample images found in all_images/.")
        else:
            choice = st.sidebar.selectbox(
                "Pick a sample image",
                options=sample_paths,
                format_func=lambda p: p.relative_to(Path.cwd()),
            )
            image_array = _load_image_from_path(choice)

    if image_array is None:
        st.info("Upload an image or select a sample to begin.")
        st.stop()

    tensor = _prepare_tensor(image_array)
    tensor = _resize_to_target(tensor)
    st.subheader("Original image")
    st.image(_tensor_to_image(tensor), clamp=True, caption="Original (normalized)")

    st.sidebar.header("Augmentation controls")
    aug_labels = {
        key: spec.get("label", key.replace("_", " ").title())
        for key, spec in AUGMENTATION_SPECS.items()
    }
    aug_choice = st.sidebar.selectbox(
        "Augmentation mode",
        options=sorted(AUGMENTATION_SPECS.keys()),
        format_func=lambda key: aug_labels.get(key, key),
    )
    canonical_mode = resolve_augmentation_mode(aug_choice)

    preview_state: Dict[str, Any] = st.session_state.setdefault("smod_preview_state", {})
    if canonical_mode == "smod":
        prev_cfg = preview_state.get("config", {})
        height, width = tensor.shape[-2:]
        st.sidebar.subheader("SMOD deformation")
        smod_enabled = st.sidebar.checkbox("Enable SMOD", value=bool(prev_cfg.get("enabled", True)))
        smod_p = st.sidebar.slider("Apply probability p", min_value=0.0, max_value=1.0, value=float(prev_cfg.get("p", 0.5)), step=0.05)
        smod_num_modes = st.sidebar.slider("PCA modes", min_value=1, max_value=64, value=int(prev_cfg.get("num_modes", 16)), step=1)
        smod_scale = st.sidebar.slider("Scale multiplier", min_value=0.1, max_value=3.0, value=float(prev_cfg.get("scale", 1.0)), step=0.05)
        smod_smooth_sigma = st.sidebar.slider(
            "Field smoothing σ", min_value=0.0, max_value=5.0, value=float(prev_cfg.get("smooth_sigma", 1.0)), step=0.1
        )
        cache_dir_input = st.sidebar.text_input("Cache directory (optional)", value=str(prev_cfg.get("cache_dir") or ""))
        fields_dir_input = st.sidebar.text_input("Fields directory (optional)", value=str(prev_cfg.get("fields_dir") or ""))
        smod_fit_subset = st.sidebar.number_input(
            "Synthetic field count", min_value=8, max_value=2048, value=int(prev_cfg.get("fit_subset", 128)), step=8
        )
        smod_model_seed = st.sidebar.number_input(
            "Model seed", min_value=0, max_value=1_000_000, value=int(prev_cfg.get("seed", 42)), step=1
        )
        smod_enable_svs = st.sidebar.checkbox("Enable scaling-and-squaring (SVS)", value=bool(prev_cfg.get("enable_svs", False)))
        smod_svs_steps = st.sidebar.slider(
            "SVS steps",
            min_value=1,
            max_value=8,
            value=int(prev_cfg.get("svs_steps", 6)),
            step=1,
        )
        smod_sample_seed = st.sidebar.number_input(
            "Sample seed",
            min_value=0,
            max_value=1_000_000,
            value=int(preview_state.get("sample_seed", 123)),
            step=1,
        )
        apply_button = st.sidebar.button("Apply SMOD once")
        refit_button = st.sidebar.button("Refit SMOD PCA model")

        cache_dir_clean = cache_dir_input.strip()
        if not cache_dir_clean or cache_dir_clean.lower() == "none":
            cache_dir_clean = None
        fields_dir_clean = fields_dir_input.strip()
        if not fields_dir_clean or fields_dir_clean.lower() == "none":
            fields_dir_clean = None

        smod_config = {
            "enabled": smod_enabled,
            "p": float(smod_p),
            "num_modes": int(smod_num_modes),
            "scale": float(smod_scale),
            "cache_dir": cache_dir_clean,
            "fields_dir": fields_dir_clean,
            "fit_subset": int(smod_fit_subset),
            "enable_svs": bool(smod_enable_svs),
            "svs_steps": int(smod_svs_steps if smod_enable_svs else max(int(prev_cfg.get("svs_steps", 6)), 1)),
            "smooth_sigma": float(smod_smooth_sigma),
            "seed": int(smod_model_seed),
        }
        preview_state["config"] = smod_config
        preview_state["sample_seed"] = int(smod_sample_seed)

        model_key = _smod_model_key(height, width, smod_config)
        if refit_button:
            _drop_smod_model(model_key)
            preview_state.pop("result", None)

        if apply_button:
            if not smod_enabled:
                preview_state["result"] = {
                    "applied": False,
                    "image": tensor.detach().cpu(),
                    "field": torch.zeros(2, height, width),
                    "seed": int(smod_sample_seed),
                }
            else:
                model = _ensure_smod_model(height, width, smod_config, tensor.device, tensor.dtype)
                if model is not None:
                    augmented, _, field = apply_smod(
                        tensor.clone(),
                        None,
                        model,
                        p=float(smod_p),
                        seed=int(smod_sample_seed),
                        smooth_sigma=float(smod_smooth_sigma),
                        return_field=True,
                    )
                    applied_flag = not torch.allclose(augmented, tensor, atol=1e-5)
                    preview_state["result"] = {
                        "applied": applied_flag,
                        "image": augmented.detach().cpu(),
                        "field": field.detach().cpu(),
                        "seed": int(smod_sample_seed),
                    }

        st.subheader("SMOD preview")
        result = preview_state.get("result")
        if result:
            col_aug, col_field = st.columns(2)
            aug_caption = f"SMOD augmented (seed {result.get('seed', '?')})"
            if not result.get("applied", False) or not smod_enabled:
                aug_caption += " [skipped]"
            col_aug.image(_tensor_to_image(result["image"]), clamp=True, caption=aug_caption)
            field_tensor: Optional[torch.Tensor] = result.get("field")
            if field_tensor is not None:
                heatmap = _field_to_heatmap(field_tensor)
                col_field.image(heatmap, clamp=True, caption="Displacement magnitude (normalized)")
                disp_mag = torch.linalg.norm(field_tensor, dim=0)
                max_disp = float(disp_mag.max().item())
                mean_disp = float(disp_mag.mean().item())
                p95_disp = float(torch.quantile(disp_mag.flatten(), 0.95).item())
                st.markdown(
                    f"- Max displacement: {max_disp:.2f} px\n"
                    f"- Mean displacement: {mean_disp:.2f} px\n"
                    f"- 95th percentile: {p95_disp:.2f} px"
                )
            else:
                col_field.info("No displacement field available.")
        else:
            st.info("Click 'Apply SMOD once' to generate a preview.")
    else:
        def _range_slider(label: str, min_val: float, max_val: float, default: Tuple[float, float], step: float = 0.1) -> Tuple[float, float]:
            low, high = st.sidebar.slider(label, min_value=min_val, max_value=max_val, value=default, step=step)
            return float(low), float(high)

        override_template: Dict[str, Any] = {}
        if canonical_mode == "smooth_displacement":
            override_template["sigma"] = st.sidebar.slider("Smooth sigma (px)", 0.5, 20.0, 6.0, 0.5)
            override_template["amplitude"] = st.sidebar.slider("Amplitude (px)", 0.0, 15.0, 4.0, 0.5)
        elif canonical_mode == "local_permutation":
            override_template["radius"] = st.sidebar.slider("Radius (px)", 1, 50, 8, 1)
            override_template["patch_size"] = st.sidebar.slider("Patch size", 1, 15, 3, 1)
            override_template["swaps"] = st.sidebar.slider("Number of swaps", 1, 5000, 500, 10)
        elif canonical_mode == "histogram_grf":
            override_template["magnitude_blur_sigma"] = st.sidebar.slider("Magnitude blur σ", 0.0, 10.0, 0.0, 0.5)
            override_template["spectral_exponent"] = st.sidebar.slider("Spectral exponent", -2.0, 2.0, 0.0, 0.1)
        elif canonical_mode == "fourier_surrogate":
            override_template["phase_mix"] = st.sidebar.slider("Phase mix (0=orig,1=random)", 0.0, 1.0, 1.0, 0.05)
            override_template["magnitude_blur_sigma"] = st.sidebar.slider("Magnitude blur σ", 0.0, 10.0, 0.0, 0.5)
            override_template["iaaft_iterations"] = st.sidebar.slider("IAAFT iterations", 0, 10, 0, 1)
        elif canonical_mode == "grid_mask":
            override_template["keep_ratio"] = st.sidebar.slider("Keep ratio", 0.1, 0.95, 0.6, 0.05)
            override_template["cell_range"] = _range_slider("Cell size range", 4.0, 128.0, (16.0, 48.0), 1.0)
        elif canonical_mode == "random_mask":
            override_template["size_range"] = _range_slider("Mask size fraction", 0.05, 0.9, (0.2, 0.5), 0.01)
            use_mean = st.sidebar.checkbox("Fill with channel mean", value=True)
            if not use_mean:
                override_template["fill_value"] = st.sidebar.slider("Fill value", 0.0, 1.0, 0.0, 0.05)
        elif canonical_mode == "cutmix":
            override_template["lambda_range"] = _range_slider("Lambda range", 0.0, 1.0, (0.3, 0.7), 0.05)
            override_template["size_range"] = _range_slider("Patch size fraction", 0.1, 0.9, (0.3, 0.6), 0.01)
        elif canonical_mode == "max_mask":
            override_template["quantile"] = st.sidebar.slider("Quantile threshold", 0.5, 0.99, 0.9, 0.01)
        elif canonical_mode == "mz_shift":
            override_template["max_shift"] = st.sidebar.slider("Max shift (pixels)", 0, 64, 5, 1)
        elif canonical_mode == "ldm":
            override_template["noise_sigma_range"] = _range_slider("Noise σ range", 0.0, 0.2, (0.02, 0.06), 0.005)
            override_template["scale_range"] = _range_slider("Random crop scale", 0.5, 1.2, (0.85, 1.0), 0.01)
            override_template["blur_sigma_range"] = _range_slider("Blur σ range", 0.0, 2.5, (0.2, 0.9), 0.05)
        elif canonical_mode == "elastic_transform":
            override_template["alpha"] = st.sidebar.slider("Alpha (px)", 1.0, 80.0, 40.0, 1.0)
            override_template["sigma"] = st.sidebar.slider("Sigma (px)", 1.0, 20.0, 6.0, 0.5)
        elif canonical_mode == "grid_distortion":
            override_template["num_steps"] = st.sidebar.slider("Number of grid steps", 2, 12, 5, 1)
            override_template["distort_limit"] = st.sidebar.slider("Distort limit", 0.0, 0.6, 0.3, 0.05)
        elif canonical_mode == "thin_plate_spline":
            override_template["num_ctrl_pts"] = st.sidebar.slider("Control points", 4, 64, 16, 1)
            override_template["strength"] = st.sidebar.slider("Strength", 0.01, 0.3, 0.08, 0.01)

        num_samples = st.sidebar.slider("Number of augmented previews", min_value=1, max_value=8, value=4)
        base_seed = st.sidebar.number_input("Base seed", min_value=0, max_value=10_000, value=123, step=1)
        refresh = st.sidebar.button("Regenerate previews")

        if refresh:
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

        st.subheader("Augmented previews")
        cols = st.columns(num_samples)
        for idx in range(num_samples):
            seed = int(base_seed + idx)
            torch.manual_seed(seed)
            override: Optional[Dict[str, Any]] = None
            if canonical_mode == "smooth_displacement":
                override = {
                    "sigma": float(override_template.get("sigma", 6.0)),
                    "amplitude": float(override_template.get("amplitude", 4.0)),
                    "seed": seed,
                }
            elif canonical_mode == "local_permutation":
                override = {
                    "radius": float(override_template.get("radius", 8)),
                    "patch_size": int(override_template.get("patch_size", 3)),
                    "swaps": int(override_template.get("swaps", 500)),
                    "seed": seed,
                }
            elif canonical_mode == "histogram_grf":
                override = {
                    "seed": seed,
                    "magnitude_blur_sigma": float(override_template.get("magnitude_blur_sigma", 0.0)),
                    "spectral_exponent": float(override_template.get("spectral_exponent", 0.0)),
                }
            elif canonical_mode == "fourier_surrogate":
                override = {
                    "seed": seed,
                    "phase_mix": float(override_template.get("phase_mix", 1.0)),
                    "magnitude_blur_sigma": float(override_template.get("magnitude_blur_sigma", 0.0)),
                    "iaaft_iterations": int(override_template.get("iaaft_iterations", 0)),
                }
            elif canonical_mode == "grid_mask":
                override = {
                    "keep_ratio": float(override_template.get("keep_ratio", 0.6)),
                    "cell_range": tuple(override_template.get("cell_range", (16.0, 48.0))),
                }
            elif canonical_mode == "random_mask":
                override = {
                    "size_range": tuple(override_template.get("size_range", (0.2, 0.5))),
                }
                if "fill_value" in override_template:
                    override["fill_value"] = float(override_template["fill_value"])
            elif canonical_mode == "cutmix":
                override = {
                    "lambda_range": tuple(override_template.get("lambda_range", (0.3, 0.7))),
                    "size_range": tuple(override_template.get("size_range", (0.3, 0.6))),
                }
            elif canonical_mode == "max_mask":
                override = {"quantile": float(override_template.get("quantile", 0.9))}
            elif canonical_mode == "mz_shift":
                override = {"max_shift": int(override_template.get("max_shift", 5))}
            elif canonical_mode == "ldm":
                override = {
                    "noise_sigma_range": tuple(override_template.get("noise_sigma_range", (0.02, 0.06))),
                    "scale_range": tuple(override_template.get("scale_range", (0.85, 1.0))),
                    "blur_sigma_range": tuple(override_template.get("blur_sigma_range", (0.2, 0.9))),
                }
            elif canonical_mode == "elastic_transform":
                override = {
                    "alpha": float(override_template.get("alpha", 40.0)),
                    "sigma": float(override_template.get("sigma", 6.0)),
                }
            elif canonical_mode == "grid_distortion":
                override = {
                    "num_steps": int(override_template.get("num_steps", 5)),
                    "distort_limit": float(override_template.get("distort_limit", 0.3)),
                }
            elif canonical_mode == "thin_plate_spline":
                override = {
                    "num_ctrl_pts": int(override_template.get("num_ctrl_pts", 16)),
                    "strength": float(override_template.get("strength", 0.08)),
                }
            augmented = apply_train_augmentations(
                tensor.clone(),
                mode=canonical_mode,
                override_params=override,
                keep_probability=1.0,
            )
            augmented = _resize_to_target(augmented)
            cols[idx].image(
                _tensor_to_image(augmented),
                clamp=True,
                caption=f"{aug_labels.get(canonical_mode, canonical_mode)} (seed {seed})",
            )

    st.markdown("---")
    st.caption(
        "Images are normalized to the 99th percentile intensity before augmentation. "
        "Sample directory: `all_images/`. PROCESSED_DIR is available at "
        f"`{PROCESSED_DIR}` for advanced integrations."
    )


if __name__ == "__main__":
    main()
