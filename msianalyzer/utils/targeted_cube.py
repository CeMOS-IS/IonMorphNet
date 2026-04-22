"""Targeted spectral cubes that cache ppm-window ion images for specific m/z lists."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Sequence, Optional

import numpy as np
import m2aia as m2

from msianalyzer.utils import msi_utils

ROUND_DECIMALS = 9
CACHE_VERSION = "targeted_cube_v1"


def _hash_payload(payload: Dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(data).hexdigest()[:16]


def _unique_preserve_order(values: Sequence[float]) -> Sequence[float]:
    seen = set()
    ordered = []
    for val in values:
        key = round(float(val), ROUND_DECIMALS)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(float(val))
    return ordered


def remove_dense_spectral_cubes(imzml_path: Path) -> None:
    """Delete legacy dense spectral cube files to avoid mixing formats."""
    cube_dir = Path(imzml_path).parent / "spectral_cubes"
    stem = Path(imzml_path).stem
    candidates = [
        cube_dir / f"{stem}_cube.dat",
        cube_dir / f"{stem}_cube.json",
        cube_dir / f"{stem}_mz.npy",
    ]
    for file_path in candidates:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


class TargetedSpectralCube:
    """Cache ppm-window slices for a selected set of m/z values."""

    def __init__(
        self,
        imzml_path: Path,
        mz_list: Sequence[float],
        ppm: float,
        *,
        extract_kwargs: Optional[Dict] = None,
        reader: Optional[m2.ImzMLReader] = None,
        build_if_missing: bool = True,
    ) -> None:
        self.imzml_path = Path(imzml_path)
        self.ppm = float(ppm)
        self.extract_kwargs = dict(extract_kwargs or {})
        self.cache_dir = self.imzml_path.parent / "spectral_cubes"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        remove_dense_spectral_cubes(self.imzml_path)

        self._requested_mz = list(_unique_preserve_order(mz_list))
        self._reader = reader
        self._arr: Optional[np.memmap] = None
        self._shape: Optional[tuple[int, int, int]] = None
        self._mz_to_idx: Dict[float, int] = {}
        self.meta_path: Optional[Path] = None
        self.data_path: Optional[Path] = None
        self.mz_values: Sequence[float] = self._requested_mz
        self._build_if_missing = build_if_missing

        if not self._requested_mz:
            raise ValueError("TargetedSpectralCube requires at least one m/z value.")

        if self._try_load_existing():
            return

        if not self._build_if_missing:
            raise FileNotFoundError(f"No targeted cube available for {self.imzml_path.name} at ppm={self.ppm}")

        self._build()

    # ------------------------------------------------------------------
    def get_image(self, mz: float) -> np.ndarray:
        idx = self._mz_to_idx.get(round(float(mz), ROUND_DECIMALS))
        if idx is None:
            raise KeyError(f"m/z {mz} not cached in targeted cube for {self.imzml_path.name}")
        if self._arr is None:
            raise RuntimeError("Targeted cube not loaded.")
        return np.asarray(self._arr[idx])

    # ------------------------------------------------------------------
    def _try_load_existing(self) -> bool:
        requested_keys = {round(float(m), ROUND_DECIMALS) for m in self._requested_mz}
        for meta_path in sorted(self.cache_dir.glob("targeted-*.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            if meta.get("type") != "targeted_cube":
                continue
            if float(meta.get("ppm", -1)) != self.ppm:
                continue
            if meta.get("extract_kwargs", {}) != self.extract_kwargs:
                continue
            saved_mz = [float(m) for m in meta.get("mz", [])]
            saved_map = {round(float(m), ROUND_DECIMALS): idx for idx, m in enumerate(saved_mz)}
            if not requested_keys.issubset(saved_map.keys()):
                continue
            data_path = meta_path.with_suffix(".dat")
            if not data_path.exists():
                continue
            shape = tuple(meta.get("shape", []))
            if len(shape) != 3:
                continue
            arr = np.memmap(data_path, mode="r", dtype=np.float32, shape=shape)
            self._arr = arr
            self._shape = shape
            self.meta_path = meta_path
            self.data_path = data_path
            self.mz_values = saved_mz
            self._mz_to_idx = saved_map
            return True
        return False

    def _build(self) -> None:
        reader = self._reader or m2.ImzMLReader(str(self.imzml_path))
        first_img = msi_utils.extract_ion_image(
            reader,
            self._requested_mz[0],
            self.ppm,
            hotspot_removal=True,
            **self.extract_kwargs,
        )
        if first_img.ndim != 2:
            first_img = np.squeeze(first_img)
        height, width = first_img.shape
        n = len(self._requested_mz)
        data_shape = (n, height, width)

        payload = {
            "ppm": self.ppm,
            "mz": [float(m) for m in self._requested_mz],
            "kwargs": self.extract_kwargs,
        }
        key = _hash_payload(payload)
        self.meta_path = self.cache_dir / f"targeted-{key}.json"
        self.data_path = self.cache_dir / f"targeted-{key}.dat"

        arr = np.memmap(self.data_path, mode="w+", dtype=np.float32, shape=data_shape)
        arr[0] = first_img.astype(np.float32, copy=False)
        for idx, mz in enumerate(self._requested_mz[1:], start=1):
            img = msi_utils.extract_ion_image(
                reader,
                mz,
                self.ppm,
                hotspot_removal=True,
                **self.extract_kwargs,
            )
            if img.ndim != 2:
                img = np.squeeze(img)
            arr[idx] = img.astype(np.float32, copy=False)
        arr.flush()

        meta = {
            "type": "targeted_cube",
            "cache_version": CACHE_VERSION,
            "created": time.time(),
            "imzml_path": str(self.imzml_path),
            "ppm": self.ppm,
            "extract_kwargs": self.extract_kwargs,
            "shape": list(data_shape),
            "mz": [float(m) for m in self._requested_mz],
        }
        with open(self.meta_path, "w", encoding="utf-8") as f_out:
            json.dump(meta, f_out, indent=2)

        self._arr = arr
        self._shape = data_shape
        self.mz_values = list(self._requested_mz)
        self._mz_to_idx = {round(float(m), ROUND_DECIMALS): idx for idx, m in enumerate(self._requested_mz)}
        self._reader = None
