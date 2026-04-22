"""Spectral cube utilities for MSI datasets.

Builds a memory-mapped 3D array storing intensities for every m/z across the
imaging grid, mirroring the dense representation used in the S3PL reference
implementation. The cube is persisted alongside lightweight metadata so
subsequent evaluations can reuse it without re-parsing the entire imzML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pyimzml.ImzMLParser import ImzMLParser


class SpectralCube:
    """Memory-mapped spectral cube for a single imzML dataset."""

    def __init__(
        self,
        imzml_path: Path,
        *,
        cache_dir: Optional[Path] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.imzml_path = Path(imzml_path)
        if cache_dir is None:
            cache_dir = self.imzml_path.parent / "spectral_cubes"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        stem = self.imzml_path.stem
        self.data_path = self.cache_dir / f"{stem}_cube.dat"
        self.meta_path = self.cache_dir / f"{stem}_cube.json"
        self.mz_path = self.cache_dir / f"{stem}_mz.npy"

        self.dtype = np.dtype(dtype)
        self._cube: Optional[np.memmap] = None
        self._mz_values: Optional[np.ndarray] = None

        if self.meta_path.exists() and self.data_path.exists() and self.mz_path.exists():
            self._load_metadata()
        else:
            self._build()
        self._load_cube()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def mz_values(self) -> np.ndarray:
        if self._mz_values is None:
            self._mz_values = np.load(self.mz_path)
        return self._mz_values

    @property
    def num_mz(self) -> int:
        return int(self._meta["num_mz"])

    @property
    def height(self) -> int:
        return int(self._meta["height"])

    @property
    def width(self) -> int:
        return int(self._meta["width"])

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.num_mz, self.height, self.width

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_image_by_index(self, idx: int) -> np.ndarray:
        cube = self._ensure_cube()
        if not (0 <= idx < self.num_mz):
            raise IndexError(f"m/z index {idx} out of range (0 <= idx < {self.num_mz})")
        return np.asarray(cube[idx])

    def get_image_by_mz(self, mz_value: float) -> np.ndarray:
        mz_axis = self.mz_values
        idx = int(np.searchsorted(mz_axis, mz_value))
        if idx >= len(mz_axis):
            idx = len(mz_axis) - 1
        return self.get_image_by_index(idx)

    def close(self) -> None:
        if self._cube is not None:
            del self._cube
            self._cube = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_cube(self) -> np.memmap:
        if self._cube is None:
            self._load_cube()
        return self._cube  # type: ignore[return-value]

    def _load_metadata(self) -> None:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)
        meta_dtype = self._meta.get("dtype")
        if meta_dtype is not None:
            self.dtype = np.dtype(meta_dtype)

    def _load_cube(self) -> None:
        self._cube = np.memmap(
            self.data_path,
            mode="r" if self.meta_path.exists() else "w+",
            dtype=self.dtype,
            shape=self.shape,
        )

    def _build(self) -> None:
        parser = ImzMLParser(str(self.imzml_path))
        mz_axis, intensities = parser.getspectrum(0)
        num_mz = len(mz_axis)

        # Determine spatial dimensions from coordinates (1-based indices)
        coords = list(parser.coordinates)
        xs, ys = [], []
        for x, y, _ in coords:
            xs.append(int(x) - 1)
            ys.append(int(y) - 1)
        width = max(xs) + 1
        height = max(ys) + 1

        print(
            f"Building spectral cube for {self.imzml_path.name}: "
            f"{len(coords)} spectra, {num_mz} m/z ({height}x{width})."
        )

        cube = np.memmap(
            self.data_path,
            mode="w+",
            dtype=self.dtype,
            shape=(num_mz, height, width),
        )
        cube[:] = 0

        # Populate cube per spectrum (fast vectorised assignment)
        for idx, (x, y, _) in enumerate(coords):
            _, intensities = parser.getspectrum(idx)
            cube[:, int(y) - 1, int(x) - 1] = np.asarray(intensities, dtype=self.dtype, copy=False)
            # if (idx + 1) % 500 == 0:
            #     print(f"  Added spectra {idx + 1}/{len(coords)}")

        cube.flush()
        del cube

        np.save(self.mz_path, np.asarray(mz_axis, dtype=np.float64))

        self._meta = {
            "imzml_path": str(self.imzml_path),
            "num_mz": num_mz,
            "height": height,
            "width": width,
            "dtype": str(self.dtype),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)
