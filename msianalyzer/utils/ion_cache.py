# msianalyzer/utils/ion_cache.py
from __future__ import annotations
import json, hashlib, time
from pathlib import Path
from typing import Sequence, Tuple, Dict, Optional, Union
import numpy as np

from msianalyzer.utils import msi_utils
from msianalyzer.config import ioncache_dir_for_imzml

def _hash_mz_list(mz_list: Sequence[float]) -> str:
    s = ",".join(f"{float(x):.9f}" for x in mz_list).encode()
    return hashlib.sha1(s).hexdigest()[:12]

class IonImageCache:
    """
    Ion image cache that can be used with or without m2aia reader.
    When reader is provided: builds cache on-demand or from existing files.
    When reader is None: loads from existing cache files only.
    """
    def __init__(
        self,
        imzml_path: Path,
        reader: Optional[object] = None,  # m2aia reader (optional for standalone loading)
        mz_list: Optional[Sequence[float]] = None,  # optional for standalone loading
        ppm: Optional[float] = None,  # optional for standalone loading
        *,
        extract_kwargs: Optional[dict] = None,
        dtype=np.float32,
        cache_key: Optional[str] = None,  # optional custom cache key
    ):
        self.imzml_path = Path(imzml_path)
        self.reader = reader
        self.dtype = dtype

        # Cache directory and file paths
        self.cache_dir = ioncache_dir_for_imzml(self.imzml_path)

        # If loading from existing cache, we need cache_key or parameters to find the right files
        if reader is None:
            if cache_key is None:
                if mz_list is None or ppm is None:
                    raise ValueError("Must provide cache_key or (mz_list, ppm) for standalone loading")
                key_payload = {"ppm": ppm, "mz": mz_list, "kwargs": extract_kwargs or {}}
                cache_key = hashlib.sha1(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

            self.meta_path = self.cache_dir / f"mz-{cache_key}.json"
            self.data_path = self.cache_dir / f"mz-{cache_key}.dat"

            if self.meta_path.exists() and self.data_path.exists():
                self._load_standalone(mz_list, ppm, extract_kwargs)
            else:
                # NEW: try to find any compatible superset cache and subset it
                found = self._find_compatible_cache(mz_list, ppm, extract_kwargs)
                if not found:
                    raise FileNotFoundError(
                        f"No exact cache ({self.meta_path.name}) and no compatible superset cache found in {self.cache_dir}"
                    )
                meta_path, data_path, meta = found
                self.meta_path = meta_path
                self.data_path = data_path
                self._load_subset_from_meta(meta, self.data_path, mz_list)

        else:
            # Building/loading cache with reader
            if mz_list is None or ppm is None:
                raise ValueError("Must provide mz_list and ppm when using reader")

            self.mz_list = [float(m) for m in mz_list]
            self.ppm = float(ppm)
            self.extract_kwargs = dict(extract_kwargs or {})
            self.mz2idx: Dict[float, int] = {round(m, 9): i for i, m in enumerate(self.mz_list)}

            key_payload = {"ppm": self.ppm, "mz": self.mz_list, "kwargs": self.extract_kwargs}
            key = hashlib.sha1(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
            self.meta_path = self.cache_dir / f"mz-{key}.json"
            self.data_path = self.cache_dir / f"mz-{key}.dat"

            self._arr = None
            self._shape: Optional[Tuple[int, int, int]] = None
            self._ensure_ready()

    

    def _load_standalone(self, mz_list: Sequence[float], ppm: float, extract_kwargs: Optional[dict]):
        """Load cache from disk without requiring m2aia reader."""
        if not (self.meta_path.exists() and self.data_path.exists()):
            raise FileNotFoundError(f"Cache files not found: {self.meta_path}, {self.data_path}")

        # Load metadata
        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load cache metadata: {e}")

        # Validate metadata
        saved_ppm = meta.get("ppm")
        saved_kwargs = meta.get("extract_kwargs", {})
        saved_mz_list = meta.get("mz", [])

        if saved_ppm != ppm:
            raise ValueError(f"PPM mismatch: cache has {saved_ppm}, requested {ppm}")
        if saved_kwargs != (extract_kwargs or {}):
            raise ValueError("Extract kwargs mismatch between cache and request")
        if saved_mz_list != list(mz_list):
            raise ValueError("M/Z list mismatch between cache and request")

        # Load the memory-mapped array
        try:
            shape = tuple(meta["shape"])
            self._arr = np.memmap(self.data_path, mode="r", dtype=self.dtype, shape=shape)
            self._shape = shape
            self.mz_list = [float(m) for m in saved_mz_list]
            self.ppm = saved_ppm
            self.extract_kwargs = saved_kwargs
            self.mz2idx = {round(m, 9): i for i, m in enumerate(self.mz_list)}
        except Exception as e:
            raise RuntimeError(f"Failed to load cache data: {e}")

    def _find_compatible_cache(self, mz_list, ppm, extract_kwargs):
        """Return (meta_path, data_path, meta) for a compatible existing cache, or None."""
        if not self.cache_dir.exists():
            return None
        for meta_file in sorted(self.cache_dir.glob("mz-*.json")):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            except Exception:
                continue
            if meta.get("ppm") != ppm:
                continue
            if meta.get("extract_kwargs", {}) != (extract_kwargs or {}):
                continue
            saved_mz = meta.get("mz", [])
            # Saved must be a **superset** of requested set
            saved_set = {round(float(m), 9) for m in saved_mz}
            req_set   = {round(float(m), 9) for m in (mz_list or [])}
            if not req_set.issubset(saved_set):
                continue
            data_file = meta_file.with_suffix(".dat")
            if data_file.exists():
                return (meta_file, data_file, meta)
        return None

    def _load_subset_from_meta(self, meta, data_path, mz_list):
        """Load memmap from an existing superset cache and subset to requested mz_list."""
        shape = tuple(meta["shape"])
        arr = np.memmap(data_path, mode="r", dtype=self.dtype, shape=shape)
        saved_mz = [float(m) for m in meta["mz"]]
        # map saved m/z to index
        saved_idx = {round(m, 9): i for i, m in enumerate(saved_mz)}
        # build index list in requested order
        idxs = [saved_idx[round(float(m), 9)] for m in mz_list]
        # create a view that indexes the first dimension lazily
        # materialize as a memmap-backed array slice via np.asarray per access
        self._arr = arr  # keep original memmap
        self._shape = (len(idxs), shape[1], shape[2])
        self.mz_list = [float(m) for m in mz_list]
        self.ppm = meta["ppm"]
        self.extract_kwargs = meta.get("extract_kwargs", {})
        # we’ll intercept get_by_index to use idxs
        self._subset_indices = idxs
        self.mz2idx = {round(m, 9): i for i, m in enumerate(self.mz_list)}

    def _imzml_mtime(self) -> float:
        try:
            return self.imzml_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _try_load_existing(self) -> bool:
        if not (self.meta_path.exists() and self.data_path.exists()):
            return False

        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False

        # Check if cache is still valid
        if abs(meta.get("imzml_mtime", 0.0) - self._imzml_mtime()) > 1e-6:
            return False
        if meta.get("ppm") != self.ppm or meta.get("extract_kwargs", {}) != self.extract_kwargs:
            return False

        try:
            shape = tuple(meta["shape"])
            self._arr = np.memmap(self.data_path, mode="r", dtype=self.dtype, shape=shape)
            self._shape = shape
            return True
        except Exception:
            return False

    def _build(self):
        """Build cache by extracting all ion images."""
        print(f"Building cache for {len(self.mz_list)} m/z values...")

        # Probe first image for H,W
        img0 = msi_utils.extract_ion_image(
            self.reader, self.mz_list[0], self.ppm,
            hotspot_removal=True, **self.extract_kwargs
        )
        H, W = int(img0.shape[0]), int(img0.shape[1])
        N = len(self.mz_list)
        self._shape = (N, H, W)

        print(f"Cache shape: {self._shape} ({self._shape[0] * self._shape[1] * self._shape[2] * 4 / 1024**3:.1f} GB)")

        # Create memory-mapped array
        self._arr = np.memmap(self.data_path, mode="w+", dtype=self.dtype, shape=self._shape)

        # Extract and cache all images
        self._arr[0, :, :] = img0.astype(self.dtype, copy=False)
        for i, mz in enumerate(self.mz_list[1:], start=1):
            if i % 10 == 0:
                print(f"  Caching image {i+1}/{N} (m/z: {mz:.6f})")
            img = msi_utils.extract_ion_image(
                self.reader, mz, self.ppm,
                hotspot_removal=True, **self.extract_kwargs
            )
            self._arr[i, :, :] = img.astype(self.dtype, copy=False)

        # Flush to disk
        self._arr.flush()

        # Save metadata
        meta = {
            "shape": list(self._shape),
            "ppm": self.ppm,
            "extract_kwargs": self.extract_kwargs,
            "mz_list": self.mz_list,
            "mz_hash": _hash_mz_list(self.mz_list),
            "created": time.time(),
            "imzml_path": str(self.imzml_path),
            "imzml_mtime": self._imzml_mtime(),
            "cache_version": "1.0"
        }
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Cache saved to {self.data_path}")

    def _ensure_ready(self):
        if self._arr is not None:
            return
        if not self._try_load_existing():
            self._build()

    def get_by_index(self, i: int) -> np.ndarray:
        if self._arr is None:
            raise RuntimeError("Cache not loaded. Use reader=None for standalone loading.")
        if 0 <= i < len(self.mz_list):
            src_i = getattr(self, "_subset_indices", None)
            j = src_i[i] if src_i is not None else i
            return np.asarray(self._arr[j, :, :])
        else:
            raise IndexError(f"Index {i} out of range for mz_list of length {len(self.mz_list)}")

    def get_by_mz(self, mz: float) -> np.ndarray:
        """Get ion image by m/z value."""
        if self._arr is None:
            raise RuntimeError("Cache not loaded. Use reader=None for standalone loading.")

        idx = self.mz2idx[round(float(mz), 9)]
        return self.get_by_index(idx)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the cache (N, H, W)."""
        if self._arr is None:
            raise RuntimeError("Cache not loaded. Use reader=None for standalone loading.")
        return self._shape

    def get_cache_info(self) -> Dict:
        """Get information about the cache."""
        if self._arr is None:
            raise RuntimeError("Cache not loaded. Use reader=None for standalone loading.")

        return {
            "shape": self._shape,
            "mz_count": len(self.mz_list),
            "ppm": self.ppm,
            "extract_kwargs": self.extract_kwargs,
            "data_path": str(self.data_path),
            "meta_path": str(self.meta_path),
            "memory_usage_gb": self._shape[0] * self._shape[1] * self._shape[2] * 4 / 1024**3 if self._shape else 0
        }
