# config.py
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "datasets"
LABELING_DIR = DATA_DIR / "labeling"
LABELING_CSV_DIR = LABELING_DIR / "csv"
METASPACE_CACHE_DIR = DATA_DIR / "metaspace_cache"
MODELS_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"

# Labels
LABELS = (
    "structured",
    "weak_structured",
    "localized",
    "fragmented",
    "unstructured",
    "negative",
)

# Target image size
TARGET_HW = (224, 224) # 224 x 224 for Resnet

# --- helpers for per-dataset cache placement ---
def dataset_id_from_imzml(imzml_path: Path) -> str:
    """
    Return dataset ID for a given imzML file based on the parent directory name.
    """
    return Path(imzml_path).parent.name

def ioncache_dir_for_imzml(imzml_path: Path) -> Path:
    """
    Return cache directory for a given imzML file based on the parent directory name.
    Always uses hotspot removal and the ppm provided by caller.
    No ppm/hotspot subfolders.
    """
    ds_dir = Path(imzml_path).parent
    cache_dir = ds_dir / "ioncache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

# Create directories if they don't exist
for directory in [PROCESSED_DIR, LABELING_CSV_DIR, METASPACE_CACHE_DIR, MODELS_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
