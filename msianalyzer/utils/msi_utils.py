# msi_utils.py
import numpy as np
from metaspace import SMInstance, image_processing

# For saving and processing metadata 
import pandas as pd 
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# For opening Metaspace.org for comparison
import webbrowser

import datetime

def get_mz_bounds(reader):
    """Return (min_mz, max_mz) for an ImzMLReader."""
    mz_axis = reader.GetXAxis()
    return float(mz_axis[0]), float(mz_axis[-1])

def extract_ion_image(reader, mz_value, ppm, hotspot_removal=True, mz_bounds=None):
    """
    Extracts a 2D ion image using m2aia.ImzMLReader.GetArray().
    This implementation aims to match METASPACE's behavior for consistency.

    Parameters:
        reader: m2aia.ImzMLReader
            ImzMLReader instance
        mz_value: float
            Center m/z value
        ppm: float
            Parts per million tolerance for m/z value
        hotspot_removal: bool
            Whether to remove hotspots (default: True)

    Returns:
        numpy.ndarray: 2D numpy array representing the ion image
    """
    # Calculate m/z window (symmetric around mz_value)
    delta_mz = mz_value * ppm * 1e-6
    min_mz = mz_value - delta_mz
    max_mz = mz_value + delta_mz
    
    if mz_bounds is not None:
        min_mz = max(min_mz, mz_bounds[0])
        max_mz = min(max_mz, mz_bounds[1])
    # Get the ion image using the calculated window
    # Note: m2aia's GetArray expects center m/z and half-window
    half_window = (max_mz - min_mz) / 2
    center_mz = (min_mz + max_mz) / 2
    ion_image = reader.GetArray(center_mz, half_window)
    try:
        ion_image = np.squeeze(ion_image)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    # DEBUG: previous_max = np.max(ion_image) # for debugging purposes
    if hotspot_removal and ion_image.size > 0:
        try:
            ion_image = image_processing.clip_hotspots(ion_image)
            # DEBUG: threshold = np.max(ion_image)
            # DEBUG: print(f"Hotspot removal applied: previous maximum: {previous_max}, threshold: {threshold}")
        except Exception as e:
            print(f"Warning during hotspot removal: {str(e)}")
    
    return ion_image

def to_target_size(img: np.ndarray, target_hw: Tuple[int, int], mode: str = "pad") -> np.ndarray:
    """
    Resize or pad an image to the target height and width.

    Parameters:
        img: numpy.ndarray
            Input image to resize or pad
        target_hw: Tuple[int, int]
            Target height and width (H, W)
        mode: str
            "pad" or "resize"

    Returns:
        numpy.ndarray: Resized or padded image
    """
    Ht, Wt = target_hw
    h, w = img.shape[:2]
    if mode == "resize":
        return cv2.resize(img, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((Ht, Wt), dtype=img.dtype)
    y0 = max((Ht - h)//2, 0); x0 = max((Wt - w)//2, 0)
    y1 = y0 + min(h, Ht);     x1 = x0 + min(w, Wt)
    src_y0 = max((h - Ht)//2, 0); src_x0 = max((w - Wt)//2, 0)
    src_y1 = src_y0 + (y1 - y0);  src_x1 = src_x0 + (x1 - x0)
    out[y0:y1, x0:x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out

def get_metaspace_id_from_imzml(imzml_path: Union[str, Path]) -> str:
        """
        Extract METASPACE ID from the imzML filename.
        
        Args:
            imzml_path: Path to the imzML file
            
        Returns:
            The METASPACE ID as a string, or empty string if not found.
        """
        try:
            filename = Path(imzml_path).name
            # METASPACE dataset IDs typically look like: 2025-05-26_08h35m04s
            pattern = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}h\d{2}m\d{2}s")
            matches = pattern.findall(filename)
            if matches:
                # If multiple matches exist, prefer the last (most specific) one.
                return matches[-1]
        except Exception as e:
            print(f"Could not read METASPACE ID from filename {imzml_path}: {e}")
        return ""

def get_metaspace_dataset_from_API(datasetid):

    """
    Retrieves a Metaspace dataset object from the API

    Parameters:
        datasetid : string
            Identifier of the dataset on METASPACE (e.g., "2025-05-26_08h35m04s").

    Returns:
        ds_metaspace : Metaspace Dataset object
    """
    sm = SMInstance()
    ds_metaspace = sm.dataset(id=datasetid)
    return ds_metaspace

# Directory to store cached API responses
from msianalyzer.config import (
    METASPACE_CACHE_DIR,
)

def get_cache_path(dataset_id: str) -> Path:
    """Get the cache file path for a dataset."""
    return METASPACE_CACHE_DIR / f"{dataset_id}.json"

def save_to_cache(dataset_id: str, data: dict) -> None:
    """Save dataset to cache."""
    cache_path = get_cache_path(dataset_id)
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_from_cache(dataset_id: str) -> Optional[dict]:
    """Load dataset from cache if it exists."""
    cache_path = get_cache_path(dataset_id)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

def get_metaspace_data(dataset_id: str, use_cache: bool = True) -> Tuple[Optional[dict], bool]:
    """
    Get METASPACE dataset, using cache if available.
    
    Args:
        dataset_id: The ID of the dataset to download
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (dataset_data, from_cache)
        dataset_data contains:
        - id: dataset ID
        - database_details: list of database details
        - config: dataset configuration including ppm
        - metadata: additional dataset metadata
        - results: placeholder for results
    """
    # Try to load from cache first
    if use_cache:
        cached_data = load_from_cache(dataset_id)
        if cached_data is not None:
            print(f"Loaded dataset {dataset_id} from cache")
            return cached_data, True
    
    # If not in cache or cache disabled, fetch from API
    print(f"Downloading dataset {dataset_id} from METASPACE API...")
    ds_metaspace = get_metaspace_dataset_from_API(dataset_id)
    if ds_metaspace is None:
        print(f"Error: Could not load dataset {dataset_id}")
        return None, False
    
    # Get dataset metadata and config
    def safe_get_attr(obj, attr, default=None):
        """Safely get attribute or dict value with nested access support"""
        if obj is None:
            return default
            
        # Handle nested attributes (e.g., 'submitter.email')
        if '.' in attr:
            parts = attr.split('.')
            current = obj
            for part in parts:
                if current is None:
                    return default
                if hasattr(current, part):
                    current = getattr(current, part, None)
                elif isinstance(current, dict) and part in current:
                    current = current.get(part)
                else:
                    return default
            return current if current is not None else default
            
        # Handle direct attribute access
        if hasattr(obj, attr):
            val = getattr(obj, attr, default)
            return val if val is not None else default
        elif isinstance(obj, dict) and attr in obj:
            val = obj.get(attr, default)
            return val if val is not None else default
        return default
    
    
    # Add metadata from get_metadata() if available
    try:
        dataset_metadata = ds_metaspace.metadata
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
    
    # Get config with fallback to empty dict
    config = getattr(ds_metaspace, 'config', {}) or {}
    
    # Convert database details to a serializable format
    database_details = []
    for db in ds_metaspace.database_details:
        try:
            # Extract database details from the MolecularDB object
            db_dict = {
                'name': str(db.name if hasattr(db, 'name') else ''),
                'version': str(db.version if hasattr(db, 'version') else ''),
                'id': str(db.id if hasattr(db, 'id') else ''),
                'is_public': bool(db.is_public if hasattr(db, 'is_public') else False),
                'archived': bool(db.archived if hasattr(db, 'archived') else False)
            }
            database_details.append(db_dict)
        except Exception as e:
            print(f"Warning: Could not process database details: {e}")
            continue

    # Prepare the data dictionary with all required fields
    data = {
        'id': dataset_id,
        'database_details': database_details,
        'config': config,
        'metadata': dataset_metadata,
        'results': {},
        'timestamp_data_retrieved': datetime.datetime.now().isoformat()
    }
    
    # Save to cache
    save_to_cache(dataset_id, data)
    
    return data, False

def process_metaspace_dataset(dataset_id: str, use_cache: bool = True) -> Optional[Dict[float, Dict[str, Any]]]:
    """
    Process a METASPACE dataset and extract annotations from all databases.
    
    Args:
        dataset_id: The ID of the dataset to process
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary containing merged annotation data by m/z value or None if processing fails
    """
    # Get dataset data (from cache or API)
    data, from_cache = get_metaspace_data(dataset_id, use_cache=use_cache)
    if data is None:
        return None

    isotope_config = (data.get('config') or {}).get('isotope_generation') or {}
    include_chem_mods = bool(isotope_config.get('chem_mods'))
    include_neutral_losses = bool(isotope_config.get('neutral_losses'))
    expected_cache_flags = {
        'include_chem_mods': include_chem_mods,
        'include_neutral_losses': include_neutral_losses,
    }

    flag_mismatch = False
    if from_cache:
        cache_flags = data.get('_cache_flags', {})
        flag_mismatch = any(cache_flags.get(flag, False) != required for flag, required in expected_cache_flags.items())
        if flag_mismatch:
            print("Refreshing cached dataset to include chemical modifications/neutral losses annotations...")
            refreshed_data, refreshed_from_cache = get_metaspace_data(dataset_id, use_cache=False)
            if refreshed_data is None:
                print(f"Warning: Unable to refresh dataset {dataset_id} from METASPACE. "
                      "Continuing with cached results, which may omit chemical modifications or neutral losses.")
            else:
                data = refreshed_data
                from_cache = refreshed_from_cache
                isotope_config = (data.get('config') or {}).get('isotope_generation') or {}
                include_chem_mods = bool(isotope_config.get('chem_mods'))
                include_neutral_losses = bool(isotope_config.get('neutral_losses'))
                expected_cache_flags = {
                    'include_chem_mods': include_chem_mods,
                    'include_neutral_losses': include_neutral_losses,
                }
        
    databases = data['database_details']
    print("\nDatabases found:", [f"{db['name']} {db['version']}" for db in databases])

    # Dictionary to store merged data by m/z value
    merged_data = {}
    
    # Get dataset object for API calls if needed
    ds_metaspace = None if from_cache else get_metaspace_dataset_from_API(dataset_id)
    if ds_metaspace is None and not from_cache:
        print(f"Error: Could not load dataset {dataset_id}")
        return None
        
    # Function to get MSM value from a result row (default to 0 if not available)
    def get_msm(row):
        try:
            return float(row.get('msm', 0))
        except (ValueError, TypeError):
            return 0

    # Process all available databases
    for db_info in databases:
        db_name = db_info['name']
        db_version = db_info['version']
        db_id = f"{db_name} {db_version}"
        
        print(f"\nProcessing database: {db_id}")
        
        try:
            # Check if we have cached results for this database
            db_key = f"{db_name}_{db_version}"
            if db_key in data['results'] and from_cache:
                print(f"Using cached results for {db_name} (version: {db_version})")
                results = pd.DataFrame(data['results'][db_key])
            else:
                # Get results from API
                print(f"Querying database: {db_name} (version: {db_version})"
                      f"{' with chemical modifications/neutral losses' if include_chem_mods or include_neutral_losses else ''}")
                if ds_metaspace is None:
                    print(f"Error: No dataset connection available to query {db_name}")
                    continue
                    
                try:
                    results = ds_metaspace.results(
                        database=(db_name, db_version),
                        include_chem_mods=include_chem_mods,
                        include_neutral_losses=include_neutral_losses
                    )
                    if results is None:
                        print(f"No results returned for {db_name} {db_version}")
                        continue
                        
                    # Convert results to a list of dicts for caching
                    results_for_cache = results.reset_index().to_dict('records')
                    
                    # Update the data dictionary with new results
                    if 'results' not in data:
                        data['results'] = {}
                    data['results'][db_key] = results_for_cache
                    data.setdefault('_cache_flags', {}).update(expected_cache_flags)
                    
                    # Save to cache if not already loading from cache
                    if not from_cache:
                        save_to_cache(dataset_id, data)
                        
                except Exception as e:
                    print(f"Error querying {db_name} {db_version}: {str(e)}")
                    continue
            print(f"Found {len(results)} results for {db_name}")
            
            if len(results) == 0:
                print(f"No results found for {db_name}")
                continue
            
            if 'fdr' not in results.columns:
                print(f"⚠️ Skipping {db_name} {db_version}: no FDR values")
                continue

            # Process each result, sorted by MSM value (highest first)
            try:
                # Ensure we have the required columns
                if 'mz' not in results.columns:
                    print(f"Warning: 'mz' column not found in results for {db_name}")
                    continue
                
                # Sort results by MSM value (highest first)
                results_sorted = results.copy()
                results_sorted['_msm'] = results_sorted.apply(get_msm, axis=1)
                results_sorted = results_sorted.sort_values('_msm', ascending=False)
                
                for _, row in results_sorted.iterrows():
                    try:
                        mz = row['mz']
                        fdr = row['fdr'] if 'fdr' in row and pd.notnull(row['fdr']) else 1.0

                        msm = get_msm(row)
                        
                        # Ensure mz is a float
                        try:
                            mz = float(mz)
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid mz value: {mz}")
                            continue
                            
                        # Use rounded m/z as key for grouping
                        mz_key = round(mz, 9)
                        
                        # Initialize entry if it doesn't exist
                        if mz_key not in merged_data:
                            merged_data[mz_key] = {
                                'mz': mz_key,
                                'databases': set(),
                                'fdr_values': [],
                                'msm': msm  # Store MSM value for sorting
                            }
                        # Update MSM if this is a better match
                        elif msm > merged_data[mz_key].get('msm', 0):
                            merged_data[mz_key]['msm'] = msm
                        
                        # Add database and FDR value
                        merged_data[mz_key]['databases'].add(db_id)
                        merged_data[mz_key]['fdr_values'].append(fdr)
                        
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing results for {db_name}: {e}")
                continue
                    
        except Exception as e:
            print(f"Error processing {db_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Get ppm from dataset config or cache
    if 'config' in data and 'image_generation' in data['config'] and 'ppm' in data['config']['image_generation']:
        ppm = data['config']['image_generation']['ppm']
        print(f"Using ppm from cache: {ppm}")
    elif ds_metaspace and hasattr(ds_metaspace, 'config'):
        try:
            ppm = ds_metaspace.config.get('ppm', None)
            print(f"Using ppm from dataset config: {ppm}")
        except Exception as e:
            print(f"Could not extract ppm from config: {e}")
    
    if ppm is None:
        #print("Warning: No ppm found in dataset config. Using default ppm=3.")
        ppm = 3
    merged_data['_ppm'] = ppm

    # Update cache flags if we populated results from cache data lacking them
    if from_cache and data.get('results') and not flag_mismatch:
        cache_flags = data.setdefault('_cache_flags', {})
        updated = False
        for flag, required in expected_cache_flags.items():
            if cache_flags.get(flag, False) != required:
                cache_flags[flag] = required
                updated = True
        if updated:
            try:
                save_to_cache(dataset_id, data)
            except Exception as e:
                print(f"Warning: Could not update cache flags for {dataset_id}: {e}")

    return merged_data

def get_ppm_from_cache_only(dataset_id: str) -> Optional[float]:
    """
    Get ppm from cache only.
    
    Args:
        dataset_id: The ID of the dataset to download (imzML file name without file extension)
        
    Returns:
        ppm: The ppm value from the cache or None if not found
    """        
    cache_path = get_cache_path(dataset_id)
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            data = json.load(f)
            if 'config' in data and 'image_generation' in data['config'] and 'ppm' in data['config']['image_generation']:
                return data['config']['image_generation']['ppm']
    else:
        pass
        #print(f"Cache not found for dataset {dataset_id}, set to standard 3 ppm\nSearched {cache_path}")
    return 3

def save_to_csv(merged_data, output_file='metaspace_merged_annotations.csv') -> pd.DataFrame:
    """
    Save merged data to a CSV file.
    
    Args:
        merged_data: Dictionary containing merged annotation data by m/z value
        output_file: Path to save the output CSV file
        
    Returns:
        DataFrame containing the merged data or None if no valid data
    """
    if not merged_data:
        print("No data to save!")
        return None
        
    # Convert the merged data to a list of dictionaries
    rows = []
    for mz_key, data in merged_data.items():
        try:
            # Get m/z value (use the original value if available, otherwise use the key)
            mz = data.get('mz', mz_key)
            
            # Get FDR values, ensuring they're floats
            fdr_values = []
            for f in data.get('fdr_values', []):
                try:
                    fdr_values.append(float(f))
                except (ValueError, TypeError):
                    continue
            
            # Get databases, ensuring they're strings
            databases = sorted(str(db) for db in data.get('databases', []))
            
            if not databases:  # Skip entries with no databases
                continue
                
            rows.append({
                'mz': mz,
                'databases': '|'.join(databases),
                'fdr_values': '|'.join(f"{f:.6f}" for f in fdr_values) if fdr_values else '',
                'num_databases': len(databases),
                'min_fdr': min(fdr_values) if fdr_values else None,
                'max_fdr': max(fdr_values) if fdr_values else None
            })
        except Exception as e:
            print(f"Error processing m/z={mz_key}: {e}")
            continue
    
    if not rows:
        print("No valid data to save!")
        return None
    
    # Create DataFrame and sort by m/z
    df = pd.DataFrame(rows).sort_values('mz')
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.9f')
    
    print(f"\nResults saved to {output_path.absolute()}")
    print(f"Total unique m/z values: {len(df)}")
    print("\nFirst few rows:")
    print(df.head().to_string())
    
    return df

# Open Metaspace.org in default browser for comparison
# the direct link is made up out of the dataset id, the database id and the fdr value
# default FDR value is 0.1, default database are all of them combined
def open_metaspace(dataset_id, directlink=False, fdr=None, database_id=None):
    if directlink:
        if fdr is None:
            webbrowser.open(f"https://metaspace2020.org/dataset/{dataset_id}/annotations?ds={dataset_id}")
        else:
            if database_id is None:
                webbrowser.open(f"https://metaspace2020.org/dataset/{dataset_id}/annotations?ds={dataset_id}&fdr={fdr}")
            else:
                webbrowser.open(f"https://metaspace2020.org/dataset/{dataset_id}/annotations?db={database_id}&ds={dataset_id}&fdr={fdr}")
    else:
        webbrowser.open(f"https://metaspace2020.org/dataset/{dataset_id}")

if __name__ == "__main__":
    import sys
    print("Error: This script is not meant to be run directly. Use label_workflow_gui.py instead.")
    sys.exit(1)

def format_metadata_for_display(metadata: dict) -> list:
    """
    Format dataset metadata into a list of strings for display.
    
    Args:
        metadata: Dictionary containing dataset metadata
        
    Returns:
        List of formatted strings with metadata information
    """
    if not metadata:
        return ["No metadata available"]
    
    lines = []
    
    # Add basic metadata
    if 'Sample_Information' in metadata:
        sample_info = metadata['Sample_Information']
        lines.append("Sample Information:")
        lines.append(f"  • Organism: {sample_info.get('Organism', 'N/A')}")
        lines.append(f"  • Organism Part: {sample_info.get('Organism_Part', 'N/A')}")
        lines.append(f"  • Sample Growth Condition: {sample_info.get('Sample_Growth_Condition', 'N/A')}")
        lines.append(f"  • Condition: {sample_info.get('Condition', 'N/A')}")
        lines.append("")

    if 'MS_Analysis' in metadata:
        ms_analysis = metadata['MS_Analysis']
        lines.append("MS Analysis:")
        lines.append(f"  • Polarity: {ms_analysis.get('Polarity', 'N/A')}")
        lines.append(f"  • Ionization: {ms_analysis.get('Ionisation_Source', 'N/A')}")
        lines.append(f"  • Analyzer: {ms_analysis.get('Analyzer', 'N/A')}")
        lines.append("")
    
    if 'Sample_Preparation' in metadata:
        sample_prep = metadata['Sample_Preparation']
        lines.append("Sample Preparation:")
        lines.append(f"  • State: {sample_prep.get('Sample_State', 'N/A')}")
        lines.append(f"  • Tissue: {sample_prep.get('Tissue_Modification', 'N/A')}")
        lines.append("")
    
    if 'Instrument' in metadata:
        instrument = metadata['Instrument']
        lines.append("Instrument:")
        lines.append(f"  • {instrument.get('Manufacturer', 'N/A')} {instrument.get('Model', 'N/A')}")
        lines.append("")
    
    return lines
