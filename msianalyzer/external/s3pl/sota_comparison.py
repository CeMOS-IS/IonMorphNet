import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Mean F1-Score over slices
datasets = ["Mouse glioblastoma", "Renal cell carcinoma", "Adenocarcinoma"]
results = [
    [0.4025, 0.10125, 0.431],
    [0.00125, 0.2375, 0.487],
    [0.24625, 0.29125, 0.298],
    [0.38125, 0.60625, 0.233],
    [0.53375, 0.5675, 0.622],
    [np.nan, np.nan, np.nan],  # placeholder for ours
]
markers = ["<", "*", "o", "s", "D", "p"]
labels = [
    "msiPL",
    "Lieb et al.",
    "MALDIquant",
    "SPUTNIK",
    "S3PL (Weigand et al.)",
    "Ours",
]

# F1-Score for GBM dataset
datasets_gbm = ['GBM12_1', 'GBM12_2', 'GBM22_1','GBM22_2', 'GBM39_1', 'GBM39_2', 'GBM108_pos', 'GBM108_neg']
results_gbm = [[0.25, 0.38, 0.41, 0.44, 0.33, 0.49, 0.4, 0.52],
           [0, 0, 0, 0, 0, 0, 0.01, 0],
           [0.14, 0.26, 0.24, 0.29, 0.22, 0.28, 0.24, 0.3],
           [0.2, 0.39, 0.37, 0.43, 0.29, 0.42, 0.37, 0.58],
           [0.27, 0.55, 0.47, 0.61, 0.38, 0.56, 0.47, 0.66],
           [np.nan] * 8]

# F1-Score for RCC dataset
datasets_rcc = ['MH0204_33', 'UH0505_12', 'UH0710_33', 'UH9610_15', 'UH9812_03', 'UH9905_18', 'UH9911_05', 'UH9912_01']
results_rcc = [[0.18, 0.14, 0.14, 0.13, 0.17, 0.15, 0.18, 0.18],
           [0.28, 0.12, 0.21, 0.02, 0.21, 0.16, 0.31, 0.28],
           [0.43, 0.32, 0.39, 0.19, 0.43, 0.36, 0.42, 0.43],
           [0.28, 0.37, 0.41, 0.26, 0.49, 0.34, 0.4, 0.43],
           [0.56, 0.44, 0.48, 0.14, 0.5, 0.51, 0.56, 0.57],
           [np.nan] * 8]

# F1-Score for Adenocarcinoma dataset by Inglese
datasets_adeno = ['40TopL','160TopL','200TopL','240TopL','280TopL','360TopL','400TopL','520TopL']
results_adeno = [[0.44, 0.43, 0.38, 0.48, 0.46, 0.41, 0.36, 0.52],
           [0.66, 0.48, 0.64, 0.61, 0.55, 0.54, 0.4, 0.69], #[0.44, 0.5, 0.59, 0.56, 0.41, 0.51, 0.4, 0.53],
           [0.35, 0.28, 0.33, 0.31, 0.3, 0.28, 0.21, 0.36],
           [0.27, 0.16, 0.16, 0.19, 0.27, 0.17, 0.42, 0.23],
           [0.68, 0.53, 0.61, 0.67, 0.63, 0.64, 0.35, 0.7],
           [np.nan] * 8]

OURS_ORDER = set(datasets_gbm + datasets_rcc + datasets_adeno)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SOTA comparison with optional 'Ours' overlay.")
    parser.add_argument("--ours-metrics", type=Path, default=None,
                        help="CSV exported by evaluate_s3pl_peak_quality.py containing columns 'dataset' and 'mSCF1'.")
    return parser.parse_args()


def load_ours_scores(csv_path: Path) -> Dict[str, float]:
    df = pd.read_csv(csv_path, index_col=False)
    if "filename" not in df.columns or "mSCF1" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'filename' and 'mSCF1' columns.")
    alias = {
        "GBM108_positive": "GBM108_pos",
        "GBM108_negative": "GBM108_neg",
    }
    mapping = {}
    for _, row in df.iterrows():
        key = str(row["filename"])
        key = alias.get(key, key)
        mapping[key] = float(row["mSCF1"])
    missing = OURS_ORDER - mapping.keys()
    if missing:
        raise ValueError(f"Ours metrics missing datasets: {sorted(missing)}")
    return mapping


def load_ours_scores_with_max(csv_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    df = pd.read_csv(csv_path, index_col=False)
    if "filename" not in df.columns or "mSCF1" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'filename' and 'mSCF1' columns.")
    alias = {
        "GBM108_positive": "GBM108_pos",
        "GBM108_negative": "GBM108_neg",
    }
    mapping: Dict[str, float] = {}
    max_mapping: Dict[str, float] = {}
    has_max = "max_mSCF1" in df.columns
    for _, row in df.iterrows():
        key = str(row["filename"])
        key = alias.get(key, key)
        mapping[key] = float(row["mSCF1"])
        if has_max:
            raw_max = row.get("max_mSCF1")
            max_mapping[key] = 1.0 if pd.isna(raw_max) else float(raw_max)
    if not has_max:
        max_mapping = {name: 1.0 for name in mapping}
    missing = OURS_ORDER - mapping.keys()
    if missing:
        raise ValueError(f"Ours metrics missing datasets: {sorted(missing)}")
    for name in OURS_ORDER:
        max_mapping.setdefault(name, 1.0)
    return mapping, max_mapping


def fill_series_from_map(mapping: Dict[str, float], names: List[str]) -> List[float]:
    return [float(mapping[name]) for name in names]


def update_with_ours(mapping: Dict[str, float]) -> None:
    ours_idx = len(results) - 1
    results_gbm[-1][:] = fill_series_from_map(mapping, datasets_gbm)
    results_rcc[-1][:] = fill_series_from_map(mapping, datasets_rcc)
    results_adeno[-1][:] = fill_series_from_map(mapping, datasets_adeno)
    gbm_mean = float(np.mean(results_gbm[-1]))
    rcc_mean = float(np.mean(results_rcc[-1]))
    adeno_mean = float(np.mean(results_adeno[-1]))
    results[ours_idx] = [gbm_mean, rcc_mean, adeno_mean]


def plot_sota(max_map: Optional[Dict[str, float]] = None):
    fig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(1, 4)
    ax1 = fig.add_subplot(gs[0, 1])
    for result, symbol, method in zip(results_gbm, markers, labels):
        ax1.scatter(np.arange(len(datasets_gbm)), result, marker=symbol, label=method)
    ax1.set_title('GBM')
    ax1.set_ylim(0,1)
    ax1.set_ylabel("mSCF1")
    ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax1.yaxis.grid()
    ax1.set_axisbelow(True)
    ax1.set_xticks(np.arange(len(datasets_gbm)))
    ax1.set_xticklabels(datasets_gbm, rotation=45, ha='right')
    _ = max_map

    ax2 = fig.add_subplot(gs[0, 2])
    for result, symbol, method in zip(results_rcc, markers, labels):
        ax2.scatter(np.arange(len(datasets_rcc)), result, marker=symbol)
    ax2.set_title('RCC')
    ax2.set_ylim(0,1)
    ax2.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax2.yaxis.grid()
    ax2.set_axisbelow(True)
    ax2.set_xticks(np.arange(len(datasets_rcc)))
    ax2.set_xticklabels(datasets_rcc, rotation=45, ha='right')
    ax3 = fig.add_subplot(gs[0, 3])
    for result, symbol, method in zip(results_adeno, markers, labels):
        ax3.scatter(np.arange(len(datasets_adeno)), result, marker=symbol)
    ax3.set_title('CAC')
    ax3.set_ylim(0,1)
    ax3.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax3.yaxis.grid()
    ax3.set_axisbelow(True)
    ax3.set_xticks(np.arange(len(datasets_adeno)))
    ax3.set_xticklabels(datasets_adeno, rotation=45, ha='right')
    axlegend = fig.add_subplot(gs[0, 0])
    axlegend.axis('off')
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(handles, legend_labels, bbox_to_anchor=(0.02, 0., 0.2, 0.85))
    fig.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.0)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    if args.ours_metrics:
        ours_map, max_map = load_ours_scores_with_max(args.ours_metrics)
        update_with_ours(ours_map)
    else:
        max_map = None
    plot_sota(max_map)
