#!/usr/bin/env python3
"""
Create an overview figure with a mean spectrum and a grid of ion images.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (
    ConnectionPatch,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
)  # noqa: E402
import numpy as np  # noqa: E402

import m2aia  # noqa: E402

from msianalyzer.utils import msi_utils  # noqa: E402


def parse_float_list(raw: str) -> List[float]:
    if not raw:
        return []
    items = [item for item in re.split(r"[,\s]+", raw.strip()) if item]
    return [float(item) for item in items]


def parse_str_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item for item in re.split(r"[,\s]+", raw.strip()) if item]


def load_table(table_path: Path, max_ions: Optional[int]) -> Tuple[List[float], List[str], List[float]]:
    mz_values: List[float] = []
    labels: List[str] = []
    scores: List[float] = []
    with table_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "mz" not in row:
                continue
            try:
                mz = float(row["mz"])
            except (TypeError, ValueError):
                continue
            label = (row.get("label") or "").strip()
            score_raw = row.get("score")
            try:
                score = float(score_raw) if score_raw not in ("", None) else float("nan")
            except (TypeError, ValueError):
                score = float("nan")
            mz_values.append(mz)
            labels.append(label)
            scores.append(score)
            if max_ions is not None and len(mz_values) >= max_ions:
                break
    return mz_values, labels, scores


def get_num_spectra(reader: m2aia.ImzMLReader) -> Optional[int]:
    for attr in ("GetNumberOfSpectra", "GetNumSpectra", "GetSpectraCount", "GetSpectrumCount"):
        if hasattr(reader, attr):
            try:
                return int(getattr(reader, attr)())
            except Exception:
                continue
    return None


def get_mz_axis(reader: m2aia.ImzMLReader) -> np.ndarray:
    if hasattr(reader, "GetXAxis"):
        return np.asarray(reader.GetXAxis())
    if hasattr(reader, "GetSpectrum"):
        mz_axis, _ = reader.GetSpectrum(0)
        return np.asarray(mz_axis)
    raise RuntimeError("Unable to read m/z axis from the imzML reader.")


def get_mean_spectrum(
    reader: m2aia.ImzMLReader,
    sample_spectra: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(reader, "GetMeanSpectrum"):
        mean_spectrum = reader.GetMeanSpectrum()
        if isinstance(mean_spectrum, tuple) and len(mean_spectrum) == 2:
            mz_axis, mean_intensity = mean_spectrum
            return np.asarray(mz_axis), np.asarray(mean_intensity)
        mz_axis = get_mz_axis(reader)
        mean_arr = np.asarray(mean_spectrum)
        min_len = min(mz_axis.shape[0], mean_arr.shape[0])
        return mz_axis[:min_len], mean_arr[:min_len]

    mz_axis = get_mz_axis(reader)
    num_spectra = get_num_spectra(reader)
    if not num_spectra:
        return mz_axis, np.zeros_like(mz_axis, dtype=np.float64)

    indices = np.arange(num_spectra)
    if sample_spectra is not None and sample_spectra < num_spectra:
        indices = np.linspace(0, num_spectra - 1, sample_spectra, dtype=int)

    mean_acc: Optional[np.ndarray] = None
    for idx in indices:
        mz, intensity = reader.GetSpectrum(int(idx))
        intensity = np.asarray(intensity, dtype=np.float64)
        if mean_acc is None:
            mz_axis = np.asarray(mz)
            mean_acc = np.zeros_like(intensity, dtype=np.float64)
        if intensity.shape[0] != mean_acc.shape[0]:
            min_len = min(mean_acc.shape[0], intensity.shape[0])
            mean_acc = mean_acc[:min_len]
            mz_axis = mz_axis[:min_len]
            intensity = intensity[:min_len]
        mean_acc += intensity

    if mean_acc is None:
        return mz_axis, np.zeros_like(mz_axis, dtype=np.float64)
    mean_acc /= max(len(indices), 1)
    return mz_axis, mean_acc


def format_panel_caption(mz_value: float, label: str, score: float) -> str:
    label_text = label.strip() if label.strip() else "n/a"
    return f"m/z={mz_value:.4f}\npred: {label_text}"


def make_msi_cube_pipeline_figure(
    out_path: Path,
    title: Optional[str] = None,
    ion_images: Optional[Sequence[np.ndarray]] = None,
    max_images: int = 3,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    box_edge = "#555555"
    arrow_color = "#333333"
    box_lw = 1.4
    arrow_lw = 1.2

    # Spectral cube (fake 3D)
    cube_x0, cube_x1 = 0.02, 0.16
    cube_y0, cube_y1 = 0.22, 0.52
    depth = 0.2 #0.06
    front = [
        (cube_x0, cube_y0),
        (cube_x1, cube_y0),
        (cube_x1, cube_y1),
        (cube_x0, cube_y1),
    ]
    top = [
        (cube_x0, cube_y1),
        (cube_x1, cube_y1),
        (cube_x1 + depth, cube_y1 + depth),
        (cube_x0 + depth, cube_y1 + depth),
    ]
    side = [
        (cube_x1, cube_y0),
        (cube_x1 + depth, cube_y0 + depth),
        (cube_x1 + depth, cube_y1 + depth),
        (cube_x1, cube_y1),
    ]
    ax.add_patch(Polygon(front, closed=True, facecolor="#f2f3f5", edgecolor=box_edge, linewidth=box_lw))
    ax.add_patch(Polygon(top, closed=True, facecolor="#e6e8eb", edgecolor=box_edge, linewidth=box_lw))
    ax.add_patch(Polygon(side, closed=True, facecolor="#e0e2e6", edgecolor=box_edge, linewidth=box_lw))
    ax.text(
        cube_x0,
        cube_y1 + depth + 0.04,
        "Spectral cube (N, H, W)",
        fontsize=12,
        ha="left",
        va="bottom",
    )
    ax.text((cube_x0 + cube_x1) / 2, cube_y0 - 0.05, "H/W (spatial)", fontsize=11, ha="center")
    ax.text(
        cube_x1 + depth + 0.03,
        cube_y1 + depth * 0.35,
        "N (m/z bins)",
        fontsize=11,
        rotation=30,
        ha="left",
        va="center",
    )

    # Slice stack
    slice_x0, slice_x1 = 0.31, 0.41
    slice_y0, slice_y1 = 0.26, 0.50
    slice_w = slice_x1 - slice_x0
    slice_h = slice_y1 - slice_y0
    draw_images = ion_images is not None and len(ion_images) > 0
    num_slices = min(max_images, len(ion_images)) if draw_images else 3
    for idx in range(num_slices):
        offset = idx * 0.02
        x0 = slice_x0 + offset
        y0 = slice_y0 + offset
        if draw_images:
            img = ion_images[idx]
            ax.imshow(
                img,
                cmap="viridis",
                extent=(x0, x0 + slice_w, y0, y0 + slice_h),
                origin="lower",
                aspect="auto",
            )
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                slice_w,
                slice_h,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor="none" if draw_images else "#f7f7f9",
                edgecolor=box_edge,
                linewidth=box_lw,
            )
        )
    ax.text(slice_x0, slice_y1 + 0.06, "Ion images at m/z_i", fontsize=11, ha="left")
    ax.text(slice_x1 + 0.02, slice_y0 + 0.04, "i = 1..N", fontsize=11, ha="left", va="bottom")

    # CNN box
    cnn_x0, cnn_x1 = 0.56, 0.68
    cnn_y0, cnn_y1 = 0.35, 0.55
    cnn_box = FancyBboxPatch(
        (cnn_x0, cnn_y0),
        cnn_x1 - cnn_x0,
        cnn_y1 - cnn_y0,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#eef1f5",
        edgecolor=box_edge,
        linewidth=box_lw,
    )
    ax.add_patch(cnn_box)
    ax.text((cnn_x0 + cnn_x1) / 2, (cnn_y0 + cnn_y1) / 2, "CNN\n(backbone + head)", ha="center", va="center", fontsize=11)

    # Outputs
    cls_x0, cls_x1 = 0.83, 0.97
    cls_y0, cls_y1 = 0.50, 0.62
    reg_x0, reg_x1 = 0.83, 0.97
    reg_y0, reg_y1 = 0.25, 0.37
    cls_box = FancyBboxPatch(
        (cls_x0, cls_y0),
        cls_x1 - cls_x0,
        cls_y1 - cls_y0,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#f7f7f9",
        edgecolor=box_edge,
        linewidth=box_lw,
    )
    reg_box = FancyBboxPatch(
        (reg_x0, reg_y0),
        reg_x1 - reg_x0,
        reg_y1 - reg_y0,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#f7f7f9",
        edgecolor=box_edge,
        linewidth=box_lw,
    )
    ax.add_patch(cls_box)
    ax.add_patch(reg_box)
    ax.text((cls_x0 + cls_x1) / 2, (cls_y0 + cls_y1) / 2, "Classification:\n6 classes", ha="center", va="center", fontsize=11)
    ax.text(
        (reg_x0 + reg_x1) / 2,
        (reg_y0 + reg_y1) / 2,
        "Regression:\n(structure, informativeness)",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Arrows and labels
    ax.add_patch(
        FancyArrowPatch(
            (cube_x1 + depth + 0.05, (cube_y0 + cube_y1) / 2 + 0.03),
            (slice_x0 - 0.04, (slice_y0 + slice_y1) / 2 + 0.03),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=arrow_lw,
            color=arrow_color,
        )
    )
    ax.text(
        (cube_x1 + slice_x0) / 2 + 0.02,
        (cube_y1 + slice_y1) / 2 + 0.10,
        "extract slices",
        fontsize=11,
        ha="center",
    )

    ax.add_patch(
        FancyArrowPatch(
            (slice_x1 + 0.05, (slice_y0 + slice_y1) / 2),
            (cnn_x0 - 0.05, (cnn_y0 + cnn_y1) / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=arrow_lw,
            color=arrow_color,
        )
    )
    ax.text(
        (slice_x1 + cnn_x0) / 2 + 0.02,
        (cnn_y1 + slice_y1) / 2 + 0.06,
        "apply CNN for all i = 1..N",
        fontsize=11,
        ha="center",
    )

    ax.add_patch(
        FancyArrowPatch(
            (cnn_x1 + 0.04, (cnn_y0 + cnn_y1) / 2 + 0.04),
            (cls_x0 - 0.04, (cls_y0 + cls_y1) / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=arrow_lw,
            color=arrow_color,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (cnn_x1 + 0.04, (cnn_y0 + cnn_y1) / 2 - 0.04),
            (reg_x0 - 0.04, (reg_y0 + reg_y1) / 2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=arrow_lw,
            color=arrow_color,
        )
    )

    if title is None:
        title = "Pipeline: applying the CNN across all m/z channels"
    ax.set_title(title, fontsize=16, pad=10)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_msi_overview_figure(
    mz_axis: np.ndarray,
    mean_spectrum: np.ndarray,
    selected_mz: Sequence[float],
    ion_images: Sequence[np.ndarray],
    labels: Sequence[str],
    scores: Sequence[float],
    out_path: Path,
    ncols: int = 4,
    title: Optional[str] = None,
    connect_mz: bool = True,
) -> Path:
    if not (len(selected_mz) == len(ion_images) == len(labels) == len(scores)):
        raise ValueError("selected_mz, ion_images, labels, and scores must match length.")
    n = len(selected_mz)
    nrows = int(math.ceil(n / ncols)) if ncols > 0 else 1

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 2.2], hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("#f7f7f9")
    ax0.plot(mz_axis, mean_spectrum, linewidth=1.3, color="#1f77b4")
    ax0.set_xlabel("m/z")
    ax0.set_ylabel("Mean intensity")
    ax0.set_title(title or "Mean spectrum with selected m/z values labeled by the CNN")
    for label in (ax0.xaxis.label, ax0.yaxis.label, ax0.title):
        label.set_zorder(3)
    ax0.grid(True, color="#e2e2e5", linewidth=0.8)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ymax = float(np.nanmax(mean_spectrum)) if np.isfinite(mean_spectrum).any() else 1.0
    if connect_mz:
        for mz in selected_mz:
            ax0.vlines(mz, 0, ymax, linewidth=1.2, color="black", linestyle="--")
    else:
        for mz in selected_mz:
            ax0.vlines(mz, 0, ymax, linewidth=1.0)

    sub = gs[1].subgridspec(nrows, ncols, wspace=0.2, hspace=0.45)
    image_axes: List[plt.Axes] = []
    for i in range(nrows * ncols):
        ax = fig.add_subplot(sub[i // ncols, i % ncols])
        ax.axis("off")
        ax.set_facecolor("#f7f7f9")
        image_axes.append(ax)
        if i >= n:
            continue
        img = ion_images[i]
        if img is None or np.size(img) == 0:
            ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=9)
            continue
        ax.imshow(img, cmap="viridis")
        label_text = format_panel_caption(selected_mz[i], labels[i], scores[i])
        ax.text(
            0.5,
            -0.06,
            label_text,
            ha="center",
            va="top",
            fontsize=9,
            transform=ax.transAxes,
            zorder=3,
        )
        if connect_mz:
            rect = plt.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                linewidth=1.5,
                edgecolor="black",
            )
            ax.add_patch(rect)
            arrow = ConnectionPatch(
                xyA=(selected_mz[i], -1300),
                coordsA=ax0.transData,
                xyB=(0.5, 1.0),
                coordsB=ax.transAxes,
                arrowstyle="->",
                shrinkA=2,
                shrinkB=2,
                mutation_scale=8,
                linewidth=1.5,
                color="black",
                clip_on=True,
                zorder=1,
            )
            fig.add_artist(arrow)

    panel_color = "#ededf0"
    panel_edge = "#9a9aa1"
    pad_x = 0.05
    pad_y = 0.06

    ax0_pos = ax0.get_position()
    spec_pad_x = pad_x + 0.02
    spec_pad_y = pad_y + 0.03
    spec_panel = FancyBboxPatch(
        (max(0.0, ax0_pos.x0 - spec_pad_x), max(0.0, ax0_pos.y0 - spec_pad_y)),
        min(1.0, ax0_pos.x1 - ax0_pos.x0 + 2 * spec_pad_x),
        min(1.0, ax0_pos.y1 - ax0_pos.y0 + 2 * spec_pad_y),
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor=panel_color,
        edgecolor=panel_edge,
        linewidth=1.0,
        zorder=0,
    )
    fig.add_artist(spec_panel)

    x0 = min(ax.get_position().x0 for ax in image_axes)
    y0 = min(ax.get_position().y0 for ax in image_axes)
    x1 = max(ax.get_position().x1 for ax in image_axes)
    y1 = max(ax.get_position().y1 for ax in image_axes)
    grid_pad_y = pad_y + 0.04
    grid_panel = FancyBboxPatch(
        (max(0.0, x0 - pad_x), max(0.0, y0 - grid_pad_y)),
        min(1.0, x1 - x0 + 2 * pad_x),
        min(1.0, y1 - y0 + 2 * grid_pad_y),
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure,
        facecolor=panel_color,
        edgecolor=panel_edge,
        linewidth=1.0,
        zorder=0,
    )
    fig.add_artist(grid_panel)

    ax0_pos = ax0.get_position()
    fig.text(ax0_pos.x0, ax0_pos.y1 + 0.02, "(A)", fontsize=12, fontweight="bold")
    grid_x0 = min(ax.get_position().x0 for ax in image_axes)
    grid_y1 = max(ax.get_position().y1 for ax in image_axes)
    fig.text(grid_x0, grid_y1 + 0.02, "(B)", fontsize=12, fontweight="bold")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mean-spectrum + ion-image overview figure from an imzML file."
    )
    parser.add_argument("imzml", type=Path, help="Path to the imzML file.")
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="CSV with columns mz,label,score to define ion images.",
    )
    parser.add_argument(
        "--mz-list",
        type=str,
        default="",
        help="Comma/space separated m/z values (used when --table is omitted).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma/space separated labels matching --mz-list (optional).",
    )
    parser.add_argument(
        "--scores",
        type=str,
        default="",
        help="Comma/space separated scores matching --mz-list (optional).",
    )
    parser.add_argument("--output", type=Path, default=Path("msi_overview.png"))
    parser.add_argument("--ppm", type=float, default=None, help="Override ppm for ion images.")
    parser.add_argument("--ncols", type=int, default=6)
    parser.add_argument("--max-ions", type=int, default=None, help="Limit number of ions.")
    parser.add_argument(
        "--sample-spectra",
        type=int,
        default=None,
        help="Approximate mean spectrum by sampling this many spectra.",
    )
    parser.add_argument(
        "--no-hotspot-removal",
        action="store_true",
        help="Disable hotspot removal during ion image extraction.",
    )
    parser.add_argument(
        "--no-connect-mz",
        action="store_true",
        help="Disable color linking between spectrum m/z lines and ion images.",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional title for the spectrum panel.")
    parser.add_argument(
        "--pipeline-output",
        type=Path,
        default=None,
        help="Optional output path for the CNN pipeline schematic figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.table is None and not args.mz_list:
        raise SystemExit("Provide --table or --mz-list to select ions.")

    if args.table is not None:
        selected_mz, labels, scores = load_table(args.table, args.max_ions)
    else:
        selected_mz = parse_float_list(args.mz_list)
        labels = parse_str_list(args.labels)
        scores = parse_float_list(args.scores)
        if labels and len(labels) != len(selected_mz):
            raise SystemExit("--labels must match --mz-list length.")
        if scores and len(scores) != len(selected_mz):
            raise SystemExit("--scores must match --mz-list length.")
        if not labels:
            labels = [""] * len(selected_mz)
        if not scores:
            scores = [float("nan")] * len(selected_mz)

    if not selected_mz:
        raise SystemExit("No m/z values found to plot.")

    combined = list(zip(selected_mz, labels, scores))
    combined.sort(key=lambda item: item[0])
    if args.max_ions is not None:
        combined = combined[: args.max_ions]
    selected_mz, labels, scores = map(list, zip(*combined))

    ppm = args.ppm
    if ppm is None:
        ppm = msi_utils.get_ppm_from_cache_only(args.imzml.stem) or 3.0

    reader = m2aia.ImzMLReader(str(args.imzml))
    mz_axis, mean_spectrum = get_mean_spectrum(reader, sample_spectra=args.sample_spectra)

    mz_bounds = None
    try:
        mz_bounds = msi_utils.get_mz_bounds(reader)
    except Exception:
        mz_bounds = None

    ion_images: List[np.ndarray] = []
    for mz in selected_mz:
        try:
            img = msi_utils.extract_ion_image(
                reader,
                mz,
                ppm=ppm,
                hotspot_removal=not args.no_hotspot_removal,
                mz_bounds=mz_bounds,
            )
        except Exception:
            img = None
        ion_images.append(img)

    ncols = min(max(args.ncols, 1), 6)
    make_msi_overview_figure(
        mz_axis=mz_axis,
        mean_spectrum=mean_spectrum,
        selected_mz=selected_mz,
        ion_images=ion_images,
        labels=labels,
        scores=scores,
        out_path=args.output,
        ncols=ncols,
        title=args.title,
        connect_mz=not args.no_connect_mz,
    )
    if args.pipeline_output is not None:
        make_msi_cube_pipeline_figure(
            args.pipeline_output,
            title=None,
            ion_images=ion_images,
            max_images=3,
        )


if __name__ == "__main__":
    main()


"""
python msianalyzer/tools/visualization/make_msi_overview.py data/processed/2025-05-26_08h35m04s/2025-05-26_08h35m04s.imzML \
--mz-list "863.565503345,467.197630941,715.575948877,794.509371639,525.191876865,889.581153409" \
--labels "structured,negative,localized,unstructured,fragmented,weak structured" \
--output msi_overview_mouse_colon.png

python msianalyzer/tools/visualization/make_msi_overview.py <file.imzML> \
--mz-list "..." --labels "..." --output overview.png --pipeline-output pipeline.png
"""
