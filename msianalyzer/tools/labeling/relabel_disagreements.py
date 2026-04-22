"""Utility to relabel high-loss ions surfaced by training disagreements.

Loads a disagreements CSV exported by the training loop, hydrates the
corresponding ion cache for quick access, and launches the existing
`IonImageLabeler` GUI focused on those m/z values so they can be
relabelled interactively.

The CLI can now consume individual CSV paths *or* derive the correct
CSV(s) directly from training run directories (via --run-dir / --runs-root),
automatically picking the best epoch recorded in run_meta.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from msianalyzer.config import LABELING_CSV_DIR
from msianalyzer.tools.labeling.label_gui import IonImageLabeler
from msianalyzer.utils import msi_utils
from msianalyzer.utils.ion_cache import IonImageCache


def _read_disagreements(csv_path: Path) -> Dict[str, List[dict]]:
    """Group disagreement rows by dataset name."""
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for row in reader:
            dataset = row.get("dataset")
            mz = row.get("mz")
            path = row.get("path")
            if not dataset or not mz or not path:
                continue
            try:
                row["mz"] = float(mz)
                row["avg_loss"] = float(row.get("avg_loss", 0.0))
                row["student_conf"] = float(row.get("student_conf", 0.0))
                row["teacher_conf"] = float(row.get("teacher_conf", 0.0))
            except ValueError:
                continue
            grouped[dataset].append(row)
    return grouped


def _prepare_metadata(rows: Sequence[dict], csv_path: Path) -> Dict[float, dict]:
    """Convert disagreement rows into label GUI metadata mapping."""
    metadata: Dict[float, dict] = {}
    metadata["_metadata"] = {
        "source": "training_disagreements",
        "csv": str(csv_path),
        "num_candidates": len(rows),
    }

    for row in sorted(rows, key=lambda r: r["avg_loss"], reverse=True):
        mz_value = float(row["mz"])
        mz_key = round(mz_value, 9)
        entry = metadata.get(mz_key, None)
        if entry and entry.get("avg_loss", 0.0) >= row["avg_loss"]:
            # Keep the highest-loss entry per m/z
            continue

        formulas = [
            f"avg_loss={row['avg_loss']:.3f}",
            f"pred={row.get('student_label', 'n/a')} (p={row.get('student_conf', 0.0):.2f})",
        ]
        if row.get("teacher_label") and row.get("teacher_label") != "N/A":
            formulas.append(
                f"teacher={row['teacher_label']} (p={row.get('teacher_conf', 0.0):.2f})"
            )
        if row.get("label_name"):
            formulas.append(f"ground_truth={row['label_name']}")

        metadata[mz_key] = {
            "mz": mz_value,
            "msm": row["avg_loss"],  # GUI sorts descending → highest loss first
            "avg_loss": row["avg_loss"],
            "label_name": row.get("label_name"),
            "ground_truth": row.get("label_name"),
            "student_label": row.get("student_label"),
            "student_conf": row.get("student_conf", 0.0),
            "teacher_label": row.get("teacher_label"),
            "teacher_conf": row.get("teacher_conf", 0.0),
            "formulas": formulas,
            "adducts": ["" for _ in formulas],
            "fdr_values": [1.0 for _ in formulas],
        }
    return metadata


def _ensure_cache(imzml_path: Path, mz_list: Iterable[float], ppm: float) -> None:
    """Warm up the ion cache for faster GUI interaction."""
    mz_values = list(dict.fromkeys(round(float(m), 9) for m in mz_list))
    if not mz_values:
        return
    try:
        IonImageCache(
            imzml_path=imzml_path,
            reader=None,
            mz_list=[float(m) for m in mz_values],
            ppm=ppm,
        )
        print(f"✓ Ion cache ready for {imzml_path.name} ({len(mz_values)} ions, ppm={ppm})")
    except FileNotFoundError as exc:
        print(
            f"⚠️  Could not locate cache for {imzml_path.name}: {exc}"
        )
        print("    The GUI will fall back to on-the-fly extraction via m2aia.")


def _resolve_dataset(grouped: Dict[str, List[dict]], requested: Optional[str]) -> Optional[str]:
    if requested:
        if requested not in grouped:
            available = ", ".join(sorted(grouped)) or "<none>"
            raise SystemExit(f"Dataset '{requested}' not present. Available: {available}")
        return requested
    return None


def _best_epoch_from_meta(meta_path: Path) -> Optional[int]:
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    for key in ("Best epoch", "Best Epoch"):
        value = meta.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _collect_best_csvs(
    run_dirs: Sequence[Path],
    split: str,
) -> List[Path]:
    csv_paths: List[Path] = []
    for run_dir in run_dirs:
        run_dir = run_dir.expanduser().resolve()
        if not run_dir.exists():
            print(f"⚠️  Run directory not found: {run_dir}")
            continue

        meta_path = run_dir / "run_meta.json"
        best_epoch = _best_epoch_from_meta(meta_path)
        if best_epoch is None:
            print(f"⚠️  Could not determine best epoch for {run_dir}")
            continue

        csv_path = run_dir / "disagreements" / f"disagreements_{split}_epoch{best_epoch:03d}.csv"
        if not csv_path.exists():
            print(f"⚠️  Disagreements CSV missing for best epoch in {run_dir}: {csv_path}")
            continue

        csv_paths.append(csv_path)
    return csv_paths


def _process_disagreements_csv(
    csv_path: Path,
    dataset: Optional[str],
    output_override: Optional[Path],
    only_unlabeled: bool,
) -> List[Dict[str, object]]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Disagreements CSV not found: {csv_path}")

    grouped = _read_disagreements(csv_path)
    if not grouped:
        raise SystemExit(f"No disagreements found in {csv_path}")

    dataset = _resolve_dataset(grouped, dataset)
    targets = [dataset] if dataset else sorted(grouped)

    if output_override is not None and len(targets) != 1:
        raise SystemExit("--output can only be used when exactly one dataset is selected")

    summaries: List[Dict[str, object]] = []

    for ds in targets:
        rows = grouped[ds]

        imzml_path = Path(rows[0]["path"]).expanduser().resolve()
        if not imzml_path.exists():
            print(f"⚠️  imzML not found for {ds}: {imzml_path}")
            continue

        dataset_id = imzml_path.stem
        ppm = msi_utils.get_ppm_from_cache_only(dataset_id)

        metadata = _prepare_metadata(rows, csv_path)
        _ensure_cache(imzml_path, (row["mz"] for row in rows), ppm)

        output_csv = output_override
        if output_csv is None:
            output_csv = (LABELING_CSV_DIR / f"{dataset_id}.csv").resolve()
        else:
            output_csv = output_csv.expanduser().resolve()

        print(f"Launching label GUI for {dataset_id}")
        print(f"  Disagreements source: {csv_path}")
        print(f"  Writing labels to:    {output_csv}")
        print(f"  Total ions queued:    {len(metadata) - 1}")

        IonImageLabeler(
            str(imzml_path),
            str(output_csv),
            dataset_id,
            ppm=ppm,
            metadata_dict=metadata,
            only_unlabeled_mode=only_unlabeled,
        )

        unique_mz = {round(float(r["mz"]), 9) for r in rows}
        mean_loss = sum(r["avg_loss"] for r in rows) / max(len(rows), 1)
        summaries.append({
            "dataset": dataset_id,
            "ions_total": len(rows),
            "ions_unique": len(unique_mz),
            "avg_loss_mean": mean_loss,
            "avg_loss_max": max(r["avg_loss"] for r in rows),
            "avg_loss_min": min(r["avg_loss"] for r in rows),
        })

    return summaries


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Relabel training disagreements via the label GUI")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        help="Path to disagreements_<split>_epochXXX.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset (imzML filename) to relabel; required if CSV contains multiple datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination CSV for labels (defaults to LABELING_CSV_DIR/<dataset>.csv)",
    )
    parser.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Skip ions that already have a label in the output CSV",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Run directory containing run_meta.json; may be used multiple times",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Folder that contains multiple run directories (each with run_meta.json)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to use when deriving disagreements from a run directory",
    )
    args = parser.parse_args(argv)

    csv_inputs: List[Path] = []

    if args.csv is not None:
        csv_inputs.append(args.csv)

    if args.run_dir:
        csv_inputs.extend(_collect_best_csvs(args.run_dir, args.split))

    if args.runs_root:
        runs_root = args.runs_root.expanduser().resolve()
        if not runs_root.exists():
            print(f"⚠️  Runs root not found: {runs_root}")
        else:
            run_dirs = [p for p in runs_root.iterdir() if (p / "run_meta.json").exists()]
            csv_inputs.extend(_collect_best_csvs(run_dirs, args.split))

    if not csv_inputs:
        raise SystemExit("No disagreements CSV specified. Provide a CSV path or run directory.")

    if args.output is not None and len(csv_inputs) > 1:
        raise SystemExit("--output cannot be used with multiple CSV inputs or run directories")

    summaries_total: List[Dict[str, object]] = []

    for idx, csv_path in enumerate(csv_inputs, start=1):
        print(f"\n=== [{idx}/{len(csv_inputs)}] Processing {csv_path} ===")
        summaries = _process_disagreements_csv(
            csv_path=csv_path,
            dataset=args.dataset,
            output_override=args.output,
            only_unlabeled=args.only_unlabeled,
        )
        summaries_total.extend(
            [{**item, "source_csv": str(csv_path)} for item in summaries]
        )

    if summaries_total:
        print("\n=== Relabel Summary ===")
        for item in summaries_total:
            print(
                f"{item['source_csv']}: {item['dataset']}: {item['ions_unique']} unique ions "
                f"({item['ions_total']} rows) | avg_loss μ={item['avg_loss_mean']:.3f} "
                f"[{item['avg_loss_min']:.3f}, {item['avg_loss_max']:.3f}]"
            )


if __name__ == "__main__":
    main()

# use example:

#python msianalyzer/tools/labeling/relabel_disagreements.py data/models/20251010-140529_resnet50_full_soft_dpr-0.05_wd-0.05_aug-default_cosine_pretrained-1_cuda/disagreements/disagreements_test_epoch012.csv
#python msianalyzer/tools/labeling/relabel_disagreements.py --run-dir data/models/20251010-140529_resnet50_full_soft_dpr-0.05_wd-0.05_aug-default_cosine_pretrained-1_cuda
#python msianalyzer/tools/labeling/relabel_disagreements.py --runs-root data/models
#python msianalyzer/tools/labeling/relabel_disagreements.py --runs-root data/models --split val
#python msianalyzer/tools/labeling/relabel_disagreements.py --run-dir data/models/20251010-140529_resnet50_full_soft_dpr-0.05_wd-0.05_aug-default_cosine_pretrained-1_cuda --dataset 2016-10-01_12h21m40s.imzML
#python msianalyzer/tools/labeling/relabel_disagreements.py data/models/20251010-140529_resnet50_full_soft_dpr-0.05_wd-0.05_aug-default_cosine_pretrained-1_cuda/disagreements/disagreements_test_epoch012.csv --output ~/labels/2016-10-01_12h21m40s.csv --only-unlabeled