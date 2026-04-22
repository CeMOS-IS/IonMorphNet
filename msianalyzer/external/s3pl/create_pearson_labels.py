"""
Adapted from CeMOS-IS/S3PL (Apache-2.0).
"""

import os
import numpy as np
import warnings
from scipy import stats
from scipy.stats import ConstantInputWarning
import math
import matplotlib.pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser


def resolve_mask_path(folderpath: str, dataname: str) -> str:
    mask_dir = os.path.join(folderpath, "masks")
    candidates = [
        os.path.join(mask_dir, f"{dataname}_mask.npy"),
        os.path.join(mask_dir, f"{dataname}.npy"),
        os.path.join(mask_dir, f"{dataname}_mask.npz"),
        os.path.join(mask_dir, f"{dataname}.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if os.path.isdir(mask_dir):
        for filename in os.listdir(mask_dir):
            if filename.startswith(dataname) and filename.endswith((".npy", ".npz")):
                return os.path.join(mask_dir, filename)
    raise FileNotFoundError(
        f'There is no mask file for {dataname} in "{mask_dir}". '
        f"Expected one of: {', '.join(os.path.basename(p) for p in candidates)}"
    )


def create_pearson_labels(dataname, folderpath, num_classes):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        labels_dir = os.path.join(folderpath, "labels")
        if not os.path.exists(labels_dir):
            os.mkdir(labels_dir)

        segmentation_classes = list(range(0, num_classes))
        mask_path = resolve_mask_path(folderpath, dataname).replace("_all_classes", "")
        mask_orig = np.load(mask_path)

        p = ImzMLParser(os.path.join(folderpath, dataname + ".imzML"))
        all_mz, _ = p.getspectrum(0)

        all_spectra = []
        XCoord = []
        YCoord = []
        for idx, (x, y, z) in enumerate(p.coordinates):
            mzs, intensities = p.getspectrum(idx)
            all_spectra.append(intensities)
            XCoord.append(x)
            YCoord.append(y)

        for class_number in segmentation_classes:
            mask = np.where(mask_orig == class_number, 1, 0)

            all_spectra = np.array(all_spectra)
            mask_flattened = []
            for i in range(len(XCoord)):
                mask_flattened.append(mask[YCoord[i] - 1, XCoord[i] - 1])

            pearson_correlations = []
            for idx, mz in enumerate(all_mz):
                ion_image = all_spectra[:, idx]
                pearson_corr = stats.pearsonr(mask_flattened, ion_image).correlation

                if math.isnan(pearson_corr):
                    pearson_correlations.append(0)
                else:
                    pearson_correlations.append(pearson_corr)

            ranking = np.argsort(pearson_correlations)[::-1]
            mz_values = [all_mz[rank] for rank in ranking]
            pearson_correlations = [pearson_correlations[rank] for rank in ranking]

            np.save(os.path.join(labels_dir, dataname + "_class" + str(class_number) + "_ranking.npy"), ranking)
            np.save(os.path.join(labels_dir, dataname + "_class" + str(class_number) + "_mz_ranking.npy"), mz_values)
            np.save(os.path.join(labels_dir, dataname + "_class" + str(class_number) + "_pearson_ranking.npy"), pearson_correlations)
