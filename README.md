# IonMorphNet: Generalizable Learning of Ion Image Morphologies for Peak Picking in Mass Spectrometry Imaging

[![arxiv.org](https://img.shields.io/badge/cs.CV-arXiv%3A2408.14131-B31B1B.svg)](https://arxiv.org/abs/2604.19369)
[![cite-bibtex](https://img.shields.io/badge/Cite-BibTeX-1f425f.svg)](#citing)
[![CC BY 4.0][cc-by-shield]][cc-by]


Official repository of the CVPR 2026 Workshop paper "IonMorphNet: Generalizable Learning of Ion Image Morphologies for Peak Picking in Mass Spectrometry Imaging".

[Philipp Weigand](https://scholar.google.com/citations?user=WBCzjgcAAAAJ&hl=de)*, Niels Nawrot\*, [Nikolas Ebert](https://scholar.google.de/citations?user=CfFwm1sAAAAJ&hl=de), [Carsten Hopf](https://scholar.google.com/citations?user=Q8T-d1MAAAAJ&hl=de&oi=ao) & [Oliver Wasenmüller](https://scholar.google.de/citations?user=GkHxKY8AAAAJ&hl=de) | *Equal Contribution \
**[CeMOS - Research and Transfer Center](https://www.cemos.hs-mannheim.de/ "CeMOS - Research and Transfer Center"), [University of Applied Sciences Mannheim](https://www.english.hs-mannheim.de/the-university.html "University of Applied Sciences Mannheim")**


<div align="center">
  <img  src="figure/ionmorphnet_teaser.png" width="500"/>
  <div>&nbsp;</div>    
</div>

<!-- IonMorphNet is a toolkit for **Mass Spectrometry Imaging (MSI)** ion-image analysis, morphology labeling, classifier training, and **S3PL (Spatial self-supervised Peak Learning)**-style peak ranking evaluation.

The project focuses on learning whether MSI ion images are morphologically informative, for example whether they are **structured** or **unstructured**, and then using these predictions for downstream peak prioritization. -->


## Installation

### 1. Clone this repository

```bash
git clone https://github.com/CeMOS-IS/IonMorphNet.git
cd IonMorphNet
```

### 2. Create and activate a conda environment

```bash
conda create -n ionmorphnet python=3.9.23
conda activate ionmorphnet
```

### 3. Install dependencies and the package

```bash
pip install -r requirements.txt
pip install -e .
```

<!-- The editable install is recommended for development and local experimentation. -->

### 4. Verify the installation

```bash
python -c "import msianalyzer; print(msianalyzer.__file__)"
```

This should print a path inside your cloned `IonMorphNet` repository.

## Datasets

The package expects MSI datasets in the following structure:

```text
data/datasets/<dataset-id>/
├── <dataset-id>.imzML
└── <dataset-id>.ibd
```

We provide our created morphology labels for each dataset at this path:

```text
data/labeling/csv/<dataset-id>.csv
```

You can find the corresponding dataset under

```text
https://metaspace2020.org/dataset/<dataset-id>
```

<!-- The training pipeline recursively scans `data/datasets/` for `.imzML` files and looks up the matching label file by dataset stem name in `data/labeling/csv/`. -->

## Training

Run the following command and specify the validation and test split with
`--val_files` and `--test_files`. The following split was used for our trainings.

```bash
python -m msianalyzer.training.train_msi_classifier \
  --timm_model "convnextv2_tiny" \
  --val_files "2017-09-25_19h48m56s.imzML,2023-07-04_10h26m22s.imzML,2025-05-20_08h21m56s.imzML,2025-06-18_03h24m09s.imzML,2025-09-24_18h51m29s.imzML" \
  --test_files "2016-10-01_12h21m40s.imzML,2016-10-01_12h27m29s.imzML,2016-10-25_14h30m16s.imzML,2017-03-10_19h59m17s.imzML,2017-03-17_17h20m49s.imzML"
```

## Evaluation

Make sure to provide the mSCF1 evaluation datasets with corresponding segmentation masks in the following folder structure:

```text
├── ionmorphnet
    ├── data
        ├── S3PL_Evaluation_Datasets
            ├── dataset1
            │   ├── masks
            │   │   ├── file1_mask.npy
            │   │   ├── file2_mask.npy
            │   │   └── ...
            │   ├── file1.imzML
            │   ├── file1.ibd
            │   ├── file2.imzML
            │   ├── file2.ibd
            │   └── ...
            ├── cac
            │   ├── masks
            │   │   ├── 40TopL_mask.npy
            │   │   ├── 40TopL_mask.npy
            │   │   └── ...
            │   ├── 40TopL.imzML
            │   ├── 40TopL.ibd
            │   └── ...
            └── ...
```

Peak Picking evaluation using the mSCF1 Score:

```text
python -m msianalyzer.evaluate.evaluate_s3pl_peak_quality \
--run-dir <model_folder_name> \
--informative-classes structured,negative,localized \
```

where --run-dir corresponds to the specific model foldername in ```/data/models/``` that should be used for evaluation. The results will be stored within that folder in `evaluation_s3pl`. To use our trained ConvNeXt-V2-tiny model, download it from [here](https://clousi.hs-mannheim.de/index.php/s/wdHdDxHasM3ATwX) and paste the folder in the ```/data/models/``` directory.


## Application: Classification of all ion images in imzML/ibd file

```text
python -m msianalyzer.evaluate.classify_ion_images \
--run-dir <model_folder_name> \
--imzml-folderpath "/path/to/folder/with/imzML/files" \
```

where --run-dir corresponds to the model foldername in /data/models/ that should be used for evaluation. The results will be stored within that folder in `morphology_predictions`. --imzml-folderpath corresponds to the folder that contains the imzML file(s) that should be anaylzed.

## Troubleshooting

No datasets are found. Verify that:

- dataset files are present under `data/datasets/`
- each dataset consists of two files:
  - `<dataset-id>.imzML`
  - `<dataset-id>.ibd`
- the file extension is exactly `.imzML` with the correct capitalization
- each dataset has a matching CSV file in `data/labeling/csv/
- the CSV filename exactly matches the dataset name

## Citing

If you use this code in academic work, cite the associated thesis, manuscript, or repository release.

```bibtex
@inproceedings{weigand2026ionmorphnet,
  title={IonMorphNet: Generalizable Learning of Ion Image Morphologies for Peak Picking in Mass Spectrometry Imaging},
  author={Weigand, Philipp and Nawrot, Niels and Ebert, Nikolas and Hopf, Carsten and Wasenm{\"u}ller, Oliver},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2026}
}
```

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg