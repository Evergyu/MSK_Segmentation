# MSK-Segmentation
**Official repository of "Musculoskeletal CT Segmentation Dataset of Lower Limbs for High-fidelity Human Digital Twin"**

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Data-blue)](https://doi.org/10.1038/s41597-026-08630-0)
[![Dataset](https://img.shields.io/badge/Dataset-DataON-green)](https://doi.org/10.22711/idr/1472)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-yellow)](https://huggingface.co/evergyu2/MSK_segmentation)

## Overview
This repository provides multi-GPU **training** and **inference/visualization** code for the
lower-limb musculoskeletal CT segmentation dataset published in our paper. It reproduces the
UPerNet benchmark using `segmentation_models_pytorch` and is compatible with the released checkpoint.

## Environment & Dependencies
This codebase has been tested in the following environment:
- **Python**: 3.11
- **PyTorch**: 2.5.1
- **CUDA**: 12.1

*Note: While tested on the above configuration, the code is likely compatible with other versions of PyTorch and Python, depending on your setup.*

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Evergyu/MSK_Segmentation.git
cd MSK_Segmentation
```

### 2. Set up the Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Repository Structure
```text
MSK_Segmentation/
├── src/
│   ├── train.py          # multi-GPU (DDP) training
│   ├── main.py           # inference / visualization
│   ├── check_data.py     # validate a dataset folder
│   ├── dataset.py        # dataset discovery
│   └── utils/            # colour map + overlay plotting
├── assets/
│   └── class_info_reference.csv   # official class table (names, sides, colours)
├── data/                 # <-- place the dataset here (Training)
├── checkpoints/          # <-- place checkpoint.pth here (Inference)
├── requirements.txt
└── README.md
```

## Dataset Preparation (for Training)
The dataset is available on our **DataON** repository (CC BY 4.0). It contains 40 lower-limb
cases (multi-planar 2D CT), **89 classes** (88 musculoskeletal structures + background), with a
subject-level split of **38 train / 2 test** cases.

### 1. Download the Data
Download the three files from [DataON](https://doi.org/10.22711/idr/1472) into a `download/` folder:
`images.zip`, `labels.zip`, `class_info.csv`.

### 2. Arrange the Directories
Unzip so that images and masks mirror each other, one sub-folder per case and plane:
```text
data/
├── images/
│   └── hm_25_001_ct/
│       ├── axial/    000000.jpg …    <-- RGB CT slices (.jpg)
│       ├── coronal/  000000.jpg …
│       └── sagittal/ 000000.jpg …
├── masks/
│   └── hm_25_001_ct/                 <-- same case / plane / filename
│       ├── axial/    000000.png …    <-- class-index masks (.png, 1 channel)
│       ├── coronal/  000000.png …
│       └── sagittal/ 000000.png …
└── class_info.csv                    <-- Structure,Side,Code,R,G,B,HEX,pixel_value
```
```bash
unzip images.zip -d data/images
unzip labels.zip -d data/masks
cp    class_info.csv data/class_info.csv
python src/check_data.py --data-root data --class-info data/class_info.csv   # verify
```
*Masks are single-channel PNGs whose pixel value is the class index (0 = background). Images and
masks may differ in resolution (same aspect ratio) and are resized to 512×512 together; the slices
are already correctly oriented.*
*For the official split, place cases under `data/train/{images,masks}` and `data/test/{images,masks}`;
a flat `data/{images,masks}` is auto-split by `--val-split`.*

## Training
Runs DistributedDataParallel on **all visible GPUs** — a plain python launch, no torchrun/slurm needed.
```bash
python src/train.py --data-root data --class-info data/class_info.csv
```
Default configuration follows the paper: **UPerNet** + `timm-efficientnet-b1`, 512×512 input,
**Adam lr 2e-4**, weight decay 1e-4, CosineAnnealing schedule with 5 warmup epochs (min lr 1e-6),
**100 epochs**, **batch 64** (8/GPU × 8 GPUs), loss `0.5·Dice + 0.5·CE`, mixed precision.

Common flags: `--arch upernet|unet|fpn|pspnet|manet`, `--gpus N`, `--batch-size` (per GPU),
`--lr`, `--val-split`, `--resume ./checkpoints/last.pth`, `--no-amp`. The best checkpoint (by test
IoU) and `last.pth` are written to `--output-dir` (default `checkpoints/`).

## Inference & Visualization

### 1. Download Pre-trained Weights
Download `checkpoint.pth` from **Hugging Face**, or use your own trained checkpoint:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('evergyu2/MSK_segmentation','checkpoint.pth',local_dir='checkpoints')"
```

### 2. Prepare Directories
Place the checkpoint and your input images in the respective folders:
```text
MSK_Segmentation/
├── checkpoints/
│   └── checkpoint.pth       <-- Place the downloaded model here
├── images/
│   ├── test_image_01.jpg    <-- Place your input images here
│   ├── test_image_02.jpg
│   └── class_info.csv       <-- class names + colours
├── results/                 <-- Segmentation results will be saved here
└── src/main.py
```

### 3. Usage
```bash
python src/main.py
```
For each image, a `<name>_visualization.png` (Original / Prediction / Overlay) is saved to `results/`.
Prediction runs at a fixed 512×512 and is resized back to each image's native resolution. Paths are
set by the constants at the top of `src/main.py`.

## Citation
```bibtex
@article{lee2026musculoskeletal,
  title   = {Musculoskeletal CT Segmentation Dataset of Lower Limbs for High-fidelity Human Digital Twin},
  author  = {Lee, In-Gyu and Kim, Gi-Wan and Lee, Jae Yong and Shim, Hyoung Seop and Kwak, Dai-Soon and
             Choi, Kwangnam and Son, Ji-Young and Han, Seung-Ho and Yang, Donghun},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-08630-0}
}
```
