# Multimodal Cancer Classification

A PyTorch solution for the **Multimodal Cancer Classification Challenge 2026** on Kaggle. The model classifies individual cells as benign or malignant using paired **Brightfield (BF)** and **Fluorescence (FL)** microscopy images, fused via a custom dual-stream ResNet-14 architecture.

---

## Overview

| | |
|---|---|
| **Task** | Binary cell-level classification (benign vs. malignant) |
| **Metric** | ROC-AUC |
| **Input** | Paired BF + FL microscopy images (224×224 px) |
| **Architecture** | Dual-stream ResNet-14 with intermediate fusion |
| **Validation** | 4-fold manual stratified cross-validation (patient-level) |

---

## Model Architecture

The core model is a custom **DualStreamResNet14** — a lightweight ResNet variant built from scratch with ~11M parameters.

**Design rationale:**

- BF and FL images carry complementary biological signals, so each modality gets its own dedicated stream (stem + stage1 + stage2) rather than being naïvely concatenated at the input.
- After stage 2, the two streams are concatenated and projected back down via a 1×1 convolution (intermediate fusion), then passed through shared stage3 and stage4 layers.
- A global average pool feeds into a two-layer classifier head with dropout.

```
BF (3ch) ──► stem ──► stage1 ──► stage2 ──┐
                                            ├──► concat ──► 1×1 fuse ──► stage3 ──► stage4 ──► GAP ──► head ──► logit
FL (3ch) ──► stem ──► stage1 ──► stage2 ──┘
```

---

## Training Details

### Data Augmentation

Training augmentations (via [Albumentations](https://albumentations.ai/)):

- Random resized crop (scale 0.7–1.0)
- Horizontal & vertical flips, random 90° rotations
- Shift/scale/rotate, color jitter, Gaussian blur
- CoarseDropout (random 32×32 patches zeroed out)
- ImageNet normalization

Validation: resize to 224×224 + normalize only.

### Loss Function

**Binary Focal Loss** (`alpha=0.75, gamma=2.0`) to handle class imbalance by down-weighting easy negatives.

### Other Training Tricks

- **Mixup** (`alpha=0.4`) applied on-the-fly during training
- **WeightedRandomSampler** for class-balanced batches
- **Mixed precision training** (`torch.cuda.amp`)
- **Gradient clipping** (max norm 1.0)
- **AdamW** optimizer with cosine annealing LR schedule (`lr=3e-4`, `weight_decay=1e-4`)

### Cross-Validation

With only 12 patients in the dataset, standard `StratifiedGroupKFold` cannot reliably guarantee both classes appear in every validation fold. A **manual round-robin patient assignment** scheme is used instead:

1. Patients are split by majority class label (malignant vs. benign-majority).
2. Each class group is shuffled with a fixed seed.
3. Patients are assigned to folds in round-robin order, ensuring every fold's validation set contains at least one patient from each class.
4. All splits are verified for zero leakage and class presence before training begins.

### Test-Time Augmentation (TTA)

At inference, 5 randomly augmented versions of each image are averaged to stabilise predictions.

---

## Configuration

Key hyperparameters (set in `CFG` dict in the notebook):

| Parameter | Value |
|---|---|
| `img_size` | 224 |
| `batch_size` | 64 |
| `num_epochs` | 20 |
| `lr` | 3e-4 |
| `weight_decay` | 1e-4 |
| `n_folds` | 4 |
| `focal_alpha` | 0.75 |
| `focal_gamma` | 2.0 |
| `tta_n` | 5 |

---

## Requirements

```
torch
torchvision
numpy
pandas
scikit-learn
albumentations
Pillow
tqdm
matplotlib
kagglehub
```

Install with:

```bash
pip install torch torchvision numpy pandas scikit-learn albumentations Pillow tqdm matplotlib kagglehub
```

---

## Data

The notebook expects data from the **Multimodal Cancer Classification Challenge 2026** Kaggle competition, downloaded automatically via `kagglehub`:

```
/kaggle/input/competitions/multimodal-cancer-classification-challenge-2026/
├── train.csv           # Name, Diagnosis columns
├── BF/
│   ├── train/          # Brightfield training images
│   └── test/           # Brightfield test images
└── FL/
    ├── train/          # Fluorescence training images
    └── test/           # Fluorescence test images
```

Each row in `train.csv` corresponds to a single cell image. Patient IDs are parsed from the filename prefix (e.g. `pat_03_cell_042.png` → patient `pat_03`).

---

## Usage

The notebook is self-contained and designed to run on Kaggle. Run cells sequentially:

1. **Cell 1** — Imports and reproducibility setup
2. **Cell 2** — Config and paths
3. **Cell 3** — Load metadata and build file paths
4. **Cell 4** — Define augmentation pipelines
5. **Cell 5** — `CancerCellDataset` (dual-channel 6ch loader)
6. **Cell 6** — `DualStreamResNet14` model definition
7. **Cell 7** — `FocalLoss` definition
8. **Cell 8** — Mixup helpers
9. **Cell 9** — Training and validation loop functions
10. **Cell 10** — 4-fold cross-validation training
11. **Cell 11** — Test inference with 4-fold ensemble + TTA → `submission.csv`
12. **Cell 12** — Save enriched checkpoints and zip for download

---

## Outputs

After running the full notebook, the following files are saved to `/kaggle/working`:

| File | Description |
|---|---|
| `best_fold{1..4}.pt` | Best checkpoint per fold (state dict) |
| `model_fold{N}_{timestamp}.pt` | Enriched checkpoints with metadata (config, AUC, etc.) |
| `all_models_{timestamp}.zip` | Zip of all enriched checkpoints + submission |
| `submission.csv` | Final predictions (Name, Diagnosis probability) |
| `prediction_analysis.png` | Ensemble distribution plots and fold correlation heatmap |

---

## Reproducibility

A fixed seed (`SEED=42`) is applied to Python's `random`, NumPy, and PyTorch (including CUDA). `cudnn.deterministic=True` and `cudnn.benchmark=False` are set for full reproducibility across runs.

---

## License

This project is released for educational and competition purposes. The dataset is subject to Kaggle competition terms.
