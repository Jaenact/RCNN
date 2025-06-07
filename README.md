# RCNN

This repository contains a single script `rcnn.py` implementing a simplified R-CNN style pipeline for the PASCAL VOC 2007 dataset. It includes selective search proposal generation, feature extraction with a custom AlexNet model, training of SVM classifiers and bounding box regressors, and evaluation.

## Requirements
- Python 3
- PyTorch
- torchvision
- OpenCV with `ximgproc` (for selective search)
- scikit-learn
- joblib
- tqdm

Install the packages via `pip`:

```bash
pip install torch torchvision opencv-python-headless scikit-learn joblib tqdm
```

## Dataset
Download the PASCAL VOC 2007 dataset and place the `JPEGImages` and `Annotations` directories under `./VOC2007` so the tree looks like:

```
VOC2007/
├── JPEGImages/
└── Annotations/
```

## Usage
Run the entire pipeline with:

```bash
python rcnn.py
```

The script will:
1. Generate or load selective search proposals.
2. Create an index file matching proposals with ground truth.
3. Train `CustomAlexNet` from scratch.
4. Extract features for each proposal.
5. Train linear SVMs for classification.
6. Train bounding box regressors.
7. Run inference, apply non‑maximum suppression, and report mAP.

All intermediate data (models, features, results) are stored in `RCNNDataCache/`.
