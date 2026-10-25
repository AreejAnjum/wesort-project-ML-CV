# TrashNet unsupervised Vision Transformer experiments

This project completes two tasks on TrashNet without using class labels to
update either Vision Transformer.

1. **Unsupervised ViT feature evaluation:** extract frozen features from
   DINOv2-small and MAE-base, then measure their quality with logistic
   regression and cosine kNN at 1, 5, 10, 20, 50, and 100 labels per class.
2. **Unsupervised fine-tuning:** continue training MAE with masked-image
   reconstruction and DINOv2 with teacher-student self-distillation. Repeat the
   same frozen-feature evaluation before and after adaptation.

Labels are used only to train the small evaluation classifiers and calculate
test accuracy. They are never passed to MAE or DINOv2 during fine-tuning.

## Method

TrashNet contains 2,527 images in six class folders. One fixed stratified split
is used throughout: 80% for unsupervised adaptation and label-budget sampling,
and 20% as a held-out test set. Test images never enter fine-tuning.

The feature test repeats each label budget five times and reports the mean and
standard deviation for logistic regression and k-nearest neighbours with
cosine distance.

## Installation

```bash
cd projects/02_trashnet_vit_unsupervised_finetuning
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Place TrashNet in this layout:

```text
data/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Task 1: frozen unsupervised ViT features

```bash
python scripts/run_vit_experiments.py features --data data/dataset-resized \
  --model facebook/dinov2-small --out outputs/dinov2_original
python scripts/run_vit_experiments.py features --data data/dataset-resized \
  --model facebook/vit-mae-base --out outputs/mae_original
```

Each run saves normalized features, the complete label-budget curve, and a
readable summary.

## Task 2A: unsupervised MAE fine-tuning

```bash
python scripts/run_vit_experiments.py mae --data data/dataset-resized \
  --epochs 20 --lr 5e-5 --out outputs/mae_adapted
python scripts/run_vit_experiments.py features --data data/dataset-resized \
  --model outputs/mae_adapted --out outputs/mae_adapted_evaluation
```

MAE hides 75% of the image patches and learns to reconstruct them. No class
label is loaded by the training dataset.

## Task 2B: unsupervised DINOv2 fine-tuning

```bash
python scripts/run_vit_experiments.py dino --data data/dataset-resized \
  --epochs 20 --lr 1e-5 --head-lr 1e-3 --out outputs/dinov2_adapted
python scripts/run_vit_experiments.py features --data data/dataset-resized \
  --model outputs/dinov2_adapted --out outputs/dinov2_adapted_evaluation
```

DINO uses a student and an exponential-moving-average teacher. The teacher sees
two global crops, the student sees the global crops plus six local crops, and
the student learns to match the teacher. Feature spread is monitored to detect
representation collapse.

## Fair comparison

Compare the `label_curve.csv` files before and after adaptation. The split,
label budgets, classifiers, and test images must remain unchanged. Repeat the
complete experiment with several seeds before interpreting small differences.
