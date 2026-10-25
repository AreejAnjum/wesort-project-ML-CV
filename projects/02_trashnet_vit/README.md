# TrashNet Vision Transformer pipeline

This project studies whether self-supervised adaptation improves a Vision
Transformer on TrashNet. It runs three controlled stages from one configuration:

1. **Frozen baseline** - train only a linear classification head.
2. **Self-supervised adaptation** - train the ViT encoder with SimCLR and no labels.
3. **Fine-tuning** - train the adapted encoder and classifier end to end.

The same stratified train, validation, and test split is reused throughout. The
test set is never used for self-supervised training or model selection.

## Folder structure

```text
02_trashnet_vit/
├── configs/              experiment settings
├── scripts/              dataset and smoke-test helpers
├── src/trashnet_vit/     data, model, training, and evaluation code
├── tests/                fast automated checks
└── pyproject.toml        package and dependency definition
```

Generated datasets, checkpoints, metrics, and reports are intentionally not
tracked by Git.

## Setup

```bash
cd projects/02_trashnet_vit
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Prepare TrashNet

Place the original class folders under `data/dataset-resized`, then create one
fixed stratified split:

```bash
python scripts/prepare_trashnet.py \
  --source data/dataset-resized \
  --output data/manifests \
  --seed 42
```

Expected source layout:

```text
data/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Run the complete experiment

```bash
python -m trashnet_vit.train --config configs/trashnet_vit.yaml
```

The command saves `metrics.json`, the best checkpoint for each stage, learning
curves, and a run summary under `outputs/trashnet_vit/`. Compare test accuracy
and macro-F1 across all three stages. Do not claim an improvement unless it is
present on the held-out test set and remains consistent across several seeds.

## Quick verification

The smoke test uses generated images and a tiny encoder. It verifies the entire
software path but is not research evidence.

```bash
python scripts/make_toy_data.py --output data/toy
python -m trashnet_vit.train --config configs/toy_smoke.yaml
pytest
```

## Design choices

- SimCLR uses two augmented views and NT-Xent contrastive loss; class labels are
  discarded during adaptation.
- Validation macro-F1 selects checkpoints because TrashNet classes are uneven.
- Fine-tuning uses a lower learning rate for the pretrained encoder than for the
  new classification head to reduce catastrophic forgetting.
- Seeds, split manifests, settings, learning curves, and per-class metrics are
  saved so results can be reproduced and audited.
