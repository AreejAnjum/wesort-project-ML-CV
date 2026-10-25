import csv
from pathlib import Path

import torch
from PIL import Image

from trashnet_vit.data import ManifestDataset, UnlabeledDataset
from trashnet_vit.model import nt_xent_loss
from trashnet_vit.train import ensure_disjoint


def manifest(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_unlabeled_dataset_returns_two_views_without_label(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (20, 20), "red").save(image)
    path = tmp_path / "train.csv"
    manifest(path, [{"path": image, "label": "plastic"}])
    first, second = UnlabeledDataset(path, 16)[0]
    assert first.shape == second.shape == (3, 16, 16)


def test_class_mapping_is_stable(tmp_path):
    rows = []
    for index, label in enumerate(("trash", "cardboard")):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (10, 10)).save(image)
        rows.append({"path": image, "label": label})
    path = tmp_path / "data.csv"
    manifest(path, rows)
    assert ManifestDataset(path, 8).class_to_idx == {"cardboard": 0, "trash": 1}


def test_contrastive_loss_prefers_matching_pairs():
    vectors = torch.eye(4)
    assert nt_xent_loss(vectors, vectors.clone(), 0.2) < nt_xent_loss(vectors, vectors.roll(1, 0), 0.2)


def test_split_overlap_is_rejected(tmp_path):
    image = tmp_path / "same.png"
    Image.new("RGB", (10, 10)).save(image)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    row = [{"path": image, "label": "paper"}]
    manifest(first, row)
    manifest(second, row)
    datasets = (ManifestDataset(first, 8), ManifestDataset(second, 8))
    try:
        ensure_disjoint(train=datasets[0], test=datasets[1])
    except ValueError as error:
        assert "Data leakage" in str(error)
    else:
        raise AssertionError("overlap should be rejected")
