from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create deterministic stratified TrashNet manifests")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    rows = []
    for class_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        for image in sorted(class_dir.rglob("*")):
            if image.suffix.lower() in SUFFIXES:
                rows.append({"path": str(image.resolve()), "label": class_dir.name})
    if not rows:
        raise ValueError(f"No class-folder images found under {args.source}")
    labels = [row["label"] for row in rows]
    train_val, test = train_test_split(rows, test_size=args.test_size, random_state=args.seed, stratify=labels)
    remaining_labels = [row["label"] for row in train_val]
    relative_val = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=args.seed, stratify=remaining_labels
    )
    for name, split in (("train", train), ("val", val), ("test", test)):
        write_manifest(args.output / f"{name}.csv", sorted(split, key=lambda row: row["path"]))
        print(f"{name}: {len(split)} images")


if __name__ == "__main__":
    main()
