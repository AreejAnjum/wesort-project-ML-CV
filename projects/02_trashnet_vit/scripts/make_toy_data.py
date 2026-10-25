from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    colors = {"paper": (225, 225, 210), "plastic": (40, 120, 220)}
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        rows = []
        for label, color in colors.items():
            for index in range(count):
                path = (args.output / "images" / split / label / f"{index}.png").resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (40, 40), color).save(path)
                rows.append({"path": str(path), "label": label})
        with (args.output / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
