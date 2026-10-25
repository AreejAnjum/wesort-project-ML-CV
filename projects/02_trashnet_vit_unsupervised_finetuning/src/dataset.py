"""TrashNet discovery, splitting, and label-free augmentation datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_class_root(root, max_depth=5):
    """Find a directory containing at least two image-bearing class folders."""
    root = Path(root)

    def has_images(directory):
        return any(path.suffix.lower() in IMAGE_EXTENSIONS for path in directory.iterdir() if path.is_file())

    def search(directory, depth):
        children = sorted(path for path in directory.iterdir() if path.is_dir())
        if sum(has_images(child) for child in children) >= 2:
            return directory
        if depth < max_depth:
            for child in children:
                found = search(child, depth + 1)
                if found:
                    return found
        return None

    found = search(root, 0)
    if found is None:
        sys.exit(f"No folder with at least two image class folders found below {root}")
    return found


def list_images(class_root):
    """Return image paths, integer labels, and sorted class names."""
    class_root = Path(class_root)
    classes = sorted(
        path.name for path in class_root.iterdir()
        if path.is_dir() and any(item.suffix.lower() in IMAGE_EXTENSIONS for item in path.iterdir() if item.is_file())
    )
    paths, labels = [], []
    for class_index, class_name in enumerate(classes):
        for path in sorted((class_root / class_name).iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
                labels.append(class_index)
    return paths, np.asarray(labels), classes


def stratified_split(labels, test_fraction=0.2, seed=0):
    """Create one deterministic class-balanced train/test index split."""
    generator = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for class_index in np.unique(labels):
        indices = np.where(labels == class_index)[0]
        generator.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])
    return np.asarray(sorted(train_indices)), np.asarray(sorted(test_indices))


def make_image_dataset(paths, training, size=224):
    """Create an image-only dataset; no class label is returned."""
    import torch
    from PIL import Image
    from torchvision import transforms

    if training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    class ImageOnlyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            return transform(Image.open(paths[index]).convert("RGB"))

    return ImageOnlyDataset()


def make_multicrop_dataset(paths, local_crops, global_size=224, local_size=98, color_strength=0.4):
    """Return two global and N local augmented views without labels."""
    import torch
    from PIL import Image
    from torchvision import transforms

    bicubic = transforms.InterpolationMode.BICUBIC

    def augmentation(size, scale, blur_probability, blur_kernel):
        operations = [
            transforms.RandomResizedCrop(size, scale=scale, interpolation=bicubic),
            transforms.RandomHorizontalFlip(),
        ]
        if color_strength > 0:
            operations.extend([
                transforms.RandomApply([transforms.ColorJitter(
                    0.8 * color_strength, 0.8 * color_strength,
                    0.4 * color_strength, 0.2 * color_strength,
                )], p=0.8),
                transforms.RandomGrayscale(0.2),
            ])
        operations.extend([
            transforms.RandomApply([transforms.GaussianBlur(blur_kernel, (0.1, 2.0))], p=blur_probability),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return transforms.Compose(operations)

    first_global = augmentation(global_size, (0.4, 1.0), 1.0, 23)
    second_global = augmentation(global_size, (0.4, 1.0), 0.1, 23)
    local = augmentation(local_size, (0.05, 0.4), 0.5, 9)

    class MultiCropDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            image = Image.open(paths[index]).convert("RGB")
            return (first_global(image), second_global(image)) + tuple(local(image) for _ in range(local_crops))

    return MultiCropDataset()
