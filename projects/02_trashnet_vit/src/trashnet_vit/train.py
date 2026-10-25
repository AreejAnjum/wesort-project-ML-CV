from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ManifestDataset, UnlabeledDataset
from .model import SimCLR, build_encoder, nt_xent_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(value: str):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset, cfg, shuffle=False, drop_last=False):
    generator = torch.Generator().manual_seed(cfg["seed"])
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg["num_workers"],
        generator=generator,
    )


def ensure_disjoint(**splits):
    paths = {name: {path.resolve() for path in dataset.paths} for name, dataset in splits.items()}
    names = list(paths)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = paths[left] & paths[right]
            if overlap:
                raise ValueError(f"Data leakage: {left} and {right} share {len(overlap)} image(s)")


def predict(encoder, head, dataset, cfg, device):
    truth, predictions = [], []
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for images, labels in make_loader(dataset, cfg):
            output = head(encoder(images.to(device))).argmax(1).cpu().tolist()
            truth.extend(labels.tolist())
            predictions.extend(output)
    names = [name for name, _ in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=list(range(len(names))),
            target_names=names,
            zero_division=0,
            output_dict=True,
        ),
    }


def train_classifier(encoder, feature_dim, train, val, cfg, device, fine_tune=False):
    encoder = encoder.to(device)
    head = nn.Linear(feature_dim, len(train.class_to_idx)).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad = fine_tune
    groups = [{"params": head.parameters(), "lr": cfg["head_learning_rate"]}]
    if fine_tune:
        groups.append({"params": encoder.parameters(), "lr": cfg["encoder_learning_rate"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    loss_function = nn.CrossEntropyLoss()
    best, best_score, history = None, -1.0, []
    for epoch in range(cfg["classifier_epochs"]):
        encoder.train(fine_tune)
        head.train()
        total = 0.0
        for images, labels in make_loader(train, cfg, shuffle=True):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if fine_tune:
                features = encoder(images)
            else:
                with torch.no_grad():
                    features = encoder(images)
            loss = loss_function(head(features), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
        metrics = predict(encoder, head, val, cfg, device)
        history.append({"epoch": epoch + 1, "loss": total / len(train), "val_macro_f1": metrics["macro_f1"]})
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best = {"encoder": deepcopy(encoder.state_dict()), "head": deepcopy(head.state_dict())}
    encoder.load_state_dict(best["encoder"])
    head.load_state_dict(best["head"])
    return encoder, head, history


def adapt_encoder(encoder, feature_dim, dataset, cfg, device):
    model = SimCLR(encoder, feature_dim, cfg["projection_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["ssl_learning_rate"], weight_decay=cfg["weight_decay"]
    )
    history = []
    data_loader = make_loader(dataset, cfg, shuffle=True, drop_last=True)
    for epoch in range(cfg["ssl_epochs"]):
        model.train()
        total, count = 0.0, 0
        for first, second in data_loader:
            first, second = first.to(device), second.to(device)
            optimizer.zero_grad()
            loss = nt_xent_loss(model(first), model(second), cfg["temperature"])
            loss.backward()
            optimizer.step()
            total += loss.item() * first.size(0)
            count += first.size(0)
        history.append({"epoch": epoch + 1, "ssl_loss": total / count})
    return model.encoder, history


def stage_result(encoder, feature_dim, train, val, test, cfg, device, fine_tune):
    encoder, head, history = train_classifier(encoder, feature_dim, train, val, cfg, device, fine_tune)
    metrics = predict(encoder, head, test, cfg, device)
    return encoder, head, {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                           "classification_report": metrics["classification_report"], "history": history}


def run(config_path: str):
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["training"]["seed"] = cfg["seed"]
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    image_size = cfg["data"]["image_size"]
    train = ManifestDataset(cfg["data"]["train_manifest"], image_size)
    val = ManifestDataset(cfg["data"]["val_manifest"], image_size, train.class_to_idx)
    test = ManifestDataset(cfg["data"]["test_manifest"], image_size, train.class_to_idx)
    unlabeled = UnlabeledDataset(cfg["data"]["train_manifest"], image_size)
    ensure_disjoint(train=train, val=val, test=test)
    if len(unlabeled) < cfg["training"]["batch_size"] or cfg["training"]["batch_size"] < 2:
        raise ValueError("Self-supervised training needs one full batch of at least two images")

    encoder, feature_dim = build_encoder(cfg["model"]["backbone"], cfg["model"]["pretrained"])
    initial = deepcopy(encoder.state_dict())
    results = {"metadata": cfg.get("metadata", {}), "device": str(device), "seed": cfg["seed"]}

    baseline_encoder, baseline_head, results["frozen_baseline"] = stage_result(
        encoder, feature_dim, train, val, test, cfg["training"], device, False
    )
    encoder.load_state_dict(initial)
    adapted, ssl_history = adapt_encoder(encoder, feature_dim, unlabeled, cfg["training"], device)
    adapted_state = deepcopy(adapted.state_dict())
    adapted, adapted_head, results["ssl_linear_probe"] = stage_result(
        adapted, feature_dim, train, val, test, cfg["training"], device, False
    )
    adapted.load_state_dict(adapted_state)
    tuned, tuned_head, results["ssl_fine_tuned"] = stage_result(
        adapted, feature_dim, train, val, test, cfg["training"], device, True
    )
    results["ssl_history"] = ssl_history

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name, stage_encoder, head in (
        ("frozen_baseline", baseline_encoder, baseline_head),
        ("ssl_linear_probe", adapted, adapted_head),
        ("ssl_fine_tuned", tuned, tuned_head),
    ):
        torch.save({"encoder": stage_encoder.state_dict(), "head": head.state_dict(),
                    "classes": train.class_to_idx, "config": cfg}, output / f"{name}.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train the TrashNet Vision Transformer pipeline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    results = run(args.config)
    printable = {key: value for key, value in results.items() if key not in {"ssl_history"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
