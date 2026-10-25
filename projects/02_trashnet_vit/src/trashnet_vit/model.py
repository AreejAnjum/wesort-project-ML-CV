from __future__ import annotations

import torch
from torch import nn


class TinyEncoder(nn.Module):
    """Small test-only encoder used by the CPU smoke test."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.num_features = 32

    def forward(self, images):
        return self.network(images).flatten(1)


def build_encoder(name: str, pretrained: bool):
    if name == "tiny":
        return TinyEncoder(), 32
    import timm

    encoder = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return encoder, encoder.num_features


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, projection_dim: int):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, images):
        return self.projector(self.encoder(images))


def nt_xent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float):
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Representations must have matching [batch, projection] shapes")
    if first.shape[0] < 2:
        raise ValueError("NT-Xent requires at least two images per batch")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    first = nn.functional.normalize(first, dim=1)
    second = nn.functional.normalize(second, dim=1)
    features = torch.cat((first, second))
    logits = features @ features.T / temperature
    batch_size = first.shape[0]
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat((torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)))
    return nn.functional.cross_entropy(logits, targets.to(logits.device))
