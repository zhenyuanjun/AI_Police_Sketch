"""Frozen ImageNet ResNet50 with a deterministic FP32 age-regression head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

MODEL_INIT_SEED = 20_261_002


class ResNet50AgeRegressor(nn.Module):
    """ResNet50 features plus a trainable 2048 -> 256 -> 1 regression head.

    The pretrained backbone is permanently kept in eval mode, including when
    ``model.train()`` is called, so BatchNorm running statistics stay fixed.
    Backbone forward passes run without autograd because all its weights are
    frozen. The model returns a float32 prediction tensor with shape ``[B]``.
    """

    def __init__(self, initialization_seed: int = MODEL_INIT_SEED):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        # Isolate initialization from caller RNG state while making the new
        # head reproducible for a fixed seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.regression_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        self.initialization_seed = initialization_seed
        self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNet50AgeRegressor":
        super().train(mode)
        # Frozen BatchNorm layers must not update running means/variances.
        self.backbone.eval()
        return self

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return frozen pooled features with shape ``[B, 2048]``."""
        with torch.no_grad():
            return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(images)
        return self.regression_head(features).squeeze(1)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count total, frozen, and trainable scalar parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "frozen": total - trainable, "trainable": trainable}


def select_device() -> torch.device:
    """Prefer Apple Silicon MPS when available; otherwise use CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
