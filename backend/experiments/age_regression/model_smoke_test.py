"""Single-batch architecture and gradient smoke check; never updates weights."""

from __future__ import annotations

import torch
from torch import nn

from age_model import MODEL_INIT_SEED, ResNet50AgeRegressor, parameter_counts, select_device
from agedb_dataset import build_dataloaders


def main() -> None:
    loaders = build_dataloaders(batch_size=2, num_workers=0)
    images, ages = next(iter(loaders["train"]))
    device = select_device()
    model = ResNet50AgeRegressor().to(device)
    model.train()
    images, ages = images.to(device), ages.to(device)

    bn_layers = [module for module in model.backbone.modules() if isinstance(module, nn.BatchNorm2d)]
    before = [(module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
              for module in bn_layers]
    features = model.extract_features(images)
    predictions = model(images)
    loss = nn.L1Loss()(predictions, ages)
    assert images.shape == (2, 3, 224, 224)
    assert features.shape == (2, 2048)
    assert predictions.shape == ages.shape == (2,)
    assert predictions.dtype == ages.dtype == loss.dtype == torch.float32
    assert torch.isfinite(loss)

    loss.backward()
    backbone_gradients = [p.grad for p in model.backbone.parameters()]
    head_gradients = [p.grad for p in model.regression_head.parameters()]
    assert all(gradient is None for gradient in backbone_gradients)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in head_gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in head_gradients)
    assert all(not module.training for module in bn_layers)
    assert all(torch.equal(mean, module.running_mean) and torch.equal(var, module.running_var)
               and torch.equal(count, module.num_batches_tracked)
               for module, (mean, var, count) in zip(bn_layers, before))

    counts = parameter_counts(model)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.regression_head.parameters())
    for parameter in model.parameters():
        parameter.grad = None

    print(f"AGE-1A split seed: 5341")
    print(f"AGE-1A hash: abf489c4cfa335fedd829753b3eb77b9bb03b99184d7fd85cdbf2bcbca2b7b99")
    print(f"Model initialization seed: {MODEL_INIT_SEED}")
    print(f"Device: {device}")
    print(f"Input: {list(images.shape)}")
    print(f"Features: {list(features.shape)}")
    print(f"Prediction: {list(predictions.shape)} {predictions.dtype}")
    print(f"Target: {list(ages.shape)} {ages.dtype}")
    print(f"One smoke loss (L1): {loss.item():.6f}")
    print(f"Parameters: total={counts['total']}, frozen={counts['frozen']}, trainable={counts['trainable']}")
    print("Backbone gradients: NONE")
    print("Head gradients: PRESENT (cleared after check)")
    print("BatchNorm running statistics: unchanged in train mode")
    print("Optimizer step / training: not performed")


if __name__ == "__main__":
    main()
