"""Train the frozen-backbone age baseline; accepts one or more explicit epochs.

The AGE-2B sanity invocation is exactly one epoch. The test split is not
iterated by this script and is reserved for a later evaluation phase.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torchvision
from PIL import __version__ as pillow_version
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from age_model import MODEL_INIT_SEED, ResNet50AgeRegressor, parameter_counts
from agedb_dataset import AGE1A_HASH, AgeDBDataset, verify_metadata_hash

SPLIT_SEED = 5341
TRAINING_SEED = 20261003
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_WORKERS = 0
RUN_DIR = Path(__file__).resolve().parent / "output" / "runs" / "age2b_one_epoch"


def select_device() -> tuple[torch.device, str]:
    """Probe basic MPS tensor/conv support; fall back to CPU on failure."""
    if not torch.backends.mps.is_available():
        return torch.device("cpu"), "MPS unavailable"
    try:
        probe = torch.nn.Conv2d(3, 4, kernel_size=3).to("mps")
        with torch.no_grad():
            result = probe(torch.zeros(1, 3, 16, 16, device="mps"))
            torch.mps.synchronize()
        if result.shape != (1, 4, 14, 14) or not torch.isfinite(result).all():
            raise RuntimeError("unexpected MPS probe result")
        return torch.device("mps"), "MPS compatibility probe passed"
    except Exception as exc:
        return torch.device("cpu"), f"MPS compatibility probe failed: {exc}"


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_epoch(model: ResNet50AgeRegressor, loader: DataLoader, optimizer: Adam,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    absolute_error_sum = 0.0
    count = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training prediction or L1 loss")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise RuntimeError("Frozen backbone unexpectedly received gradients")
        if not all(parameter.grad is not None for parameter in model.regression_head.parameters()):
            raise RuntimeError("Regression head is missing gradients")
        optimizer.step()
        batch_size = targets.numel()
        absolute_error_sum += loss.detach().item() * batch_size
        count += batch_size
    if count != len(loader.dataset):
        raise RuntimeError(f"Training consumed {count} rows; expected {len(loader.dataset)}")
    return absolute_error_sum / count


@torch.no_grad()
def validate_epoch(model: ResNet50AgeRegressor, loader: DataLoader,
                   criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    absolute_errors: list[float] = []
    predictions_all: list[float] = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images)
        loss = criterion(predictions, targets)
        if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation prediction or L1 loss")
        absolute_errors.extend((predictions - targets).abs().cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
    if len(absolute_errors) != len(loader.dataset):
        raise RuntimeError(f"Validation consumed {len(absolute_errors)} rows; expected {len(loader.dataset)}")
    return {
        "mae": statistics.mean(absolute_errors),
        "absolute_error_std": statistics.pstdev(absolute_errors),
        "prediction_min": min(predictions_all),
        "prediction_max": max(predictions_all),
        "prediction_mean": statistics.mean(predictions_all),
        "prediction_count": len(predictions_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive, learning-rate positive, num-workers nonnegative")

    metadata_hash = verify_metadata_hash()
    if metadata_hash != AGE1A_HASH:
        raise RuntimeError("AGE-1A metadata hash differs from the confirmed baseline")
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    try:
        import numpy as np
        np.random.seed(args.training_seed)
    except ImportError:
        pass

    # Build only the train and validation datasets/loaders. The test split is
    # deliberately not instantiated or iterated during training/model choice.
    root_value = os.environ.get("AGEDB_ROOT")
    if not root_value:
        raise ValueError("Set AGEDB_ROOT to the AgeDB directory containing images/")
    dataset_root = Path(root_value).expanduser()
    train_dataset = AgeDBDataset("train", dataset_root)
    validation_dataset = AgeDBDataset("validation", dataset_root)
    if len(train_dataset) != 11_457 or len(validation_dataset) != 2_685:
        raise RuntimeError("Train/validation counts do not match the AGE-1A baseline")
    generator = torch.Generator().manual_seed(args.training_seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False)

    device, device_note = select_device()
    model = ResNet50AgeRegressor(initialization_seed=MODEL_INIT_SEED).to(device)
    model.train()
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Backbone parameters must all be frozen")
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable_parameters) != sum(1 for _ in model.regression_head.parameters()):
        raise RuntimeError("Optimizer parameter selection would include unexpected parameters")
    backbone_initial = {name: value.detach().cpu().clone()
                        for name, value in model.backbone.state_dict().items()}
    head_initial = {name: value.detach().cpu().clone()
                    for name, value in model.regression_head.state_dict().items()}
    optimizer = Adam(trainable_parameters, lr=args.learning_rate)
    criterion = nn.L1Loss()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_validation_mae = math.inf
    total_start = time.perf_counter()
    final_validation: dict = {}
    training_seconds = validation_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
        training_seconds += time.perf_counter() - start
        start = time.perf_counter()
        validation = validate_epoch(model, validation_loader, criterion, device)
        validation_seconds += time.perf_counter() - start
        final_validation = validation
        learning_rate = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch, "train_mae": train_mae,
            "validation_mae": validation["mae"],
            "validation_absolute_error_std": validation["absolute_error_std"],
            "learning_rate": learning_rate,
        }
        if not all(math.isfinite(value) for key, value in record.items() if key != "epoch"):
            raise FloatingPointError("Non-finite value in epoch metrics")
        history.append(record)
        atomic_json(args.run_dir / "training_history.json", history)
        print(json.dumps(record, sort_keys=True), flush=True)

        if validation["mae"] < best_validation_mae:
            best_validation_mae = validation["mae"]
            checkpoint = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": {
                    "architecture": "resnet50_frozen_age_regressor",
                    "backbone": "torchvision.models.resnet50",
                    "backbone_weights": "ResNet50_Weights.DEFAULT",
                    "backbone_frozen": True,
                    "backbone_batchnorm_eval": True,
                    "head": ["Linear(2048,256)", "ReLU", "Linear(256,1)"],
                    "output_shape": "[B]", "output_clamped": False,
                    "loss": "torch.nn.L1Loss",
                },
                "epoch": epoch, "validation_mae": validation["mae"],
                "model_initialization_seed": MODEL_INIT_SEED,
                "training_seed": args.training_seed, "split_seed": SPLIT_SEED,
                "agedb_metadata_sha256": metadata_hash,
                "training_config": {
                    "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                    "optimizer": "Adam", "loss": "L1Loss",
                    "num_workers": args.num_workers, "device": str(device),
                    "epochs": args.epochs,
                },
                "runtime": {"python": sys.version.split()[0],
                            "torch": torch.__version__, "torchvision": torchvision.__version__,
                            "Pillow": pillow_version},
            }
            torch.save(checkpoint, args.run_dir / "best_checkpoint.pt")

    backbone_unchanged = all(torch.equal(value, model.backbone.state_dict()[name].detach().cpu())
                             for name, value in backbone_initial.items())
    head_changed = any(not torch.equal(value, model.regression_head.state_dict()[name].detach().cpu())
                       for name, value in head_initial.items())
    backbone_frozen = all(not parameter.requires_grad and parameter.grad is None
                          for parameter in model.backbone.parameters()) and backbone_unchanged
    if not backbone_frozen or not head_changed:
        raise RuntimeError(f"Post-run parameter validation failed: backbone_frozen={backbone_frozen}, head_changed={head_changed}")

    elapsed = time.perf_counter() - total_start
    summary = {
        "epochs_executed": args.epochs, "device": str(device), "device_note": device_note,
        "training_seconds": training_seconds, "validation_seconds": validation_seconds,
        "total_seconds": elapsed, "final_validation": final_validation,
        "backbone_frozen_and_unchanged": backbone_frozen, "head_updated": head_changed,
        "test_set_used": False, "agedb_metadata_sha256": metadata_hash,
    }
    atomic_json(args.run_dir / "run_summary.json", summary)
    print(json.dumps({"summary": summary, "checkpoint": str(args.run_dir / "best_checkpoint.pt"),
                      "history": str(args.run_dir / "training_history.json")}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
