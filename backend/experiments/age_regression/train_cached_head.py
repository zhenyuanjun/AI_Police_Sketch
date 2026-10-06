"""Train only the FP32 age-regression head from AGE-2D cached features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from agedb_dataset import AGE1A_HASH, EXPECTED_COUNTS, OUTPUT_DIR as AGE_OUTPUT_DIR, verify_metadata_hash

MODEL_INIT_SEED = 20_261_002
TRAINING_SEED = 20_261_003
SPLIT_SEED = 5341
CACHE_DIR = Path(__file__).resolve().parent / "output" / "features"
TRAIN_CACHE_SHA256 = "9d9226061e3e88f1bbef932993bddae819c1c68d36205a77b8074229bdc8edac"
VALIDATION_CACHE_SHA256 = "fd97df3e18ba12c069900748844d3de3ec747cad72574541ac7e871a7ff4843f"
RUN_DIR = Path(__file__).resolve().parent / "output" / "runs" / "age3a_cached_head"
MAX_EPOCHS = 20
PATIENCE = 5
MIN_DELTA = 1e-4  # years; used only to reset stopping patience


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def select_device() -> tuple[torch.device, str]:
    """Use MPS when available; otherwise keep cached head training on CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "MPS available"
    return torch.device("cpu"), "MPS unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(split: str, expected_hash: str, metadata_hash: str) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    path = CACHE_DIR / f"agedb_{split}_features.pt"
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{split} feature cache SHA-256 mismatch: {actual_hash}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("split") != split or metadata.get("agedb_metadata_sha256") != metadata_hash:
        raise ValueError(f"{split} cache metadata does not match AGE-1A")
    features, ages, samples = payload["features"], payload["ages"], payload["samples"]
    count = EXPECTED_COUNTS[split]
    if features.shape != (count, 2048) or features.dtype != torch.float32:
        raise ValueError(f"Invalid {split} feature shape/dtype: {tuple(features.shape)}, {features.dtype}")
    if ages.shape != (count,) or ages.dtype != torch.float32 or len(samples) != count:
        raise ValueError(f"Invalid {split} target or metadata dimensions")
    if not torch.isfinite(features).all() or not torch.isfinite(ages).all():
        raise FloatingPointError(f"{split} cache contains NaN/Inf")
    if not torch.all((ages >= 1) & (ages <= 101)):
        raise ValueError(f"{split} ages outside expected 1..101 range")
    if any(row.get("split") != split or int(row["age"]) != int(age)
           for row, age in zip(samples, ages.tolist())):
        raise ValueError(f"{split} cached sample metadata does not match cached targets")
    return features.contiguous(), ages.contiguous(), samples


def verify_sample_rows(train_samples: list[dict], validation_samples: list[dict]) -> None:
    # Compare the two cache row lists to only their corresponding AGE-1A rows.
    with (AGE_OUTPUT_DIR / "splits.csv").open(newline="", encoding="utf-8") as handle:
        splits = {row["image_id"]: row["split"] for row in csv.DictReader(handle)}
    if any(splits.get(row["image_id"]) != "train" for row in train_samples):
        raise ValueError("Train cache contains a non-train AGE-1A sample")
    if any(splits.get(row["image_id"]) != "validation" for row in validation_samples):
        raise ValueError("Validation cache contains a non-validation AGE-1A sample")
    train_ids = [row["image_id"] for row in train_samples]
    val_ids = [row["image_id"] for row in validation_samples]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate image IDs in cached split")
    if {row["identity"] for row in train_samples} & {row["identity"] for row in validation_samples}:
        raise ValueError("Train and validation identities overlap")


def make_head() -> nn.Sequential:
    # Mirrors ResNet50AgeRegressor's isolated initialization without importing
    # or constructing the ResNet50 backbone.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_INIT_SEED)
        return nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1))


def train_epoch(head: nn.Module, loader: DataLoader, optimizer: Adam,
                criterion: nn.Module, device: torch.device) -> float:
    head.train()
    absolute_sum = 0.0
    count = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("Non-finite training features or targets")
        optimizer.zero_grad(set_to_none=True)
        predictions = head(features).squeeze(1)
        loss = criterion(predictions, targets)
        if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training prediction or loss")
        loss.backward()
        if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in head.parameters()):
            raise FloatingPointError("Missing or non-finite regression-head gradient")
        optimizer.step()
        absolute_sum += float((predictions.detach() - targets).abs().sum().cpu())
        count += targets.numel()
    if count != len(loader.dataset):
        raise RuntimeError(f"Training consumed {count} of {len(loader.dataset)} samples")
    return absolute_sum / count


@torch.no_grad()
def validate_epoch(head: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> dict[str, float]:
    head.eval()
    errors: list[float] = []
    predictions_all: list[float] = []
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("Non-finite validation features or targets")
        predictions = head(features).squeeze(1)
        loss = criterion(predictions, targets)
        if not torch.isfinite(predictions).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation prediction or loss")
        errors.extend((predictions - targets).abs().cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
    if len(errors) != len(loader.dataset):
        raise RuntimeError("Validation did not consume the expected sample count")
    return {"mae": statistics.mean(errors),
            "absolute_error_std": statistics.pstdev(errors),
            "prediction_min": min(predictions_all),
            "prediction_max": max(predictions_all),
            "prediction_mean": statistics.mean(predictions_all)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite AGE-3A run artifacts: {args.run_dir}")
    run_start = time.perf_counter()

    metadata_hash = verify_metadata_hash()
    if metadata_hash != AGE1A_HASH:
        raise ValueError("AGE-1A metadata hash differs from the confirmed value")
    report = json.loads((CACHE_DIR / "cache_report.json").read_text(encoding="utf-8"))
    if report.get("agedb_metadata_sha256") != metadata_hash or report.get("test_cache_created") is not False:
        raise ValueError("AGE-2D cache report has invalid AGE-1A/test-cache metadata")
    if report["train"].get("sha256") != TRAIN_CACHE_SHA256 or report["validation"].get("sha256") != VALIDATION_CACHE_SHA256:
        raise ValueError("AGE-2D cache report hashes do not match the confirmed feature caches")

    train_x, train_y, train_samples = load_cache("train", TRAIN_CACHE_SHA256, metadata_hash)
    val_x, val_y, val_samples = load_cache("validation", VALIDATION_CACHE_SHA256, metadata_hash)
    verify_sample_rows(train_samples, val_samples)

    random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    generator = torch.Generator().manual_seed(TRAINING_SEED)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True,
                              drop_last=False, num_workers=0, generator=generator)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False,
                            drop_last=False, num_workers=0)
    device, device_note = select_device()
    head = make_head().to(device)
    optimized_parameters = list(head.parameters())
    if len(optimized_parameters) != 4 or any(not p.requires_grad for p in optimized_parameters):
        raise RuntimeError("Optimizer selection is not exactly the regression head")
    optimizer = Adam(optimized_parameters, lr=1e-3)
    criterion = nn.L1Loss()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_checkpoint_mae = math.inf
    stopping_best_mae = math.inf
    epochs_without_improvement = 0
    best_epoch = None
    best_validation = None
    train_seconds_total = 0.0
    validation_seconds_total = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.perf_counter()
        train_mae = train_epoch(head, train_loader, optimizer, criterion, device)
        train_seconds = time.perf_counter() - started
        train_seconds_total += train_seconds
        started = time.perf_counter()
        validation = validate_epoch(head, val_loader, criterion, device)
        validation_seconds = time.perf_counter() - started
        validation_seconds_total += validation_seconds
        lr = optimizer.param_groups[0]["lr"]
        record = {"epoch": epoch, "train_mae": train_mae,
                  "validation_mae": validation["mae"],
                  "validation_absolute_error_std": validation["absolute_error_std"],
                  "learning_rate": lr, "training_seconds": train_seconds,
                  "validation_seconds": validation_seconds}
        if not all(math.isfinite(v) for key, v in record.items() if key != "epoch"):
            raise FloatingPointError("Non-finite epoch metrics")
        history.append(record)
        atomic_json(args.run_dir / "training_history.json", history)
        print(json.dumps(record, sort_keys=True), flush=True)

        # Checkpoint on the literal minimum; min_delta controls only patience.
        if validation["mae"] < best_checkpoint_mae:
            best_checkpoint_mae = validation["mae"]
            best_epoch, best_validation = epoch, validation.copy()
            checkpoint = {
                "epoch": epoch,
                "validation_mae": validation["mae"],
                "validation_absolute_error_std": validation["absolute_error_std"],
                "validation_prediction_min": validation["prediction_min"],
                "validation_prediction_max": validation["prediction_max"],
                "validation_prediction_mean": validation["prediction_mean"],
                "model_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": {"head": ["Linear(2048,256)", "ReLU", "Linear(256,1)"],
                                 "input": "cached raw FP32 ResNet50 features [B,2048]",
                                 "output": "unclamped FP32 age in years [B]"},
                "model_initialization_seed": MODEL_INIT_SEED,
                "training_seed": TRAINING_SEED,
                "split_seed": SPLIT_SEED,
                "agedb_metadata_sha256": metadata_hash,
                "train_feature_cache_sha256": TRAIN_CACHE_SHA256,
                "validation_feature_cache_sha256": VALIDATION_CACHE_SHA256,
                "training_config": {"optimizer": "Adam", "learning_rate": 1e-3,
                                    "loss": "L1Loss", "batch_size": 32,
                                    "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE,
                                    "minimum_improvement_years": MIN_DELTA,
                                    "training_seed": TRAINING_SEED, "num_workers": 0,
                                    "device": str(device)},
            }
            temporary = args.run_dir / "best_checkpoint.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.run_dir / "best_checkpoint.pt")

        if validation["mae"] < stopping_best_mae - MIN_DELTA:
            stopping_best_mae = validation["mae"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break

    total_seconds = time.perf_counter() - run_start
    stopped_early = len(history) < MAX_EPOCHS
    summary = {
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation": best_validation, "early_stopping_triggered": stopped_early,
        "patience": PATIENCE, "minimum_improvement_years": MIN_DELTA,
        "device": str(device), "device_note": device_note,
        "total_wall_seconds": total_seconds,
        "average_training_seconds_per_epoch": train_seconds_total / len(history),
        "average_validation_seconds_per_epoch": validation_seconds_total / len(history),
        "training_seconds_total": train_seconds_total,
        "validation_seconds_total": validation_seconds_total,
        "train_cache_sha256": TRAIN_CACHE_SHA256,
        "validation_cache_sha256": VALIDATION_CACHE_SHA256,
        "agedb_metadata_sha256": metadata_hash,
        "resnet50_executed_during_training": False,
        "test_set_accessed": False,
        "optimized_parameter_count": sum(p.numel() for p in optimized_parameters),
        "optimized_modules": "regression head only",
        "numerical_validation": "PASS",
    }
    atomic_json(args.run_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
