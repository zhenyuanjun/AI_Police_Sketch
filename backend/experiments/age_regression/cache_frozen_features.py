"""Cache AGE-1A train/validation raw ResNet50 features (never test features)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from age_model import MODEL_INIT_SEED, ResNet50AgeRegressor
from agedb_dataset import AGE1A_HASH, AgeDBDataset, EXPECTED_COUNTS, verify_metadata_hash
from train_age_regressor import select_device

CACHE_DIR = Path(__file__).resolve().parent / "output" / "features"
EXPECTED_FEATURE_DIM = 2048
SPLIT_SEED = 5341


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def extract_split(dataset: AgeDBDataset, split: str, model: ResNet50AgeRegressor,
                  device: torch.device, batch_size: int, num_workers: int,
                  output_dir: Path) -> tuple[dict, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    feature_batches: list[torch.Tensor] = []
    age_batches: list[torch.Tensor] = []
    start = time.perf_counter()
    model.backbone.eval()
    with torch.inference_mode():
        for batch_index, (images, ages) in enumerate(loader, start=1):
            features = model.extract_features(images.to(device)).to(device="cpu", dtype=torch.float32)
            feature_batches.append(features.contiguous())
            age_batches.append(ages.to(device="cpu", dtype=torch.float32))
            if batch_index % 50 == 0 or batch_index == len(loader):
                print(f"{split}: extracted {min(batch_index * batch_size, len(dataset))}/{len(dataset)} images",
                      flush=True)
    extraction_seconds = time.perf_counter() - start
    features = torch.cat(feature_batches, dim=0)
    ages = torch.cat(age_batches, dim=0)
    samples = [
        {"image_id": row["image_id"], "relative_path": row["relative_path"],
         "identity": row["identity"], "age": int(row["age"]),
         "gender": row["gender"], "split": split}
        for row in dataset.rows
    ]
    if len(samples) != EXPECTED_COUNTS[split] or features.shape != (EXPECTED_COUNTS[split], EXPECTED_FEATURE_DIM):
        raise RuntimeError(f"{split} cache dimensions/count do not match AGE-1A: {len(samples)}, {tuple(features.shape)}")
    if features.dtype != torch.float32 or ages.dtype != torch.float32:
        raise RuntimeError(f"{split} cache must retain FP32 features and ages")
    if not torch.isfinite(features).all() or not torch.isfinite(ages).all():
        raise FloatingPointError(f"{split} cache contains NaN or Inf")
    if not torch.all((ages >= 1) & (ages <= 101)):
        raise ValueError(f"{split} ages fall outside 1..101")
    if [row["image_id"] for row in samples] != [row["image_id"] for row in dataset.rows]:
        raise RuntimeError(f"{split} cache metadata order changed")

    cache_path = output_dir / f"agedb_{split}_features.pt"
    payload = {
        "features": features,
        "ages": ages,
        "samples": samples,
        "metadata": {
            "split": split,
            "sample_count": len(samples),
            "feature_shape": list(features.shape),
            "feature_dtype": "float32",
            "age_dtype": "float32",
            "agedb_metadata_sha256": AGE1A_HASH,
            "split_seed": SPLIT_SEED,
            "model_architecture": "torchvision ResNet50, classifier replaced by Identity",
            "pretrained_weights": str(ResNet50_Weights.DEFAULT),
            "preprocessing": "ResNet50_Weights.DEFAULT.transforms() (RGB, resize 232, center crop 224, ImageNet normalization)",
            "backbone_frozen": True,
            "backbone_eval_mode": True,
            "model_initialization_seed": MODEL_INIT_SEED,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "pillow_version": pillow_version,
            "device": str(device),
            "extraction_batch_size": batch_size,
            "num_workers": num_workers,
            "ordering": "AGE-1A manifest row order (filename lexical order)",
            "dataset_root": str(dataset.dataset_root.resolve()),
            "extraction_seconds": extraction_seconds,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    atomic_torch_save(payload, cache_path)
    return payload, extraction_seconds


@torch.inference_mode()
def parity_check(dataset: AgeDBDataset, payload: dict, model: ResNet50AgeRegressor,
                 device: torch.device) -> dict[str, float]:
    # Fixed endpoints and midpoint: deterministic and only three images/split.
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    cache_ids = [sample["image_id"] for sample in payload["samples"]]
    direct_values, cached_values = [], []
    model.backbone.eval()
    for index in indices:
        row = dataset.rows[index]
        if cache_ids[index] != row["image_id"]:
            raise RuntimeError(f"Cached metadata ordering mismatch at {dataset.split} index {index}")
        image, _ = dataset[index]
        direct = model.extract_features(image.unsqueeze(0).to(device)).cpu()[0]
        cached = payload["features"][index]
        direct_values.append(direct)
        cached_values.append(cached)
    direct_tensor = torch.stack(direct_values)
    cached_tensor = torch.stack(cached_values)
    absolute = (direct_tensor - cached_tensor).abs()
    if not torch.allclose(direct_tensor, cached_tensor, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"{dataset.split} direct/cached feature parity exceeded FP32 tolerance")
    return {"max_absolute_difference": float(absolute.max()),
            "mean_absolute_difference": float(absolute.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")

    metadata_hash = verify_metadata_hash()
    if metadata_hash != AGE1A_HASH:
        raise RuntimeError(f"Unexpected AGE-1A metadata hash: {metadata_hash}")
    dataset_root_value = os.environ.get("AGEDB_ROOT")
    if not dataset_root_value:
        raise ValueError("Set AGEDB_ROOT to the AgeDB directory containing images/")
    dataset_root = Path(dataset_root_value).expanduser()

    # Intentionally instantiate only train and validation. No test Dataset or
    # DataLoader is created, and no test image is opened by this script.
    train_dataset = AgeDBDataset("train", dataset_root)
    validation_dataset = AgeDBDataset("validation", dataset_root)
    if len(train_dataset) != 11_457 or len(validation_dataset) != 2_685:
        raise RuntimeError("Train/validation sizes differ from the confirmed AGE-1A split")
    train_identities = {row["identity"] for row in train_dataset.rows}
    validation_identities = {row["identity"] for row in validation_dataset.rows}
    if train_identities & validation_identities:
        raise RuntimeError("AGE-1A train and validation identities overlap")

    device, device_note = select_device()
    model = ResNet50AgeRegressor(initialization_seed=MODEL_INIT_SEED).to(device)
    model.eval()
    model.backbone.eval()
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("ResNet50 backbone is not fully frozen")
    if model.backbone.training:
        raise RuntimeError("ResNet50 backbone is not in eval mode")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_files = [args.output_dir / "agedb_train_features.pt",
                      args.output_dir / "agedb_validation_features.pt",
                      args.output_dir / "cache_report.json"]
    if any(path.exists() for path in expected_files):
        raise FileExistsError(f"Refusing to overwrite existing feature-cache artifacts in {args.output_dir}")
    total_start = time.perf_counter()
    train_payload, train_seconds = extract_split(train_dataset, "train", model, device,
                                                 args.batch_size, args.num_workers, args.output_dir)
    validation_payload, validation_seconds = extract_split(
        validation_dataset, "validation", model, device, args.batch_size,
        args.num_workers, args.output_dir)
    train_parity = parity_check(train_dataset, train_payload, model, device)
    validation_parity = parity_check(validation_dataset, validation_payload, model, device)
    process_seconds = time.perf_counter() - total_start
    total_seconds = train_seconds + validation_seconds

    cache_paths = {
        "train": args.output_dir / "agedb_train_features.pt",
        "validation": args.output_dir / "agedb_validation_features.pt",
    }
    cache_hashes = {split: sha256_file(path) for split, path in cache_paths.items()}
    cache_sizes = {split: path.stat().st_size for split, path in cache_paths.items()}
    report = {
        "agedb_metadata_sha256": metadata_hash,
        "split_seed": SPLIT_SEED,
        "feature_dimension": EXPECTED_FEATURE_DIM,
        "feature_dtype": "float32",
        "device": str(device), "device_note": device_note,
        "extraction_batch_size": args.batch_size, "num_workers": args.num_workers,
        "train": {"samples": len(train_payload["samples"]), "shape": list(train_payload["features"].shape),
                  "dtype": str(train_payload["features"].dtype), "path": str(cache_paths["train"]),
                  "size_bytes": cache_sizes["train"], "sha256": cache_hashes["train"],
                  "extraction_seconds": train_seconds, "parity": train_parity},
        "validation": {"samples": len(validation_payload["samples"]),
                       "shape": list(validation_payload["features"].shape),
                       "dtype": str(validation_payload["features"].dtype),
                       "path": str(cache_paths["validation"]),
                       "size_bytes": cache_sizes["validation"], "sha256": cache_hashes["validation"],
                       "extraction_seconds": validation_seconds, "parity": validation_parity},
        "train_validation_identity_overlap": 0,
        "test_cache_created": False, "test_images_accessed": False,
        "total_extraction_seconds": total_seconds,
        "cache_validation_and_parity_seconds": process_seconds - total_seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.output_dir / "cache_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
