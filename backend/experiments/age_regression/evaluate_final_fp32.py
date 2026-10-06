"""One-time final AGE-1A test evaluation for the finalized AGE-3A FP32 head."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import PIL
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from age_model import MODEL_INIT_SEED, ResNet50AgeRegressor, select_device
from agedb_dataset import AGE1A_HASH, AgeDBDataset, EXPECTED_COUNTS, OUTPUT_DIR, verify_metadata_hash
from train_cached_head import TRAIN_CACHE_SHA256, VALIDATION_CACHE_SHA256, TRAINING_SEED, SPLIT_SEED

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_DIR / "output" / "runs" / "age3a_cached_head" / "best_checkpoint.pt"
AGE3A_SUMMARY = PROJECT_DIR / "output" / "runs" / "age3a_cached_head" / "run_summary.json"
RUN_DIR = PROJECT_DIR / "output" / "runs" / "age3b_final_test"
EXPECTED_BEST_EPOCH = 8
EXPECTED_VALIDATION_MAE = 9.630586811683697
EXPECTED_IDENTITIES = 85
EXPECTED_TEST_IMAGES = 2346


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(cpu.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(cpu.shape)).encode("ascii") + b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoint(checkpoint: dict, summary: dict, metadata_hash: str) -> None:
    errors: list[str] = []
    expected_architecture = ["Linear(2048,256)", "ReLU", "Linear(256,1)"]
    if checkpoint.get("architecture", {}).get("head") != expected_architecture:
        errors.append("head architecture differs")
    if checkpoint.get("epoch") != EXPECTED_BEST_EPOCH or summary.get("best_epoch") != EXPECTED_BEST_EPOCH:
        errors.append("best epoch is not 8")
    val_mae = checkpoint.get("validation_mae")
    if not isinstance(val_mae, (int, float)) or not math.isclose(
            val_mae, EXPECTED_VALIDATION_MAE, rel_tol=0.0, abs_tol=1e-8):
        errors.append("checkpoint validation MAE differs from finalized AGE-3A value")
    if not math.isclose(summary.get("best_validation", {}).get("mae", math.nan),
                         EXPECTED_VALIDATION_MAE, rel_tol=0.0, abs_tol=1e-8):
        errors.append("AGE-3A summary validation MAE differs")
    provenance = {
        "agedb_metadata_sha256": metadata_hash,
        "train_feature_cache_sha256": TRAIN_CACHE_SHA256,
        "validation_feature_cache_sha256": VALIDATION_CACHE_SHA256,
        "split_seed": SPLIT_SEED,
        "model_initialization_seed": MODEL_INIT_SEED,
        "training_seed": TRAINING_SEED,
    }
    errors.extend(f"{key} provenance mismatch" for key, expected in provenance.items()
                  if checkpoint.get(key) != expected)
    if summary.get("agedb_metadata_sha256") != metadata_hash:
        errors.append("AGE-3A summary AGE-1A hash differs")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or set(state) != {"0.weight", "0.bias", "2.weight", "2.bias"}:
        errors.append("checkpoint does not contain the expected head state_dict")
    if errors:
        raise ValueError("AGE-3A checkpoint provenance validation failed: " + "; ".join(errors))


def age_bin(age: int) -> str:
    if age >= 91:
        return "91+"
    low = ((age - 1) // 10) * 10 + 1
    return f"{low}–{low + 9}"


def main() -> None:
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite final test-evaluation artifacts: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    checkpoint_hash_before = sha256_file(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    summary = json.loads(AGE3A_SUMMARY.read_text(encoding="utf-8"))
    metadata_hash = verify_metadata_hash()
    if metadata_hash != AGE1A_HASH:
        raise ValueError("AGE-1A metadata hash differs from the confirmed split")
    validate_checkpoint(checkpoint, summary, metadata_hash)

    # The root comes from AGE-1A's own record so this evaluation follows the
    # exact dataset used to generate the immutable split metadata.
    report = json.loads((OUTPUT_DIR / "report.json").read_text(encoding="utf-8"))
    dataset_root = Path(report["source_dataset_path"]).expanduser()
    train_dataset = AgeDBDataset("train", dataset_root)
    validation_dataset = AgeDBDataset("validation", dataset_root)
    test_dataset = AgeDBDataset("test", dataset_root)
    train_ids = {row["identity"] for row in train_dataset.rows}
    validation_ids = {row["identity"] for row in validation_dataset.rows}
    test_ids = {row["identity"] for row in test_dataset.rows}
    train_overlap = train_ids & test_ids
    validation_overlap = validation_ids & test_ids
    if len(test_dataset) != EXPECTED_TEST_IMAGES or len(test_ids) != EXPECTED_IDENTITIES:
        raise ValueError(f"Unexpected test count/identity count: {len(test_dataset)}, {len(test_ids)}")
    if train_overlap or validation_overlap:
        raise ValueError("AGE-1A test identities overlap train or validation")

    device = select_device()
    model = ResNet50AgeRegressor(initialization_seed=MODEL_INIT_SEED).to(device)
    model.regression_head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if model.training or model.backbone.training or any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("Backbone must be frozen and in eval mode")
    params_before = state_sha256(model)

    # AgeDBDataset uses ResNet50_Weights.DEFAULT.transforms(), the canonical
    # preprocessing paired with the pretrained weights. Only the authorized
    # test split is iterated for images/features/predictions.
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)
    features_batches: list[torch.Tensor] = []
    predictions: list[float] = []
    targets: list[float] = []
    inference_start = time.perf_counter()
    model.backbone.eval()
    with torch.inference_mode():
        for images, ages in loader:
            if not torch.isfinite(images).all() or not torch.isfinite(ages).all():
                raise FloatingPointError("Non-finite test image tensor or target")
            features = model.extract_features(images.to(device))
            prediction = model.regression_head(features).squeeze(1)
            if not torch.isfinite(features).all() or not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite test feature or prediction")
            features_batches.append(features.to(device="cpu", dtype=torch.float32).contiguous())
            predictions.extend(prediction.to(device="cpu", dtype=torch.float64).tolist())
            targets.extend(ages.to(dtype=torch.float64).tolist())
    inference_seconds = time.perf_counter() - inference_start
    features_all = torch.cat(features_batches, dim=0)
    if features_all.shape != (EXPECTED_TEST_IMAGES, 2048) or features_all.dtype != torch.float32:
        raise RuntimeError(f"Unexpected test feature shape/dtype: {tuple(features_all.shape)}, {features_all.dtype}")
    if len(predictions) != EXPECTED_TEST_IMAGES or len(targets) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Test prediction count is not 2,346")

    signed = [pred - target for pred, target in zip(predictions, targets)]
    absolute = [abs(error) for error in signed]
    mae = statistics.mean(absolute)
    error_std = statistics.pstdev(absolute)
    rmse = math.sqrt(statistics.mean(error * error for error in signed))
    prediction_stats = {"minimum": min(predictions), "maximum": max(predictions),
                        "mean": statistics.mean(predictions), "median": statistics.median(predictions)}
    target_stats = {"minimum": min(targets), "maximum": max(targets),
                    "mean": statistics.mean(targets), "median": statistics.median(targets)}
    bins: dict[str, list[int]] = {label: [] for label in
                                  ("1–10", "11–20", "21–30", "31–40", "41–50", "51–60",
                                   "61–70", "71–80", "81–90", "91+")}
    records = []
    for sample, target, prediction, error, abs_error in zip(
            test_dataset.rows, targets, predictions, signed, absolute):
        bins[age_bin(int(target))].append(abs_error)
        records.append({"image_id": sample["image_id"], "relative_path": sample["relative_path"],
                        "identity": sample["identity"], "ground_truth_age": target,
                        "predicted_age": prediction, "absolute_error": abs_error,
                        "signed_error": error})
    age_bin_metrics = {
        label: {"sample_count": len(values),
                "mae": statistics.mean(values) if values else None,
                "absolute_error_std": statistics.pstdev(values) if values else None}
        for label, values in bins.items()
    }

    params_after = state_sha256(model)
    checkpoint_hash_after = sha256_file(CHECKPOINT)
    parameters_changed = params_before != params_after
    checkpoint_changed = checkpoint_hash_before != checkpoint_hash_after
    if parameters_changed or checkpoint_changed:
        raise RuntimeError("Model parameters or finalized checkpoint changed during evaluation")

    features_path = RUN_DIR / "test_features.pt"
    feature_payload = {
        "features": features_all,
        "ages": torch.tensor(targets, dtype=torch.float32),
        "samples": [{"image_id": row["image_id"], "relative_path": row["relative_path"],
                     "identity": row["identity"], "age": int(row["age"]),
                     "gender": row["gender"], "split": "test"} for row in test_dataset.rows],
        "metadata": {"split": "test", "sample_count": EXPECTED_TEST_IMAGES,
                     "feature_shape": list(features_all.shape), "feature_dtype": "float32",
                     "agedb_metadata_sha256": metadata_hash,
                     "checkpoint_sha256": checkpoint_hash_before,
                     "checkpoint_epoch": checkpoint["epoch"],
                     "model_initialization_seed": MODEL_INIT_SEED,
                     "training_seed": TRAINING_SEED, "split_seed": SPLIT_SEED,
                     "backbone": "torchvision ResNet50, ResNet50_Weights.DEFAULT",
                     "preprocessing": "ResNet50_Weights.DEFAULT.transforms()",
                     "backbone_frozen": True, "backbone_eval_mode": True,
                     "created_utc": datetime.now(timezone.utc).isoformat()},
    }
    temporary = features_path.with_suffix(".pt.tmp")
    torch.save(feature_payload, temporary)
    temporary.replace(features_path)
    feature_hash = sha256_file(features_path)
    feature_size = features_path.stat().st_size

    result = {
        "phase": "AGE-3B",
        "architecture": {"backbone": "torchvision ResNet50 / ResNet50_Weights.DEFAULT",
                          "backbone_frozen": True, "head": ["Linear(2048,256)", "ReLU", "Linear(256,1)"],
                          "output_clamped": False},
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_hash_before,
                       "epoch": checkpoint["epoch"], "validation_mae": checkpoint["validation_mae"],
                       "provenance_validation": "PASS"},
        "split": {"agedb_metadata_sha256": metadata_hash, "split_seed": SPLIT_SEED,
                  "test_images": len(test_dataset), "test_identities": len(test_ids),
                  "train_test_identity_overlap": len(train_overlap),
                  "validation_test_identity_overlap": len(validation_overlap)},
        "metrics": {"mae_years": mae, "absolute_error_std_years": error_std,
                    "median_absolute_error_years": statistics.median(absolute),
                    "rmse_years": rmse, "mean_signed_error_years": statistics.mean(signed)},
        "test_predictions": prediction_stats,
        "ground_truth": target_stats,
        "age_bin_metrics": age_bin_metrics,
        "test_feature_cache": {"created": True, "path": str(features_path),
                               "shape": list(features_all.shape), "dtype": "float32",
                               "size_bytes": feature_size, "sha256": feature_hash},
        "timing": {"feature_extraction_and_inference_seconds": inference_seconds,
                   "evaluation_wall_seconds": time.perf_counter() - wall_start,
                   "note": "Dataset-processing time; not a production-latency benchmark."},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "torch": torch.__version__, "torchvision": torchvision.__version__,
                    "Pillow": PIL.__version__, "device": str(device),
                    "weights": str(ResNet50_Weights.DEFAULT)},
        "numerical_validation": {"prediction_count": len(predictions),
                                 "all_predictions_finite": all(math.isfinite(x) for x in predictions),
                                 "all_targets_finite": all(math.isfinite(x) for x in targets),
                                 "all_errors_finite": all(math.isfinite(x) for x in signed),
                                 "identity_count": len(test_ids),
                                 "train_overlap": len(train_overlap),
                                 "validation_overlap": len(validation_overlap),
                                 "model_parameters_changed": parameters_changed,
                                 "checkpoint_changed": checkpoint_changed},
    }
    atomic_json(RUN_DIR / "evaluation_report.json", result)
    atomic_json(RUN_DIR / "predictions.json", records)


if __name__ == "__main__":
    main()
