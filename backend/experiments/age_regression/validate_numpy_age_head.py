#!/usr/bin/env python3
"""Validate NumPy-head parity against PyTorch on random features and real images."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.age_estimation_service import (  # noqa: E402
    DEFAULT_BACKBONE_PATH,
    DEFAULT_HEAD_PATH,
    EXPECTED_BACKBONE_SHA256,
    EXPECTED_HEAD_SHA256,
    EXPECTED_HEAD_TENSORS,
    EXPECTED_INPUT_SCALE,
    EXPECTED_INPUT_ZERO_POINT,
    EXPECTED_OUTPUT_SCALE,
    EXPECTED_OUTPUT_ZERO_POINT,
    EXPECTED_OUTPUT_SHAPE,
    AgeEstimationService,
    sha256_file,
)

SOURCE_HEAD = BACKEND_DIR / "data" / "models" / "age_regression_head.pt"
EXPECTED_SOURCE_HEAD_SHA256 = "c9d757ee0dee4ce1aff96ec8526d001e3072e30ca27264991f0db0c4b4320fe8"
TOLERANCE = 5e-5


def numpy_head(features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    hidden = features @ arrays["linear1_weight"].T
    hidden = np.maximum(hidden + arrays["linear1_bias"], np.float32(0.0))
    return (hidden @ arrays["linear2_weight"].T + arrays["linear2_bias"]).reshape(-1)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    differences = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "samples": int(differences.size),
        "maximum_absolute_difference": float(np.max(differences)),
        "mean_absolute_difference": float(np.mean(differences)),
        "tolerance": TOLERANCE,
        "pass": bool(np.max(differences) <= TOLERANCE),
    }


def main() -> None:
    if sha256_file(DEFAULT_BACKBONE_PATH) != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError("Finalized LiteRT backbone hash mismatch")
    if sha256_file(DEFAULT_HEAD_PATH) != EXPECTED_HEAD_SHA256:
        raise RuntimeError("NumPy age-head artifact hash mismatch")
    checkpoint_hash = sha256_file(SOURCE_HEAD)
    if checkpoint_hash != EXPECTED_SOURCE_HEAD_SHA256:
        raise RuntimeError("Finalized PyTorch age-head checkpoint hash mismatch")
    checkpoint = torch.load(SOURCE_HEAD, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    torch_head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1)).eval()
    torch_head.load_state_dict(state, strict=True)
    with np.load(DEFAULT_HEAD_PATH, allow_pickle=False) as artifact:
        arrays = {name: np.asarray(artifact[name]).copy() for name in EXPECTED_HEAD_TENSORS}
    source_tensor_names = {
        "linear1_weight": "0.weight",
        "linear1_bias": "0.bias",
        "linear2_weight": "2.weight",
        "linear2_bias": "2.bias",
    }
    weights_bitwise_identical = all(
        np.array_equal(arrays[name], state[source_name].numpy())
        for name, source_name in source_tensor_names.items()
    )
    if not weights_bitwise_identical:
        raise RuntimeError("Exported NPZ tensors differ from the finalized checkpoint")
    torch.set_num_threads(1)

    rng = np.random.default_rng(534)
    random_features = rng.normal(0.0, 1.0, size=(512, 2048)).astype(np.float32)
    with torch.inference_mode():
        torch_random = torch_head(torch.from_numpy(random_features.copy())).squeeze(1).numpy()
    numpy_random = numpy_head(random_features, arrays)
    random_metrics = metrics(torch_random, numpy_random)

    service = AgeEstimationService(num_threads=1)
    interpreter = service._interpreter
    weights_transform = ResNet50_Weights.DEFAULT.transforms()
    image_paths = sorted((BACKEND_DIR / "data" / "generated_sketches").glob("*.png"))
    if not image_paths:
        raise RuntimeError("No persisted generated sketch PNGs available for image parity")

    predictions_torch: list[float] = []
    predictions_numpy: list[float] = []
    quantized_inputs_equal = 0
    preprocessing_max_difference = 0.0
    for path in image_paths:
        image_bytes = path.read_bytes()
        with Image.open(path) as source:
            canonical = weights_transform(source.convert("RGB")).unsqueeze(0).contiguous().numpy()
        lightweight = service._prepare_input(image_bytes)
        preprocessing_max_difference = max(
            preprocessing_max_difference,
            float(np.max(np.abs(canonical.astype(np.float64) - lightweight.astype(np.float64)))),
        )
        quantized_reference = np.clip(
            np.rint(canonical / EXPECTED_INPUT_SCALE + EXPECTED_INPUT_ZERO_POINT), -128, 127
        ).astype(np.int8)
        quantized_lightweight = np.clip(
            np.rint(lightweight / service._input_scale + service._input_zero_point), -128, 127
        ).astype(np.int8)
        if np.array_equal(quantized_reference, quantized_lightweight):
            quantized_inputs_equal += 1

        numpy_result = service.estimate_from_image_bytes(image_bytes)
        predictions_numpy.append(numpy_result.raw_prediction)

        # Reference the same LiteRT output with the original checkpoint's PyTorch head.
        interpreter.set_tensor(service._input_detail["index"], quantized_reference)
        interpreter.invoke()
        raw_feature = np.asarray(interpreter.get_tensor(service._output_detail["index"])).copy()
        if raw_feature.shape != EXPECTED_OUTPUT_SHAPE or raw_feature.dtype != np.int8:
            raise RuntimeError("LiteRT output contract mismatch during parity validation")
        feature = (raw_feature.astype(np.float32) - EXPECTED_OUTPUT_ZERO_POINT) * EXPECTED_OUTPUT_SCALE
        with torch.inference_mode():
            torch_prediction = float(torch_head(torch.from_numpy(feature.copy())).squeeze().item())
        predictions_torch.append(torch_prediction)

    image_metrics = metrics(np.asarray(predictions_torch), np.asarray(predictions_numpy))
    report = {
        "architecture": "Linear(2048,256) -> ReLU -> Linear(256,1)",
        "source_checkpoint_sha256": checkpoint_hash,
        "lightweight_artifact_sha256": sha256_file(DEFAULT_HEAD_PATH),
        "backbone_sha256": sha256_file(DEFAULT_BACKBONE_PATH),
        "exported_weights_bitwise_identical_to_checkpoint": weights_bitwise_identical,
        "random_feature_parity": random_metrics,
        "real_image_parity": image_metrics,
        "real_image_count": len(image_paths),
        "preprocessing_max_abs_difference": preprocessing_max_difference,
        "real_images_with_identical_quantized_inputs": quantized_inputs_equal,
        "preprocessing_quantized_input_parity": quantized_inputs_equal == len(image_paths),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not random_metrics["pass"] or not image_metrics["pass"]:
        raise SystemExit("NumPy age-head parity failed; do not use the lightweight head in production")
    if quantized_inputs_equal != len(image_paths):
        raise SystemExit("Preprocessing quantized-input parity failed; do not use the lightweight path")


if __name__ == "__main__":
    main()
