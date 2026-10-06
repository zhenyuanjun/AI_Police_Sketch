#!/usr/bin/env python3
"""AGE-5B standalone, read-only smoke test for ``AgeEstimationService``."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

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
    EXPECTED_INPUT_SCALE,
    EXPECTED_INPUT_ZERO_POINT,
    EXPECTED_OUTPUT_SCALE,
    EXPECTED_OUTPUT_ZERO_POINT,
    EXPECTED_OUTPUT_SHAPE,
    AgeEstimationService,
    sha256_file,
)


def select_image() -> tuple[str, bytes, tuple[int, int]]:
    """Select one persisted PNG deterministically, or build a technical fallback."""
    image_dir = BACKEND_DIR / "data" / "generated_sketches"
    candidates = sorted(path for path in image_dir.glob("*.png") if path.is_file())
    if candidates:
        chosen = candidates[0]
        with Image.open(chosen) as image:
            dimensions = image.size
        return str(chosen.relative_to(PROJECT_DIR)), chosen.read_bytes(), dimensions

    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(224, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.arange(224, dtype=np.uint8)[:, None]
    pixels[..., 2] = 127
    stream = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    return "deterministic technical fallback image (no persisted sketch found)", stream.getvalue(), (224, 224)


def reference_prediction(image_bytes: bytes) -> float:
    """Independent direct implementation of the finalized INT8->head inference chain."""
    from ai_edge_litert.interpreter import Interpreter

    with Image.open(BytesIO(image_bytes)) as image:
        tensor = ResNet50_Weights.DEFAULT.transforms()(image.convert("RGB"))
    input_array = tensor.unsqueeze(0).contiguous().numpy().astype(np.float32, copy=False)
    quantized = np.clip(np.rint(input_array / EXPECTED_INPUT_SCALE + EXPECTED_INPUT_ZERO_POINT), -128, 127).astype(np.int8)
    interpreter = Interpreter(model_path=str(DEFAULT_BACKBONE_PATH), num_threads=4)
    interpreter.allocate_tensors()
    input_detail, output_detail = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], quantized)
    interpreter.invoke()
    raw_feature = np.asarray(interpreter.get_tensor(output_detail["index"])).copy()
    if raw_feature.shape != EXPECTED_OUTPUT_SHAPE or raw_feature.dtype != np.int8:
        raise RuntimeError("Reference LiteRT output contract mismatch")
    feature = (raw_feature.astype(np.float32) - EXPECTED_OUTPUT_ZERO_POINT) * EXPECTED_OUTPUT_SCALE
    checkpoint = torch.load(DEFAULT_HEAD_PATH, map_location="cpu", weights_only=True)
    head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1)).cpu().eval()
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    with torch.inference_mode():
        return float(head(torch.from_numpy(feature)).squeeze().item())


def main() -> None:
    image_label, image_bytes, dimensions = select_image()
    service = AgeEstimationService()
    result = service.estimate_from_image_bytes(image_bytes)
    reference = reference_prediction(image_bytes)
    difference = abs(result.raw_prediction - reference)
    report = {
        "phase": "AGE-5B",
        "smoke_test": {
            "image": image_label,
            "dimensions": list(dimensions),
            "input_shape": list(result.preprocessing_shape),
            "raw_int8_feature_shape": list(result.raw_int8_feature_shape),
            "dequantized_feature_shape": list(result.dequantized_feature_shape),
            "estimated_age": result.estimated_age,
            "out_of_training_range": result.out_of_training_range,
            "result": "PASS",
            "note": "Application-path smoke test only; the prediction is not interpreted as ground truth.",
        },
        "parity_check": {
            "samples": 1,
            "reference": "direct finalized INT8 LiteRT backbone -> dequantization -> finalized FP32 head",
            "maximum_age_prediction_difference": difference,
            "tolerance": 1e-5,
            "result": "PASS" if difference <= 1e-5 else "FAIL",
        },
        "provenance": {
            "backbone_sha256": sha256_file(DEFAULT_BACKBONE_PATH),
            "age_head_sha256": sha256_file(DEFAULT_HEAD_PATH),
        },
        "service_result": asdict(result),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["parity_check"]["result"] != "PASS":
        raise SystemExit("AGE-5B parity check failed")


if __name__ == "__main__":
    main()
