"""Read-only inference service for the finalized AGE INT8 backbone and FP32 head."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BACKBONE_PATH = (
    BACKEND_DIR / "experiments" / "age_regression" / "output" / "age4b2_int8" / "resnet50_backbone_w8a8.tflite"
)
DEFAULT_HEAD_PATH = (
    BACKEND_DIR / "experiments" / "age_regression" / "output" / "runs" / "age3a_cached_head" / "best_checkpoint.pt"
)

EXPECTED_BACKBONE_SHA256 = "c3bdfb8002a40fe4892a2179616ea80fa2a2437d5c6a4402c01d0d15b9f6d5a0"
EXPECTED_HEAD_SHA256 = "c9d757ee0dee4ce1aff96ec8526d001e3072e30ca27264991f0db0c4b4320fe8"
EXPECTED_INPUT_SHAPE = (1, 3, 224, 224)
EXPECTED_OUTPUT_SHAPE = (1, 2048)
EXPECTED_INPUT_SCALE = 0.016827620565891266
EXPECTED_INPUT_ZERO_POINT = -11
EXPECTED_OUTPUT_SCALE = 0.021488072350621223
EXPECTED_OUTPUT_ZERO_POINT = -128
TRAINING_AGE_MIN, TRAINING_AGE_MAX = 1.0, 101.0
MODEL_IDENTIFIER = "agedb-resnet50-w8a8-backbone-fp32-age-head"
TRAINING_EVALUATION_DOMAIN = "AgeDB photographs"
APPLICATION_INTERPRETATION = "exploratory sketch-domain signal"


class AgeEstimationError(RuntimeError):
    """Base class for age-service failures that callers can handle cleanly."""


class ArtifactNotFoundError(AgeEstimationError):
    pass


class ArtifactProvenanceError(AgeEstimationError):
    pass


class LiteRTContractError(AgeEstimationError):
    pass


class InvalidImageError(AgeEstimationError):
    pass


class BackboneInferenceError(AgeEstimationError):
    pass


class HeadInferenceError(AgeEstimationError):
    pass


@dataclass(frozen=True)
class AgeEstimationResult:
    """Inference data only; API/UI wording belongs to a later integration phase."""

    estimated_age: float
    raw_prediction: float
    out_of_training_range: bool
    model_identifier: str
    backbone_artifact_hash: str
    age_head_artifact_hash: str
    training_evaluation_domain: str
    application_interpretation: str
    preprocessing_shape: tuple[int, ...]
    raw_int8_feature_shape: tuple[int, ...]
    dequantized_feature_shape: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_artifact(path: Path, expected_hash: str, label: str) -> str:
    if not path.is_file():
        raise ArtifactNotFoundError(f"Finalized {label} artifact is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ArtifactProvenanceError(
            f"Finalized {label} artifact SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )
    return actual_hash


def _quantization(detail: dict, label: str) -> tuple[float, int]:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    try:
        return float(scale), int(zero_point)
    except (TypeError, ValueError) as exc:
        raise LiteRTContractError(f"{label} tensor quantization metadata is invalid") from exc


class AgeEstimationService:
    """Loads finalized artifacts once and performs one image-to-age inference."""

    def __init__(
        self,
        backbone_path: Path | str = DEFAULT_BACKBONE_PATH,
        head_path: Path | str = DEFAULT_HEAD_PATH,
        *,
        num_threads: int | None = 4,
    ) -> None:
        self.backbone_path = Path(backbone_path)
        self.head_path = Path(head_path)
        self.backbone_artifact_hash = _checked_artifact(
            self.backbone_path, EXPECTED_BACKBONE_SHA256, "INT8 backbone"
        )
        self.age_head_artifact_hash = _checked_artifact(
            self.head_path, EXPECTED_HEAD_SHA256, "FP32 age head"
        )
        self._interpreter = self._load_interpreter(num_threads)
        self._input_detail, self._output_detail = self._validate_litert_contract()
        self._input_scale, self._input_zero_point = _quantization(self._input_detail, "input")
        self._output_scale, self._output_zero_point = _quantization(self._output_detail, "output")
        self._interpreter_lock = Lock()
        self._head = self._load_head()
        self._preprocess = ResNet50_Weights.DEFAULT.transforms()

    def _load_interpreter(self, num_threads: int | None):
        try:
            from ai_edge_litert.interpreter import Interpreter

            interpreter = Interpreter(model_path=str(self.backbone_path), num_threads=num_threads)
            interpreter.allocate_tensors()
            return interpreter
        except AgeEstimationError:
            raise
        except Exception as exc:
            raise LiteRTContractError("Could not load finalized INT8 LiteRT backbone") from exc

    def _validate_litert_contract(self) -> tuple[dict, dict]:
        inputs, outputs = self._interpreter.get_input_details(), self._interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise LiteRTContractError(f"Expected one LiteRT input/output, got {len(inputs)}/{len(outputs)}")
        input_detail, output_detail = inputs[0], outputs[0]
        if tuple(int(x) for x in input_detail["shape"]) != EXPECTED_INPUT_SHAPE:
            raise LiteRTContractError(f"Unexpected LiteRT input shape: {input_detail['shape']}")
        if np.dtype(input_detail["dtype"]) != np.dtype(np.int8):
            raise LiteRTContractError(f"Unexpected LiteRT input dtype: {input_detail['dtype']}")
        if tuple(int(x) for x in output_detail["shape"]) != EXPECTED_OUTPUT_SHAPE:
            raise LiteRTContractError(f"Unexpected LiteRT output shape: {output_detail['shape']}")
        if np.dtype(output_detail["dtype"]) != np.dtype(np.int8):
            raise LiteRTContractError(f"Unexpected LiteRT output dtype: {output_detail['dtype']}")
        input_scale, input_zero = _quantization(input_detail, "input")
        output_scale, output_zero = _quantization(output_detail, "output")
        if not math.isfinite(input_scale) or input_scale <= 0 or not np.isclose(
            input_scale, EXPECTED_INPUT_SCALE, rtol=0.0, atol=1e-12
        ) or input_zero != EXPECTED_INPUT_ZERO_POINT:
            raise LiteRTContractError("Unexpected LiteRT input quantization contract")
        if not math.isfinite(output_scale) or output_scale <= 0 or not np.isclose(
            output_scale, EXPECTED_OUTPUT_SCALE, rtol=0.0, atol=1e-12
        ) or output_zero != EXPECTED_OUTPUT_ZERO_POINT:
            raise LiteRTContractError("Unexpected LiteRT output quantization contract")
        return input_detail, output_detail

    def _load_head(self) -> nn.Sequential:
        try:
            checkpoint = torch.load(self.head_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["model_state_dict"]
            expected_keys = {"0.weight", "0.bias", "2.weight", "2.bias"}
            if set(state_dict) != expected_keys:
                raise ValueError("unexpected state-dict keys")
            if tuple(state_dict["0.weight"].shape) != (256, 2048) or tuple(state_dict["0.bias"].shape) != (256,):
                raise ValueError("unexpected first linear-layer dimensions")
            if tuple(state_dict["2.weight"].shape) != (1, 256) or tuple(state_dict["2.bias"].shape) != (1,):
                raise ValueError("unexpected output linear-layer dimensions")
            head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1)).cpu()
            head.load_state_dict(state_dict, strict=True)
            head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            return head
        except Exception as exc:
            raise HeadInferenceError("Could not load finalized FP32 age-head checkpoint") from exc

    def _prepare_input(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                tensor = self._preprocess(source.convert("RGB"))
        except (OSError, ValueError, TypeError) as exc:
            raise InvalidImageError("Image bytes cannot be decoded as an RGB image") from exc
        if tuple(tensor.shape) != EXPECTED_INPUT_SHAPE[1:] or tensor.dtype != torch.float32:
            raise InvalidImageError(f"Canonical preprocessing returned {tensor.shape} {tensor.dtype}")
        prepared = tensor.unsqueeze(0).contiguous().numpy().astype(np.float32, copy=False)
        if prepared.shape != EXPECTED_INPUT_SHAPE or not np.isfinite(prepared).all():
            raise InvalidImageError("Canonical preprocessing produced an invalid input tensor")
        return prepared

    def estimate_from_image_bytes(self, image_bytes: bytes) -> AgeEstimationResult:
        prepared = self._prepare_input(image_bytes)
        quantized = np.clip(
            np.rint(prepared / self._input_scale + self._input_zero_point), -128, 127
        ).astype(np.int8)
        try:
            with self._interpreter_lock:
                self._interpreter.set_tensor(self._input_detail["index"], quantized)
                self._interpreter.invoke()
                raw_feature = np.asarray(self._interpreter.get_tensor(self._output_detail["index"])).copy()
        except Exception as exc:
            raise BackboneInferenceError("Finalized INT8 LiteRT backbone inference failed") from exc
        if raw_feature.shape != EXPECTED_OUTPUT_SHAPE or raw_feature.dtype != np.int8:
            raise BackboneInferenceError("LiteRT backbone returned an unexpected output contract")
        dequantized = (raw_feature.astype(np.float32) - self._output_zero_point) * self._output_scale
        if not np.isfinite(dequantized).all():
            raise BackboneInferenceError("LiteRT backbone dequantization produced non-finite features")
        try:
            with torch.inference_mode():
                raw_prediction = float(self._head(torch.from_numpy(dequantized)).squeeze().item())
        except Exception as exc:
            raise HeadInferenceError("Finalized FP32 age-head inference failed") from exc
        if not math.isfinite(raw_prediction):
            raise HeadInferenceError("Finalized FP32 age head returned a non-finite prediction")
        return AgeEstimationResult(
            estimated_age=raw_prediction,
            raw_prediction=raw_prediction,
            out_of_training_range=raw_prediction < TRAINING_AGE_MIN or raw_prediction > TRAINING_AGE_MAX,
            model_identifier=MODEL_IDENTIFIER,
            backbone_artifact_hash=self.backbone_artifact_hash,
            age_head_artifact_hash=self.age_head_artifact_hash,
            training_evaluation_domain=TRAINING_EVALUATION_DOMAIN,
            application_interpretation=APPLICATION_INTERPRETATION,
            preprocessing_shape=tuple(prepared.shape),
            raw_int8_feature_shape=tuple(raw_feature.shape),
            dequantized_feature_shape=tuple(dequantized.shape),
        )
