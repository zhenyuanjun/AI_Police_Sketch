#!/usr/bin/env python3
"""Smoke-test and compare the AGE FP32 LiteRT backbone against PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = PROJECT_DIR / "output" / "age4b1_litert" / "resnet50_backbone_fp32.tflite"
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 2048)
SEEDS = (534401, 42, 123, 456)


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    delta = reference - actual
    return {
        "max_absolute_difference": float(np.max(np.abs(delta))),
        "mean_absolute_difference": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": float(np.dot(reference.ravel(), actual.ravel()) /
                                    (np.linalg.norm(reference.ravel()) * np.linalg.norm(actual.ravel()))),
        "allclose_rtol_1e-3_atol_1e-4": bool(np.allclose(reference, actual, rtol=1e-3, atol=1e-4)),
    }


def get_delegate_observation(interpreter) -> dict:
    observation = {"status": "runtime does not expose delegate selection"}
    delegates = getattr(interpreter, "_delegates", None)
    if delegates is not None:
        observation = {"interpreter_delegates": [repr(item) for item in delegates]}
    get_ops = getattr(interpreter, "_get_ops_details", None)
    if callable(get_ops):
        try:
            names = sorted({str(item.get("op_name")) for item in get_ops() if item.get("op_name")})
            observation["runtime_operator_names"] = names
            matches = [name for name in names if "xnnpack" in name.lower()]
            observation["xnnpack_operator_observed"] = matches or None
            observation["delegate_nodes_observed"] = [name for name in names if "DELEGATE" in name.upper()]
        except Exception as exc:  # optional private runtime diagnostics
            observation["operator_diagnostics_error"] = str(exc)
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    export_report_path = artifact.with_suffix(".export.json")
    if not artifact.is_file() or not export_report_path.is_file():
        raise FileNotFoundError(f"Need LiteRT artifact and export report: {artifact}")
    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_hash = state_sha256(model)
    source_hash = export_report["source"]["state_sha256"]
    if model_hash != source_hash:
        raise RuntimeError(f"Backbone state hash differs from export source: {model_hash} != {source_hash}")

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(artifact))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(f"Expected single input/output, got {len(input_details)}/{len(output_details)}")
    input_info, output_info = input_details[0], output_details[0]
    in_shape = tuple(int(x) for x in input_info["shape"])
    out_shape = tuple(int(x) for x in output_info["shape"])
    if in_shape != INPUT_SHAPE or np.dtype(input_info["dtype"]) != np.dtype(np.float32):
        raise RuntimeError(f"Unexpected LiteRT input contract: {in_shape} {input_info['dtype']}")
    if out_shape != OUTPUT_SHAPE or np.dtype(output_info["dtype"]) != np.dtype(np.float32):
        raise RuntimeError(f"Unexpected LiteRT output contract: {out_shape} {output_info['dtype']}")

    results = []
    differences = []
    with torch.inference_mode():
        for seed in args.seeds:
            torch.manual_seed(seed)
            sample = torch.randn(INPUT_SHAPE, dtype=torch.float32)
            reference_t = model(sample)
            if (tuple(reference_t.shape) != OUTPUT_SHAPE or reference_t.dtype != torch.float32 or
                    not torch.isfinite(reference_t).all().item()):
                raise RuntimeError(f"PyTorch reference contract failed for seed {seed}")
            interpreter.set_tensor(input_info["index"], sample.numpy())
            interpreter.invoke()
            actual = np.asarray(interpreter.get_tensor(output_info["index"]))
            if actual.shape != OUTPUT_SHAPE or actual.dtype != np.float32 or not np.isfinite(actual).all():
                raise RuntimeError(f"LiteRT output contract or finite check failed for seed {seed}")
            metrics = compare(reference_t.numpy(), actual)
            results.append({"seed": seed, "metrics": metrics})
            differences.append(reference_t.numpy() - actual)

    flat_diff = np.concatenate([diff.ravel() for diff in differences])
    max_abs = float(np.max(np.abs(flat_diff)))
    mean_abs = float(np.mean(np.abs(flat_diff)))
    rmse = float(np.sqrt(np.mean(flat_diff * flat_diff)))
    cosines = [float(result["metrics"]["cosine_similarity"]) for result in results]
    passed = all(bool(result["metrics"]["allclose_rtol_1e-3_atol_1e-4"]) for result in results)

    try:
        runtime_version = importlib.metadata.version("ai-edge-litert")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = "unknown"
    report = {
        "phase": "AGE-4B.1 FP32 LiteRT smoke and parity validation",
        "backbone_state_sha256": model_hash,
        "artifact_sha256": export_report["artifact"]["sha256"],
        "artifact_path": str(artifact),
        "runtime": {"ai_edge_litert": runtime_version, **get_delegate_observation(interpreter)},
        "contract": {"input_shape": list(in_shape), "input_dtype": str(np.dtype(input_info["dtype"])),
                     "output_shape": list(out_shape), "output_dtype": str(np.dtype(output_info["dtype"]))},
        "inputs_tested": len(results),
        "per_input": results,
        "aggregate": {"max_absolute_difference": max_abs,
                      "mean_absolute_difference": mean_abs, "rmse": rmse,
                      "mean_cosine_similarity": float(np.mean(cosines)),
                      "minimum_cosine_similarity": min(cosines)},
        "tolerance": {"criterion": "np.allclose(rtol=1e-3, atol=1e-4) for every input",
                      "passed": passed},
        "smoke_test": "PASS",
        "parity": "PASS" if passed else "FAIL",
        "real_image_parity": "not run; no AGE data accessed",
    }
    report_path = artifact.with_suffix(".validation.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("FP32 LiteRT parity failed the declared allclose tolerance")


if __name__ == "__main__":
    main()
