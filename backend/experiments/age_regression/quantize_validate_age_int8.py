#!/usr/bin/env python3
"""Calibrate, quantize and technically validate the AGE ResNet50 as W8A8."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights

from agedb_dataset import AGE1A_HASH, EXPECTED_COUNTS, OUTPUT_DIR, verify_metadata_hash

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "output" / "age4b2_int8"
FP32_ARTIFACT = PROJECT_DIR / "output" / "age4b1_litert" / "resnet50_backbone_fp32.tflite"
FP32_HASH = "c8bdaad0c0c85d45d80ee69b34bc656a3f6972c921b1772c18075bd7fb5df471"
BACKBONE_HASH = "da7fcbf478583233248fcefdd6979ed1c0c35cd2751f11e6e5b8550898af7f1a"
EXPECTED_AGE1A_HASH = AGE1A_HASH
CALIBRATION_SEED = 5341
PARITY_SEED = 5342
CALIBRATION_COUNT = 200
PARITY_COUNT = 50
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 2048)
INT8_ARTIFACT = OUTPUT / "resnet50_backbone_w8a8.tflite"
CALIBRATION_MANIFEST = OUTPUT / "calibration_manifest.json"
RANGE_AUDIT = OUTPUT / "calibration_range_audit.json"
QUANTIZATION_REPORT = OUTPUT / "quantization_report.json"
VALIDATION_REPORT = OUTPUT / "technical_validation_report.json"
HEAD_CHECKPOINT = OUTPUT_DIR / "runs" / "age3a_cached_head" / "best_checkpoint.pt"
HEAD_SUMMARY = OUTPUT_DIR / "runs" / "age3a_cached_head" / "run_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def verify_fp32_source():
    if not FP32_ARTIFACT.is_file():
        raise FileNotFoundError(f"AGE-4B.1 FP32 LiteRT source is missing: {FP32_ARTIFACT}")
    actual_hash = sha256_file(FP32_ARTIFACT)
    if actual_hash != FP32_HASH:
        raise RuntimeError(f"STOP: FP32 LiteRT source hash {actual_hash} != required {FP32_HASH}")
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(FP32_ARTIFACT))
    interpreter.allocate_tensors()
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError(f"FP32 source must have one input/output, got {len(ins)}/{len(outs)}")
    if tuple(int(x) for x in ins[0]["shape"]) != INPUT_SHAPE or np.dtype(ins[0]["dtype"]) != np.float32:
        raise RuntimeError("FP32 source input contract differs from [1,3,224,224] float32")
    if tuple(int(x) for x in outs[0]["shape"]) != OUTPUT_SHAPE or np.dtype(outs[0]["dtype"]) != np.float32:
        raise RuntimeError("FP32 source output contract differs from [1,2048] float32")
    return actual_hash


def load_metadata_and_root() -> tuple[list[dict], dict[str, str], dict[str, str], Path, str]:
    metadata_hash = verify_metadata_hash()
    if metadata_hash != EXPECTED_AGE1A_HASH:
        raise RuntimeError(f"Unexpected AGE-1A metadata hash {metadata_hash}")
    with (OUTPUT_DIR / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    with (OUTPUT_DIR / "splits.csv").open(newline="", encoding="utf-8") as stream:
        split_rows = list(csv.DictReader(stream))
    with (OUTPUT_DIR / "identity_splits.csv").open(newline="", encoding="utf-8") as stream:
        identity_rows = list(csv.DictReader(stream))
    by_id = {row["image_id"]: row for row in manifest}
    if len(by_id) != len(manifest) or len(manifest) != sum(EXPECTED_COUNTS.values()):
        raise RuntimeError("AGE-1A image metadata does not have unique IDs and expected split count")
    split_by_id = {row["image_id"]: row["split"] for row in split_rows}
    if len(split_by_id) != len(split_rows) or set(split_by_id) != set(by_id):
        raise RuntimeError("AGE-1A split assignments do not uniquely match manifest IDs")
    identity_split = {row["identity"]: row["split"] for row in identity_rows}
    if len(identity_split) != len(identity_rows):
        raise RuntimeError("AGE-1A identity split metadata has duplicate identities")
    for image_id, row in by_id.items():
        if row["identity"] not in identity_split or identity_split[row["identity"]] != split_by_id[image_id]:
            raise RuntimeError(f"AGE-1A image/identity split mismatch for {image_id}")
    report = json.loads((OUTPUT_DIR / "report.json").read_text(encoding="utf-8"))
    root = Path(report["source_dataset_path"]).expanduser().resolve()
    if not (root / "images").is_dir():
        raise FileNotFoundError(f"AgeDB images directory is missing: {root / 'images'}")
    return manifest, split_by_id, identity_split, root, metadata_hash


def select_rows(manifest: list[dict], split_by_id: dict[str, str], identity_split: dict[str, str],
                split: str, count: int, seed: int) -> list[dict]:
    rows = [row for row in manifest if split_by_id[row["image_id"]] == split]
    if len(rows) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"AGE-1A {split} count changed: {len(rows)}")
    selected = random.Random(seed).sample(rows, count)
    ids = [row["image_id"] for row in selected]
    identities = [row["identity"] for row in selected]
    if len(ids) != count or len(set(ids)) != count:
        raise RuntimeError(f"{split} sample does not have {count} unique image IDs")
    if any(split_by_id[image_id] != split or identity_split.get(identity) != split
           for image_id, identity in zip(ids, identities)):
        raise RuntimeError(f"{split} sample includes a cross-split identity")
    for row in selected:
        if not 1 <= int(row["age"]) <= 101:
            raise RuntimeError(f"Invalid age metadata for image {row['image_id']}")
    return selected


def preprocess_rows(rows: list[dict], root: Path, transform) -> list[np.ndarray]:
    arrays = []
    for row in rows:
        path = root / "images" / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"AGE-1A {row.get('split', 'selected split')} image is missing: {row['relative_path']}")
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB"))
        if tuple(tensor.shape) != INPUT_SHAPE[1:] or tensor.dtype != torch.float32:
            raise RuntimeError(f"Canonical transform contract failed for image_id={row['image_id']}")
        if not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"Non-finite preprocessed image_id={row['image_id']}")
        arrays.append(tensor.unsqueeze(0).contiguous().numpy().astype(np.float32, copy=False))
    return arrays


def calibration_range_audit(qsv: dict) -> dict:
    audit = {
        "total_ranges": len(qsv), "finite_ranges": 0, "invalid_ranges": 0,
        "empty_ranges": 0, "nan_values": 0, "positive_infinity_values": 0,
        "negative_infinity_values": 0, "min_greater_than_max": 0,
        "zero_width_ranges": 0, "invalid_entries": [], "zero_width_entries": [],
    }
    for name, value in qsv.items():
        lo = np.asarray(value.get("min", []))
        hi = np.asarray(value.get("max", []))
        entry_issues = []
        if lo.size == 0 or hi.size == 0:
            audit["empty_ranges"] += 1
            entry_issues.append("empty min/max")
        joined = np.concatenate((lo.reshape(-1), hi.reshape(-1))) if lo.size and hi.size else np.asarray([])
        audit["nan_values"] += int(np.isnan(joined).sum())
        audit["positive_infinity_values"] += int(np.isposinf(joined).sum())
        audit["negative_infinity_values"] += int(np.isneginf(joined).sum())
        if joined.size and not np.isfinite(joined).all():
            entry_issues.append("non-finite min/max")
        if lo.size and hi.size:
            try:
                lower, upper = np.broadcast_arrays(lo, hi)
            except ValueError:
                entry_issues.append("min/max shapes do not broadcast")
            else:
                reversed_count = int(np.count_nonzero(lower > upper))
                audit["min_greater_than_max"] += reversed_count
                if reversed_count:
                    entry_issues.append("min greater than max")
                zero_count = int(np.count_nonzero(lower == upper))
                if zero_count:
                    audit["zero_width_ranges"] += 1
                    audit["zero_width_entries"].append({"name": str(name), "element_count": zero_count})
                if np.isfinite(joined).all() and reversed_count == 0:
                    audit["finite_ranges"] += 1
        if entry_issues:
            audit["invalid_ranges"] += 1
            audit["invalid_entries"].append({"name": str(name), "issues": entry_issues,
                                             "min": lo.tolist(), "max": hi.tolist()})
    audit["result"] = "PASS" if audit["invalid_ranges"] == 0 else "FAIL"
    audit["zero_width_note"] = "Reported as suspicious; finite ordered zero-width ranges are not automatically invalid."
    return audit


def interpreter_contract(interpreter) -> tuple[dict, dict]:
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError(f"INT8 artifact must have one input/output, got {len(ins)}/{len(outs)}")
    in_detail, out_detail = ins[0], outs[0]
    input_q = in_detail.get("quantization_parameters", {})
    output_q = out_detail.get("quantization_parameters", {})
    input_scales = np.asarray(input_q.get("scales", []), dtype=np.float64)
    output_scales = np.asarray(output_q.get("scales", []), dtype=np.float64)
    input_zeros = np.asarray(input_q.get("zero_points", []), dtype=np.int64)
    output_zeros = np.asarray(output_q.get("zero_points", []), dtype=np.int64)
    if tuple(int(x) for x in in_detail["shape"]) != INPUT_SHAPE or np.dtype(in_detail["dtype"]) != np.int8:
        raise RuntimeError(f"INT8 input contract mismatch: {in_detail['shape']} {in_detail['dtype']}")
    if tuple(int(x) for x in out_detail["shape"]) != OUTPUT_SHAPE or np.dtype(out_detail["dtype"]) != np.int8:
        raise RuntimeError(f"INT8 output contract mismatch: {out_detail['shape']} {out_detail['dtype']}")
    if len(input_scales) != 1 or not np.isfinite(input_scales).all() or not (input_scales > 0).all():
        raise RuntimeError(f"INT8 input quantization scale is invalid: {input_scales}")
    if len(output_scales) != 1 or not np.isfinite(output_scales).all() or not (output_scales > 0).all():
        raise RuntimeError(f"INT8 output quantization scale is invalid: {output_scales}")
    if len(input_zeros) != 1 or not -128 <= int(input_zeros[0]) <= 127:
        raise RuntimeError(f"INT8 input zero point is invalid: {input_zeros}")
    if len(output_zeros) != 1 or not -128 <= int(output_zeros[0]) <= 127:
        raise RuntimeError(f"INT8 output zero point is invalid: {output_zeros}")
    def item(detail, scales, zeros):
        return {"shape": [int(x) for x in detail["shape"]], "dtype": str(np.dtype(detail["dtype"])),
                "scale": float(scales[0]), "zero_point": int(zeros[0])}
    return item(in_detail, input_scales, input_zeros), item(out_detail, output_scales, output_zeros)


def structure_audit(interpreter) -> dict:
    details = interpreter.get_tensor_details()
    counts: dict[str, int] = {}
    for item in details:
        dtype_name = str(np.dtype(item["dtype"]))
        counts[dtype_name] = counts.get(dtype_name, 0) + 1
    float_ops = []
    ops_getter = getattr(interpreter, "_get_ops_details", None)
    if callable(ops_getter):
        index_to_detail = {int(item["index"]): item for item in details}
        for op in ops_getter():
            name = str(op.get("op_name", "unknown"))
            if name.upper() == "DELEGATE":
                continue
            float_tensors = []
            for tensor_index in list(op.get("inputs", [])) + list(op.get("outputs", [])):
                tensor_index = int(tensor_index)
                if tensor_index < 0:
                    continue
                detail = index_to_detail.get(tensor_index)
                if detail is not None and np.dtype(detail["dtype"]) == np.float32:
                    float_tensors.append({"index": tensor_index, "name": detail.get("name", "")})
            if float_tensors:
                float_ops.append({"operator": name, "float32_tensors": float_tensors})
    return {"tensor_count": len(details), "tensor_dtype_counts": counts,
            "visible_float32_compute_ops": float_ops,
            "float_fallback_detected": bool(float_ops),
            "operator_inspection": "LiteRT _get_ops_details when exposed; delegated subgraphs may hide internal op detail."}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.astype(np.float64, copy=False).ravel(), b.astype(np.float64, copy=False).ravel()
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    return float(np.dot(a_flat, b_flat) / denom) if denom else 0.0


def main() -> None:
    targets = [INT8_ARTIFACT, CALIBRATION_MANIFEST, RANGE_AUDIT, QUANTIZATION_REPORT, VALIDATION_REPORT]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite AGE-4B.2 outputs: {[str(p) for p in targets if p.exists()]}")

    # Mandatory source validation is completed before AGE-1A image metadata or pixels are read.
    source_hash = verify_fp32_source()
    manifest, split_by_id, identity_split, dataset_root, metadata_hash = load_metadata_and_root()
    calibration_rows = select_rows(manifest, split_by_id, identity_split, "train", CALIBRATION_COUNT,
                                   CALIBRATION_SEED)
    calibration_ids = {row["image_id"] for row in calibration_rows}
    train_ids = {image_id for image_id, split in split_by_id.items() if split == "train"}
    validation_ids = {image_id for image_id, split in split_by_id.items() if split == "validation"}
    test_ids = {image_id for image_id, split in split_by_id.items() if split == "test"}
    train_identities = {identity for identity, split in identity_split.items() if split == "train"}
    validation_identities = {identity for identity, split in identity_split.items() if split == "validation"}
    test_identities = {identity for identity, split in identity_split.items() if split == "test"}
    if calibration_ids - train_ids or calibration_ids & (validation_ids | test_ids):
        raise RuntimeError("Calibration image IDs include validation or test assignments")
    calibration_identities = {row["identity"] for row in calibration_rows}
    if calibration_identities - train_identities or calibration_identities & (validation_identities | test_identities):
        raise RuntimeError("Calibration identities include validation or test identities")

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()
    calibration_inputs = preprocess_rows(calibration_rows, dataset_root, transform)
    if len(calibration_inputs) != CALIBRATION_COUNT or any(
            x.shape != INPUT_SHAPE or x.dtype != np.float32 or not np.isfinite(x).all()
            for x in calibration_inputs):
        raise RuntimeError("Calibration tensors do not all satisfy finite FP32 [1,3,224,224]")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    calibration_manifest = {
        "phase": "AGE-4B.2 TRAIN-only calibration",
        "calibration_seed": CALIBRATION_SEED,
        "sampling": "random.Random(5341).sample over AGE-1A manifest rows in stored order; without replacement",
        "sample_count": len(calibration_rows),
        "unique_image_ids": len(calibration_ids),
        "split_counts": {name: sum(split == name for split in split_by_id.values()) for name in EXPECTED_COUNTS},
        "train_only": all(split_by_id[row["image_id"]] == "train"
                           and identity_split[row["identity"]] == "train" for row in calibration_rows),
        "validation_test_image_id_overlap_count": len(calibration_ids & (validation_ids | test_ids)),
        "validation_test_identity_overlap_count": len(calibration_identities & (validation_identities | test_identities)),
        "age1a_metadata_sha256": metadata_hash,
        "preprocessing": "ResNet50_Weights.DEFAULT.transforms(); RGB, canonical resize/crop and ImageNet normalization",
        "model_input_contract": {"shape": list(INPUT_SHAPE), "dtype": "float32", "layout": "NCHW"},
        "images": [{key: (int(row[key]) if key == "age" else row[key])
                    for key in ("image_id", "relative_path", "identity", "age")}
                   | {"split": "train"} for row in calibration_rows],
    }
    atomic_json(CALIBRATION_MANIFEST, calibration_manifest)
    if not calibration_manifest["train_only"] or len(calibration_ids) != CALIBRATION_COUNT:
        raise RuntimeError("Calibration manifest failed exact count/train-only gates")

    from ai_edge_quantizer import calibrator, quantizer, recipe

    qt = quantizer.Quantizer(str(FP32_ARTIFACT))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    qsv = qt.calibrate({"serving_default": [{"args_0": array} for array in calibration_inputs]},
                       num_threads=4,
                       mode=calibrator.CalibrationMode.CALIBRATION_PRESERVE_ALL_TENSORS)
    range_audit = calibration_range_audit(qsv)
    atomic_json(RANGE_AUDIT, range_audit)
    if range_audit["invalid_ranges"] != 0 or range_audit["total_ranges"] == 0:
        raise RuntimeError("STOP: required calibration range is invalid; INT8 artifact was not created")

    # Quantization is deliberately unreachable until every required calibration range passes.
    qt.quantize(calibration_result=qsv, enable_progress_report=True).export_model(str(INT8_ARTIFACT))
    if not INT8_ARTIFACT.is_file() or INT8_ARTIFACT.stat().st_size == 0:
        raise RuntimeError("Quantizer did not write a non-empty INT8 artifact")
    int8_hash = sha256_file(INT8_ARTIFACT)

    from ai_edge_litert.interpreter import Interpreter

    int8_interpreter = Interpreter(model_path=str(INT8_ARTIFACT))
    int8_interpreter.allocate_tensors()
    input_contract, output_contract = interpreter_contract(int8_interpreter)
    structure = structure_audit(int8_interpreter)

    # Validation samples are sampled only from AGE-1A VALIDATION, disjoint from TRAIN calibration.
    parity_rows = select_rows(manifest, split_by_id, identity_split, "validation", PARITY_COUNT, PARITY_SEED)
    if {row["image_id"] for row in parity_rows} & calibration_ids:
        raise RuntimeError("Validation parity IDs overlap train calibration IDs")
    parity_inputs = preprocess_rows(parity_rows, dataset_root, transform)
    if any(x.shape != INPUT_SHAPE or x.dtype != np.float32 or not np.isfinite(x).all()
           for x in parity_inputs):
        raise RuntimeError("Validation parity tensors do not satisfy finite FP32 input contract")

    fp32_interpreter = Interpreter(model_path=str(FP32_ARTIFACT))
    fp32_interpreter.allocate_tensors()
    fp_in = fp32_interpreter.get_input_details()[0]
    fp_out = fp32_interpreter.get_output_details()[0]
    int_in = int8_interpreter.get_input_details()[0]
    int_out = int8_interpreter.get_output_details()[0]
    input_scale, input_zero = input_contract["scale"], input_contract["zero_point"]
    output_scale, output_zero = output_contract["scale"], output_contract["zero_point"]
    feature_diffs, cosine_values, prediction_diffs = [], [], []
    failed_samples, nonfinite_samples = [], []

    if not HEAD_CHECKPOINT.is_file() or not HEAD_SUMMARY.is_file():
        raise FileNotFoundError("Finalized AGE-3A head checkpoint/summary is missing")
    head_checkpoint_hash = sha256_file(HEAD_CHECKPOINT)
    checkpoint = torch.load(HEAD_CHECKPOINT, map_location="cpu", weights_only=False)
    head_summary = json.loads(HEAD_SUMMARY.read_text(encoding="utf-8"))
    if checkpoint.get("epoch") != 8 or head_summary.get("best_epoch") != 8 or \
            checkpoint.get("agedb_metadata_sha256") != metadata_hash or \
            head_summary.get("agedb_metadata_sha256") != metadata_hash:
        raise RuntimeError("AGE-3A head checkpoint does not match the finalized AGE-1A/AGE-3A provenance")
    if checkpoint.get("architecture", {}).get("head") != ["Linear(2048,256)", "ReLU", "Linear(256,1)"]:
        raise RuntimeError("AGE-3A checkpoint has an unexpected head architecture")
    head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1)).cpu().eval()
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head_hash = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        value = tensor.detach().contiguous()
        head_hash.update(name.encode("utf-8") + b"\0")
        head_hash.update(str(value.dtype).encode("ascii") + b"\0")
        head_hash.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        head_hash.update(value.numpy().tobytes())

    for row, fp32_input in zip(parity_rows, parity_inputs):
        image_id = row["image_id"]
        try:
            fp32_interpreter.set_tensor(fp_in["index"], fp32_input)
            fp32_interpreter.invoke()
            fp32_feature = np.asarray(fp32_interpreter.get_tensor(fp_out["index"]))
            if fp32_feature.shape != OUTPUT_SHAPE or fp32_feature.dtype != np.float32:
                raise RuntimeError("FP32 output contract failure")

            quantized = np.rint(fp32_input / input_scale + input_zero)
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
            int8_interpreter.set_tensor(int_in["index"], quantized)
            int8_interpreter.invoke()
            raw_feature = np.asarray(int8_interpreter.get_tensor(int_out["index"]))
            if raw_feature.shape != OUTPUT_SHAPE or raw_feature.dtype != np.int8:
                raise RuntimeError("INT8 raw output contract failure")
            dequantized = (raw_feature.astype(np.float32) - output_zero) * output_scale
            if not np.isfinite(fp32_feature).all() or not np.isfinite(dequantized).all():
                nonfinite_samples.append(image_id)
                continue
            feature_diffs.append((fp32_feature - dequantized).astype(np.float64))
            cosine_values.append(cosine_similarity(fp32_feature, dequantized))
            with torch.inference_mode():
                p_fp = float(head(torch.from_numpy(fp32_feature.copy())).squeeze().item())
                p_i8 = float(head(torch.from_numpy(dequantized.copy())).squeeze().item())
            if not (math.isfinite(p_fp) and math.isfinite(p_i8)):
                nonfinite_samples.append(image_id)
            else:
                prediction_diffs.append(p_i8 - p_fp)
        except Exception as exc:
            failed_samples.append({"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})

    usable = len(feature_diffs)
    if usable:
        flattened = np.concatenate([diff.reshape(-1) for diff in feature_diffs])
        feature_metrics = {
            "samples": usable,
            "max_absolute_difference": float(np.max(np.abs(flattened))),
            "mean_absolute_difference": float(np.mean(np.abs(flattened))),
            "rmse": float(np.sqrt(np.mean(flattened * flattened))),
            "mean_cosine_similarity": float(np.mean(cosine_values)),
            "minimum_cosine_similarity": float(np.min(cosine_values)),
            "non_finite_failures": len(nonfinite_samples),
            "failed_samples": len(failed_samples),
        }
    else:
        feature_metrics = {"samples": 0, "max_absolute_difference": None,
                           "mean_absolute_difference": None, "rmse": None,
                           "mean_cosine_similarity": None, "minimum_cosine_similarity": None,
                           "non_finite_failures": len(nonfinite_samples), "failed_samples": len(failed_samples)}
    if prediction_diffs:
        abs_predictions = np.abs(np.asarray(prediction_diffs, dtype=np.float64))
        prediction_metrics = {
            "samples": len(prediction_diffs),
            "mean_absolute_difference": float(np.mean(abs_predictions)),
            "median_absolute_difference": float(np.median(abs_predictions)),
            "p95_absolute_difference": float(np.percentile(abs_predictions, 95)),
            "maximum_absolute_difference": float(np.max(abs_predictions)),
            "rmse": float(np.sqrt(np.mean(np.square(prediction_diffs)))),
            "mean_signed_difference_int8_minus_fp32": float(np.mean(prediction_diffs)),
        }
    else:
        prediction_metrics = {"samples": 0, "mean_absolute_difference": None,
                              "median_absolute_difference": None, "p95_absolute_difference": None,
                              "maximum_absolute_difference": None, "rmse": None,
                              "mean_signed_difference_int8_minus_fp32": None}

    runtime = {"delegate_nodes": [], "xnnpack_status": "not exposed"}
    ops_getter = getattr(int8_interpreter, "_get_ops_details", None)
    if callable(ops_getter):
        op_names = sorted({str(op.get("op_name")) for op in ops_getter() if op.get("op_name")})
        runtime["delegate_nodes"] = [name for name in op_names if name.upper() == "DELEGATE"]
        runtime["operator_names"] = op_names
        if runtime["delegate_nodes"]:
            runtime["delegate_status"] = "delegate node observed; identity not exposed by interpreter"
    package_versions = {name: importlib.metadata.version(dist) for name, dist in {
        "torch": "torch", "torchvision": "torchvision", "litert_torch": "litert-torch",
        "ai_edge_litert": "ai-edge-litert", "ai_edge_quantizer": "ai-edge-quantizer",
        "numpy": "numpy"}.items()}
    required_gates_passed = (
        source_hash == FP32_HASH and len(calibration_rows) == CALIBRATION_COUNT
        and len(calibration_ids) == CALIBRATION_COUNT and calibration_manifest["train_only"]
        and range_audit["result"] == "PASS" and input_contract["dtype"] == "int8"
        and output_contract["dtype"] == "int8" and usable == PARITY_COUNT
        and len(prediction_diffs) == PARITY_COUNT and not failed_samples and not nonfinite_samples
    )
    report = {
        "phase": "AGE-4B.2 train-only W8A8 calibration and technical parity",
        "source_chain": {
            "int8_artifact_derived_from_fp32_litert_sha256": source_hash,
            "fp32_litert_derived_from_backbone_state_sha256": BACKBONE_HASH,
            "backbone_weight_enum": "ResNet50_Weights.IMAGENET1K_V2",
        },
        "source_fp32_artifact": {"path": str(FP32_ARTIFACT), "sha256": source_hash,
                                  "verified": source_hash == FP32_HASH},
        "calibration": {
            "sample_count": len(calibration_rows), "seed": CALIBRATION_SEED,
            "train_only": calibration_manifest["train_only"], "unique_ids": len(calibration_ids),
            "manifest": str(CALIBRATION_MANIFEST), "manifest_sha256": sha256_file(CALIBRATION_MANIFEST),
            "age1a_metadata_sha256": metadata_hash,
            "method": "CALIBRATION_PRESERVE_ALL_TENSORS",
            "preprocessing": "ResNet50_Weights.DEFAULT.transforms()",
            "range_audit": range_audit,
        },
        "quantization": {"recipe": "static_wi8_ai8", "weights": "INT8", "activations": "INT8",
                         "package_versions": package_versions,
                         "source_fp32_sha256": source_hash},
        "int8_artifact": {"path": str(INT8_ARTIFACT), "size_bytes": INT8_ARTIFACT.stat().st_size,
                           "sha256": int8_hash},
        "contract": {"input": input_contract, "output": output_contract},
        "structure_audit": structure,
        "lite_rt_invocation": {"result": "PASS" if usable > 0 else "FAIL",
                               "xnnpack": runtime},
        "technical_feature_parity": {
            **feature_metrics, "split": "validation", "seed": PARITY_SEED,
            "sample_ids": [row["image_id"] for row in parity_rows],
        },
        "age_head_parity": {
            **prediction_metrics,
            "head_checkpoint_sha256": head_checkpoint_hash,
            "head_state_sha256": head_hash.hexdigest(),
            "head_architecture": ["Linear(2048,256)", "ReLU", "Linear(256,1)"],
            "age3a_best_epoch": checkpoint["epoch"],
        },
        "test_set_accessed": False,
        "final_test_mae_calculated": False,
        "benchmark_performed": False,
        "overall_validation": "PASS" if required_gates_passed else "FAIL",
        "go_for_age4b3_final_int8_test_evaluation": "GO" if required_gates_passed else "NO-GO",
    }
    atomic_json(VALIDATION_REPORT, report)
    atomic_json(QUANTIZATION_REPORT, {
        "phase": report["phase"], "source_fp32_sha256": source_hash,
        "age1a_metadata_sha256": metadata_hash,
        "calibration_manifest_sha256": report["calibration"]["manifest_sha256"],
        "calibration_range_audit": range_audit,
        "method": "static_wi8_ai8", "weights": "INT8", "activations": "INT8",
        "package_versions": package_versions, "int8_artifact": report["int8_artifact"],
        "source_hash_chain": report["source_chain"],
        "overall_validation": report["overall_validation"],
    })
    print(json.dumps(report, indent=2, sort_keys=True))

    if usable != PARITY_COUNT or failed_samples or nonfinite_samples or not range_audit["result"] == "PASS":
        raise SystemExit("AGE-4B.2 validation failed one or more required technical gates")


if __name__ == "__main__":
    main()
