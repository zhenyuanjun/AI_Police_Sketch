#!/usr/bin/env python3
"""One-time final AGE-1A test evaluation for the frozen AGE W8A8 backbone."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agedb_dataset import AGE1A_HASH, EXPECTED_COUNTS, OUTPUT_DIR, AgeDBDataset, verify_metadata_hash

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = OUTPUT_DIR / "runs" / "age4b3_int8_final_test"
FP32_ARTIFACT = OUTPUT_DIR / "age4b1_litert" / "resnet50_backbone_fp32.tflite"
FP32_EXPORT_REPORT = OUTPUT_DIR / "age4b1_litert" / "resnet50_backbone_fp32.export.json"
INT8_ARTIFACT = OUTPUT_DIR / "age4b2_int8" / "resnet50_backbone_w8a8.tflite"
INT8_QUANT_REPORT = OUTPUT_DIR / "age4b2_int8" / "quantization_report.json"
INT8_VALIDATION_REPORT = OUTPUT_DIR / "age4b2_int8" / "technical_validation_report.json"
CALIBRATION_MANIFEST = OUTPUT_DIR / "age4b2_int8" / "calibration_manifest.json"
HEAD_CHECKPOINT = OUTPUT_DIR / "runs" / "age3a_cached_head" / "best_checkpoint.pt"
HEAD_SUMMARY = OUTPUT_DIR / "runs" / "age3a_cached_head" / "run_summary.json"
FP32_RUN_DIR = OUTPUT_DIR / "runs" / "age3b_final_test"
FP32_REPORT = FP32_RUN_DIR / "evaluation_report.json"
FP32_PREDICTIONS = FP32_RUN_DIR / "predictions.json"

EXPECTED_FP32_HASH = "c8bdaad0c0c85d45d80ee69b34bc656a3f6972c921b1772c18075bd7fb5df471"
EXPECTED_AGE1A_HASH = AGE1A_HASH
EXPECTED_IMAGES = 2_346
EXPECTED_IDENTITIES = 85
EXPECTED_CALIBRATION_COUNT = 200
INPUT_SHAPE = (1, 3, 224, 224)
FEATURE_SHAPE = (1, 2048)
EXPECTED_INPUT_SCALE = 0.016827620565891266
EXPECTED_INPUT_ZERO = -11
EXPECTED_OUTPUT_SCALE = 0.021488072350621223
EXPECTED_OUTPUT_ZERO = -128
FP32_BASELINE = {
    "mae_years": 9.6941,
    "absolute_error_std_years": 7.6447,
    "median_absolute_error_years": 8.0905,
    "rmse_years": 12.3458,
    "mean_signed_error_years": 0.9571,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required AGE provenance/evaluation file missing: {path}")


def validate_provenance() -> dict:
    """Validate all locked artifacts and split metadata before loading test data."""
    for path in (FP32_ARTIFACT, FP32_EXPORT_REPORT, INT8_ARTIFACT, INT8_QUANT_REPORT,
                 INT8_VALIDATION_REPORT, CALIBRATION_MANIFEST, HEAD_CHECKPOINT, HEAD_SUMMARY,
                 FP32_REPORT, FP32_PREDICTIONS):
        require_file(path)

    int8_hash = sha256_file(INT8_ARTIFACT)
    source_hash = sha256_file(FP32_ARTIFACT)
    head_checkpoint_hash = sha256_file(HEAD_CHECKPOINT)
    quant_report = json.loads(INT8_QUANT_REPORT.read_text(encoding="utf-8"))
    int8_validation = json.loads(INT8_VALIDATION_REPORT.read_text(encoding="utf-8"))
    fp32_export = json.loads(FP32_EXPORT_REPORT.read_text(encoding="utf-8"))
    head_summary = json.loads(HEAD_SUMMARY.read_text(encoding="utf-8"))
    fp32_report = json.loads(FP32_REPORT.read_text(encoding="utf-8"))
    calibration_manifest = json.loads(CALIBRATION_MANIFEST.read_text(encoding="utf-8"))
    calibration_hash = sha256_file(CALIBRATION_MANIFEST)
    metadata_hash = verify_metadata_hash()

    if int8_hash != quant_report.get("int8_artifact", {}).get("sha256"):
        raise RuntimeError("STOP: INT8 artifact hash does not match AGE-4B.2 quantization report")
    if int8_hash != int8_validation.get("int8_artifact", {}).get("sha256"):
        raise RuntimeError("STOP: INT8 artifact hash does not match AGE-4B.2 validation report")
    if source_hash != EXPECTED_FP32_HASH:
        raise RuntimeError("STOP: source FP32 artifact differs from the locked AGE-4B.1 SHA-256")
    if source_hash != quant_report.get("source_fp32_sha256") or \
            source_hash != fp32_export.get("artifact", {}).get("sha256"):
        raise RuntimeError("STOP: source FP32 artifact SHA-256 does not match provenance records")
    if quant_report.get("source_hash_chain", {}).get("int8_artifact_derived_from_fp32_litert_sha256") != source_hash:
        raise RuntimeError("STOP: AGE-4B.2 source hash chain does not link to this FP32 artifact")
    if metadata_hash != EXPECTED_AGE1A_HASH or metadata_hash != quant_report.get("age1a_metadata_sha256"):
        raise RuntimeError("STOP: AGE-1A metadata hash differs from AGE-4B.2 calibration provenance")
    if calibration_hash != quant_report.get("calibration_manifest_sha256"):
        raise RuntimeError("STOP: calibration manifest SHA-256 differs from AGE-4B.2 report")
    calibration_rows = calibration_manifest.get("images", [])
    calibration_ids = [row.get("image_id") for row in calibration_rows]
    if (calibration_manifest.get("sample_count") != EXPECTED_CALIBRATION_COUNT or
            len(calibration_rows) != EXPECTED_CALIBRATION_COUNT or
            len(set(calibration_ids)) != EXPECTED_CALIBRATION_COUNT or
            not calibration_manifest.get("train_only") or
            any(row.get("split") != "train" for row in calibration_rows)):
        raise RuntimeError("STOP: AGE-4B.2 calibration provenance is not exactly 200 unique TRAIN images")
    if calibration_manifest.get("age1a_metadata_sha256") != metadata_hash:
        raise RuntimeError("STOP: calibration manifest uses a different AGE-1A metadata version")
    with (OUTPUT_DIR / "identity_splits.csv").open(newline="", encoding="utf-8") as stream:
        identity_splits = {row["identity"]: row["split"] for row in csv.DictReader(stream)}
    if (len(identity_splits) == 0 or any(identity_splits.get(row.get("identity")) != "train"
                                         for row in calibration_rows)):
        raise RuntimeError("STOP: calibration manifest contains a non-TRAIN AGE-1A identity")
    if head_checkpoint_hash != int8_validation.get("age_head_parity", {}).get("head_checkpoint_sha256"):
        raise RuntimeError("STOP: finalized AGE-3A head checkpoint differs from AGE-4B.2 head provenance")
    if head_summary.get("best_epoch") != 8:
        raise RuntimeError("STOP: AGE-3A summary best epoch is not 8")
    if fp32_report.get("checkpoint", {}).get("sha256") != head_checkpoint_hash or \
            fp32_report.get("checkpoint", {}).get("epoch") != 8:
        raise RuntimeError("STOP: AGE-3B FP32 baseline did not use the finalized AGE-3A epoch-8 checkpoint")
    if fp32_report.get("split", {}).get("agedb_metadata_sha256") != metadata_hash:
        raise RuntimeError("STOP: AGE-3B FP32 evaluation used a different AGE-1A metadata hash")
    if fp32_report.get("split", {}).get("test_images") != EXPECTED_IMAGES or \
            fp32_report.get("split", {}).get("test_identities") != EXPECTED_IDENTITIES:
        raise RuntimeError("STOP: AGE-3B baseline has unexpected test count/identity count")
    for name, expected in FP32_BASELINE.items():
        actual = fp32_report.get("metrics", {}).get(name)
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, abs_tol=1e-4, rel_tol=0):
            raise RuntimeError(f"STOP: AGE-3B {name} differs from finalized baseline: {actual} vs {expected}")

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(INT8_ARTIFACT))
    interpreter.allocate_tensors()
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError("STOP: INT8 model does not have one input and one output")
    in_info, out_info = ins[0], outs[0]
    if tuple(int(x) for x in in_info["shape"]) != INPUT_SHAPE or np.dtype(in_info["dtype"]) != np.int8:
        raise RuntimeError("STOP: INT8 model input contract changed")
    if tuple(int(x) for x in out_info["shape"]) != FEATURE_SHAPE or np.dtype(out_info["dtype"]) != np.int8:
        raise RuntimeError("STOP: INT8 model output contract changed")
    input_q, output_q = in_info["quantization_parameters"], out_info["quantization_parameters"]
    input_scale = float(np.asarray(input_q["scales"])[0])
    input_zero = int(np.asarray(input_q["zero_points"])[0])
    output_scale = float(np.asarray(output_q["scales"])[0])
    output_zero = int(np.asarray(output_q["zero_points"])[0])
    if (not math.isclose(input_scale, EXPECTED_INPUT_SCALE, rel_tol=0, abs_tol=1e-9) or
            input_zero != EXPECTED_INPUT_ZERO or
            not math.isclose(output_scale, EXPECTED_OUTPUT_SCALE, rel_tol=0, abs_tol=1e-9) or
            output_zero != EXPECTED_OUTPUT_ZERO):
        raise RuntimeError("STOP: INT8 quantization scales/zero points differ from AGE-4B.2 contract")

    head_state_hash = int8_validation.get("age_head_parity", {}).get("head_state_sha256")
    if not head_state_hash or int8_validation.get("age_head_parity", {}).get("age3a_best_epoch") != 8:
        raise RuntimeError("STOP: AGE-4B.2 validation lacks finalized AGE-3A head state provenance")
    return {
        "int8_artifact_sha256": int8_hash,
        "source_fp32_artifact_sha256": source_hash,
        "age1a_metadata_sha256": metadata_hash,
        "calibration_manifest_sha256": calibration_hash,
        "calibration_sample_count": len(calibration_rows),
        "calibration_train_only": True,
        "age3a_head_checkpoint_sha256": head_checkpoint_hash,
        "age3a_head_state_sha256": head_state_hash,
        "age3a_best_epoch": head_summary["best_epoch"],
        "fp32_baseline_report": str(FP32_REPORT),
        "test_contract": {"input_shape": list(INPUT_SHAPE), "input_dtype": "float32",
                           "feature_shape": list(FEATURE_SHAPE), "feature_dtype": "int8"},
        "int8_contract": {
            "input": {"shape": [int(x) for x in in_info["shape"]], "dtype": str(np.dtype(in_info["dtype"])),
                      "scale": input_scale, "zero_point": input_zero},
            "output": {"shape": [int(x) for x in out_info["shape"]], "dtype": str(np.dtype(out_info["dtype"])),
                       "scale": output_scale, "zero_point": output_zero},
        },
        "quant_report": quant_report,
    }


def age_bin(age: int) -> str:
    if age >= 91:
        return "91+"
    low = ((age - 1) // 10) * 10 + 1
    return f"{low}–{low + 9}"


def error_metrics(targets: list[float], predictions: list[float]) -> dict:
    signed = [prediction - target for target, prediction in zip(targets, predictions)]
    absolute = [abs(error) for error in signed]
    return {
        "mae_years": statistics.mean(absolute),
        "absolute_error_std_years": statistics.pstdev(absolute),
        "median_absolute_error_years": statistics.median(absolute),
        "rmse_years": math.sqrt(statistics.mean(error * error for error in signed)),
        "mean_signed_error_years": statistics.mean(signed),
        "prediction_statistics": {"minimum": min(predictions), "maximum": max(predictions),
                                  "mean": statistics.mean(predictions), "median": statistics.median(predictions)},
        "signed_errors": signed,
        "absolute_errors": absolute,
    }


def main() -> None:
    for name in ("evaluation_report.json", "int8_predictions.json", "paired_prediction_comparison.json",
                 "age_bin_comparison.json"):
        if (OUT_DIR / name).exists():
            raise FileExistsError(f"Refusing to overwrite final AGE-4B.3 result: {OUT_DIR / name}")

    # Fail closed: all provenance gates run before prediction records or test images are loaded.
    provenance = validate_provenance()
    from ai_edge_litert.interpreter import Interpreter

    int8_interpreter = Interpreter(model_path=str(INT8_ARTIFACT))
    int8_interpreter.allocate_tensors()
    int8_in = int8_interpreter.get_input_details()[0]
    int8_out = int8_interpreter.get_output_details()[0]
    in_scale = provenance["int8_contract"]["input"]["scale"]
    in_zero = provenance["int8_contract"]["input"]["zero_point"]
    out_scale = provenance["int8_contract"]["output"]["scale"]
    out_zero = provenance["int8_contract"]["output"]["zero_point"]

    checkpoint_hash_before = sha256_file(HEAD_CHECKPOINT)
    int8_hash_before = sha256_file(INT8_ARTIFACT)
    fp32_hash_before = sha256_file(FP32_ARTIFACT)
    checkpoint = torch.load(HEAD_CHECKPOINT, map_location="cpu", weights_only=False)
    head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 1)).cpu().eval()
    head.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head_state_before = state_sha256(head)
    if head_state_before != provenance["age3a_head_state_sha256"]:
        raise RuntimeError("AGE-3A head state changed relative to available quantization provenance")

    split_report = json.loads((OUTPUT_DIR / "report.json").read_text(encoding="utf-8"))
    dataset_root = Path(split_report["source_dataset_path"]).expanduser()
    train_dataset = AgeDBDataset("train", dataset_root)
    validation_dataset = AgeDBDataset("validation", dataset_root)
    test_dataset = AgeDBDataset("test", dataset_root)
    train_identities = {row["identity"] for row in train_dataset.rows}
    validation_identities = {row["identity"] for row in validation_dataset.rows}
    test_identities = {row["identity"] for row in test_dataset.rows}
    if (len(test_dataset) != EXPECTED_IMAGES or len(test_identities) != EXPECTED_IDENTITIES or
            train_identities & test_identities or validation_identities & test_identities):
        raise RuntimeError("AGE-1A test counts or identity disjointness gate failed; test evaluation stopped")

    # AGE-3B records are loaded only after source/checkpoint/split provenance passes.
    fp32_records = json.loads(FP32_PREDICTIONS.read_text(encoding="utf-8"))
    if len(fp32_records) != EXPECTED_IMAGES or len(test_dataset.rows) != EXPECTED_IMAGES:
        raise RuntimeError("AGE-3B prediction records or AGE-1A test rows are not exactly 2,346")
    for index, (row, record) in enumerate(zip(test_dataset.rows, fp32_records)):
        if (record.get("image_id") != row["image_id"] or record.get("relative_path") != row["relative_path"] or
                record.get("identity") != row["identity"] or
                int(record.get("ground_truth_age")) != int(row["age"])):
            raise RuntimeError(f"AGE-3B prediction record/order mismatch at test row {index}")
        if not math.isfinite(float(record["predicted_age"])):
            raise RuntimeError(f"Non-finite existing FP32 prediction at row {index}")

    output_details = int8_interpreter.get_output_details()
    input_details = int8_interpreter.get_input_details()
    transform = test_dataset.transform
    targets: list[float] = []
    int8_predictions: list[float] = []
    fp32_predictions = [float(record["predicted_age"]) for record in fp32_records]
    int8_records: list[dict] = []
    raw_outputs_valid = True
    features_finite = True
    predictions_finite = True
    wall_start = time.perf_counter()
    for index, row in enumerate(test_dataset.rows):
        image, age = test_dataset[index]  # Reads each authorized test image exactly once.
        if image.shape != (3, 224, 224) or image.dtype != torch.float32 or not torch.isfinite(image).all():
            raise FloatingPointError(f"Invalid canonical test tensor at row {index}, image_id={row['image_id']}")
        target = float(age.item())
        if not math.isfinite(target):
            raise FloatingPointError(f"Non-finite target at test row {index}")
        normalized = image.unsqueeze(0).contiguous().numpy()
        quantized_input = np.clip(np.rint(normalized / in_scale + in_zero), -128, 127).astype(np.int8)
        int8_interpreter.set_tensor(input_details[0]["index"], quantized_input)
        int8_interpreter.invoke()
        raw_feature = np.asarray(int8_interpreter.get_tensor(output_details[0]["index"]))
        if raw_feature.shape != FEATURE_SHAPE or raw_feature.dtype != np.int8:
            raw_outputs_valid = False
            raise RuntimeError(f"Invalid INT8 output contract at row {index}, image_id={row['image_id']}")
        feature = (raw_feature.astype(np.float32) - out_zero) * out_scale
        if not np.isfinite(feature).all():
            features_finite = False
            raise FloatingPointError(f"Non-finite dequantized feature at row {index}, image_id={row['image_id']}")
        with torch.inference_mode():
            prediction = float(head(torch.from_numpy(feature.copy())).squeeze().item())
        if not math.isfinite(prediction):
            predictions_finite = False
            raise FloatingPointError(f"Non-finite INT8-backbone age prediction at row {index}")
        targets.append(target)
        int8_predictions.append(prediction)
        signed_error = prediction - target
        int8_records.append({"image_id": row["image_id"], "relative_path": row["relative_path"],
                            "identity": row["identity"], "ground_truth_age": target,
                            "predicted_age": prediction, "signed_error": signed_error,
                            "absolute_error": abs(signed_error), "int8_raw_output_dtype": "int8",
                            "int8_raw_output_shape": list(raw_feature.shape),
                            "dequantized_features_finite": True})
        if (index + 1) % 250 == 0 or index + 1 == EXPECTED_IMAGES:
            print(f"processed_test_images: {index + 1}/{EXPECTED_IMAGES}", flush=True)
    evaluation_wall_seconds = time.perf_counter() - wall_start

    if len(targets) != EXPECTED_IMAGES or len(int8_predictions) != EXPECTED_IMAGES:
        raise RuntimeError("INT8 evaluation did not produce exactly 2,346 predictions")
    fp32_metrics = error_metrics(targets, fp32_predictions)
    int8_metrics = error_metrics(targets, int8_predictions)
    accuracy_deltas = {
        "mae_years": int8_metrics["mae_years"] - fp32_metrics["mae_years"],
        "mae_relative_change_percent": (int8_metrics["mae_years"] - fp32_metrics["mae_years"])
        / fp32_metrics["mae_years"] * 100.0,
        "absolute_error_std_years": int8_metrics["absolute_error_std_years"] - fp32_metrics["absolute_error_std_years"],
        "median_absolute_error_years": int8_metrics["median_absolute_error_years"] - fp32_metrics["median_absolute_error_years"],
        "rmse_years": int8_metrics["rmse_years"] - fp32_metrics["rmse_years"],
        "mean_signed_error_years": int8_metrics["mean_signed_error_years"] - fp32_metrics["mean_signed_error_years"],
    }
    signed_prediction_differences = [i8 - fp32 for i8, fp32 in zip(int8_predictions, fp32_predictions)]
    absolute_prediction_differences = [abs(x) for x in signed_prediction_differences]
    correlation = float(np.corrcoef(np.asarray(fp32_predictions), np.asarray(int8_predictions))[0, 1])
    paired_metrics = {
        "sample_count": EXPECTED_IMAGES,
        "mean_absolute_difference_years": statistics.mean(absolute_prediction_differences),
        "median_absolute_difference_years": statistics.median(absolute_prediction_differences),
        "p95_absolute_difference_years": float(np.percentile(absolute_prediction_differences, 95)),
        "maximum_absolute_difference_years": max(absolute_prediction_differences),
        "rmse_years": math.sqrt(statistics.mean(x * x for x in signed_prediction_differences)),
        "mean_signed_difference_int8_minus_fp32_years": statistics.mean(signed_prediction_differences),
        "correlation": correlation,
    }

    bin_order = ("1–10", "11–20", "21–30", "31–40", "41–50", "51–60",
                 "61–70", "71–80", "81–90", "91+")
    per_bin: dict[str, dict[str, list[float]]] = {label: {"fp32": [], "int8": []} for label in bin_order}
    for target, fp32_prediction, int8_prediction in zip(targets, fp32_predictions, int8_predictions):
        label = age_bin(int(target))
        per_bin[label]["fp32"].append(abs(fp32_prediction - target))
        per_bin[label]["int8"].append(abs(int8_prediction - target))
    age_bin_rows = []
    for label in bin_order:
        fp_errors, int8_errors = per_bin[label]["fp32"], per_bin[label]["int8"]
        fp_mae = statistics.mean(fp_errors) if fp_errors else None
        int8_mae = statistics.mean(int8_errors) if int8_errors else None
        age_bin_rows.append({"age_bin": label, "n": len(fp_errors), "fp32_mae_years": fp_mae,
                             "int8_mae_years": int8_mae,
                             "delta_int8_minus_fp32_years": int8_mae - fp_mae if fp_errors else None})

    fp32_size, int8_size = FP32_ARTIFACT.stat().st_size, INT8_ARTIFACT.stat().st_size
    size_reduction = fp32_size - int8_size
    size_metrics = {"fp32_bytes": fp32_size, "int8_bytes": int8_size,
                    "reduction_bytes": size_reduction,
                    "reduction_percent": size_reduction / fp32_size * 100.0,
                    "fp32_to_int8_ratio": fp32_size / int8_size}

    checkpoint_hash_after = sha256_file(HEAD_CHECKPOINT)
    int8_hash_after = sha256_file(INT8_ARTIFACT)
    fp32_hash_after = sha256_file(FP32_ARTIFACT)
    head_state_after = state_sha256(head)
    parameters_changed = head_state_before != head_state_after
    int8_changed = provenance["int8_artifact_sha256"] != int8_hash_after
    fp32_changed = provenance["source_fp32_artifact_sha256"] != fp32_hash_after
    checkpoint_changed = checkpoint_hash_before != checkpoint_hash_after
    finite_errors = all(math.isfinite(value) for value in fp32_metrics["signed_errors"] +
                        int8_metrics["signed_errors"] + signed_prediction_differences)
    numeric_pass = (len(int8_predictions) == EXPECTED_IMAGES and len(test_identities) == EXPECTED_IDENTITIES
                    and raw_outputs_valid and features_finite and predictions_finite and finite_errors
                    and not parameters_changed and not checkpoint_changed and not int8_changed and not fp32_changed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired_records = [{"image_id": row["image_id"], "identity": row["identity"],
                       "ground_truth_age": targets[index], "fp32_prediction": fp32_predictions[index],
                       "int8_prediction": int8_predictions[index],
                       "signed_difference_int8_minus_fp32": signed_prediction_differences[index],
                       "absolute_difference": absolute_prediction_differences[index]}
                      for index, row in enumerate(test_dataset.rows)]
    report = {
        "phase": "AGE-4B.3 final INT8 test evaluation",
        "provenance_status": "PASS",
        "provenance": {key: value for key, value in provenance.items() if key != "quant_report"},
        "test": {"images": len(int8_predictions), "identities": len(test_identities),
                 "train_identity_overlap": len(train_identities & test_identities),
                 "validation_identity_overlap": len(validation_identities & test_identities),
                 "agedb_metadata_sha256": provenance["age1a_metadata_sha256"]},
        "fp32": {key: value for key, value in fp32_metrics.items()
                 if key not in ("signed_errors", "absolute_errors")},
        "fp32_baseline_reference": FP32_BASELINE,
        "int8": {key: value for key, value in int8_metrics.items()
                 if key not in ("signed_errors", "absolute_errors")},
        "accuracy_deltas_int8_minus_fp32": accuracy_deltas,
        "paired_predictions": paired_metrics,
        "age_bin_comparison": age_bin_rows,
        "model_size": size_metrics,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "evaluation_timing_note": "Complete dataset processing wall time; not controlled inference latency.",
        "numerical_validation": {
            "prediction_count": len(int8_predictions), "expected_prediction_count": EXPECTED_IMAGES,
            "test_identity_count": len(test_identities), "expected_test_identity_count": EXPECTED_IDENTITIES,
            "all_int8_raw_outputs_valid": raw_outputs_valid,
            "all_dequantized_features_finite": features_finite,
            "all_predictions_finite": predictions_finite,
            "all_errors_finite": finite_errors,
            "fp32_model_parameters_changed": parameters_changed,
            "age_head_checkpoint_changed": checkpoint_changed,
            "int8_artifact_changed": int8_changed,
            "fp32_artifact_changed": fp32_changed,
            "result": "PASS" if numeric_pass else "FAIL",
        },
        "model_parameters_changed": parameters_changed,
        "quantization_changed": int8_changed,
        "int8_status": "FROZEN",
        "test_set_accessed": True,
        "final_test_mae_calculated": True,
        "benchmark_performed": False,
        "overall_validation": "PASS" if numeric_pass else "FAIL",
        "next_recommended_phase": "AGE-4C Controlled FP32 vs INT8 Latency Benchmark",
    }
    atomic_json(OUT_DIR / "int8_predictions.json", int8_records)
    atomic_json(OUT_DIR / "paired_prediction_comparison.json", {
        "metrics": paired_metrics, "predictions": paired_records,
        "pairing_order_verified_against_age3b": True,
    })
    atomic_json(OUT_DIR / "age_bin_comparison.json", {"bins": age_bin_rows})
    atomic_json(OUT_DIR / "evaluation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not numeric_pass:
        raise SystemExit("AGE-4B.3 numerical validation failed")


if __name__ == "__main__":
    main()
