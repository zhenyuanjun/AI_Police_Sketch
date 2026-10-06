#!/usr/bin/env python3
"""AGE-4C controlled LiteRT ResNet50 backbone invoke-only latency benchmark.

This program is deliberately read-only with respect to the finalized FP32 and
INT8 LiteRT artifacts.  It refuses to start unless their hashes and sizes match
the AGE-4B.3 provenance values, and writes results only below the ignored
``output/age4c_latency_benchmark`` directory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "output"
BENCHMARK_DIR = OUTPUT_ROOT / "age4c_latency_benchmark"
JSON_REPORT = BENCHMARK_DIR / "benchmark_results.json"
MARKDOWN_REPORT = BENCHMARK_DIR / "benchmark_report.md"

FP32_MODEL = OUTPUT_ROOT / "age4b1_litert" / "resnet50_backbone_fp32.tflite"
INT8_MODEL = OUTPUT_ROOT / "age4b2_int8" / "resnet50_backbone_w8a8.tflite"

FP32_SHA256 = "c8bdaad0c0c85d45d80ee69b34bc656a3f6972c921b1772c18075bd7fb5df471"
INT8_SHA256 = "c3bdfb8002a40fe4892a2179616ea80fa2a2437d5c6a4402c01d0d15b9f6d5a0"
FP32_SIZE_BYTES = 94_020_956
INT8_SIZE_BYTES = 24_299_360

INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 2048)
INT8_INPUT_SCALE = 0.0168276206
INT8_INPUT_ZERO_POINT = -11
THREADS = 4
WARMUPS = 10
ITERATIONS = 50
ROUNDS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_provenance(path: Path, expected_hash: str, expected_size: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"STOP: finalized artifact is missing: {path}")
    actual_hash, actual_size = sha256(path), path.stat().st_size
    if actual_hash != expected_hash or actual_size != expected_size:
        raise RuntimeError(
            "STOP: finalized artifact provenance mismatch for "
            f"{path.name}: sha256={actual_hash}, size={actual_size}"
        )
    return {
        "path": str(path.relative_to(PROJECT_DIR)),
        "sha256": actual_hash,
        "size_bytes": actual_size,
        "expected_sha256": expected_hash,
        "expected_size_bytes": expected_size,
        "verified": True,
    }


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def fixed_validation_input() -> tuple[np.ndarray, dict[str, str]]:
    """Load and preprocess the first recorded AGE-1A validation image before timing."""
    report_path = OUTPUT_ROOT / "report.json"
    splits_path = OUTPUT_ROOT / "splits.csv"
    manifest_path = OUTPUT_ROOT / "manifest.csv"
    for path in (report_path, splits_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"AGE-4C needs existing AGE-1A metadata: {path}")
    dataset_root = Path(json.loads(report_path.read_text(encoding="utf-8"))["source_dataset_path"])
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = {row["image_id"]: row for row in csv.DictReader(stream)}
    with splits_path.open(newline="", encoding="utf-8") as stream:
        validation_id = next(
            row["image_id"] for row in csv.DictReader(stream) if row["split"] == "validation"
        )
    row = manifest[validation_id]
    image_path = dataset_root / "images" / row["relative_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Selected AGE-1A validation image is missing: {image_path}")
    transform = ResNet50_Weights.DEFAULT.transforms()
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB"))
    if tuple(tensor.shape) != INPUT_SHAPE[1:] or tensor.dtype != torch.float32:
        raise RuntimeError(f"Canonical preprocessing contract failed: {tensor.shape} {tensor.dtype}")
    array = tensor.unsqueeze(0).contiguous().numpy().astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise FloatingPointError("Fixed preprocessed input contains non-finite values")
    return array, {
        "split": "validation",
        "image_id": validation_id,
        "relative_path": row["relative_path"],
        "preprocessing": "torchvision ResNet50_Weights.DEFAULT.transforms() before timing",
    }


def percentile(values: np.ndarray, value: int) -> float:
    # NumPy 1.22+ spells the estimator argument "method".
    return float(np.percentile(values, value, method="linear"))


def latency_statistics(values: list[float]) -> dict[str, float]:
    raw = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(raw)),
        "median_ms": float(np.median(raw)),
        "stddev_ms_population": float(np.std(raw, ddof=0)),
        "min_ms": float(np.min(raw)),
        "max_ms": float(np.max(raw)),
        "p90_ms": percentile(raw, 90),
        "p95_ms": percentile(raw, 95),
    }


def io_contract(detail: dict[str, Any]) -> dict[str, Any]:
    scale, zero_point = detail["quantization"]
    return {
        "shape": [int(item) for item in detail["shape"]],
        "dtype": str(np.dtype(detail["dtype"])),
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def benchmark_one_runtime(name: str, model_path: Path, prepared_input: np.ndarray) -> dict[str, Any]:
    """Run independent fresh-interpreter rounds, timing only ``invoke()``."""
    from ai_edge_litert.interpreter import Interpreter

    wanted_dtype = np.dtype(np.float32 if name == "fp32" else np.int8)
    wanted_output_dtype = wanted_dtype
    round_values: list[list[float]] = []
    input_contract: dict[str, Any] | None = None
    output_contract: dict[str, Any] | None = None
    output_validation: dict[str, Any] | None = None

    for round_index in range(ROUNDS):
        # Passing num_threads is accepted by the installed public LiteRT API. The
        # public API does not expose an effective-thread-count getter.
        interpreter = Interpreter(model_path=str(model_path), num_threads=THREADS)
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"{name}: expected one input and one output, got {len(inputs)}/{len(outputs)}")
        input_detail, output_detail = inputs[0], outputs[0]
        if tuple(int(x) for x in input_detail["shape"]) != INPUT_SHAPE:
            raise RuntimeError(f"{name}: input shape is {input_detail['shape']}, expected {INPUT_SHAPE}")
        if tuple(int(x) for x in output_detail["shape"]) != OUTPUT_SHAPE:
            raise RuntimeError(f"{name}: output shape is {output_detail['shape']}, expected {OUTPUT_SHAPE}")
        if np.dtype(input_detail["dtype"]) != wanted_dtype:
            raise RuntimeError(f"{name}: input dtype is {input_detail['dtype']}, expected {wanted_dtype}")
        if np.dtype(output_detail["dtype"]) != wanted_output_dtype:
            raise RuntimeError(f"{name}: output dtype is {output_detail['dtype']}, expected {wanted_output_dtype}")

        input_contract, output_contract = io_contract(input_detail), io_contract(output_detail)
        if name == "int8":
            if not np.isclose(input_contract["scale"], INT8_INPUT_SCALE, rtol=0.0, atol=1e-10):
                raise RuntimeError(f"INT8 input scale mismatch: {input_contract['scale']}")
            if input_contract["zero_point"] != INT8_INPUT_ZERO_POINT:
                raise RuntimeError(f"INT8 input zero point mismatch: {input_contract['zero_point']}")

        # set_tensor, an untimed sanity invoke, and output access precede warmups.
        interpreter.set_tensor(input_detail["index"], prepared_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if name == "fp32":
            if not np.isfinite(output).all():
                raise FloatingPointError("FP32 output contains non-finite values")
            output_validation = {"shape": list(output.shape), "dtype": str(output.dtype), "finite": True}
        else:
            scale, zero_point = output_detail["quantization"]
            dequantized = (output.astype(np.float32) - zero_point) * scale
            if not np.isfinite(dequantized).all():
                raise FloatingPointError("Dequantized INT8 output contains non-finite values")
            output_validation = {
                "shape": list(output.shape), "dtype": str(output.dtype), "dequantized_finite": True,
            }

        for _ in range(WARMUPS):
            interpreter.invoke()
        measured: list[float] = []
        for _ in range(ITERATIONS):
            start_ns = time.perf_counter_ns()
            interpreter.invoke()
            elapsed_ns = time.perf_counter_ns() - start_ns
            measured.append(elapsed_ns / 1_000_000.0)
        round_values.append(measured)

    flattened = [value for values in round_values for value in values]
    return {
        "runtime": "ai-edge-litert",
        "input": input_contract,
        "output": output_contract,
        "output_validation": output_validation,
        "thread_configuration": {
            "requested": THREADS,
            "passed_to_interpreter_constructor": True,
            "verified_effective_thread_count": False,
            "verification_note": "LiteRT Python public API exposes no effective-thread-count getter.",
        },
        "delegate_notes": (
            "No explicit delegate was supplied. During this process LiteRT emitted "
            "'Created TensorFlow Lite XNNPACK delegate for CPU.' The installed Python API does "
            "not expose per-interpreter delegate assignment or effective delegate thread count."
        ),
        "raw_latency_ms_by_round": round_values,
        "round_statistics_ms": [latency_statistics(values) for values in round_values],
        "statistics_ms": latency_statistics(flattened),
    }


def fmt(value: float) -> str:
    return f"{value:.4f} ms"


def markdown_report(result: dict[str, Any]) -> str:
    fp32 = result["runtimes"]["fp32"]
    int8 = result["runtimes"]["int8"]
    fstats, istats = fp32["statistics_ms"], int8["statistics_ms"]
    lines = [
        "# AGE-4C Controlled FP32 vs INT8 ResNet50 Backbone Latency Benchmark",
        "",
        "PROVENANCE:",
        "PASS",
        "",
        "BENCHMARK PROTOCOL:",
        f"Runtime: ai-edge-litert {result['environment']['packages']['ai-edge-litert']}",
        "Batch size: 1",
        f"Threads requested: {THREADS}",
        "Threads verified: No (constructor accepted num_threads=4; public API has no effective-count getter)",
        f"Warmups: {WARMUPS}",
        f"Timed iterations: {ITERATIONS}",
        f"Rounds: {ROUNDS}",
        f"Measurements per runtime: {ROUNDS * ITERATIONS}",
        "Timing boundary: interpreter.invoke() only; fixed input was preprocessed, quantized (INT8), and set before timing.",
        "",
        "FP32 BACKBONE:",
        f"Mean latency: {fmt(fstats['mean_ms'])}", f"Median latency: {fmt(fstats['median_ms'])}",
        f"Latency STD: {fmt(fstats['stddev_ms_population'])}", f"Min: {fmt(fstats['min_ms'])}",
        f"Max: {fmt(fstats['max_ms'])}", f"P90: {fmt(fstats['p90_ms'])}", f"P95: {fmt(fstats['p95_ms'])}",
        "",
        "INT8 BACKBONE:",
        f"Mean latency: {fmt(istats['mean_ms'])}", f"Median latency: {fmt(istats['median_ms'])}",
        f"Latency STD: {fmt(istats['stddev_ms_population'])}", f"Min: {fmt(istats['min_ms'])}",
        f"Max: {fmt(istats['max_ms'])}", f"P90: {fmt(istats['p90_ms'])}", f"P95: {fmt(istats['p95_ms'])}",
        "",
        "PER-ROUND RESULTS:",
    ]
    for name, runtime in (("FP32", fp32), ("INT8", int8)):
        for index, values in enumerate(runtime["round_statistics_ms"], start=1):
            lines.append(f"{name} round {index}: mean {fmt(values['mean_ms'])}; median {fmt(values['median_ms'])}")
    lines += [
        "",
        "SPEED COMPARISON:",
        f"Mean speedup: {result['speed_comparison']['mean_speedup_ratio']:.4f}x (FP32 mean / INT8 mean)",
        f"Latency reduction: {result['speed_comparison']['latency_reduction_percent']:.2f}%",
        "",
        "MODEL SIZE:",
        f"FP32: {FP32_SIZE_BYTES:,} bytes ({FP32_SIZE_BYTES / 1024**2:.2f} MiB)",
        f"INT8: {INT8_SIZE_BYTES:,} bytes ({INT8_SIZE_BYTES / 1024**2:.2f} MiB)",
        f"Reduction: {result['model_size']['reduction_percent']:.2f}%",
        f"Size ratio: {result['model_size']['fp32_to_int8_ratio']:.4f}x",
        "",
        "OUTPUT VALIDATION:",
        "PASS",
        "",
        "DELEGATE / RUNTIME NOTES:",
        f"FP32: {fp32['delegate_notes']}",
        f"INT8: {int8['delegate_notes']}",
        "",
        "SECONDARY END-TO-END BENCHMARK:",
        "Not performed",
        "",
        "ACCURACY REFERENCE:",
        "FP32 Test MAE: 9.6941 years", "INT8 Test MAE: 9.7239 years", "MAE delta: +0.0297 years",
        "FP32 Error STD: 7.6447 years", "INT8 Error STD: 7.7231 years",
        "",
        "ARTIFACTS MODIFIED:",
        "NO",
        "",
        "OVERALL VALIDATION:",
        "PASS",
        "",
        "FINAL AGE-4 CONCLUSION:",
        "The finalized W8A8 ResNet50 backbone completed the controlled LiteRT invoke-only benchmark with validated artifacts and outputs. The reported ratio is backbone invoke-only speedup, not an application-wide speedup.",
        "",
        "NEXT RECOMMENDED PHASE:",
        "AGE-5 Police Sketch Integration",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if JSON_REPORT.exists() or MARKDOWN_REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing AGE-4C report directory: {BENCHMARK_DIR}")
    fp32_provenance = artifact_provenance(FP32_MODEL, FP32_SHA256, FP32_SIZE_BYTES)
    int8_provenance = artifact_provenance(INT8_MODEL, INT8_SHA256, INT8_SIZE_BYTES)
    fixed_input, fixed_input_info = fixed_validation_input()
    int8_input = np.clip(
        np.rint(fixed_input / INT8_INPUT_SCALE + INT8_INPUT_ZERO_POINT), -128, 127
    ).astype(np.int8)

    fp32 = benchmark_one_runtime("fp32", FP32_MODEL, fixed_input)
    int8 = benchmark_one_runtime("int8", INT8_MODEL, int8_input)
    # A final read-only verification proves the benchmark did not modify either artifact.
    artifact_provenance(FP32_MODEL, FP32_SHA256, FP32_SIZE_BYTES)
    artifact_provenance(INT8_MODEL, INT8_SHA256, INT8_SIZE_BYTES)

    fp32_mean = fp32["statistics_ms"]["mean_ms"]
    int8_mean = int8["statistics_ms"]["mean_ms"]
    result = {
        "phase": "AGE-4C",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {"result": "PASS", "artifacts": {"fp32": fp32_provenance, "int8": int8_provenance}},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(),
            "physical_cpu_count": command("sysctl", "-n", "hw.physicalcpu"),
            "logical_cpu_count": command("sysctl", "-n", "hw.logicalcpu"),
            "cpu_brand": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "packages": {name: package_version(distribution) for name, distribution in {
                "ai-edge-litert": "ai-edge-litert", "numpy": "numpy", "torch": "torch", "torchvision": "torchvision",
            }.items()},
        },
        "protocol": {
            "batch_size": 1, "threads_requested": THREADS, "warmups_per_round": WARMUPS,
            "timed_iterations_per_round": ITERATIONS, "rounds": ROUNDS,
            "measurements_per_runtime": ROUNDS * ITERATIONS, "timer": "time.perf_counter_ns()",
            "timing_boundary": "interpreter.invoke() only",
            "excluded": ["disk I/O", "image loading", "preprocessing", "input quantization", "set_tensor", "output retrieval", "output dequantization", "Age head", "metric calculation"],
            "fixed_input": fixed_input_info,
        },
        "runtimes": {"fp32": fp32, "int8": int8},
        "speed_comparison": {
            "mean_speedup_ratio": fp32_mean / int8_mean,
            "latency_reduction_percent": (fp32_mean - int8_mean) / fp32_mean * 100.0,
            "scope": "ResNet50 backbone invoke-only speedup",
        },
        "model_size": {
            "fp32_bytes": FP32_SIZE_BYTES, "int8_bytes": INT8_SIZE_BYTES,
            "fp32_mib": FP32_SIZE_BYTES / 1024**2, "int8_mib": INT8_SIZE_BYTES / 1024**2,
            "reduction_percent": (FP32_SIZE_BYTES - INT8_SIZE_BYTES) / FP32_SIZE_BYTES * 100.0,
            "fp32_to_int8_ratio": FP32_SIZE_BYTES / INT8_SIZE_BYTES,
        },
        "accuracy_reference": {
            "fp32_test_mae_years": 9.6941, "int8_test_mae_years": 9.7239,
            "mae_delta_years": 0.0297, "fp32_error_std_years": 7.6447, "int8_error_std_years": 7.7231,
        },
        "artifacts_modified": False,
        "secondary_end_to_end_benchmark": "Not performed",
        "overall_validation": "PASS",
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=False)
    JSON_REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))
    print(f"\nDetailed report: {JSON_REPORT}")


if __name__ == "__main__":
    main()
