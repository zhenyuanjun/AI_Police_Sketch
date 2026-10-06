#!/usr/bin/env python3
"""Export the finalized AME534 ImageNet ResNet50 backbone as FP32 LiteRT."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "age4b1_litert" / "resnet50_backbone_fp32.tflite"
INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_SHAPE = (1, 2048)
SEED = 534401


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".export.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval().cpu()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Backbone is not frozen and in eval mode")

    torch.manual_seed(args.seed)
    sample = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        reference = model(sample)
    if tuple(reference.shape) != OUTPUT_SHAPE or reference.dtype != torch.float32:
        raise RuntimeError(f"Unexpected PyTorch contract: {tuple(reference.shape)}, {reference.dtype}")
    if not torch.isfinite(reference).all().item():
        raise RuntimeError("PyTorch reference output contains NaN or Inf")

    import litert_torch

    edge_model = litert_torch.convert(model, (sample,))
    edge_model.export(str(output))
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"LiteRT export was not created: {output}")

    report = {
        "phase": "AGE-4B.1 FP32 export only",
        "export_command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "seed": args.seed,
        "source": {
            "architecture": "torchvision.models.resnet50",
            "weight_enum": "ResNet50_Weights.DEFAULT",
            "resolved_weight_enum": str(weights),
            "weight_url": weights.url,
            "classifier": "torch.nn.Identity",
            "state_sha256": state_sha256(model),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "python": platform.python_version(),
            "litert_torch": getattr(litert_torch, "__version__", "version not exposed"),
            "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
            "ai_edge_quantizer": importlib.metadata.version("ai-edge-quantizer"),
        },
        "pytorch_reference": {
            "shape": list(reference.shape),
            "dtype": str(reference.dtype),
            "all_finite": bool(torch.isfinite(reference).all().item()),
            "model_eval": not model.training,
        },
        "contract": {
            "input_shape": list(INPUT_SHAPE), "input_dtype": "float32", "input_layout": "NCHW",
            "output_shape": list(OUTPUT_SHAPE), "output_dtype": "float32",
        },
        "artifact": {
            "path": str(output), "size_bytes": output.stat().st_size,
            "sha256": file_sha256(output),
        },
        "source_hash_chain": {
            "weight_enum": str(weights),
            "backbone_state_sha256": state_sha256(model),
            "fp32_litert_artifact_sha256": file_sha256(output),
            "future_int8_quantization_source_sha256": file_sha256(output),
        },
        "quantization": "none; FP32 artifact",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
