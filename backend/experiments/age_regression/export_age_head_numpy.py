#!/usr/bin/env python3
"""Offline-only deterministic export of the finalized FP32 age head to NPZ."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parents[2]
SOURCE = BACKEND_DIR / "data" / "models" / "age_regression_head.pt"
DESTINATION = BACKEND_DIR / "data" / "models" / "age_regression_head.npz"
EXPECTED_SOURCE_SHA256 = "c9d757ee0dee4ce1aff96ec8526d001e3072e30ca27264991f0db0c4b4320fe8"
EXPECTED_TENSORS = {
    "0.weight": ("linear1_weight", (256, 2048)),
    "0.bias": ("linear1_bias", (256,)),
    "2.weight": ("linear2_weight", (1, 256)),
    "2.bias": ("linear2_bias", (1,)),
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Refusing export: source checkpoint hash mismatch: {source_hash}")

    checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture", {}).get("head") != [
        "Linear(2048,256)", "ReLU", "Linear(256,1)"
    ]:
        raise RuntimeError("Refusing export: finalized checkpoint architecture does not match")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or set(state) != set(EXPECTED_TENSORS):
        raise RuntimeError("Refusing export: checkpoint state-dict keys do not match")

    arrays: dict[str, np.ndarray] = {}
    for source_name, (export_name, expected_shape) in EXPECTED_TENSORS.items():
        tensor = state[source_name]
        if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float32:
            raise RuntimeError(f"Refusing export: invalid tensor contract for {source_name}")
        array = tensor.detach().cpu().numpy().copy()
        if not np.isfinite(array).all():
            raise RuntimeError(f"Refusing export: non-finite tensor in {source_name}")
        arrays[export_name] = array

    # Write standard .npy members with fixed ZIP timestamps so repeated exports
    # from the same source produce the same artifact bytes and SHA-256.
    with zipfile.ZipFile(DESTINATION, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            buffer = BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    print(f"source_sha256={source_hash}")
    print(f"export_sha256={sha256(DESTINATION)}")
    print(f"export_size_bytes={DESTINATION.stat().st_size}")
    print(f"tensors={','.join(arrays)}")


if __name__ == "__main__":
    main()
