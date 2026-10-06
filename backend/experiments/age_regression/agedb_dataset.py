"""AgeDB Dataset and loaders backed exclusively by the AGE-1A split files."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

AGE1A_HASH = "abf489c4cfa335fedd829753b3eb77b9bb03b99184d7fd85cdbf2bcbca2b7b99"
EXPECTED_COUNTS = {"train": 11_457, "validation": 2_685, "test": 2_346}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def verify_metadata_hash(output_dir: Path = OUTPUT_DIR) -> str:
    files = ["manifest.csv", "splits.csv", "identity_splits.csv", "report.json"]
    digest = hashlib.sha256()
    for filename in files:
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"AGE-1A metadata is missing {path}; run the AGE-1A generator explicitly.")
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    actual = digest.hexdigest()
    expected_file = output_dir / "metadata.sha256"
    if not expected_file.is_file() or expected_file.read_text(encoding="ascii").strip() != actual:
        raise ValueError("AGE-1A metadata hash file is missing or does not match the metadata files.")
    if actual != AGE1A_HASH:
        raise ValueError(f"Unexpected AGE-1A metadata hash {actual}; expected {AGE1A_HASH}.")
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("random_seed") != 5341:
        raise ValueError("AGE-1A report has an unexpected random seed.")
    return actual


class AgeDBDataset(Dataset):
    """One AGE-1A split. Age targets are scalar float32 tensors in years."""

    def __init__(self, split: str, dataset_root: Path, transform: Any | None = None,
                 output_dir: Path = OUTPUT_DIR):
        if split not in EXPECTED_COUNTS:
            raise ValueError(f"split must be one of {tuple(EXPECTED_COUNTS)}, got {split!r}")
        self.split = split
        self.dataset_root = Path(dataset_root)
        self.transform = transform or ResNet50_Weights.DEFAULT.transforms()
        verify_metadata_hash(output_dir)
        with (output_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        with (output_dir / "splits.csv").open(newline="", encoding="utf-8") as f:
            split_rows = list(csv.DictReader(f))
        with (output_dir / "identity_splits.csv").open(newline="", encoding="utf-8") as f:
            identity_rows = list(csv.DictReader(f))

        split_for_id: dict[str, str] = {}
        for row in split_rows:
            image_id, name = row["image_id"], row["split"]
            if image_id in split_for_id:
                raise ValueError(f"AGE-1A split assigns image_id {image_id!r} more than once")
            if name not in EXPECTED_COUNTS:
                raise ValueError(f"AGE-1A split contains unknown split {name!r}")
            split_for_id[image_id] = name
        if len(manifest) != sum(EXPECTED_COUNTS.values()) or len(split_for_id) != len(manifest):
            raise ValueError("AGE-1A manifest/split row counts are inconsistent")
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        if Path(report["source_dataset_path"]).expanduser().resolve() != self.dataset_root.resolve():
            raise ValueError("AGEDB_ROOT does not match the dataset root recorded by AGE-1A")
        manifest_ids = {row["image_id"] for row in manifest}
        if len(manifest_ids) != len(manifest) or manifest_ids != set(split_for_id):
            raise ValueError("AGE-1A manifest IDs and split IDs do not match uniquely")

        identities: dict[str, set[str]] = defaultdict(set)
        selected = []
        for row in manifest:
            name = split_for_id[row["image_id"]]
            identities[row["identity"]].add(name)
            if name == split:
                parsed_age = int(row["age"])
                if not 1 <= parsed_age <= 101:
                    raise ValueError(f"Invalid age {parsed_age} for image {row['image_id']}")
                selected.append({**row, "age": parsed_age})
        overlaps = {identity: names for identity, names in identities.items() if len(names) != 1}
        if overlaps:
            raise ValueError(f"AGE-1A identities cross splits; examples: {list(overlaps.items())[:3]}")
        declared_identity_split = {row["identity"]: row["split"] for row in identity_rows}
        actual_identity_split = {identity: next(iter(names)) for identity, names in identities.items()}
        if len(declared_identity_split) != len(identity_rows) or declared_identity_split != actual_identity_split:
            raise ValueError("AGE-1A identity_splits.csv does not match the image manifest and split assignments")
        counts = {name: sum(value == name for value in split_for_id.values()) for name in EXPECTED_COUNTS}
        if counts != EXPECTED_COUNTS:
            raise ValueError(f"AGE-1A image counts changed: {counts}; expected {EXPECTED_COUNTS}")
        self.rows = selected
        if len(self.rows) != EXPECTED_COUNTS[split]:
            raise ValueError(f"AGE-1A {split} count is {len(self.rows)}, expected {EXPECTED_COUNTS[split]}")

        self.image_dir = self.dataset_root / "images"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"AgeDB images directory not found: {self.image_dir} (set AGEDB_ROOT)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = self.image_dir / row["relative_path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        tensor = self.transform(image)
        if tensor.shape != (3, 224, 224) or tensor.dtype != torch.float32:
            raise ValueError(f"Unexpected transformed image {tensor.shape} {tensor.dtype} for {image_path}")
        age = torch.tensor(row["age"], dtype=torch.float32)
        return tensor, age


def build_dataloaders(dataset_root: Path | None = None, batch_size: int = 8,
                      num_workers: int = 0) -> dict[str, DataLoader]:
    """Create loaders without changing AGE-1A membership or rebalancing samples."""
    root_value = os.environ.get("AGEDB_ROOT")
    root = Path(dataset_root or root_value).expanduser() if (dataset_root or root_value) else None
    if root is None:
        raise ValueError("Set AGEDB_ROOT to the AgeDB directory containing images/.")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    datasets = {name: AgeDBDataset(name, root) for name in EXPECTED_COUNTS}
    return {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=(name == "train"),
                         num_workers=num_workers, drop_last=False)
        for name, dataset in datasets.items()
    }
