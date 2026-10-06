#!/usr/bin/env python3
"""Build a deterministic, subject-disjoint AgeDB manifest and split."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_IMAGES = 16_488
SEED = 534_1
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
PARSER_VERSION = "agedb-filename-v1"
# Greedy parsing preserves identity exactly, including underscores, Unicode, and spaces.
FILENAME_RE = re.compile(r"^(\d+)_(.+)_(\d+)_([mf])\.jpg$", re.DOTALL)
AGE_BINS = [(i, i + 9, f"{i}–{i+9}") for i in range(1, 91, 10)] + [(91, 101, "91+")]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def parse_filename(path: Path) -> dict:
    match = FILENAME_RE.fullmatch(path.name)
    if not match:
        fail(f"Invalid AgeDB filename: {path.name!r}")
    image_id, identity, age, gender = match.groups()
    if not identity:
        fail(f"Empty identity in filename: {path.name!r}")
    return {"image_id": image_id, "relative_path": path.name,
            "identity": identity, "age": int(age), "gender": gender}


def validate(rows: list[dict]) -> None:
    if len(rows) != EXPECTED_IMAGES:
        fail(f"Expected {EXPECTED_IMAGES} JPG images, found {len(rows)}")
    for field in ("image_id", "relative_path"):
        values = [row[field] for row in rows]
        if len(set(values)) != len(values):
            fail(f"Duplicate {field} detected")
    for row in rows:
        if not row["identity"]:
            fail("Empty identity detected")
        if not isinstance(row["age"], int) or not 1 <= row["age"] <= 101:
            fail(f"Age outside 1..101 in {row['relative_path']!r}")
        if row["gender"] not in {"m", "f"}:
            fail(f"Invalid gender in {row['relative_path']!r}")


def create_split(rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    identities = sorted({row["identity"] for row in rows})
    shuffled = identities[:]
    random.Random(SEED).shuffle(shuffled)
    n_train = round(len(identities) * SPLIT_RATIOS["train"])
    n_val = round(len(identities) * SPLIT_RATIOS["validation"])
    assigned = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }
    owner = {identity: split for split, members in assigned.items() for identity in members}
    splits = {name: [row for row in rows if owner[row["identity"]] == name]
              for name in SPLIT_RATIOS}
    sets = {name: set(members) for name, members in assigned.items()}
    names = list(sets)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if sets[first] & sets[second]:
                fail(f"Identity overlap between {first} and {second}")
    if sum(map(len, splits.values())) != len(rows):
        fail("Manifest rows are not assigned exactly once")
    return splits, assigned


def age_bins(rows: list[dict]) -> dict[str, int]:
    return {label: sum(lo <= row["age"] <= hi for row in rows)
            for lo, hi, label in AGE_BINS}


def main() -> None:
    root_value = os.environ.get("AGEDB_ROOT")
    if not root_value:
        fail("Set AGEDB_ROOT to the AgeDB directory containing images/")
    root = Path(root_value).expanduser().resolve()
    images = root / "images"
    if not images.is_dir():
        fail(f"Image directory does not exist: {images}")

    # Include all jpg/JPG variants in validation; fail on anything that does not parse.
    image_paths = sorted((p for p in images.iterdir()
                          if p.is_file() and p.suffix.lower() == ".jpg"),
                         key=lambda p: p.name)
    rows = [parse_filename(path) for path in image_paths]
    validate(rows)
    splits, assigned = create_split(rows)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "relative_path", "identity", "age", "gender"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    split_rows = [{"image_id": row["image_id"], "split": split_name}
                  for split_name, members in splits.items() for row in members]
    split_rows.sort(key=lambda item: item["image_id"])
    split_path = out_dir / "splits.csv"
    with split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(split_rows)

    identity_split_path = out_dir / "identity_splits.csv"
    with identity_split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["identity", "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"identity": ident, "split": name}
                         for name, identities in assigned.items() for ident in sorted(identities))

    stats = {
        "total_images": len(rows), "unique_identities": len(assigned["train"] + assigned["validation"] + assigned["test"]),
        "age_min": min(r["age"] for r in rows), "age_max": max(r["age"] for r in rows),
        "age_mean": statistics.mean(r["age"] for r in rows),
        "age_median": statistics.median(r["age"] for r in rows),
        "gender_counts": dict(sorted(Counter(r["gender"] for r in rows).items())),
        "images_per_identity": {
            "min": min(Counter(r["identity"] for r in rows).values()),
            "max": max(Counter(r["identity"] for r in rows).values()),
            "mean": statistics.mean(Counter(r["identity"] for r in rows).values()),
            "median": statistics.median(Counter(r["identity"] for r in rows).values()),
        },
        "splits": {},
    }
    for name, members in splits.items():
        stats["splits"][name] = {
            "identities": len(assigned[name]), "images": len(members),
            "image_percent": 100 * len(members) / len(rows),
            "age_min": min(r["age"] for r in members), "age_max": max(r["age"] for r in members),
            "age_mean": statistics.mean(r["age"] for r in members),
            "age_median": statistics.median(r["age"] for r in members),
            "age_bins": age_bins(members),
        }
    metadata = {
        "random_seed": SEED, "split_ratios": SPLIT_RATIOS,
        "parsing_rule_version": PARSER_VERSION,
        "filename_pattern": "<numeric_image_id>_<identity>_<numeric_age>_<m|f>.jpg; identity captured greedily and preserved verbatim",
        "source_dataset_path": str(root), "stats": stats,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256()
    for path in (manifest_path, split_path, identity_split_path, report_path):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    hash_path = out_dir / "metadata.sha256"
    hash_path.write_text(digest.hexdigest() + "\n", encoding="ascii")

    print(json.dumps({"output_dir": str(out_dir), "metadata_sha256": digest.hexdigest(),
                      "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
