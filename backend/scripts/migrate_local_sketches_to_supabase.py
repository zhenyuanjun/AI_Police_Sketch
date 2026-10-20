#!/usr/bin/env python3
"""Idempotently copy the existing filesystem sketch store into Supabase."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.image_storage import LocalSketchImageStorage, create_sketch_image_storage  # noqa: E402

logger = logging.getLogger("sketch_migration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=BACKEND_DIR / "data" / "generated_sketches",
        help="Local generated_sketches directory (default: backend/data/generated_sketches)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(BACKEND_DIR / ".env")

    source = LocalSketchImageStorage(root=args.source)
    destination = create_sketch_image_storage()
    if isinstance(destination, LocalSketchImageStorage):
        parser.error("Set SKETCH_STORAGE_BACKEND=supabase before running this migration.")

    copied = 0
    missing = 0
    for metadata_path in sorted(source.root.glob("*.json")):
        try:
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            sketch_id = document["sketch_id"]
            versions = document.get("versions", [])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            logger.warning("Skipping unreadable metadata file: %s", metadata_path)
            missing += 1
            continue

        for version in sorted(versions, key=lambda item: item.get("version", 0)):
            image_path = source.root / version.get("filename", "")
            if not image_path.is_file():
                logger.warning("Missing image; skipped sketch_id=%s version=%s path=%s", sketch_id, version.get("version"), image_path)
                missing += 1
                continue
            destination.save(
                sketch_id=sketch_id,
                version=int(version["version"]),
                image_bytes=image_path.read_bytes(),
                generation_type=version.get("generation_type", "unknown"),
                description=version.get("description"),
                refinement_instruction=version.get("refinement_instruction"),
                parent_version=version.get("parent_version"),
                witness_age=document.get("witness_age"),
                created_at=version.get("created_at"),
                age_analysis=version.get("age_analysis"),
            )
            copied += 1
            logger.info("Migrated sketch_id=%s version=%s", sketch_id, version["version"])

    logger.info("Migration complete: copied_versions=%s skipped_items=%s destination=%s", copied, missing, destination.root)
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
