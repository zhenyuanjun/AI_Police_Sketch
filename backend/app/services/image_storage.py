from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional


class LocalSketchImageStorage:
    """Small filesystem-backed store for generated sketch image versions."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2] / "data" / "generated_sketches"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        sketch_id: str,
        version: int,
        image_bytes: bytes,
        generation_type: str,
        description: Optional[str] = None,
        refinement_instruction: Optional[str] = None,
        parent_version: Optional[int] = None,
        witness_age: Optional[float] = None,
    ) -> Path:
        filename = f"{sketch_id}.png" if version == 1 else f"{sketch_id}_v{version}.png"
        image_path = self.root / filename
        image_path.write_bytes(image_bytes)

        metadata_path = self.root / f"{sketch_id}.json"
        metadata: dict[str, Any] = {"sketch_id": sketch_id, "versions": []}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {"sketch_id": sketch_id, "versions": []}
        if witness_age is not None:
            metadata["witness_age"] = float(witness_age)
        versions = [item for item in metadata.get("versions", []) if item.get("version") != version]
        version_metadata: dict[str, Any] = {
            "version": version,
            "filename": filename,
            "generation_type": generation_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if description is not None:
            version_metadata["description"] = description
        if refinement_instruction is not None:
            version_metadata["refinement_instruction"] = refinement_instruction
        if parent_version is not None:
            version_metadata["parent_version"] = parent_version
        versions.append(version_metadata)
        metadata["versions"] = sorted(versions, key=lambda item: item["version"])
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return image_path

    def save_age_analysis(self, sketch_id: str, version: int, analysis: dict[str, Any]) -> bool:
        """Replace the successful analysis record for one existing image version."""
        metadata_path = self.root / f"{sketch_id}.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record = next(item for item in metadata.get("versions", []) if item.get("version") == version)
            record["age_analysis"] = {**analysis, "analyzed_at": datetime.now(timezone.utc).isoformat()}
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return True
        except (OSError, KeyError, TypeError, StopIteration, ValueError, json.JSONDecodeError):
            return False

    def load_latest(self, sketch_id: str) -> Optional[bytes]:
        return self.load_version(sketch_id)

    def load_version(self, sketch_id: str, version: Optional[int] = None) -> Optional[bytes]:
        metadata_path = self.root / f"{sketch_id}.json"
        if not metadata_path.exists():
            if version not in (None, 1):
                return None
            image_path = self.root / f"{sketch_id}.png"
            return image_path.read_bytes() if image_path.exists() else None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            versions = metadata.get("versions", [])
            selected = max(versions, key=lambda item: item["version"]) if version is None else next(
                item for item in versions if item["version"] == version
            )
            image_path = self.root / selected["filename"]
        except (OSError, KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError):
            return None
        return image_path.read_bytes() if image_path.exists() else None

    def get_version_metadata(self, sketch_id: str, version: int) -> Optional[dict[str, Any]]:
        metadata_path = self.root / f"{sketch_id}.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return next(item for item in metadata.get("versions", []) if item.get("version") == version)
        except (OSError, KeyError, TypeError, StopIteration, json.JSONDecodeError):
            return None

    def get_latest_version_metadata(self, sketch_id: str) -> Optional[dict[str, Any]]:
        metadata_path = self.root / f"{sketch_id}.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return max(metadata.get("versions", []), key=lambda item: item["version"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def get_witness_age(self, sketch_id: str) -> Optional[float]:
        try:
            metadata = json.loads((self.root / f"{sketch_id}.json").read_text(encoding="utf-8"))
            value = metadata.get("witness_age")
            return float(value) if value is not None else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def list_sketches(self) -> list[dict[str, Any]]:
        saved = []
        for metadata_path in sorted(self.root.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                versions = metadata.get("versions", [])
                latest = max(versions, key=lambda item: item["version"])
                if not (self.root / latest["filename"]).exists():
                    continue
                saved.append(
                    {
                        "sketch_id": metadata["sketch_id"],
                        "latest_version": latest["version"],
                        "generation_type": latest["generation_type"],
                        "created_at": latest["created_at"],
                        "image_url": f"/api/sketches/{metadata['sketch_id']}/image",
                        "witness_age": metadata.get("witness_age"),
                        "versions": [
                            {
                                "version": item["version"],
                                "generation_type": item.get("generation_type", "unknown"),
                                "refinement_instruction": item.get("refinement_instruction"),
                                "image_url": f"/api/sketches/{metadata['sketch_id']}/image?version={item['version']}",
                                "age_analysis": item.get("age_analysis"),
                            }
                            for item in sorted(versions, key=lambda item: item["version"])
                        ],
                    }
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(saved, key=lambda item: item["created_at"], reverse=True)
