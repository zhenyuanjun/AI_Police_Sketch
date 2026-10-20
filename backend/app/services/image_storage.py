from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class LocalSketchImageStorage:
    """Small filesystem-backed store for generated sketch image versions."""

    def __init__(self, root: Optional[Path] = None) -> None:
        configured_root = os.getenv("SKETCH_STORAGE_DIR")
        default_root = Path(__file__).resolve().parents[2] / "data" / "generated_sketches"
        self.root = Path(root or configured_root or default_root).expanduser().resolve()
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
        created_at: Optional[str] = None,
        age_analysis: Optional[dict[str, Any]] = None,
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
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        }
        if description is not None:
            version_metadata["description"] = description
        if refinement_instruction is not None:
            version_metadata["refinement_instruction"] = refinement_instruction
        if parent_version is not None:
            version_metadata["parent_version"] = parent_version
        if age_analysis is not None:
            version_metadata["age_analysis"] = age_analysis
        versions.append(version_metadata)
        metadata["versions"] = sorted(versions, key=lambda item: item["version"])
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "Saved sketch version: sketch_id=%s version=%s image_path=%s metadata_path=%s process_id=%s",
            sketch_id,
            version,
            image_path,
            metadata_path,
            os.getpid(),
        )
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
                logger.warning(
                    "Sketch version lookup missing metadata: sketch_id=%s requested_version=%s "
                    "metadata_path=%s storage_root=%s process_id=%s",
                    sketch_id,
                    version,
                    metadata_path,
                    self.root,
                    os.getpid(),
                )
                return None
            image_path = self.root / f"{sketch_id}.png"
            if image_path.exists():
                return image_path.read_bytes()
            logger.warning(
                "Sketch version lookup missing image and metadata: sketch_id=%s requested_version=%s "
                "image_path=%s metadata_path=%s storage_root=%s process_id=%s",
                sketch_id,
                version,
                image_path,
                metadata_path,
                self.root,
                os.getpid(),
            )
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            versions = metadata.get("versions", [])
            selected = max(versions, key=lambda item: item["version"]) if version is None else next(
                item for item in versions if item["version"] == version
            )
            image_path = self.root / selected["filename"]
        except StopIteration:
            logger.warning(
                "Sketch version lookup missing requested version: sketch_id=%s requested_version=%s "
                "metadata_path=%s available_versions=%s storage_root=%s process_id=%s",
                sketch_id,
                version,
                metadata_path,
                [item.get("version") for item in versions if isinstance(item, dict)],
                self.root,
                os.getpid(),
            )
            return None
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.exception(
                "Sketch version metadata could not be read: sketch_id=%s requested_version=%s "
                "metadata_path=%s storage_root=%s process_id=%s",
                sketch_id,
                version,
                metadata_path,
                self.root,
                os.getpid(),
            )
            return None
        if image_path.exists():
            return image_path.read_bytes()
        logger.warning(
            "Sketch version lookup found metadata but image is missing: sketch_id=%s requested_version=%s "
            "image_path=%s metadata_path=%s storage_root=%s process_id=%s",
            sketch_id,
            version,
            image_path,
            metadata_path,
            self.root,
            os.getpid(),
        )
        return None

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


class SupabaseSketchImageStorage:
    """Store JSON metadata in Supabase Postgres and PNGs in a private bucket."""

    def __init__(
        self,
        project_url: str,
        service_role_key: str,
        bucket: str = "sketch-images",
        *,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        if not project_url.strip() or not service_role_key.strip():
            raise RuntimeError("Supabase persistence requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
        self.project_url = project_url.rstrip("/")
        self.bucket = bucket.strip()
        if not self.bucket:
            raise RuntimeError("SUPABASE_STORAGE_BUCKET must not be empty.")
        self.root = f"supabase://{self.bucket}"
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={
                "apikey": service_role_key,
                "Authorization": f"Bearer {service_role_key}",
            },
        )

    def _table_url(self) -> str:
        return f"{self.project_url}/rest/v1/sketches"

    def _object_url(self, sketch_id: str, filename: str) -> str:
        object_key = f"{quote(sketch_id, safe='')}/{quote(filename, safe='')}"
        return f"{self.project_url}/storage/v1/object/{quote(self.bucket, safe='')}/{object_key}"

    def _get_metadata(self, sketch_id: str) -> Optional[dict[str, Any]]:
        response = self._client.get(
            self._table_url(),
            params={"select": "metadata", "sketch_id": f"eq.{sketch_id}"},
        )
        response.raise_for_status()
        rows = response.json()
        return rows[0]["metadata"] if rows else None

    def _put_metadata(self, sketch_id: str, metadata: dict[str, Any]) -> None:
        response = self._client.post(
            self._table_url(),
            params={"on_conflict": "sketch_id"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
            json={
                "sketch_id": sketch_id,
                "metadata": metadata,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        response.raise_for_status()

    def _download(self, sketch_id: str, filename: str) -> Optional[bytes]:
        response = self._client.get(self._object_url(sketch_id, filename))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

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
        created_at: Optional[str] = None,
        age_analysis: Optional[dict[str, Any]] = None,
    ) -> str:
        filename = f"{sketch_id}.png" if version == 1 else f"{sketch_id}_v{version}.png"
        upload = self._client.post(
            self._object_url(sketch_id, filename),
            headers={"Content-Type": "image/png", "x-upsert": "true"},
            content=image_bytes,
        )
        upload.raise_for_status()

        metadata = self._get_metadata(sketch_id) or {"sketch_id": sketch_id, "versions": []}
        if witness_age is not None:
            metadata["witness_age"] = float(witness_age)
        versions = [item for item in metadata.get("versions", []) if item.get("version") != version]
        version_metadata: dict[str, Any] = {
            "version": version,
            "filename": filename,
            "generation_type": generation_type,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        }
        if description is not None:
            version_metadata["description"] = description
        if refinement_instruction is not None:
            version_metadata["refinement_instruction"] = refinement_instruction
        if parent_version is not None:
            version_metadata["parent_version"] = parent_version
        if age_analysis is not None:
            version_metadata["age_analysis"] = age_analysis
        versions.append(version_metadata)
        metadata["versions"] = sorted(versions, key=lambda item: item["version"])
        self._put_metadata(sketch_id, metadata)
        logger.info("Saved sketch to Supabase: sketch_id=%s version=%s bucket=%s", sketch_id, version, self.bucket)
        return self._object_url(sketch_id, filename)

    def save_age_analysis(self, sketch_id: str, version: int, analysis: dict[str, Any]) -> bool:
        metadata = self._get_metadata(sketch_id)
        if metadata is None:
            return False
        try:
            record = next(item for item in metadata.get("versions", []) if item.get("version") == version)
        except (StopIteration, TypeError):
            return False
        record["age_analysis"] = {**analysis, "analyzed_at": datetime.now(timezone.utc).isoformat()}
        self._put_metadata(sketch_id, metadata)
        return True

    def load_latest(self, sketch_id: str) -> Optional[bytes]:
        return self.load_version(sketch_id)

    def load_version(self, sketch_id: str, version: Optional[int] = None) -> Optional[bytes]:
        metadata = self._get_metadata(sketch_id)
        if metadata is None:
            return None
        versions = metadata.get("versions", [])
        try:
            selected = max(versions, key=lambda item: item["version"]) if version is None else next(
                item for item in versions if item.get("version") == version
            )
        except (StopIteration, TypeError, ValueError):
            return None
        filename = selected.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            logger.error("Invalid stored sketch filename: sketch_id=%s version=%s", sketch_id, version)
            return None
        return self._download(sketch_id, filename)

    def get_version_metadata(self, sketch_id: str, version: int) -> Optional[dict[str, Any]]:
        metadata = self._get_metadata(sketch_id)
        if metadata is None:
            return None
        try:
            return next(item for item in metadata.get("versions", []) if item.get("version") == version)
        except (StopIteration, TypeError):
            return None

    def get_latest_version_metadata(self, sketch_id: str) -> Optional[dict[str, Any]]:
        metadata = self._get_metadata(sketch_id)
        try:
            return max(metadata.get("versions", []), key=lambda item: item["version"]) if metadata else None
        except (TypeError, ValueError):
            return None

    def get_witness_age(self, sketch_id: str) -> Optional[float]:
        metadata = self._get_metadata(sketch_id)
        try:
            value = metadata.get("witness_age") if metadata else None
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def list_sketches(self) -> list[dict[str, Any]]:
        response = self._client.get(
            self._table_url(),
            params={"select": "sketch_id,metadata", "order": "updated_at.desc"},
        )
        response.raise_for_status()
        saved = []
        for row in response.json():
            metadata = row.get("metadata") or {}
            versions = metadata.get("versions", [])
            try:
                latest = max(versions, key=lambda item: item["version"])
                saved.append(
                    {
                        "sketch_id": row["sketch_id"],
                        "latest_version": latest["version"],
                        "generation_type": latest.get("generation_type", "unknown"),
                        "created_at": latest["created_at"],
                        "image_url": f"/api/sketches/{row['sketch_id']}/image",
                        "witness_age": metadata.get("witness_age"),
                        "versions": [
                            {
                                "version": item["version"],
                                "generation_type": item.get("generation_type", "unknown"),
                                "refinement_instruction": item.get("refinement_instruction"),
                                "image_url": f"/api/sketches/{row['sketch_id']}/image?version={item['version']}",
                                "age_analysis": item.get("age_analysis"),
                            }
                            for item in sorted(versions, key=lambda item: item["version"])
                        ],
                    }
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed Supabase sketch metadata: sketch_id=%s", row.get("sketch_id"))
        return sorted(saved, key=lambda item: item["created_at"], reverse=True)


def create_sketch_image_storage():
    """Select remote persistence explicitly, retaining local storage for development."""
    backend = os.getenv("SKETCH_STORAGE_BACKEND", "local").strip().lower()
    if backend == "local":
        return LocalSketchImageStorage()
    if backend == "supabase":
        project_url = os.getenv("SUPABASE_URL", "").strip()
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not project_url or not service_role_key:
            logger.warning(
                "SKETCH_STORAGE_BACKEND=supabase was selected without SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY; falling back to local sketch storage."
            )
            return LocalSketchImageStorage()
        return SupabaseSketchImageStorage(
            project_url=project_url,
            service_role_key=service_role_key,
            bucket=os.getenv("SUPABASE_STORAGE_BUCKET", "").strip() or "sketch-images",
        )
    raise RuntimeError("SKETCH_STORAGE_BACKEND must be 'local' or 'supabase'.")
