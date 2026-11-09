from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
from io import BytesIO
import logging
import os
from typing import Dict, Optional
from uuid import uuid4

from app.services.image_storage import LocalSketchImageStorage

logger = logging.getLogger(__name__)


@dataclass
class Sketch:
    id: str
    version: int
    description: str
    refinement_instruction: Optional[str] = None
    image_url: Optional[str] = None
    witness_age: Optional[float] = None


class SketchGenerationService(ABC):
    @abstractmethod
    def generate(self, description: str, witness_age: Optional[float] = None) -> Sketch: ...

    @abstractmethod
    def refine(self, sketch_id: str, instruction: str) -> Optional[Sketch]: ...

    @abstractmethod
    def get_image_bytes(self, sketch_id: str) -> Optional[bytes]: ...


class MockSketchGenerationService(SketchGenerationService):
    """In-memory provider used until a real image generation adapter is selected."""

    def __init__(self) -> None:
        self._sketches: Dict[str, Sketch] = {}

    def generate(self, description: str, witness_age: Optional[float] = None) -> Sketch:
        sketch = Sketch(id=str(uuid4()), version=1, description=description.strip(), witness_age=witness_age)
        self._sketches[sketch.id] = sketch
        return sketch

    def refine(self, sketch_id: str, instruction: str) -> Optional[Sketch]:
        current = self._sketches.get(sketch_id)
        if current is None:
            return None
        refined = Sketch(
            id=current.id,
            version=current.version + 1,
            description=current.description,
            refinement_instruction=instruction.strip(),
            witness_age=current.witness_age,
        )
        self._sketches[sketch_id] = refined
        return refined

    def get_image_bytes(self, sketch_id: str) -> Optional[bytes]:
        return None


class OpenAISketchGenerationService(SketchGenerationService):
    """OpenAI provider for text-to-image generation and iterative image edits."""

    MODEL = "gpt-image-2.5-flare"
    # gpt-image-1 was shut down on 2026-10-23. Keep edits on the active image
    # model used by generation so the account supports both operations.
    EDIT_MODEL = MODEL

    def __init__(self, image_storage=None) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key) if api_key else None
        self._sketches: Dict[str, Sketch] = {}
        self._image_bytes: Dict[str, bytes] = {}
        self._image_storage = image_storage or LocalSketchImageStorage()

    @staticmethod
    def _image_from_response(result) -> tuple[str, bytes]:
        image_data = getattr(result.data[0], "b64_json", None) if getattr(result, "data", None) else None
        if not image_data:
            raise RuntimeError("OpenAI returned no generated image data.")
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except (ValueError, TypeError):
            raise RuntimeError("OpenAI returned malformed image data.") from None
        if not image_bytes:
            raise RuntimeError("OpenAI returned an empty image.")
        return f"data:image/png;base64,{image_data}", image_bytes

    def generate(self, description: str, witness_age: Optional[float] = None) -> Sketch:
        cleaned_description = description.strip()
        if not cleaned_description:
            raise ValueError("A witness description is required.")
        if self._client is None:
            raise ValueError("OPENAI_API_KEY is required when SKETCH_PROVIDER=openai.")
        prompt = (
            "Create a forensic-style police composite facial sketch for a research prototype. "
            "Depict one person only, with a frontal or near-frontal head-and-shoulders composition, "
            "a neutral expression, a neutral simple background, and clearly visible facial features. "
            "Use a consistent monochrome police composite / forensic sketch presentation. "
            "Include no text, labels, watermark, props, or additional people. "
            "Preserve the witness-provided visible characteristics as closely as possible. "
            "This is an interpretive visual reconstruction, not a claim that the person shown is an actual suspect.\n\n"
            f"Witness description:\n{cleaned_description}"
        )
        try:
            result = self._client.images.generate(model=self.MODEL, prompt=prompt)
        except Exception as exc:
            raise RuntimeError("OpenAI image generation failed. Check the backend logs for details.") from exc
        image_url, image_bytes = self._image_from_response(result)
        sketch = Sketch(id=str(uuid4()), version=1, description=cleaned_description, image_url=image_url, witness_age=witness_age)
        self._sketches[sketch.id] = sketch
        self._image_bytes[sketch.id] = image_bytes
        self._image_storage.save(
            sketch.id,
            sketch.version,
            image_bytes,
            "initial",
            description=sketch.description,
            witness_age=witness_age,
        )
        return sketch

    def refine(self, sketch_id: str, instruction: str) -> Optional[Sketch]:
        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            raise ValueError("A refinement instruction is required.")
        if self._client is None:
            raise ValueError("OPENAI_API_KEY is required when SKETCH_PROVIDER=openai.")
        current = self._sketches.get(sketch_id)
        image_bytes = self._image_bytes.get(sketch_id)
        source_from_memory = image_bytes is not None
        source_metadata = None
        stage = "source_image_lookup"
        if image_bytes is None:
            try:
                image_bytes = self._image_storage.load_latest(sketch_id)
            except Exception:
                logger.exception(
                    "Sketch refinement failed: stage=%s sketch_id=%s storage_backend=%s",
                    stage, sketch_id, type(self._image_storage).__name__,
                )
                raise
        if current is None:
            source_metadata = self._image_storage.get_latest_version_metadata(sketch_id)
            if source_metadata is not None:
                current = Sketch(
                    id=sketch_id,
                    version=source_metadata["version"],
                    description=source_metadata.get("description", ""),
                    refinement_instruction=source_metadata.get("refinement_instruction"),
                    witness_age=self._image_storage.get_witness_age(sketch_id),
                )
        if current is None or image_bytes is None:
            logger.error(
                "Sketch refinement source unavailable: stage=%s sketch_id=%s image_exists=%s "
                "image_size_bytes=%s storage_backend=%s storage_root=%s",
                stage, sketch_id, image_bytes is not None,
                len(image_bytes) if image_bytes is not None else 0,
                type(self._image_storage).__name__,
                getattr(self._image_storage, "root", None),
            )
            raise ValueError("No generated image is available to refine.")

        source_filename = (source_metadata or {}).get("filename")
        storage_root = getattr(self._image_storage, "root", None)
        if source_filename is None and storage_root is not None:
            source_filename = (
                f"{sketch_id}.png" if current.version == 1
                else f"{sketch_id}_v{current.version}.png"
            )
        source_path = storage_root / source_filename if storage_root is not None and source_filename else None

        source_format = source_mode = source_dimensions = "unknown"
        try:
            from PIL import Image
            with Image.open(BytesIO(image_bytes)) as source_image:
                source_format = source_image.format or "unknown"
                source_mode = source_image.mode
                source_dimensions = f"{source_image.width}x{source_image.height}"
                source_image.verify()
        except Exception as exc:
            logger.exception(
                "Sketch refinement source image validation failed: stage=source_image_validation "
                "sketch_id=%s image_exists=true image_size_bytes=%s image_format=%s "
                "source_path=%s storage_backend=%s",
                sketch_id, len(image_bytes), source_format, source_path,
                type(self._image_storage).__name__,
            )
            raise ValueError("The generated image is not a valid image to refine.") from exc
        logger.info(
            "Sketch refinement source ready: sketch_id=%s source_version=%s source_from_memory=%s "
            "image_exists=true image_size_bytes=%s image_format=%s image_mode=%s "
            "image_dimensions=%s source_path=%s storage_backend=%s storage_root=%s",
            sketch_id, current.version, source_from_memory, len(image_bytes), source_format,
            source_mode, source_dimensions, source_path, type(self._image_storage).__name__,
            getattr(self._image_storage, "root", None),
        )

        prompt = (
            "Edit this existing forensic police composite sketch while preserving the identity of the same person. "
            "Preserve the person's pose, framing, composition, approximate age, gender presentation, ethnicity or skin appearance, "
            "hairstyle, clothing, background, expression, and every facial feature not explicitly mentioned in the instruction. "
            "Modify only the requested visible characteristics. Maintain the same monochrome forensic sketch style. "
            "Keep one person only and produce no text, labels, or watermark.\n\n"
            f"Refinement instruction:\n{cleaned_instruction}"
        )
        image_file = BytesIO(image_bytes)
        image_file.name = "caseid-sketch.jpg" if source_format == "JPEG" else "caseid-sketch.png"
        stage = "openai_image_edit"
        try:
            result = self._client.images.edit(model=self.EDIT_MODEL, image=image_file, prompt=prompt)
        except Exception as exc:
            logger.exception(
                "Sketch refinement failed: stage=%s sketch_id=%s model=%s exception_class=%s "
                "http_status=%s error_message=%s image_exists=true image_size_bytes=%s "
                "image_format=%s image_mode=%s image_dimensions=%s",
                stage, sketch_id, self.EDIT_MODEL, type(exc).__name__,
                getattr(exc, "status_code", None), str(getattr(exc, "message", str(exc)))[:1000],
                len(image_bytes), source_format, source_mode, source_dimensions,
            )
            raise RuntimeError("OpenAI image refinement failed. Check the backend logs for details.") from exc

        stage = "openai_response_decode"
        try:
            image_url, edited_image_bytes = self._image_from_response(result)
        except Exception:
            logger.exception(
                "Sketch refinement failed: stage=%s sketch_id=%s model=%s",
                stage, sketch_id, self.EDIT_MODEL,
            )
            raise
        refined = Sketch(
            id=current.id,
            version=current.version + 1,
            description=current.description,
            refinement_instruction=cleaned_instruction,
            image_url=image_url,
            witness_age=current.witness_age,
        )
        self._sketches[sketch_id] = refined
        self._image_bytes[sketch_id] = edited_image_bytes
        stage = "version_history_storage"
        try:
            self._image_storage.save(
                refined.id,
                refined.version,
                edited_image_bytes,
                "refinement",
                description=refined.description or None,
                refinement_instruction=refined.refinement_instruction,
                parent_version=current.version,
            )
        except Exception:
            logger.exception(
                "Sketch refinement failed: stage=%s sketch_id=%s source_version=%s target_version=%s "
                "storage_backend=%s",
                stage, sketch_id, current.version, refined.version, type(self._image_storage).__name__,
            )
            raise
        return refined

    def get_image_bytes(self, sketch_id: str) -> Optional[bytes]:
        persisted = self._image_storage.load_latest(sketch_id)
        return persisted if persisted is not None else self._image_bytes.get(sketch_id)
