from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
from io import BytesIO
import os
from typing import Dict, Optional
from uuid import uuid4


@dataclass
class Sketch:
    id: str
    version: int
    description: str
    refinement_instruction: Optional[str] = None
    image_url: Optional[str] = None


class SketchGenerationService(ABC):
    @abstractmethod
    def generate(self, description: str) -> Sketch: ...

    @abstractmethod
    def generate_from_reference(self, filename: str, content_type: str, image_bytes: bytes) -> Sketch: ...

    @abstractmethod
    def refine(self, sketch_id: str, instruction: str) -> Optional[Sketch]: ...


class MockSketchGenerationService(SketchGenerationService):
    """In-memory provider used until a real image generation adapter is selected."""

    def __init__(self) -> None:
        self._sketches: Dict[str, Sketch] = {}

    def generate(self, description: str) -> Sketch:
        sketch = Sketch(id=str(uuid4()), version=1, description=description.strip())
        self._sketches[sketch.id] = sketch
        return sketch

    def generate_from_reference(self, filename: str, content_type: str, image_bytes: bytes) -> Sketch:
        # The bytes are inspected only for this request and are never written to disk.
        description = "Reference image upload: {} ({})".format(filename, content_type)
        sketch = Sketch(id=str(uuid4()), version=1, description=description)
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
        )
        self._sketches[sketch_id] = refined
        return refined


class OpenAISketchGenerationService(SketchGenerationService):
    """OpenAI provider for text-to-image generation and iterative image edits."""

    MODEL = "gpt-image-2.5-flare"
    EDIT_MODEL = "gpt-image-1"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key) if api_key else None
        self._sketches: Dict[str, Sketch] = {}
        self._image_bytes: Dict[str, bytes] = {}

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

    def generate(self, description: str) -> Sketch:
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
        sketch = Sketch(id=str(uuid4()), version=1, description=cleaned_description, image_url=image_url)
        self._sketches[sketch.id] = sketch
        self._image_bytes[sketch.id] = image_bytes
        return sketch

    def generate_from_reference(self, filename: str, content_type: str, image_bytes: bytes) -> Sketch:
        raise NotImplementedError("Reference-image generation is not available in OpenAI Phase 2A.")

    def refine(self, sketch_id: str, instruction: str) -> Optional[Sketch]:
        cleaned_instruction = instruction.strip()
        if not cleaned_instruction:
            raise ValueError("A refinement instruction is required.")
        if self._client is None:
            raise ValueError("OPENAI_API_KEY is required when SKETCH_PROVIDER=openai.")
        current = self._sketches.get(sketch_id)
        image_bytes = self._image_bytes.get(sketch_id)
        if current is None or image_bytes is None:
            raise ValueError("No generated image is available to refine.")

        prompt = (
            "Edit this existing forensic police composite sketch while preserving the identity of the same person. "
            "Preserve the person's pose, framing, composition, approximate age, gender presentation, ethnicity or skin appearance, "
            "hairstyle, clothing, background, expression, and every facial feature not explicitly mentioned in the instruction. "
            "Modify only the requested visible characteristics. Maintain the same monochrome forensic sketch style. "
            "Keep one person only and produce no text, labels, or watermark.\n\n"
            f"Refinement instruction:\n{cleaned_instruction}"
        )
        image_file = BytesIO(image_bytes)
        image_file.name = "caseid-sketch.png"
        try:
            result = self._client.images.edit(model=self.EDIT_MODEL, image=image_file, prompt=prompt)
        except Exception as exc:
            raise RuntimeError("OpenAI image refinement failed. Check the backend logs for details.") from exc

        image_url, edited_image_bytes = self._image_from_response(result)
        refined = Sketch(
            id=current.id,
            version=current.version + 1,
            description=current.description,
            refinement_instruction=cleaned_instruction,
            image_url=image_url,
        )
        self._sketches[sketch_id] = refined
        self._image_bytes[sketch_id] = edited_image_bytes
        return refined
