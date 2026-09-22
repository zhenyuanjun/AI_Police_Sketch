from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
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
    """OpenAI text-to-image provider for Phase 2A."""

    MODEL = "gpt-image-2.5-flare"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key) if api_key else None

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
        image_data = getattr(result.data[0], "b64_json", None) if getattr(result, "data", None) else None
        if not image_data:
            raise RuntimeError("OpenAI returned no generated image data.")
        try:
            base64.b64decode(image_data, validate=True)
        except (ValueError, TypeError):
            raise RuntimeError("OpenAI returned malformed image data.") from None
        return Sketch(id=str(uuid4()), version=1, description=cleaned_description, image_url=f"data:image/png;base64,{image_data}")

    def generate_from_reference(self, filename: str, content_type: str, image_bytes: bytes) -> Sketch:
        raise NotImplementedError("Reference-image generation is not available in OpenAI Phase 2A.")

    def refine(self, sketch_id: str, instruction: str) -> Optional[Sketch]:
        raise NotImplementedError("Image refinement is not available in OpenAI Phase 2A.")
