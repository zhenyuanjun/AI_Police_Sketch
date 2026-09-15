from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
from uuid import uuid4


@dataclass
class Sketch:
    id: str
    version: int
    description: str
    refinement_instruction: Optional[str] = None


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
