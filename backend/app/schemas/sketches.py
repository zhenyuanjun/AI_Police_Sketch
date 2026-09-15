from typing import Optional

from pydantic import BaseModel, Field


class GenerateSketchRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)


class RefineSketchRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=1000)


class SketchResponse(BaseModel):
    id: str
    version: int
    description: str
    refinement_instruction: Optional[str] = None
    image_url: Optional[str] = None
    status: str
