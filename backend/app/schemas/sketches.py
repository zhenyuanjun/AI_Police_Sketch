from typing import Optional

from pydantic import BaseModel, Field


class GenerateSketchRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    witness_age: Optional[float] = Field(default=None, ge=1, le=120)


class RefineSketchRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=1000)


class SketchResponse(BaseModel):
    id: str
    version: int
    description: str
    refinement_instruction: Optional[str] = None
    image_url: Optional[str] = None
    status: str
    witness_age: Optional[float] = None


class FaceAnalysisResponse(BaseModel):
    model: str
    embedding_dimension: int
    inference_time_ms: float
    device: str


class AgeArtifactHashes(BaseModel):
    backbone: str
    age_head: str


class AgeAnalysisResponse(BaseModel):
    sketch_id: str
    version: int
    status: str
    ml_estimated_age: float
    out_of_training_range: bool
    model: str
    training_domain: str
    interpretation: str
    artifact_sha256: AgeArtifactHashes


class BenchmarkResponse(BaseModel):
    model: str
    runtime: str
    device: str
    embedding_dimension: int
    warmup_runs: int
    measured_runs: int
    mean_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float


class SavedSketchResponse(BaseModel):
    sketch_id: str
    latest_version: int
    generation_type: str
    created_at: str
    image_url: str
    witness_age: Optional[float] = None
    versions: list["SavedVersionResponse"]


class SavedVersionResponse(BaseModel):
    version: int
    generation_type: str
    refinement_instruction: Optional[str] = None
    image_url: str
    age_analysis: Optional[dict] = None


class SketchComparisonResponse(BaseModel):
    sketch_id: str
    version_a: int
    version_b: int
    model: str
    embedding_dimension: int
    cosine_similarity: float
    cosine_distance: float
    refinement_instruction: Optional[str] = None
