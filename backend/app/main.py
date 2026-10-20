import base64
import binascii
import os
from pathlib import Path
import logging
from typing import Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Path as ApiPath, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from threading import Lock

from app.schemas.sketches import AgeAnalysisResponse, BenchmarkResponse, FaceAnalysisResponse, GenerateSketchRequest, RefineSketchRequest, SavedSketchResponse, SketchComparisonResponse, SketchResponse
from app.services.image_storage import create_sketch_image_storage
from app.services.sketch_generation import OpenAISketchGenerationService, MockSketchGenerationService, SketchGenerationService

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
logger = logging.getLogger(__name__)
app = FastAPI(title="AI Police Sketch API", version="0.1.0")
MAX_IMAGE_DATA_URL_LENGTH = 22 * 1024 * 1024
origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
def create_generation_service(storage) -> SketchGenerationService:
    provider = os.getenv("SKETCH_PROVIDER", "mock").strip().lower()
    logger.info("Sketch provider: %s", provider)
    if provider == "mock":
        return MockSketchGenerationService()
    if provider == "openai":
        return OpenAISketchGenerationService(image_storage=storage)
    raise RuntimeError("SKETCH_PROVIDER must be either 'mock' or 'openai'.")


image_storage = create_sketch_image_storage()
service: SketchGenerationService = create_generation_service(image_storage)
logger.info("Sketch storage backend location: %s", image_storage.root)
analysis_service = None
age_estimation_service = None
_analysis_service_lock = Lock()
_age_estimation_service_lock = Lock()


def get_analysis_service():
    """Load the optional ResNet visual-analysis model on its first requested use."""
    global analysis_service
    if analysis_service is None:
        with _analysis_service_lock:
            if analysis_service is None:
                from app.services.face_analysis_service import FaceAnalysisService

                analysis_service = FaceAnalysisService()
    return analysis_service


def get_age_estimation_service():
    """Load the finalized age runtime only when age analysis is first requested."""
    global age_estimation_service
    if age_estimation_service is None:
        with _age_estimation_service_lock:
            if age_estimation_service is None:
                from app.services.age_estimation_service import AgeEstimationService

                try:
                    age_estimation_service = AgeEstimationService()
                except Exception:
                    logger.exception("Age estimation service is unavailable.")
    return age_estimation_service


def response_from_sketch(sketch):
    status = "openai" if sketch.image_url else "mock"
    return SketchResponse(id=sketch.id, version=sketch.version, description=sketch.description, refinement_instruction=sketch.refinement_instruction, image_url=sketch.image_url, status=status, witness_age=sketch.witness_age)


def decode_image_data_url(image_data_url: str) -> bytes:
    """Decode a bounded PNG/JPEG data URL supplied for cross-instance analysis."""
    if len(image_data_url) > MAX_IMAGE_DATA_URL_LENGTH:
        raise HTTPException(status_code=413, detail="Image data is too large for age analysis.")
    header, separator, encoded = image_data_url.partition(",")
    if not separator or header not in ("data:image/png;base64", "data:image/jpeg;base64"):
        raise HTTPException(status_code=422, detail="Age analysis requires a PNG or JPEG data URL.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Image data is malformed.") from exc
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Image data is empty.")
    return image_bytes


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sketches/saved", response_model=list[SavedSketchResponse])
def list_saved_sketches() -> list[SavedSketchResponse]:
    return [SavedSketchResponse(**item) for item in image_storage.list_sketches()]


@app.get("/api/sketches/{sketch_id}/image")
def get_saved_sketch_image(sketch_id: str, version: Optional[int] = Query(default=None, ge=1)) -> Response:
    image_bytes = image_storage.load_version(sketch_id, version)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="Saved sketch image not found.")
    return Response(content=image_bytes, media_type="image/png")


@app.get("/api/sketches/{sketch_id}/compare", response_model=SketchComparisonResponse)
def compare_sketch_versions(
    sketch_id: str,
    version_a: int = Query(ge=1),
    version_b: int = Query(ge=1),
) -> SketchComparisonResponse:
    image_a = image_storage.load_version(sketch_id, version_a)
    image_b = image_storage.load_version(sketch_id, version_b)
    if image_a is None or image_b is None:
        raise HTTPException(status_code=404, detail="One or both sketch versions were not found.")
    try:
        result = get_analysis_service().compare(image_a, image_b)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metadata_b = image_storage.get_version_metadata(sketch_id, version_b) or {}
    return SketchComparisonResponse(
        sketch_id=sketch_id,
        version_a=version_a,
        version_b=version_b,
        refinement_instruction=metadata_b.get("refinement_instruction"),
        **result,
    )


@app.post("/api/sketches/generate", response_model=SketchResponse)
def generate_sketch(request: GenerateSketchRequest) -> SketchResponse:
    try:
        return response_from_sketch(service.generate(request.description, request.witness_age))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/sketches/{sketch_id}/refine", response_model=SketchResponse)
def refine_sketch(sketch_id: str, request: RefineSketchRequest) -> SketchResponse:
    try:
        sketch = service.refine(sketch_id, request.instruction)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if sketch is None:
        raise HTTPException(status_code=404, detail="Sketch not found")
    return response_from_sketch(sketch)


@app.post("/api/sketches/{sketch_id}/analyze", response_model=FaceAnalysisResponse)
def analyze_sketch(sketch_id: str) -> FaceAnalysisResponse:
    image_bytes = service.get_image_bytes(sketch_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="No generated image is available for this sketch.")
    try:
        return FaceAnalysisResponse(**get_analysis_service().analyze(image_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/sketches/{sketch_id}/benchmark", response_model=BenchmarkResponse)
def benchmark_sketch(sketch_id: str) -> BenchmarkResponse:
    image_bytes = service.get_image_bytes(sketch_id)
    if image_bytes is None:
        raise HTTPException(status_code=404, detail="No generated image is available for this sketch.")
    try:
        return BenchmarkResponse(**get_analysis_service().benchmark(image_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/sketches/{sketch_id}/versions/{version}/age-analysis", response_model=AgeAnalysisResponse)
def analyze_sketch_age(
    sketch_id: str,
    version: int = ApiPath(ge=1),
    image_data_url: Optional[str] = Body(default=None, embed=True),
) -> AgeAnalysisResponse:
    """Run exploratory age inference on exactly one persisted sketch version."""
    from app.services.age_estimation_service import (
        AgeEstimationError,
        BackboneInferenceError,
        HeadInferenceError,
        InvalidImageError,
    )

    image_bytes = image_storage.load_version(sketch_id, version)
    if image_bytes is None:
        if image_data_url is None:
            logger.warning(
                "Age analysis 404: sketch_id=%s version=%s image_data_url_supplied=false "
                "storage_root=%s process_id=%s",
                sketch_id,
                version,
                image_storage.root,
                os.getpid(),
            )
            raise HTTPException(status_code=404, detail="Saved sketch version not found.")
        image_bytes = decode_image_data_url(image_data_url)
        logger.warning(
            "Age analysis using client image fallback because sketch version is missing from storage: "
            "sketch_id=%s version=%s storage_root=%s process_id=%s",
            sketch_id,
            version,
            image_storage.root,
            os.getpid(),
        )
    estimator = get_age_estimation_service()
    if estimator is None:
        raise HTTPException(status_code=503, detail="Age analysis service is unavailable.")
    try:
        result = estimator.estimate_from_image_bytes(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail="Saved sketch image cannot be analyzed.") from exc
    except (BackboneInferenceError, HeadInferenceError, AgeEstimationError) as exc:
        logger.exception("Age analysis failed for sketch %s version %s", sketch_id, version)
        raise HTTPException(status_code=500, detail="Age analysis failed.") from exc
    persisted = image_storage.save_age_analysis(sketch_id, version, {
        "ml_estimated_age": result.estimated_age,
        "out_of_training_range": result.out_of_training_range,
        "model": result.model_identifier,
        "training_domain": result.training_evaluation_domain,
        "interpretation": result.application_interpretation,
        "backbone_hash": result.backbone_artifact_hash,
        "age_head_hash": result.age_head_artifact_hash,
    })
    if not persisted:
        logger.warning(
            "Age analysis succeeded but could not be persisted: sketch_id=%s version=%s "
            "storage_root=%s process_id=%s",
            sketch_id,
            version,
            image_storage.root,
            os.getpid(),
        )
    return AgeAnalysisResponse(
        sketch_id=sketch_id,
        version=version,
        status="available",
        ml_estimated_age=result.estimated_age,
        out_of_training_range=result.out_of_training_range,
        model=result.model_identifier,
        training_domain=result.training_evaluation_domain,
        interpretation=result.application_interpretation,
        artifact_sha256={
            "backbone": result.backbone_artifact_hash,
            "age_head": result.age_head_artifact_hash,
        },
    )
