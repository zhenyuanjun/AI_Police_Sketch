import os
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.sketches import GenerateSketchRequest, RefineSketchRequest, SketchResponse
from app.services.sketch_generation import OpenAISketchGenerationService, MockSketchGenerationService, SketchGenerationService

app = FastAPI(title="AI Police Sketch API", version="0.1.0")
origins = [os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
def create_generation_service() -> SketchGenerationService:
    provider = os.getenv("SKETCH_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockSketchGenerationService()
    if provider == "openai":
        return OpenAISketchGenerationService()
    raise RuntimeError("SKETCH_PROVIDER must be either 'mock' or 'openai'.")


service: SketchGenerationService = create_generation_service()
MAX_REFERENCE_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_REFERENCE_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def response_from_sketch(sketch):
    status = "openai" if sketch.image_url else "mock"
    return SketchResponse(id=sketch.id, version=sketch.version, description=sketch.description, refinement_instruction=sketch.refinement_instruction, image_url=sketch.image_url, status=status)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/sketches/generate", response_model=SketchResponse)
def generate_sketch(request: GenerateSketchRequest) -> SketchResponse:
    try:
        return response_from_sketch(service.generate(request.description))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/sketches/generate-from-reference", response_model=SketchResponse)
async def generate_from_reference(file: UploadFile = File(...)) -> SketchResponse:
    if file.content_type not in ALLOWED_REFERENCE_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Reference image must be JPEG, PNG, or WebP.")
    image_bytes = await file.read(MAX_REFERENCE_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_REFERENCE_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Reference image must be 10 MB or smaller.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Reference image cannot be empty.")
    try:
        return response_from_sketch(service.generate_from_reference(file.filename or "reference-image", file.content_type, image_bytes))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@app.post("/api/sketches/{sketch_id}/refine", response_model=SketchResponse)
def refine_sketch(sketch_id: str, request: RefineSketchRequest) -> SketchResponse:
    try:
        sketch = service.refine(sketch_id, request.instruction)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    if sketch is None:
        raise HTTPException(status_code=404, detail="Sketch not found")
    return response_from_sketch(sketch)
