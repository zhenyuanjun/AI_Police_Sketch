# AI Police Sketch

AI Police Sketch is a semester-project research prototype for exploring how natural-language eyewitness descriptions can become iterative facial sketches. Phase 1 provides a complete local workflow with a mock generator; it does not use an external AI model or identify real people.

## Current MVP

- Choose between a witness description or a reference face image.
- Submit a witness description or a JPEG, PNG, or WebP reference image (maximum 10 MB) to the Next.js interface.
- Generate a versioned mock sketch through FastAPI.
- Refine the same sketch with natural-language instructions.
- Review and select prior versions in session history.
- Keep all data in memory; there are no accounts, analytics, or permanent storage.

## Architecture

`Browser → Next.js App Router → REST → FastAPI → SketchGenerationService → MockSketchGenerationService`

The API routes depend on the service interface rather than a particular model, so a future provider can replace the mock service without rewriting the UI or routes.

## Repository structure

```text
frontend/                 Next.js + React + TypeScript + Tailwind UI
  app/                    App Router page and global styles
  components/             Workspace UI components
  lib/api.ts              Typed API client
backend/                  FastAPI application
  app/main.py             Application and routes
  app/schemas/            Pydantic request/response models
  app/services/           Generator abstraction and mock provider
```

## Requirements

- Python 3.9+
- Node.js 18.17+
- npm

## Run the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

## Run the frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## Environment variables

`frontend/.env.example` contains `NEXT_PUBLIC_API_URL=http://localhost:8000`. Change it when the API is hosted elsewhere.

## API endpoints

- `GET /health` → `{ "status": "ok" }`
- `POST /api/sketches/generate` with `{ "description": "..." }` for Mode A.
- `POST /api/sketches/generate-from-reference` as multipart form data with a `file` field for Mode B.
- `POST /api/sketches/{id}/refine` with `{ "instruction": "..." }`

Reference uploads are validated on both client and server, read in memory for the mock request, and never persisted. Supported MIME types are `image/jpeg`, `image/png`, and `image/webp`; the maximum size is 10 MB.

## Limitations and responsible use

The current image is an illustrative SVG-style placeholder generated from mock state. No facial recognition, identity matching, suspect search, biometric identification, or production persistence is implemented. Generated sketches are interpretive visualizations and should not be treated as identification evidence or claims of forensic accuracy. Uploaded images are not permanently stored.

## Planned Phase 2

Evaluate candidate text-to-image or structured face-generation approaches, define a police-sketch style and attribute-preservation benchmark, add provider adapters behind the existing service interface, and measure refinement consistency with human evaluation and controlled prompt sets.
