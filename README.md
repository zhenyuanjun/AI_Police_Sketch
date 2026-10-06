# AI Police Sketch

AI Police Sketch is an interactive research application combining generative AI with a machine-learning age-analysis pipeline. The Next.js frontend connects to a FastAPI backend for OpenAI police-sketch generation, text-based refinement, sketch history/version management, and saved-sketch restoration. Witness age is entered as a structured value and can be compared with ML analysis for each sketch version.

## Application pipeline

`Witness Description + Witness Age → AI Police Sketch → INT8 ResNet50 → 2048-D Features → FP32 Age Regression Head → ML Estimated Age → Visual Age Difference → Text-based Refinement → ML Re-analysis`

**ML Estimated Age is an exploratory visual signal and should not be interpreted as a forensic or true-age estimate.** The age model was trained and evaluated on AgeDB photographs, while the application analyzes generated police sketches. This photo-to-sketch domain gap means AgeDB test accuracy does not establish accuracy on generated sketches.

## Final results

The ResNet50 age model uses a frozen pretrained backbone with a `2048 → 256 → 1` ReLU regression head. On the AgeDB photo test set, FP32 MAE was 9.6941 years and INT8 MAE was 9.7239 years (+0.31%). The INT8 backbone measured 8.33 ms versus 33.97 ms for FP32 (4.08× speedup; 75.47% lower backbone latency), with model size reduced from 89.67 MiB to 23.17 MiB (74.16%). Latency is backbone invoke-only at batch size 1.

In the final 12-trial police-sketch experiment (12 base sketches, four at each witness age of 20, 40, and 60; six Younger and six Older refinements), Pearson correlation between witness age and base ML Estimated Age was 0.963 and Spearman correlation was 0.946. Mean absolute visual age difference was 9.508 years. Refinement moved the ML signal in the requested direction in 5/6 Younger trials and 5/6 Older trials (10/12 overall, 83.33%); mean changes were −8.111 and +9.862 years, respectively. These are exploratory results on a small formal experiment, not evidence of forensic validity or true-age estimation.

Lightweight result tables, JSON summaries, figures, and experiment scripts are in `backend/experiments/age_regression/`. Raw AgeDB images and local model/cache artifacts are excluded from Git.

## Run locally

Requirements: Python 3.9+, Node.js 18.17+, and npm. Configure `backend/.env` with `SKETCH_PROVIDER=openai` and `OPENAI_API_KEY` for OpenAI generation. The model artifacts required for age analysis are local and ignored by Git; without them, sketch generation remains available but age analysis is unavailable.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The API runs at `http://localhost:8000` and the web app at `http://localhost:3000`.

## Future work

3D reconstruction and multi-view generation are future work; they are not part of the current final implementation.
