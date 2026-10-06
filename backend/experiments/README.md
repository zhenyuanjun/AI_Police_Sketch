# Face landmark feasibility experiment

Run `test_face_landmarks.py` from the backend environment after installing
MediaPipe there, without changing the existing PyTorch/torchvision versions.

The script expects the official MediaPipe Face Landmarker task bundle at:

```text
backend/data/models/face_landmarker.task
```

Official source: [MediaPipe Face Landmarker task documentation](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker).
The task bundle is a small model asset (typically a few megabytes; verify the
exact size after download). It is intentionally not included in Git.

Outputs are written under `backend/data/face_landmark_experiments/`, one
directory per source image, containing `landmarks.json` and
`landmarks_debug.png`. Landmark `z` values are model-relative coordinates, not
metric depth.

## AgeDB age-regression metadata (AGE-1A)

The dataset stays outside the repository. Set `AGEDB_ROOT` to the AgeDB
directory containing `images/`, then run:

```sh
AGEDB_ROOT=/Users/chenyuanjun/Downloads/AgeDB \
  python backend/experiments/age_regression/build_agedb_metadata.py
```

The script validates every JPG filename and writes only CSV/JSON metadata to
`backend/experiments/age_regression/output/`. It groups by the exact filename
identity and assigns whole identities deterministically (seed `5341`) to
70%/15%/15% train/validation/test splits. `metadata.sha256` fingerprints the
manifest, image split, identity split, and report. Generated output and common
local dataset/training artifact locations are ignored by Git.

### AGE-1B PyTorch input pipeline

Create the isolated Python 3.11 environment and install the pinned minimal
dependencies with:

```sh
python3.11 -m venv backend/.venv-age
backend/.venv-age/bin/python -m pip install -r backend/experiments/age_regression/requirements.txt
```

Run the no-training smoke test (optional cached ResNet50 feature check included):

```sh
AGEDB_ROOT=/Users/chenyuanjun/Downloads/AgeDB \
  backend/.venv-age/bin/python backend/experiments/age_regression/smoke_test.py \
  --batch-size 4 --num-workers 0 --check-resnet
```

`agedb_dataset.py` consumes the AGE-1A manifest and split files, verifies their
recorded SHA-256 and subject-disjoint membership, and uses
`ResNet50_Weights.DEFAULT.transforms()` for RGB preprocessing. The default
loader batch size is 8 and worker count is 0; both are configurable.

### AGE-2A FP32 model definition

The baseline model is `ResNet50AgeRegressor` in `age_model.py`: cached
ImageNet-pretrained ResNet50 features are frozen and kept in eval mode so
BatchNorm running statistics do not change; a seeded, trainable
`Linear(2048, 256) -> ReLU -> Linear(256, 1)` head predicts an unclamped age.
The output and age targets are one-dimensional float32 tensors. The initial
regression loss is `L1Loss`; this phase only checks one batch and gradients.

Run the architecture smoke check with:

```sh
AGEDB_ROOT=/Users/chenyuanjun/Downloads/AgeDB \
  backend/.venv-age/bin/python backend/experiments/age_regression/model_smoke_test.py
```

### AGE-2B training pipeline

The training entry point uses Adam (`1e-3`), `L1Loss`, batch size 32, training
seed `20261003`, and zero workers by default. It accepts explicit configuration
flags and writes a best-validation checkpoint, JSON history, and run summary
under the Git-ignored `output/runs/age2b_one_epoch/` directory. The sanity run
is restricted to one epoch and uses train/validation only:

```sh
AGEDB_ROOT=/Users/chenyuanjun/Downloads/AgeDB \
  backend/.venv-age/bin/python backend/experiments/age_regression/train_age_regressor.py \
  --epochs 1 --batch-size 32 --learning-rate 1e-3 --num-workers 0
```

The script probes basic MPS convolution support before selecting MPS; it falls
back to CPU if MPS is unavailable or the probe fails. Validation reports the
population standard deviation of per-image absolute errors.

### AGE-2D frozen feature cache

`cache_frozen_features.py` writes separate PyTorch `.pt` artifacts for the
AGE-1A train and validation splits under the Git-ignored
`age_regression/output/features/` directory. Each cache stores an FP32
`[N, 2048]` feature tensor, FP32 ages, ordered per-image metadata, and a
configuration record. The script performs direct-vs-cached parity checks on
three deterministic samples from each split. It creates no test Dataset,
DataLoader, or feature cache.

```sh
AGEDB_ROOT=/Users/chenyuanjun/Downloads/AgeDB \
  backend/.venv-age/bin/python backend/experiments/age_regression/cache_frozen_features.py \
  --batch-size 32 --num-workers 0
```
