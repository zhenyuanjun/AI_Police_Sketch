# Production age-estimation runtime

The standard `backend/requirements.txt` contains the production age path and does not install PyTorch or torchvision. `AgeEstimationService` uses NumPy, Pillow, the finalized LiteRT INT8 backbone, and `data/models/age_regression_head.npz`.

The NumPy head is the exact learned inference graph from the finalized checkpoint:

```text
Linear(2048, 256) -> ReLU -> Linear(256, 1)
```

The offline converter reads the existing `age_regression_head.pt` checkpoint, checks its locked SHA-256 and tensor architecture, then writes only the four FP32 weight/bias arrays. It is not run at server startup. The service verifies the LiteRT and NPZ artifact hashes and checks NPZ member names, shapes, dtypes, and finite values.

To reproduce the export from the experiment environment:

```sh
cd backend
.venv-age-int8/bin/python experiments/age_regression/export_age_head_numpy.py
```

To verify numerical parity against the source checkpoint using representative random features and all locally persisted generated PNGs:

```sh
cd backend
.venv-age-int8/bin/python experiments/age_regression/validate_numpy_age_head.py
```

The optional `FaceAnalysisService` still uses PyTorch and torchvision and remains lazy-loaded. Its routes return HTTP 503 with an install instruction when those optional dependencies are absent. To install them, use `pip install -r requirements-face-analysis.txt` in addition to the standard requirements. The experiment requirements and lock files remain unchanged.

## Locked artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `resnet50_backbone_w8a8.tflite` | `c3bdfb8002a40fe4892a2179616ea80fa2a2437d5c6a4402c01d0d15b9f6d5a0` |
| `age_regression_head.pt` source checkpoint | `c9d757ee0dee4ce1aff96ec8526d001e3072e30ca27264991f0db0c4b4320fe8` |
| `age_regression_head.npz` production head | `e9616bc044ab5a3308c61fbc132b5a692157ef7ac6d0cbba2461800614b55548` |
