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

Visual version comparison reuses the production INT8 LiteRT ResNet50 backbone and compares its dequantized 2048-D pooled features with cosine similarity. The backbone uses the same frozen `ResNet50_Weights.IMAGENET1K_V2` weights and canonical preprocessing as the original FP32 comparison extractor. The quantization validation report measured mean cosine similarity 0.988 between FP32 and INT8 features on its 50-image validation sample. This avoids adding PyTorch and torchvision to the production comparison path while retaining the same feature representation up to the measured INT8 approximation.

The separate optional `FaceAnalysisService` still uses PyTorch and torchvision and remains lazy-loaded for the `/analyze` and `/benchmark` routes. Those routes return HTTP 503 when the optional dependencies are absent. To install them for those routes, use `pip install -r requirements-face-analysis.txt` in addition to the standard requirements. The experiment requirements and lock files remain unchanged.

## Locked artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| `resnet50_backbone_w8a8.tflite` | `c3bdfb8002a40fe4892a2179616ea80fa2a2437d5c6a4402c01d0d15b9f6d5a0` |
| `age_regression_head.pt` source checkpoint | `c9d757ee0dee4ce1aff96ec8526d001e3072e30ca27264991f0db0c4b4320fe8` |
| `age_regression_head.npz` production head | `e9616bc044ab5a3308c61fbc132b5a692157ef7ac6d0cbba2461800614b55548` |
