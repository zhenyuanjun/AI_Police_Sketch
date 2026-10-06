"""One-image, CPU-only 3DDFA_V2 feasibility experiment.

This is intentionally outside the production FastAPI application. It reads an
already persisted sketch and writes only under face_landmark_experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 3DDFA_V2's legacy Cython NMS wrapper still references np.int. Keep this
# compatibility shim local to the experiment instead of modifying the vendor.
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]
if "long" not in np.__dict__:
    np.long = int  # type: ignore[attr-defined]


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "backend" / "experiments" / "vendor" / "3DDFA_V2"
DEFAULT_IMAGE = ROOT / "backend" / "data" / "generated_sketches" / (
    "49324c0a-f3c9-4321-8321-644e56e64766.png"
)
DEFAULT_OUTPUT = ROOT / "backend" / "data" / "face_landmark_experiments" / "3ddfa" / (
    "49324c0a-f3c9-4321-8321-644e56e64766"
)

sys.path.insert(0, str(VENDOR))

from FaceBoxes import FaceBoxes  # noqa: E402
from TDDFA import TDDFA  # noqa: E402
from utils.serialization import ser_to_obj  # noqa: E402


def landmark_fallback_box(image_path: Path) -> list[list[float]] | None:
    """Return a FaceBoxes-compatible box from the existing 68-point result."""
    landmark_path = (
        ROOT / "backend" / "data" / "face_landmark_experiments" / "face_alignment" /
        image_path.stem / "landmarks.json"
    )
    if not landmark_path.exists():
        return None
    data = json.loads(landmark_path.read_text())
    points = data.get("landmarks", [])
    if not points:
        return None
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return [[min(xs), min(ys), max(xs), max(ys), 1.0]]


def save_preview(vertices: np.ndarray, triangles: np.ndarray, output: Path, angle: float) -> None:
    """Save a simple shaded projection for qualitative geometry inspection."""
    points = vertices.T.astype(float)
    center = points.mean(axis=0)
    points -= center
    radians = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(radians), 0.0, np.sin(radians)], [0.0, 1.0, 0.0],
         [-np.sin(radians), 0.0, np.cos(radians)]]
    )
    points = points @ rotation.T
    faces = [points[triangle] for triangle in triangles.astype(int)]

    fig = plt.figure(figsize=(6, 6), dpi=140)
    axis = fig.add_subplot(111, projection="3d")
    axis.add_collection3d(
        Poly3DCollection(faces, alpha=0.78, linewidths=0.05, edgecolors="0.25")
    )
    axis.set_axis_off()
    limits = np.max(np.abs(points), axis=0).max() or 1.0
    axis.set_xlim(-limits, limits)
    axis.set_ylim(-limits, limits)
    axis.set_zlim(-limits, limits)
    # 3DDFA uses x/y image coordinates and z as relative depth. Looking down
    # the z axis gives the expected front-facing projection.
    axis.view_init(elev=90, azim=-90)
    fig.tight_layout(pad=0)
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    print(f"source: {image_path}")
    print("device: cpu")
    detector = FaceBoxes()
    boxes = detector(image)
    detection_path = "FaceBoxes"
    if not boxes:
        boxes = landmark_fallback_box(image_path)
        detection_path = "existing face-alignment 68-landmark bounding box"
    if not boxes:
        raise RuntimeError("No face detected and no existing landmark fallback is available")
    print(f"detection_path: {detection_path}")
    print(f"face_count: {len(boxes)}")

    config = VENDOR / "configs" / "mb1_120x120.yml"
    import yaml

    cfg = yaml.safe_load(config.read_text())
    start = time.perf_counter()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    params, roi_boxes = tddfa(image, boxes)
    vertices_list = tddfa.recon_vers(params, roi_boxes, dense_flag=True)
    inference_ms = (time.perf_counter() - start) * 1000.0
    vertices = vertices_list[0]
    triangles = np.asarray(tddfa.tri)

    geometry = {
        "source": "3DDFA_V2",
        "approximate": True,
        "source_image": str(image_path),
        "detector": detection_path,
        "device": "cpu",
        "inference_time_ms": round(inference_ms, 2),
        "vertex_count": int(vertices.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "z_is_metric": False,
        "vertices": vertices.T.astype(float).round(6).tolist(),
        "triangles": triangles.astype(int).tolist(),
    }
    (output / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    (output / "parameters.json").write_text(
        json.dumps({
            "source": "3DDFA_V2",
            "device": "cpu",
            "inference_time_ms": round(inference_ms, 2),
            "roi_boxes": np.asarray(roi_boxes).astype(float).tolist(),
            "parameters": [np.asarray(param).astype(float).tolist() for param in params],
        }, indent=2) + "\n"
    )
    ser_to_obj(image, vertices_list, triangles, height=image.shape[0], wfp=str(output / "face.obj"))
    save_preview(vertices, triangles, output / "front.png", 0.0)
    save_preview(vertices, triangles, output / "45deg.png", 45.0)
    save_preview(vertices, triangles, output / "side.png", 90.0)
    print(f"vertices: {vertices.shape[1]}")
    print(f"triangles: {triangles.shape[0]}")
    print(f"inference_time_ms: {inference_ms:.2f}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
