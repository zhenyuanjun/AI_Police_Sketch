"""CPU-only 3D landmark feasibility test using face-alignment.

Run from backend/.venv-3d. This experiment reads persisted PNGs only and does
not call OpenAI, the FastAPI application, or the production ResNet pipeline.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
IMAGE_ROOT = BACKEND_ROOT / "data" / "generated_sketches"
OUTPUT_ROOT = BACKEND_ROOT / "data" / "face_landmark_experiments" / "face_alignment"


def find_images(paths: Iterable[Path], process_all: bool) -> List[Path]:
    selected = list(paths) if paths else sorted(IMAGE_ROOT.glob("*.png"))
    return selected[:3] if process_all else selected[:1]


def process_image(image_path: Path, output_root: Path) -> None:
    try:
        import face_alignment
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "This experiment requires face-alignment, Pillow, and NumPy in backend/.venv-3d."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D,
        device="cpu",
        flip_input=False,
        compile=False,
    )
    start = time.perf_counter()
    predictions = detector.get_landmarks(image_array)
    inference_time_ms = round((time.perf_counter() - start) * 1000, 2)
    detected = bool(predictions)
    points = predictions[0] if detected else np.empty((0, 3), dtype=float)
    landmarks = [
        {"index": index, "x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
        for index, point in enumerate(points)
    ]

    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_image": str(image_path.relative_to(PROJECT_ROOT)),
        "detector": "face-alignment",
        "landmark_model": "3D-FAN",
        "device": "cpu",
        "face_detected": detected,
        "landmark_count": len(landmarks),
        "inference_time_ms": inference_time_ms,
        "landmarks": landmarks,
        "z_note": "z is a model-derived relative coordinate, not metric physical depth.",
    }
    (output_dir / "landmarks.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    debug_image = image.copy()
    if detected:
        draw = ImageDraw.Draw(debug_image)
        for point in points:
            x, y = round(float(point[0])), round(float(point[1]))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(220, 40, 40))
    debug_image.save(output_dir / "landmarks_debug.png")
    print(
        f"{image_path.name}: face_detected={str(detected).lower()} "
        f"landmark_count={len(landmarks)} inference_time_ms={inference_time_ms}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="*", type=Path, help="Optional persisted sketch PNG path.")
    parser.add_argument("--all", action="store_true", help="Process up to three persisted sketches.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    images = find_images(args.image, args.all)
    if not images:
        raise SystemExit(f"No persisted PNG files found under {IMAGE_ROOT}.")
    for image in images:
        image_path = image.resolve()
        if not image_path.exists():
            raise SystemExit(f"Image not found: {image_path}")
        process_image(image_path, args.output_root.resolve())


if __name__ == "__main__":
    main()
