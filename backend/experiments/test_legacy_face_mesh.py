"""Feasibility test for MediaPipe's legacy Face Mesh API.

Run this script from the isolated backend/.venv-3d environment. It only reads
persisted PNGs and writes experimental landmark outputs; it does not call the
backend, OpenAI, PyTorch, or torchvision.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
IMAGE_ROOT = BACKEND_ROOT / "data" / "generated_sketches"
OUTPUT_ROOT = BACKEND_ROOT / "data" / "face_landmark_experiments" / "legacy"


def find_images(paths: Iterable[Path], process_all: bool) -> List[Path]:
    selected = list(paths) if paths else sorted(IMAGE_ROOT.glob("*.png"))
    return selected[:3] if process_all else selected[:1]


def landmarks_as_json(face_landmarks: Iterable[object]) -> list[dict[str, float | int]]:
    return [
        {
            "index": index,
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
        }
        for index, landmark in enumerate(face_landmarks)
    ]


def draw_connections(image, landmarks: list[dict[str, float | int]], connections: Iterable[tuple[int, int]]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    points = [(round(item["x"] * width), round(item["y"] * height)) for item in landmarks]
    for start, end in connections:
        if start < len(points) and end < len(points):
            draw.line((points[start], points[end]), fill=(30, 180, 40), width=1)
    for x, y in points:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(220, 40, 40))


def process_image(image_path: Path, output_root: Path) -> None:
    try:
        import mediapipe as mp
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "This experiment requires mediapipe 0.10.21 and Pillow in backend/.venv-3d."
        ) from exc

    image = Image.open(image_path).convert("RGB")
    start = time.perf_counter()
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(__import__("numpy").asarray(image))
    inference_time_ms = round((time.perf_counter() - start) * 1000, 2)

    detected = bool(result.multi_face_landmarks)
    landmarks = landmarks_as_json(result.multi_face_landmarks[0].landmark) if detected else []
    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_image": str(image_path.relative_to(PROJECT_ROOT)),
        "face_detected": detected,
        "landmark_count": len(landmarks),
        "inference_time_ms": inference_time_ms,
        "api": "mediapipe.solutions.face_mesh.FaceMesh",
        "landmarks": landmarks,
        "z_note": "MediaPipe z is relative/model-derived and is not metric physical depth.",
    }
    (output_dir / "landmarks.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    debug_image = image.copy()
    if detected:
        draw_connections(debug_image, landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
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
