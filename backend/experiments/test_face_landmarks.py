"""Run a local MediaPipe Face Landmarker feasibility test on saved sketches.

This experiment does not call the API, generate meshes, or modify source images.
Install MediaPipe in the intended backend environment and provide the official
face_landmarker.task asset before running this script.
"""

import argparse
import json
import platform
from pathlib import Path
import sys
from typing import Iterable, List, Union


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DEFAULT_IMAGE_ROOT = BACKEND_ROOT / "data" / "generated_sketches"
DEFAULT_MODEL_PATH = BACKEND_ROOT / "data" / "models" / "face_landmarker.task"
DEFAULT_OUTPUT_ROOT = BACKEND_ROOT / "data" / "face_landmark_experiments"


def image_paths(root: Path, limit: int) -> List[Path]:
    return sorted(root.glob("*.png"))[:limit]


def normalized_landmarks(face_landmarks: Iterable[object]) -> list[dict[str, Union[float, int]]]:
    return [
        {
            "index": index,
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
        }
        for index, landmark in enumerate(face_landmarks)
    ]


def run(image_path: Path, model_path: Path, output_root: Path) -> dict[str, object]:
    try:
        import mediapipe as mp
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe and Pillow are required. Install MediaPipe in the intended "
            "backend environment without changing the existing PyTorch versions."
        ) from exc

    delegate_type = getattr(mp.tasks.BaseOptions, "Delegate", None)
    cpu_delegate = getattr(delegate_type, "CPU", None)
    if cpu_delegate is None:
        raise RuntimeError(
            "This MediaPipe build does not expose BaseOptions.Delegate.CPU; "
            "the experiment will not guess a delegate."
        )

    print(
        "runtime: "
        f"platform={platform.system()} "
        f"release={platform.release()} "
        f"machine={platform.machine()} "
        f"python={sys.version.split()[0]} "
        f"mediapipe={getattr(mp, '__version__', 'unknown')} "
        "requested_delegate=CPU"
    )

    image = Image.open(image_path).convert("RGB")
    image_array = __import__("numpy").asarray(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path), delegate=cpu_delegate)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    faces = result.face_landmarks
    landmarks = normalized_landmarks(faces[0]) if faces else []
    output_dir = output_root / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_image": str(image_path.relative_to(PROJECT_ROOT)),
        "face_detected": bool(faces),
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
    }
    (output_dir / "landmarks.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    debug_image = image.copy()
    draw = ImageDraw.Draw(debug_image)
    for landmark in landmarks:
        x = round(landmark["x"] * debug_image.width)
        y = round(landmark["y"] * debug_image.height)
        radius = 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(220, 40, 40))
    debug_image.save(output_dir / "landmarks_debug.png")
    print(f"{image_path.name}: face_detected={bool(faces)} landmark_count={len(landmarks)}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="*", type=Path, help="Optional sketch PNG path(s); defaults to up to three saved PNGs.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the official face_landmarker.task asset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Model asset not found: {args.model}\n"
            "Download the official MediaPipe Face Landmarker task asset separately "
            "and place it at backend/data/models/face_landmarker.task."
        )
    paths = args.image[:3] if args.image else image_paths(DEFAULT_IMAGE_ROOT, 3)
    if not paths:
        raise SystemExit("No persisted sketch PNG files were found.")
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Image not found: {path}")
        run(path.resolve(), args.model.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
