"""Convert saved face-alignment 3D landmarks into previewable geometry JSON.

This is an approximate landmark-derived representation, not a reconstructed
3D face. It performs no ML inference and does not call the application or
OpenAI.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "backend" / "data" / "face_landmark_experiments" / "face_alignment"

REGIONS = {
    "jaw": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_lips": list(range(48, 60)),
    "inner_lips": list(range(60, 68)),
}

EDGE_REGIONS = [
    REGIONS["jaw"],
    REGIONS["left_eyebrow"],
    REGIONS["right_eyebrow"],
    REGIONS["nose"],
    REGIONS["left_eye"] + [REGIONS["left_eye"][0]],
    REGIONS["right_eye"] + [REGIONS["right_eye"][0]],
    REGIONS["outer_lips"] + [REGIONS["outer_lips"][0]],
    REGIONS["inner_lips"] + [REGIONS["inner_lips"][0]],
]


def edge_list() -> list[list[int]]:
    edges: list[list[int]] = []
    for region in EDGE_REGIONS:
        edges.extend([[a, b] for a, b in zip(region, region[1:])])
    return edges


def normalize(raw_landmarks: list[dict[str, Any]]) -> tuple[list[list[float]], dict[str, Any]]:
    if len(raw_landmarks) != 68:
        raise ValueError(f"Expected 68 landmarks, received {len(raw_landmarks)}")
    eye_points = raw_landmarks[36:48]
    center_x = sum(point["x"] for point in eye_points) / len(eye_points)
    center_y = sum(point["y"] for point in eye_points) / len(eye_points)
    center_z = sum(point["z"] for point in eye_points) / len(eye_points)
    jaw_width = math.dist(
        (raw_landmarks[0]["x"], raw_landmarks[0]["y"]),
        (raw_landmarks[16]["x"], raw_landmarks[16]["y"]),
    )
    if jaw_width == 0:
        raise ValueError("Cannot normalize landmarks with zero jaw width.")

    vertices = [
        [
            round((point["x"] - center_x) / jaw_width, 6),
            round(-(point["y"] - center_y) / jaw_width, 6),
            round((point["z"] - center_z) / jaw_width, 6),
        ]
        for point in raw_landmarks
    ]
    normalization = {
        "center": "mean of eye-region landmarks 36-47",
        "scale": "distance between jaw landmarks 0 and 16",
        "axes": "x right, y up (image y inverted), z preserved from 3D-FAN relative coordinate orientation",
        "z_is_relative": True,
    }
    return vertices, normalization


def preview(geometry: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices = geometry["vertices"]
    edges = geometry["edges"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, angle in (("front", 0), ("45deg", 45), ("side", 90)):
        radians = math.radians(angle)
        points = []
        for x, y, z in vertices:
            points.append([x * math.cos(radians) + z * math.sin(radians), y, -x * math.sin(radians) + z * math.cos(radians)])
        figure = plt.figure(figsize=(6, 7))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter([p[0] for p in points], [p[2] for p in points], [p[1] for p in points], s=10, c="#d62828")
        for start, end in edges:
            axis.plot(
                [points[start][0], points[end][0]],
                [points[start][2], points[end][2]],
                [points[start][1], points[end][1]],
                color="#2a9d8f",
                linewidth=0.8,
            )
        axis.set_title(f"Approximate landmark geometry — {label}")
        axis.set_xlabel("x")
        axis.set_ylabel("depth")
        axis.set_zlabel("y")
        axis.set_box_aspect((1, 0.65, 1.25))
        axis.view_init(elev=8, azim=-90)
        figure.tight_layout()
        figure.savefig(output_dir / f"{label}.png", dpi=160)
        plt.close(figure)


def convert(input_path: Path, output_dir: Path) -> None:
    metadata = json.loads(input_path.read_text(encoding="utf-8"))
    vertices, normalization = normalize(metadata["landmarks"])
    geometry = {
        "source": "face-alignment-3d-fan",
        "approximate": True,
        "landmark_count": len(vertices),
        "coordinate_system": {
            "centered": True,
            "normalized": True,
            **normalization,
        },
        "vertices": vertices,
        "regions": REGIONS,
        "edges": edge_list(),
        "faces": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
    preview(geometry, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="*", type=Path, help="landmarks.json path(s); defaults to all saved face-alignment outputs")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    inputs = args.input or sorted(args.input_root.glob("*/landmarks.json"))
    if not inputs:
        raise SystemExit(f"No landmarks.json files found under {args.input_root}.")
    for input_path in inputs:
        input_path = input_path.resolve()
        output_root = args.output_root.resolve() if args.output_root else input_path.parent
        convert(input_path, output_root)
        print(f"{input_path.name} -> {output_root / 'geometry.json'}")


if __name__ == "__main__":
    main()
