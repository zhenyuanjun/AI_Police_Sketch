from io import BytesIO
from statistics import mean, median, pstdev
import time
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class FaceAnalysisService:
    """Local baseline visual feature extractor for generated sketch images."""

    MODEL_NAME = "resnet50"

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        weights = ResNet50_Weights.DEFAULT
        self._model = resnet50(weights=weights)
        self._model.fc = torch.nn.Identity()
        self._model.eval()
        self._preprocess = weights.transforms()

    def _prepare_tensor(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (OSError, ValueError) as exc:
            raise ValueError("The generated sketch is not a valid image.") from exc
        return self._preprocess(image).unsqueeze(0).to(self.device)

    def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        tensor = self._prepare_tensor(image_bytes)
        start = time.perf_counter()
        with torch.inference_mode():
            embedding = self.extract_embedding(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "model": self.MODEL_NAME,
            "embedding_dimension": int(embedding.shape[-1]),
            "inference_time_ms": round(elapsed_ms, 2),
            "device": self.device.type,
        }

    def extract_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self._model(tensor)

    def compare(self, image_a: bytes, image_b: bytes) -> dict[str, Any]:
        tensor_a = self._prepare_tensor(image_a)
        tensor_b = self._prepare_tensor(image_b)
        with torch.inference_mode():
            embedding_a = self.extract_embedding(tensor_a)
            embedding_b = self.extract_embedding(tensor_b)
            similarity = float(F.cosine_similarity(embedding_a, embedding_b).item())
        return {
            "model": self.MODEL_NAME,
            "embedding_dimension": int(embedding_a.shape[-1]),
            "cosine_similarity": round(similarity, 4),
            "cosine_distance": round(1 - similarity, 4),
        }

    def benchmark(self, image_bytes: bytes) -> dict[str, Any]:
        tensor = self._prepare_tensor(image_bytes)
        warmup_runs = 5
        measured_runs = 30

        with torch.inference_mode():
            for _ in range(warmup_runs):
                self._model(tensor)

            latencies_ms = []
            for _ in range(measured_runs):
                start_ns = time.perf_counter_ns()
                self._model(tensor)
                latencies_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000)

        return {
            "model": self.MODEL_NAME,
            "runtime": "pytorch",
            "device": self.device.type,
            "embedding_dimension": 2048,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mean_latency_ms": round(mean(latencies_ms), 2),
            "median_latency_ms": round(median(latencies_ms), 2),
            "min_latency_ms": round(min(latencies_ms), 2),
            "max_latency_ms": round(max(latencies_ms), 2),
            "std_latency_ms": round(pstdev(latencies_ms), 2),
        }
