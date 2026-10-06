"""Lightweight AGE-1B input pipeline smoke test; performs no training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50

from agedb_dataset import AGE1A_HASH, build_dataloaders, verify_metadata_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--check-resnet", action="store_true")
    args = parser.parse_args()
    digest = verify_metadata_hash()
    loaders = build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    first_batches = {}
    for name, loader in loaders.items():
        images, ages = next(iter(loader))
        assert images.shape == (args.batch_size, 3, 224, 224), (name, images.shape)
        assert images.dtype == torch.float32 and torch.isfinite(images).all()
        assert ages.shape == (args.batch_size,) and ages.dtype == torch.float32
        assert torch.all((ages >= 1) & (ages <= 101))
        first_batches[name] = (images, ages)

    resnet_result = "NOT RUN"
    if args.check_resnet:
        weights = ResNet50_Weights.DEFAULT
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        if not checkpoint.is_file():
            resnet_result = f"NOT RUN: pretrained checkpoint not cached at {checkpoint}; no download attempted"
        else:
            model = resnet50(weights=None)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.fc = torch.nn.Identity()
            model.eval()
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            with torch.no_grad():
                features = model(first_batches["validation"][0][:1].to(device))
            assert features.shape == (1, 2048)
            resnet_result = f"PASS: {list(features.shape)} on {device}; cached weights {checkpoint.name}"

    print(f"AGE-1A metadata hash verified: {digest} (expected {AGE1A_HASH})")
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"torch: {torch.__version__}")
    import torchvision
    import PIL
    print(f"torchvision: {torchvision.__version__}")
    print(f"Pillow: {PIL.__version__}")
    print(f"batch_size={args.batch_size}, num_workers={args.num_workers}")
    print(f"preprocessing: {ResNet50_Weights.DEFAULT.transforms()}")
    for name, (images, ages) in first_batches.items():
        print(f"{name}: images={list(images.shape)} {images.dtype}, ages={list(ages.shape)} {ages.dtype}")
    print(f"ResNet input compatibility: {resnet_result}")


if __name__ == "__main__":
    main()
