#!/usr/bin/env python3
"""Generate a dense planted-low-rank dataset shared by CUDA and PyTorch."""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

DATA_MAGIC = b"SSTEPDS1"
def generate_split(n, loadings, spectrum, planted_weights, residual_scale, generator, device):
    rank, d = loadings.shape
    latent = torch.randn(n, rank, device=device, dtype=torch.float32, generator=generator)
    features = (latent * spectrum).matmul(loadings)
    if residual_scale:
        features.add_(torch.randn(n, d, device=device, dtype=torch.float32, generator=generator), alpha=residual_scale)
    logits = (latent * spectrum).matmul(planted_weights)
    labels = torch.bernoulli(torch.sigmoid(logits), generator=generator).mul_(2).sub_(1).to(torch.int8)
    return features, labels


def write_dataset(path, features, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = features.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    y = labels.detach().cpu().contiguous().numpy().astype("i1", copy=False)
    with path.open("wb") as handle:
        handle.write(DATA_MAGIC)
        handle.write(struct.pack("<QQ", x.shape[0], x.shape[1]))
        x.tofile(handle)
        y.tofile(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=98_304)
    parser.add_argument("--n-test", type=int, default=16_384)
    parser.add_argument("--d", type=int, default=10_000)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--condition-number", type=float, default=100.0)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    basis = torch.randn(args.d, args.rank, device=device, dtype=torch.float32, generator=generator)
    loadings = torch.linalg.qr(basis, mode="reduced")[0].T.contiguous()
    spectrum = torch.exp(-0.5 * math.log(args.condition_number) * torch.linspace(0, 1, args.rank, device=device))
    planted_weights = torch.randn(args.rank, device=device, dtype=torch.float32, generator=generator)

    train_x, train_y = generate_split(args.n_train, loadings, spectrum, planted_weights, args.residual_scale, generator, device)
    test_x, test_y = generate_split(args.n_test, loadings, spectrum, planted_weights, args.residual_scale, generator, device)
    train_path = args.output_dir / "train.sstep"
    test_path = args.output_dir / "test.sstep"
    write_dataset(train_path, train_x, train_y)
    write_dataset(test_path, test_x, test_y)

    metadata = vars(args).copy()
    metadata["output_dir"] = str(args.output_dir)
    metadata.update(train_file=str(train_path), test_file=str(test_path))
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
