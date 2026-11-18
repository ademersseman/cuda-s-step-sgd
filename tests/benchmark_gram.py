"""Compare fixed-width, end-to-end CUDA training times and final weights."""

import argparse
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile

import numpy as np


def run(args, mode, width, weights):
    command = [
        str(args.trainer.resolve()), "--dataset", str(args.dataset),
        "--gram-kernel", mode, "--fixed-l", str(width),
        "--batch-size", str(args.batch_size), "--s", str(args.s),
        "--n-iters", str(args.iters), "--eta", str(args.eta),
        "--sketch-seed", "0", "--weights-out", str(weights),
    ]
    stdout = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout
    time_ms = float(re.search(r"^Training Time: (\S+) ms$", stdout, re.M)[1])
    loss = float(re.search(r"^Final Objective: (\S+)$", stdout, re.M)[1])
    model = np.fromfile(weights, dtype=np.float32)
    if model.size == 0 or not np.isfinite(model).all() or not np.isfinite(loss):
        raise ValueError(f"Invalid training output for {mode}, width {width}")
    return time_ms, loss, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--s", type=int, default=128)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--baseline", choices=("cublas", "fused"), default="cublas")
    parser.add_argument("--candidate", choices=("fused", "factor"), default="fused")
    args = parser.parse_args()
    if args.s < 2:
        parser.error("--s must be at least 2 to exercise a Gram recurrence")
    if args.baseline == args.candidate:
        parser.error("choose distinct kernels")
    modes = (args.baseline, args.candidate)
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    print(f"width,{args.baseline}_median_ms,{args.candidate}_median_ms,speedup,max_abs_weight_diff,max_abs_loss_diff")
    with tempfile.TemporaryDirectory() as directory:
        for width in (2, 16, 64):
            times = {mode: [] for mode in modes}
            weight_diff = loss_diff = 0.0
            for pair in range(args.pairs + 1):
                results = {}
                for mode in (modes if pair % 2 == 0 else reversed(modes)):
                    elapsed, loss, model = run(args, mode, width, Path(directory) / mode)
                    results[mode] = (loss, model)
                    if pair > 0:
                        times[mode].append(elapsed)
                reference_loss, reference = results[args.baseline]
                candidate_loss, candidate = results[args.candidate]
                np.testing.assert_allclose(candidate, reference, atol=1e-5, rtol=1e-4)
                np.testing.assert_allclose(candidate_loss, reference_loss, atol=1e-6, rtol=1e-5)
                weight_diff = max(weight_diff, float(np.max(np.abs(candidate - reference))))
                loss_diff = max(loss_diff, abs(candidate_loss - reference_loss))
            baseline, candidate_time = median(times[args.baseline]), median(times[args.candidate])
            print(f"{width},{baseline:g},{candidate_time:g},{baseline/candidate_time:.3f},"
                  f"{weight_diff:.3g},{loss_diff:.3g}", flush=True)


if __name__ == "__main__":
    main()
