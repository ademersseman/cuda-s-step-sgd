"""Check LIBSVM width, binary label encodings, and adaptive statistics on padded blocks."""

import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def check_wide_libsvm(trainer, root):
    src = root / "wide.libsvm"
    weights = root / "wide.f32"
    src.write_text("+1 1:2 300:1\n-1 301:2 2000:4\n")
    subprocess.run([trainer, "--dataset", str(src), "--batch-size", "2", "--s", "1",
                    "--n-iters", "1", "--eta", "1", "--weights-out", str(weights)], check=True)
    # At zero weights a single mean-gradient step gives w = (x_pos - x_neg) / 4.
    expected = [0.0] * 2000
    expected[0], expected[299], expected[300], expected[1999] = 0.5, 0.25, -0.5, -1.0
    payload = weights.read_bytes()
    assert len(payload) == 4 * len(expected), f"expected 2000 float32 weights, got {len(payload) // 4}"
    assert list(struct.unpack("=2000f", payload)) == expected


def main():
    trainer = sys.argv[1]
    features = (1, -1, 1, -1, 1)
    signed_labels = (1, -1, 1, -1, 1)
    # After signing, all five true feature rows equal one. At zero weights,
    # each preceding real row contributes (eta / b) / 2 to the margin.
    residuals = [1 / (1 + math.exp(0.1 * (i // 2 * 2))) for i in range(5)]
    expected_ratio = math.sqrt(sum((r - 0.5) ** 2 for r in residuals) / 1.25)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "data"
        weights = root / "weights"
        for mode in ("cublas", "fused", "factor"):
            command = [trainer, "--dataset", str(src), "--gram-kernel", mode,
                       "--batch-size", "2", "--s", "4", "--n-iters", "8", "--eta", "0.4",
                       "--sampled-features", "2", "--trace-controller", "--weights-out", str(weights)]
            reference = None
            for binary in (False, True):
                for labels in (signed_labels, (1, 0, 1, 0, 1), (1, -1, 1, 0, 1)):
                    if binary:
                        src.write_bytes(b"SSTEPDS1" + struct.pack("=QQ5f5b", 5, 1, *features, *labels))
                    else:
                        src.write_text("".join(f"{y} 1:{x}\n" for x, y in zip(features, labels)))
                    run = subprocess.run(command, capture_output=True, text=True)
                    if "no CUDA-capable device" in run.stderr:
                        return 77
                    if -1 in labels and 0 in labels:
                        assert run.returncode != 0 and "binary labels" in run.stderr, run.stderr
                        continue
                    assert run.returncode == 0, run.stderr
                    trace = next(line for line in run.stdout.splitlines()
                                 if line.startswith("correction_ratio_history:"))
                    assert math.isclose(float(trace.split()[1]), expected_ratio, rel_tol=2e-4), trace
                    weight, = struct.unpack("=f", weights.read_bytes())
                    assert math.isfinite(weight) and weight > 0
                    if reference is None:
                        reference = weight
                    assert math.isclose(weight, reference, rel_tol=1e-5)
        check_wide_libsvm(trainer, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
