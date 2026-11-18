"""Check binary label encodings and adaptive statistics on a partially padded block."""

import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
