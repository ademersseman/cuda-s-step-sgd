# CUDA s-step SGD baseline benchmark

The canonical baseline is Slurm job `59460981`, run on 2026-10-06 on one
NVIDIA A100-SXM4-80GB. The current source was rebuilt in Release mode with
CUDA 13.2.78 and GCC 14.3.0 immediately before testing, and both project tests
passed.

## Configuration

- Dataset: 98,304 training and 16,384 test samples with 10,000 features
- Batch size `b=256`; `s` in `{32, 128, 384}`
- Leverage-sampled features `l=64`; rank 32; 512 leverage rows; seed 0
- 120 s-step block iterations; learning rate 0.001
- Current sampling: leverage probabilities computed once, then a distinct
  feature sample used by every block
- One warmup followed by three interleaved measured trials per configuration

## Baseline results

Training time is the median of the internal GPU training-loop timer. The
current logs confirm `feature_sketch_count: 120`.

| s | Training median [range] | Wall median [range] | Test loss | Test accuracy |
|---:|---:|---:|---:|---:|
| 32 | 58.077 ms [58.050, 58.408] | 3.59 s [3.58, 3.62] | 0.58886027 | 74.865723% |
| 128 | 211.854 ms [211.530, 212.365] | 3.70 s [3.67, 3.90] | 0.51036823 | 76.257324% |
| 384 | 601.478 ms [601.242, 602.050] | 4.19 s [4.16, 4.25] | 0.47653022 | 77.130127% |

The avoided full Gram sizes are 0.25, 4, and 36 GiB for `s=32`, `128`, and
`384`. Median peak host RSS is approximately 4.12 GiB.

Full methodology, control measurements, hashes, and artifact names are in
`results/per_block_resampling_20261006/BENCHMARK.md`.

## Progressive `l` comparison

Slurm job `59462499` compared the baseline's fixed `l=64` binary against the
current `l={8,16,32,64,64,...}` implementation on the same A100-SXM4-80GB,
dataset, and training parameters. Medians are from three alternating trials.

| s | Fixed `l=64` | Progressive `l` | Runtime change | Fixed accuracy | Progressive accuracy |
|---:|---:|---:|---:|---:|---:|
| 32 | 58.749 ms | 62.628 ms | +6.60% | 74.865723% | 74.865723% |
| 128 | 212.250 ms | 215.458 ms | +1.51% | 76.257324% | 76.263428% |
| 384 | 601.842 ms | 601.312 ms | -0.09% | 77.130127% | 77.130127% |

Only three of 120 blocks use fewer than 64 features, reducing total
`l`-dependent work by 1.77%. That saving is too small to offset the overhead of
using four GEMM shapes in the shorter cases. Test predictions are unchanged at
`s=32` and `s=384`; `s=128` changes by one correct sample. Detailed ranges,
losses, hashes, and raw artifacts are in
`results/adaptive_l_20261006/BENCHMARK.md`.

## Thirty-iteration progressive `l`

Slurm job `59462885` tested the revised schedule with 30 blocks each at
`l={8,16,32,64}` and prewarmed each recurrence shape before the internal
training timer. It used the same A100-SXM4-80GB, dataset, parameters, and fixed
`l=64` control as the baseline. Medians are from three alternating trials.

| s | Fixed `l=64` | Progressive `l` | Reduction | Fixed accuracy | Progressive accuracy |
|---:|---:|---:|---:|---:|---:|
| 32 | 58.324 ms | 52.049 ms | 10.76% | 74.865723% | 74.865723% |
| 128 | 212.100 ms | 200.070 ms | 5.67% | 76.257324% | 76.251221% |
| 384 | 601.323 ms | 576.964 ms | 4.05% | 77.130127% | 77.124023% |

The progressive schedule reduces `l`-dependent work by 53.125% and wins on
GPU loop time for every `s`. Test accuracy is unchanged at `s=32` and changes
by one sample at each larger `s`; test loss is slightly lower in every case.
Process wall time remains effectively flat because dataset loading, setup, and
the explicit prewarm dominate the saved loop time. Full trial ranges, wall
times, losses, hashes, and raw artifacts are in
`results/progressive_l_30_20261006/BENCHMARK.md`.

## Stochastic residual-correction controller

Slurm job `59487993` compared fixed `l=64`, the 30-block progressive schedule,
and the stochastic residual-correction controller on the same A100-SXM4-80GB,
dataset, and training parameters. Medians are from three trials run in rotating
method order.

| s | Fixed `l=64` | Progressive `l` | Stochastic `l` | Stochastic vs fixed | Fixed accuracy | Progressive accuracy | Stochastic accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 58.967 ms | 51.876 ms | 52.938 ms | 10.22% faster | 74.865723% | 74.865723% | 74.865723% |
| 128 | 211.885 ms | 200.301 ms | 198.467 ms | 6.33% faster | 76.257324% | 76.251221% | 76.257324% |
| 384 | 601.426 ms | 576.740 ms | 575.089 ms | 4.38% faster | 77.130127% | 77.124023% | 77.142334% |

The stochastic controller selected average widths of 12.267, 16.600, and
19.733 for `s=32`, `128`, and `384`. It is 2.05% slower than progressive at
`s=32`, but 0.92% and 0.29% faster at the two larger `s` values. It matches the
fixed baseline's held-out accuracy at `s=32` and `s=128` and gains two correct
predictions out of 16,384 at `s=384`. Full trial ranges, losses, wall times,
hashes, and raw artifacts are in
`results/stochastic_controller_20261007/BENCHMARK.md`.
