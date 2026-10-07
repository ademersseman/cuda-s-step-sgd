# CUDA s-step SGD

Build with CMake:

```bash
module load cudatoolkit
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

## Factored leverage approximation

The executable estimates rank-32 column leverage probabilities from 512
training rows, independently samples columns for every s-step block, and
weights column `j` by `1/sqrt(l*p[j])`:

```bash
./build/s_step_sgd --batch-size 256 --s 352 --n-iters 100 \
  --dataset build/synthetic_data_e6.txt --sampled-features 64 --eta 0.001
```

`--sampled-features` sets the maximum `l`. Training starts from `l=8` and uses
the smoothed relative change between base and recurrence-corrected residuals to
stochastically select `l` from `8`, `16`, `32`, and `64` for later blocks.

Run `./build/s_step_sgd --help` for all named options. The original positional
arguments remain available for existing benchmark scripts.

This mode does not construct the `(batch_size*s)^2` Gram matrix. For weighted
sample matrix `B`, it computes each base residual in parallel and applies the
strictly lower block Gram through
`B_i * sum_{j<i}(B_j^T*r_j)`. A CUDA prefix kernel accumulates the sampled
gradients and strided-batched cuBLAS products apply both factors. The result is
the same sampled-Gram action apart from floating-point order, with linear
`O(batch_size*s*l)` sampled storage instead of quadratic Gram storage.

The `leverage` argument is retained for compatibility with existing benchmark
scripts. The full-Gram, uniform, and scoring paths have been removed. Sampling
is with replacement, so repeated columns remain valid separate contributions
to the sampled Gram.
