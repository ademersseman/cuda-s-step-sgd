# CUDA s-step SGD

Build with CMake:

```bash
module load cudatoolkit
module load gcc-native/14
cmake -S . -B builds/release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)"
cmake --build builds/release -j
```

## Factored leverage approximation

The executable estimates rank-32 column leverage probabilities from 512
training rows, independently samples columns for every s-step block, and
weights sampled column `j` by `1/sqrt(l*p[j])`:

```bash
./builds/release/s_step_sgd --batch-size 256 --s 352 --n-iters 100 \
  --dataset path/to/train.sstep --sampled-features 64 --eta 0.001
```

`--sampled-features` sets the maximum `l`. Adaptive training starts from `l=64`,
doubles the width when the smoothed residual-correction ratio exceeds `0.20`,
and halves it after three consecutive updates below `0.05`.

The loader accepts binary labels encoded as either `{-1,+1}` or `{0,1}` in
LIBSVM and SSTEPDS1 files. Padding rows are excluded from the adaptive
correction ratio.

Run `./builds/release/s_step_sgd --help` for all named options. The original positional
arguments remain available for existing benchmark scripts.

For the weighted feature matrix `B`, the `cublas` (default) and `fused`
backends use cuBLAS SYRK on the prefetch stream to construct the lower
triangle of the sampled Gram `G=B*B^T`. Two slots store the sampled features and square Gram buffers so one block
can be prepared while the other is consumed. The recurrence references only
blocks `G_ij` with `j<i`. The fused kernel applies these blocks together.
With `--gram-kernel factor`, the recurrence uses exclusive minibatch prefixes
of the weighted sampled features and omits the square Gram buffers. All three
backends use residuals frozen at block entry.

The `leverage` argument is retained for compatibility with existing benchmark
scripts. The uniform and scoring paths have been removed. Sampling is with
replacement through a persistent Walker-Vose alias table, so repeated columns
remain valid separate contributions to the sampled Gram.
