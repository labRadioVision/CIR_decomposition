# Matrix Factorization (Deep Chain + Low-Rank Variants)

This repository provides a Python implementation of matrix factorization for rectangular targets

\[
B \approx \hat B
\]

with three model families:

1. `chain`: `B ~= H_1 H_2 ... H_L`
2. `uv`: `B ~= U V`
3. `ugv`: `B ~= U G V`

The `uv` and `ugv` models are intended for matrices with very similar rows, few informative columns, and very low effective rank.

## Main implementation

- `deep_matrix_factorization.py`
  - `DeepMatrixFactorization`: original chain model.
  - `LowRankMatrixFactorization`: lightweight `uv` / `ugv` model.
  - `build_factor_dims(...)`: helper for chain-model dimensions.
  - `build_low_rank_shapes(...)`: helper for low-rank factor shapes.
  - `build_model_for_matrix(...)`: unified model builder.
  - `factorize_from_mat(...)`: end-to-end MAT input/output pipeline.

## Models

### Chain model

For a target matrix `B` with shape `M x N`, you can build an `L`-factor chain with shared hidden size `Q`:

\[
H_1 \in \mathbb{R}^{M \times Q}, \quad
H_2,\dots,H_{L-1} \in \mathbb{R}^{Q \times Q}, \quad
H_L \in \mathbb{R}^{Q \times N}
\]

Use `build_factor_dims(M, N, L, Q)` to generate the chain dimensions.

### Low-rank UV model

\[
B \approx U V, \quad
U \in \mathbb{R}^{M \times r}, \quad
V \in \mathbb{R}^{r \times N}
\]

This is the simplest choice for very low-rank matrices.

### Low-rank UGV model

\[
B \approx U G V, \quad
U \in \mathbb{R}^{M \times r}, \quad
G \in \mathbb{R}^{r \times r}, \quad
V \in \mathbb{R}^{r \times N}
\]

This keeps a small "deep flavor" while staying much simpler than a long chain.

## Loss function

The training objective is mean squared error:

\[
\mathcal{L}=\frac{1}{MN}\|B-\hat B\|_F^2
\]

Optional L2 regularization is supported.

## Positivity / PSD handling strategies

Four practical constraint mechanisms are implemented:

1. **`softplus`**
   - Elementwise-positive parameterization.
   - Good when factors should stay strictly positive.

2. **`cholesky`**
   - Square PSD factors via `H = L L^T`.
   - Only valid when every factor is square.

3. **`projected_psd`**
   - Raw optimization plus PSD projection for square factors.
   - Rectangular factors are clipped elementwise to be nonnegative.

4. **`projected_nonnegative`**
   - Raw optimization plus elementwise nonnegative projection.
   - Recommended for sparse rectangular nonnegative matrices.

## Quick usage

### Chain model

```python
import torch
from deep_matrix_factorization import DeepMatrixFactorization, build_factor_dims

M, N = 6, 10
L = 4
Q = 3
dims = build_factor_dims(M, N, L, Q)

B = torch.rand(M, N, dtype=torch.float64)
model = DeepMatrixFactorization(dims=dims, constraint="projected_nonnegative", seed=0)
result = model.fit(B, lr=0.05, max_steps=2000, optimizer="adam")
```

### UV model

```python
import torch
from deep_matrix_factorization import LowRankMatrixFactorization

M, N, r = 20, 30, 2
B = torch.rand(M, N, dtype=torch.float64)

model = LowRankMatrixFactorization(
    rows=M,
    cols=N,
    rank=r,
    model="uv",
    constraint="projected_nonnegative",
    seed=0,
)
result = model.fit(B, lr=0.05, max_steps=2000, optimizer="adam")
```

### UGV model

```python
import torch
from deep_matrix_factorization import LowRankMatrixFactorization

M, N, r = 20, 30, 2
B = torch.rand(M, N, dtype=torch.float64)

model = LowRankMatrixFactorization(
    rows=M,
    cols=N,
    rank=r,
    model="ugv",
    constraint="projected_nonnegative",
    seed=0,
)
result = model.fit(B, lr=0.05, max_steps=2000, optimizer="adam")
```

## MAT input/output workflow

```python
from deep_matrix_factorization import factorize_from_mat

result = factorize_from_mat(
    input_mat_path="input_B.mat",
    output_mat_path="factors_out.mat",
    model="ugv",
    rank=2,
    input_key="B",
    constraint="projected_nonnegative",
    lr=0.03,
    max_steps=3000,
    optimizer="adam",
)
```

Saved MAT variables include:

- `H_1`, `H_2`, ...: learned factors
- `loss_history`: optimization history
- `B_reconstruction`: reconstructed matrix

## Command-line usage

### Low-rank UGV

```bash
python deep_matrix_factorization.py \
  --input-mat input_B.mat \
  --output-mat factors_out.mat \
  --model ugv \
  --rank 2 \
  --input-key B \
  --constraint projected_nonnegative \
  --optimizer adam \
  --lr 0.03 \
  --max-steps 3000
```

### Deep chain

```bash
python deep_matrix_factorization.py \
  --input-mat input_B.mat \
  --output-mat factors_out.mat \
  --model chain \
  --num-factors 4 \
  --inner-dim 12
```

## Recommendations

- For sparse rectangular matrices with very similar rows: start with `model="uv"` and a very small `rank`.
- If you want a little extra flexibility while staying low-rank: use `model="ugv"`.
- Use the full `chain` model only when the shallow low-rank models are clearly too restrictive.
- For sparse nonnegative targets: prefer `constraint="projected_nonnegative"`.

## Run demo

```bash
python deep_matrix_factorization.py
```

This runs a small synthetic low-rank `ugv` example.
