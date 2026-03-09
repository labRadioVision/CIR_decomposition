# Deep Matrix Factorization (Gradient-Based, PSD-Aware)

This repository provides a Python implementation of **deep matrix factorization**:

\[
B \approx H_1 H_2 \cdots H_L
\]

where each factor is optimized via gradient methods by vectorizing its entries.

## Main implementation

- `deep_matrix_factorization.py`
  - `DeepMatrixFactorization`: model class.
  - `build_factor_dims(...)`: helper that builds an `L`-factor chain for a target matrix of shape `M x N` using a shared hidden size `Q`.
  - `resolve_factor_dims(...)`: validates explicit `dims` or infers them from `(M, N, L, Q)`.
  - `fit(...)`: MSE-based optimization with Adam or L-BFGS.
  - `FitResult`: output container with learned factors, loss history, and reconstruction.
  - `load_matrix_from_mat(...)`: loads `B` from `.mat` using `scipy.io.loadmat`.
  - `save_factorization_to_mat(...)`: exports factors as `.mat` via `scipy.io.savemat`.
  - `factorize_from_mat(...)`: end-to-end MAT input/output pipeline.

## Loss function

The training objective is mean squared error:

\[
\mathcal{L}=\frac{1}{MN}\|B-\hat B\|_F^2,
\quad \hat B=\prod_{i=1}^L H_i
\]

Optional L2 regularization is supported.

## Positivity / PSD handling strategies

Three practical constraint mechanisms are implemented:

1. **`softplus`** (elementwise-positive factors)
   - Parameterize unconstrained matrix `A_i` and set `H_i = softplus(A_i)`.
   - Works for rectangular factors, including `M x Q` and `Q x N` endpoint factors.

2. **`cholesky`** (strictly PSD/PD square factors)
   - Parameterize unconstrained `A_i` and build lower-triangular `L_i` with positive diagonal.
   - Set `H_i = L_i L_i^T`.
   - This requires every factor to be square, so it is not suitable for rectangular endpoint factors.

3. **`projected_psd`** (projected gradient)
   - Optimize raw matrices directly.
   - After every update, symmetrize and project each square factor to PSD cone using eigenvalue clipping.
   - For rectangular factors under this mode, fallback projection is elementwise non-negativity.

## Rectangular factorizations with custom `L` and `Q`

If `B` has shape `M x N`, you can ask for `L` factors with shared hidden size `Q`:

\[
H_1 \in \mathbb{R}^{M \times Q}, \quad
H_2,\dots,H_{L-1} \in \mathbb{R}^{Q \times Q}, \quad
H_L \in \mathbb{R}^{Q \times N}
\]

Use `build_factor_dims(M, N, L, Q)` to generate the corresponding dimension chain.

## Quick usage

```python
import torch
from deep_matrix_factorization import DeepMatrixFactorization, build_factor_dims

M, N = 6, 10
L = 4
Q = 3
dims = build_factor_dims(M, N, L, Q)  # [6, 3, 3, 3, 10]

B = torch.rand(M, N, dtype=torch.float64)
model = DeepMatrixFactorization(dims=dims, constraint="softplus", seed=0)
result = model.fit(B, lr=0.05, max_steps=2000, optimizer="adam")

print(result.loss_history[-1])
print(result.reconstruction.shape)
```

## MAT input/output workflow

You can directly load `B` from a `.mat` file and either:
- pass an explicit dimension chain with `dims`, or
- let the code infer the factor shapes from `B.shape`, `num_factors=L`, and `inner_dim=Q`.

```python
from deep_matrix_factorization import factorize_from_mat

result = factorize_from_mat(
    input_mat_path="input_B.mat",
    output_mat_path="factors_out.mat",
    num_factors=4,
    inner_dim=12,
    input_key="B",
    constraint="softplus",
    lr=0.03,
    max_steps=3000,
    optimizer="adam",
)
```

Saved MAT variables include:
- `H_1`, `H_2`, ..., `H_L`: learned factors
- `loss_history`: optimization history
- `B_reconstruction`: reconstructed matrix product

## Command-line usage

```bash
python deep_matrix_factorization.py \
  --input-mat input_B.mat \
  --output-mat factors_out.mat \
  --num-factors 4 \
  --inner-dim 12 \
  --input-key B \
  --constraint softplus \
  --optimizer adam \
  --lr 0.03 \
  --max-steps 3000
```

You can still provide an explicit chain if you prefer:

```bash
python deep_matrix_factorization.py \
  --input-mat input_B.mat \
  --output-mat factors_out.mat \
  --dims 20 12 12 30
```

## Recommendations

- If factors must be PSD and square: use **`constraint="cholesky"`**.
- If you want a straightforward feasibility-enforcing baseline: use **`constraint="projected_psd"`**.
- For rectangular deep factorization with `M x Q`, `Q x Q`, ..., `Q x N`: use **`constraint="softplus"`**.

## Run demo

```bash
python deep_matrix_factorization.py
```

This creates a synthetic rectangular factorization, composes `B`, and recovers a low-error deep factorization.
