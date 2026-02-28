# Deep Matrix Factorization (Gradient-Based, PSD-Aware)

This repository provides a Python implementation of **deep matrix factorization**:

\[
B \approx H_1 H_2 \cdots H_L
\]

where each factor is optimized via gradient methods by vectorizing its entries.

## Main implementation

- `deep_matrix_factorization.py`
  - `DeepMatrixFactorization`: model class.
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
   - Works for rectangular factors.

2. **`cholesky`** (strictly PSD/PD square factors)
   - Parameterize unconstrained `A_i` and build lower-triangular `L_i` with positive diagonal.
   - Set `H_i = L_i L_i^T`.
   - This is usually the most stable PSD parameterization for gradient methods.

3. **`projected_psd`** (projected gradient)
   - Optimize raw matrices directly.
   - After every update, symmetrize and project each square factor to PSD cone using eigenvalue clipping.
   - For rectangular factors under this mode, fallback projection is elementwise non-negativity.

## Why these methods are efficient

- **Cholesky parameterization** avoids expensive per-step eigendecompositions and guarantees PSD by construction.
- **Projected PSD** is simple and robust when you need explicit post-step feasibility.
- **Vectorized parameters** are packed in 1-D trainable tensors and reshaped only in forward passes, which integrates cleanly with PyTorch optimizers.

## Quick usage

```python
import torch
from deep_matrix_factorization import DeepMatrixFactorization

n = 8
dims = [n, n, n, n]  # L=3 factors
B = torch.randn(n, n, dtype=torch.float64)
B = B @ B.T + 1e-1 * torch.eye(n, dtype=torch.float64)  # PSD target example

model = DeepMatrixFactorization(dims=dims, constraint="cholesky", seed=0)
result = model.fit(B, lr=0.05, max_steps=2000, optimizer="adam")

print(result.loss_history[-1])
print(result.reconstruction.shape)
```

## MAT input/output workflow (requested)

You can directly load `B` from a `.mat` file and save all learned factors back to another `.mat` file:

```python
from deep_matrix_factorization import factorize_from_mat

result = factorize_from_mat(
    input_mat_path="input_B.mat",
    output_mat_path="factors_out.mat",
    dims=[32, 32, 32, 32],
    input_key="B",         # variable name in input .mat
    constraint="cholesky", # PSD square factors
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
  --dims 32 32 32 32 \
  --input-key B \
  --constraint cholesky \
  --optimizer adam \
  --lr 0.03 \
  --max-steps 3000
```

## Recommendations

- If factors must be PSD and square: use **`constraint="cholesky"`**.
- If you want a straightforward feasibility-enforcing baseline: use **`constraint="projected_psd"`**.
- For rectangular deep factorization with positivity only: use **`constraint="softplus"`**.

## Run demo

```bash
python deep_matrix_factorization.py
```

This creates synthetic PSD factors, composes `B`, and recovers a low-error deep factorization.
