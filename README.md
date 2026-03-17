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

M, N, r = 20, 30, 7
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

M, N, r = 20, 30, 7
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
    rank=7,
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
  --rank 7 \
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

- For sparse rectangular matrices with very similar rows: start with `model="uv"` and a small `rank`, with `7` as the current default.
- If you want a little extra flexibility while staying low-rank: use `model="ugv"`, also with default `rank=7`.
- Use the full `chain` model only when the shallow low-rank models are clearly too restrictive.
- For sparse nonnegative targets: prefer `constraint="projected_nonnegative"`.

## Run demo

```bash
python deep_matrix_factorization.py
```

This runs a small synthetic low-rank `ugv` example.

---

## Scripts

### `compare_factorizations.py` — pairwise factorization comparison

Factorizes two MAT files independently with the UGV model and compares their factors.

```bash
python compare_factorizations.py \
  --input-a input_B.mat \
  --input-b input_B_v2.mat \
  --input-key CIR_linear \
  --output-dir comparison_results \
  --rank 7
```

Outputs (in `--output-dir`):

- `input_B_ugv_rank7.mat`, `input_B_v2_ugv_rank7.mat` — per-file UGV factors
- `factorization_comparison_rank7.json` — relative reconstruction errors, Frobenius factor differences, singular value spectra
- `factorization_comparison_rank7.mat` — same metrics in MAT format

---

### `run_window_factorization.py` — sliding-window factorization

Factorizes consecutive (or overlapping) windows of a single matrix using the UGV model.

```bash
python run_window_factorization.py \
  --input-mat input_B.mat \
  --source-key CIR_linear \
  --output-dir window_factorization_results \
  --workspace-tmp tmp_window_factorization \
  --window-size 300 \
  --step 300 \
  --max-cols 2100 \
  --rank 7 \
  --output-prefix factors_out \
  --aggregate-name factors_out_windows_ugv_rank7.mat
```

Pass `--compute-overlap-metrics` when `step < window-size` to also report overlap-region consistency between adjacent windows.

Outputs (in `--output-dir`):

- `factors_out_window_NN.mat` — per-window factors (`H_1`, `H_2`, `H_3`, `B_reconstruction`) plus window metadata
- `factors_out_windows_ugv_rank7.mat` — aggregate MAT with all windows' factors and optional overlap metrics

---

### `run_overlap_rank_sweep.py` — rank sweep for overlapping windows

Sweeps UGV rank from `--rank-min` to `--rank-max` and measures reconstruction consistency in overlapping regions between adjacent windows. Identifies the rank that minimizes overlap disagreement.

```bash
python run_overlap_rank_sweep.py \
  --input-mat input_B.mat \
  --source-key CIR_linear \
  --workspace-tmp tmp_rank_sweep \
  --summary-json rank_sweep_overlap_summary.json \
  --window-size 300 \
  --step 150 \
  --max-cols 2100 \
  --rank-min 2 \
  --rank-max 21
```

Output:

- `rank_sweep_overlap_summary.json` — per-rank mean/min/max overlap disagreement, best overall rank, and best rank per overlap pair

---

### `compare_overlap_windows.py` — three-material windowed comparison

Factorizes overlapping windows of three material CIR matrices (metal, paper, PTFE) in parallel and computes pairwise factor differences across all windows. Useful for identifying where in the travel range the materials are most distinguishable.

Place the three source MAT files in the working directory, then run:

```bash
python compare_overlap_windows.py \
  --input-metal CIR_20260209T162335_Pit5x10_NS21x1_RF18001_RF28001_MetalCyl_LinOnly.mat \
  --input-paper CIR_20260209T163709_Pit5x10_NS21x1_RF18001_RF28001_PaperCyl_LinOnly.mat \
  --input-ptfe  CIR_20260209T165435_Pit5x10_NS21x1_RF18001_RF28001_PTFECyl_LinOnly.mat \
  --input-key CIR_linear \
  --output-dir overlap_window_results \
  --window-size 300 \
  --step 150 \
  --max-cols 2100 \
  --rank 7
```

Per-window metrics reported for every material pair (`metal_vs_paper`, `metal_vs_ptfe`, `paper_vs_ptfe`):

- relative reconstruction error per material
- Frobenius norm difference of raw factors (U, G, V)
- L2 distance between sorted singular value spectra of U, G, V
- combined singular distance (sum of the three spectral L2 distances)
- raw input relative difference

Outputs (in `--output-dir`):

- `overlap_window_comparison_rank7.json` — full per-window results, strongest/weakest window by mean combined singular distance
- `overlap_window_comparison_rank7.mat` — per-pair combined singular distances and raw input differences across all windows
