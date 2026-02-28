"""Deep matrix factorization with gradient-based optimization.

Given a target matrix B and a chain of factor dimensions
(n0, n1, ..., nL), this module learns factors H1..HL such that

    B \approx H1 @ H2 @ ... @ HL

using mean-squared-error loss and first-order gradient methods.

The implementation supports multiple positivity/PSD constraints:
1) softplus: elementwise positivity (works for rectangular factors)
2) cholesky: positive semidefinite/definite square factors H = L L^T + eps I
3) projected_psd: projected-gradient updates with eigenvalue clipping

The trainable variables are stored as vectors and reshaped to matrices during
forward passes, matching the "vectorize each factor and optimize" workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Literal, Sequence

import torch
import torch.nn.functional as F

Constraint = Literal["softplus", "cholesky", "projected_psd"]
OptimizerName = Literal["adam", "lbfgs"]


@dataclass
class FitResult:
    factors: List[torch.Tensor]
    loss_history: List[float]
    reconstruction: torch.Tensor


def _import_scipy_io():
    """Import scipy.io lazily so core module import stays lightweight."""
    try:
        from scipy import io as scipy_io
    except ImportError as exc:
        raise ImportError(
            "scipy is required for .mat input/output. Install it with `pip install scipy`."
        ) from exc
    return scipy_io


class DeepMatrixFactorization:
    def __init__(
        self,
        dims: Sequence[int],
        constraint: Constraint = "softplus",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        jitter: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if len(dims) < 2:
            raise ValueError("dims must contain at least [input_dim, output_dim].")

        self.dims = list(dims)
        self.L = len(dims) - 1
        self.constraint = constraint
        self.device = device
        self.dtype = dtype
        self.jitter = jitter

        torch.manual_seed(seed)
        self._init_parameters()

    def _init_parameters(self) -> None:
        self.params = torch.nn.ParameterList()

        for i in range(self.L):
            n_in, n_out = self.dims[i], self.dims[i + 1]

            if self.constraint == "cholesky":
                if n_in != n_out:
                    raise ValueError(
                        "Cholesky constraint requires square factors; "
                        f"got factor shape {(n_in, n_out)} at index {i}."
                    )
                size = n_in * n_in
            else:
                size = n_in * n_out

            p = torch.nn.Parameter(
                0.01 * torch.randn(size, dtype=self.dtype, device=self.device)
            )
            self.params.append(p)

    def _raw_factor(self, idx: int) -> torch.Tensor:
        n_in, n_out = self.dims[idx], self.dims[idx + 1]
        return self.params[idx].view(n_in, n_out)

    def factors(self) -> List[torch.Tensor]:
        mats: List[torch.Tensor] = []
        for i in range(self.L):
            A = self._raw_factor(i)

            if self.constraint == "softplus":
                H = F.softplus(A)
            elif self.constraint == "cholesky":
                # Build lower-triangular matrix with strictly positive diagonal.
                L = torch.tril(A)
                d = torch.diag(F.softplus(torch.diag(L)) + self.jitter)
                L = torch.tril(L, diagonal=-1) + d
                H = L @ L.T
            elif self.constraint == "projected_psd":
                H = A
            else:
                raise ValueError(f"Unknown constraint: {self.constraint}")

            mats.append(H)
        return mats

    @staticmethod
    def _chain_product(mats: Sequence[torch.Tensor]) -> torch.Tensor:
        out = mats[0]
        for M in mats[1:]:
            out = out @ M
        return out

    def _project_psd_(self) -> None:
        if self.constraint != "projected_psd":
            return

        with torch.no_grad():
            for i in range(self.L):
                n_in, n_out = self.dims[i], self.dims[i + 1]
                if n_in != n_out:
                    # For rectangular factors, fallback to elementwise positivity projection.
                    M = self.params[i].view(n_in, n_out)
                    M.copy_(M.clamp_min(0.0))
                    continue

                M = self.params[i].view(n_in, n_out)
                S = 0.5 * (M + M.T)
                eigvals, eigvecs = torch.linalg.eigh(S)
                eigvals = eigvals.clamp_min(0.0)
                S_psd = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
                M.copy_(S_psd)

    def fit(
        self,
        B: torch.Tensor,
        lr: float = 1e-2,
        max_steps: int = 5_000,
        optimizer: OptimizerName = "adam",
        tol: float = 1e-10,
        l2_reg: float = 0.0,
        verbose: bool = False,
    ) -> FitResult:
        B = B.to(device=self.device, dtype=self.dtype)
        if B.shape != (self.dims[0], self.dims[-1]):
            raise ValueError(
                f"B shape {tuple(B.shape)} incompatible with dims chain "
                f"{self.dims[0]} -> {self.dims[-1]}"
            )

        if optimizer == "adam":
            opt = torch.optim.Adam(self.params, lr=lr)
        elif optimizer == "lbfgs":
            opt = torch.optim.LBFGS(self.params, lr=lr, max_iter=20, history_size=30)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        loss_history: List[float] = []

        def closure() -> torch.Tensor:
            opt.zero_grad()
            mats = self.factors()
            B_hat = self._chain_product(mats)
            loss = torch.mean((B_hat - B) ** 2)
            if l2_reg > 0:
                reg = sum(torch.sum(p**2) for p in self.params)
                loss = loss + l2_reg * reg
            loss.backward()
            return loss

        prev = float("inf")
        for step in range(max_steps):
            if optimizer == "lbfgs":
                loss_t = opt.step(closure)
            else:
                loss_t = closure()
                opt.step()

            self._project_psd_()

            loss = float(loss_t.detach().cpu())
            loss_history.append(loss)

            if verbose and (step % max(1, max_steps // 20) == 0):
                print(f"step={step:5d}  loss={loss:.6e}")

            if abs(prev - loss) < tol:
                break
            prev = loss

        with torch.no_grad():
            mats = self.factors()
            recon = self._chain_product(mats)

        return FitResult(factors=mats, loss_history=loss_history, reconstruction=recon)


def load_matrix_from_mat(
    mat_path: str,
    key: str = "B",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Load target matrix B from a MAT file.

    Args:
        mat_path: Path to .mat file.
        key: Variable name containing matrix B in the MAT file.
    """
    scipy_io = _import_scipy_io()
    data = scipy_io.loadmat(mat_path)
    if key not in data:
        raise KeyError(f"Variable '{key}' not found in MAT file: {mat_path}")

    B = torch.as_tensor(data[key], dtype=dtype, device=device)
    if B.ndim != 2:
        raise ValueError(f"Loaded '{key}' must be a 2D matrix, got shape {tuple(B.shape)}")
    return B


def save_factorization_to_mat(
    output_path: str,
    result: FitResult,
    factor_prefix: str = "H",
    include_reconstruction: bool = True,
) -> None:
    """Save learned factors and optimization metadata to a MAT file."""
    scipy_io = _import_scipy_io()

    payload = {
        f"{factor_prefix}_{i+1}": H.detach().cpu().numpy()
        for i, H in enumerate(result.factors)
    }
    payload["loss_history"] = torch.tensor(result.loss_history, dtype=torch.float64).numpy()
    if include_reconstruction:
        payload["B_reconstruction"] = result.reconstruction.detach().cpu().numpy()

    scipy_io.savemat(output_path, payload)


def factorize_from_mat(
    input_mat_path: str,
    output_mat_path: str,
    dims: Sequence[int],
    input_key: str = "B",
    constraint: Constraint = "cholesky",
    lr: float = 1e-2,
    max_steps: int = 5000,
    optimizer: OptimizerName = "adam",
    tol: float = 1e-10,
    l2_reg: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> FitResult:
    """Load B from MAT, run deep factorization, and save factors back to MAT."""
    B = load_matrix_from_mat(input_mat_path, key=input_key, device=device, dtype=dtype)

    model = DeepMatrixFactorization(
        dims=dims,
        constraint=constraint,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    result = model.fit(
        B,
        lr=lr,
        max_steps=max_steps,
        optimizer=optimizer,
        tol=tol,
        l2_reg=l2_reg,
        verbose=verbose,
    )
    save_factorization_to_mat(output_mat_path, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep matrix factorization from MAT input.")
    parser.add_argument("--input-mat", type=str, required=False, help="Path to input MAT file")
    parser.add_argument("--output-mat", type=str, required=False, help="Path to output MAT file")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        required=False,
        help="Factor chain dimensions, e.g. --dims 20 20 20 20 for 3 square factors",
    )
    parser.add_argument("--input-key", type=str, default="B")
    parser.add_argument("--constraint", choices=["softplus", "cholesky", "projected_psd"], default="cholesky")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def demo() -> None:
    """Small self-check demo on synthetic PSD factors."""
    torch.manual_seed(1)

    # Example with L=3 square PSD factors (dimension 6x6)
    n = 6
    dims = [n, n, n, n]

    true_factors = []
    for _ in range(3):
        X = torch.randn(n, n, dtype=torch.float64)
        true_factors.append(X @ X.T + 0.1 * torch.eye(n, dtype=torch.float64))

    B = true_factors[0] @ true_factors[1] @ true_factors[2]

    model = DeepMatrixFactorization(dims=dims, constraint="cholesky", seed=42)
    result = model.fit(B, lr=0.08, max_steps=2500, optimizer="adam", verbose=False)

    rel_err = torch.linalg.norm(result.reconstruction - B) / torch.linalg.norm(B)
    print(f"Final loss: {result.loss_history[-1]:.3e}")
    print(f"Relative reconstruction error: {float(rel_err):.3e}")


if __name__ == "__main__":
    args = _parse_args()
    if args.input_mat and args.output_mat and args.dims:
        out = factorize_from_mat(
            input_mat_path=args.input_mat,
            output_mat_path=args.output_mat,
            dims=args.dims,
            input_key=args.input_key,
            constraint=args.constraint,
            lr=args.lr,
            max_steps=args.max_steps,
            optimizer=args.optimizer,
            tol=args.tol,
            l2_reg=args.l2_reg,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(f"Saved {len(out.factors)} factors to {args.output_mat}")
    else:
        demo()
