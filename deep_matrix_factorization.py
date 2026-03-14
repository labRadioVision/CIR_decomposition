"""Matrix factorization with deep-chain and lightweight low-rank models.

This module supports three factorization families for a target matrix ``B``:

1) ``chain``:     B ~= H1 @ H2 @ ... @ HL
2) ``uv``:        B ~= U @ V
3) ``ugv``:       B ~= U @ G @ V

The low-rank ``uv`` and ``ugv`` variants are useful when rows are very similar
and only a few columns carry signal, since such matrices are typically very
low-rank and do not need a long factor chain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Constraint = Literal["softplus", "cholesky", "projected_psd", "projected_nonnegative"]
OptimizerName = Literal["adam", "lbfgs"]
ModelName = Literal["chain", "uv", "ugv"]


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


def build_factor_dims(
    rows: int,
    cols: int,
    num_factors: int,
    inner_dim: Optional[int] = None,
) -> List[int]:
    """Build a chain-model dimension list for a target matrix in R^(rows x cols)."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive, got {(rows, cols)}.")
    if num_factors < 1:
        raise ValueError(f"num_factors must be at least 1, got {num_factors}.")
    if num_factors == 1:
        return [rows, cols]
    if inner_dim is None:
        raise ValueError("inner_dim is required when num_factors >= 2.")
    if inner_dim <= 0:
        raise ValueError(f"inner_dim must be positive, got {inner_dim}.")

    return [rows] + [inner_dim] * (num_factors - 1) + [cols]


def build_low_rank_shapes(rows: int, cols: int, rank: int, model: ModelName) -> List[Tuple[int, int]]:
    """Build factor shapes for lightweight low-rank models."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive, got {(rows, cols)}.")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    if model == "uv":
        return [(rows, rank), (rank, cols)]
    if model == "ugv":
        return [(rows, rank), (rank, rank), (rank, cols)]
    raise ValueError(f"Low-rank shapes are only defined for model='uv' or 'ugv', got {model}.")


def resolve_factor_dims(
    target_shape: Sequence[int],
    dims: Optional[Sequence[int]] = None,
    num_factors: Optional[int] = None,
    inner_dim: Optional[int] = None,
) -> List[int]:
    """Resolve a valid chain-model dimension list for a 2D target matrix."""
    if len(target_shape) != 2:
        raise ValueError(f"target_shape must have length 2, got {tuple(target_shape)}.")

    rows, cols = int(target_shape[0]), int(target_shape[1])

    if dims is not None:
        if num_factors is not None or inner_dim is not None:
            raise ValueError("Provide either dims or (num_factors, inner_dim), not both.")

        resolved = [int(d) for d in dims]
        if len(resolved) < 2:
            raise ValueError("dims must contain at least [input_dim, output_dim].")
        if any(d <= 0 for d in resolved):
            raise ValueError(f"All dims must be positive, got {resolved}.")
        if resolved[0] != rows or resolved[-1] != cols:
            raise ValueError(
                f"dims chain {resolved[0]} -> {resolved[-1]} does not match target shape "
                f"{rows} -> {cols}."
            )
        return resolved

    if num_factors is None:
        raise ValueError("Provide dims or specify num_factors to infer the chain.")

    return build_factor_dims(rows=rows, cols=cols, num_factors=num_factors, inner_dim=inner_dim)


class _FactorizationBase:
    def __init__(
        self,
        factor_shapes: Sequence[Tuple[int, int]],
        constraint: Constraint = "softplus",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        jitter: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if len(factor_shapes) < 1:
            raise ValueError("factor_shapes must contain at least one factor.")

        self.factor_shapes = [(int(n_in), int(n_out)) for n_in, n_out in factor_shapes]
        if any(n_in <= 0 or n_out <= 0 for n_in, n_out in self.factor_shapes):
            raise ValueError(f"All factor dimensions must be positive, got {self.factor_shapes}.")

        self.L = len(self.factor_shapes)
        self.constraint = constraint
        self.device = device
        self.dtype = dtype
        self.jitter = jitter
        self.target_shape = (self.factor_shapes[0][0], self.factor_shapes[-1][1])

        torch.manual_seed(seed)
        self._init_parameters()

    def _init_parameters(self) -> None:
        self.params = torch.nn.ParameterList()

        for i, (n_in, n_out) in enumerate(self.factor_shapes):
            if self.constraint == "cholesky":
                if n_in != n_out:
                    raise ValueError(
                        "Cholesky constraint requires square factors; "
                        f"got factor shape {(n_in, n_out)} at index {i}."
                    )
                size = n_in * n_in
            else:
                size = n_in * n_out

            if self.constraint == "projected_nonnegative":
                init = 0.05 * torch.rand(size, dtype=self.dtype, device=self.device) + 1e-3
            else:
                init = 0.01 * torch.randn(size, dtype=self.dtype, device=self.device)

            self.params.append(
                torch.nn.Parameter(init)
            )

    def _raw_factor(self, idx: int) -> torch.Tensor:
        n_in, n_out = self.factor_shapes[idx]
        return self.params[idx].view(n_in, n_out)

    def factors(self) -> List[torch.Tensor]:
        mats: List[torch.Tensor] = []
        for i in range(self.L):
            A = self._raw_factor(i)

            if self.constraint == "softplus":
                H = F.softplus(A)
            elif self.constraint == "cholesky":
                L = torch.tril(A)
                d = torch.diag(F.softplus(torch.diag(L)) + self.jitter)
                L = torch.tril(L, diagonal=-1) + d
                H = L @ L.T
            elif self.constraint in {"projected_psd", "projected_nonnegative"}:
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
            for i, (n_in, n_out) in enumerate(self.factor_shapes):
                M = self.params[i].view(n_in, n_out)
                if n_in != n_out:
                    M.copy_(M.clamp_min(0.0))
                    continue

                S = 0.5 * (M + M.T)
                eigvals, eigvecs = torch.linalg.eigh(S)
                eigvals = eigvals.clamp_min(0.0)
                S_psd = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
                M.copy_(S_psd)

    def _project_nonnegative_(self) -> None:
        if self.constraint != "projected_nonnegative":
            return

        with torch.no_grad():
            for i, (n_in, n_out) in enumerate(self.factor_shapes):
                M = self.params[i].view(n_in, n_out)
                M.copy_(M.clamp_min(0.0))

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
        if tuple(B.shape) != self.target_shape:
            raise ValueError(
                f"B shape {tuple(B.shape)} incompatible with factorization target "
                f"{self.target_shape}."
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
            self._project_nonnegative_()

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


class DeepMatrixFactorization(_FactorizationBase):
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
        if any(d <= 0 for d in dims):
            raise ValueError(f"All dims must be positive, got {list(dims)}.")

        self.dims = [int(d) for d in dims]
        factor_shapes = list(zip(self.dims[:-1], self.dims[1:]))
        super().__init__(
            factor_shapes=factor_shapes,
            constraint=constraint,
            device=device,
            dtype=dtype,
            jitter=jitter,
            seed=seed,
        )

    @classmethod
    def from_target_shape(
        cls,
        target_shape: Sequence[int],
        num_factors: int,
        inner_dim: Optional[int] = None,
        **kwargs,
    ) -> "DeepMatrixFactorization":
        dims = resolve_factor_dims(
            target_shape=target_shape,
            num_factors=num_factors,
            inner_dim=inner_dim,
        )
        return cls(dims=dims, **kwargs)


class LowRankMatrixFactorization(_FactorizationBase):
    def __init__(
        self,
        rows: int,
        cols: int,
        rank: int,
        model: Literal["uv", "ugv"] = "uv",
        constraint: Constraint = "projected_nonnegative",
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        jitter: float = 1e-6,
        seed: int = 0,
    ) -> None:
        self.rows = int(rows)
        self.cols = int(cols)
        self.rank = int(rank)
        self.model = model
        factor_shapes = build_low_rank_shapes(self.rows, self.cols, self.rank, model)
        super().__init__(
            factor_shapes=factor_shapes,
            constraint=constraint,
            device=device,
            dtype=dtype,
            jitter=jitter,
            seed=seed,
        )

    @classmethod
    def from_target_shape(
        cls,
        target_shape: Sequence[int],
        rank: int,
        model: Literal["uv", "ugv"] = "uv",
        **kwargs,
    ) -> "LowRankMatrixFactorization":
        if len(target_shape) != 2:
            raise ValueError(f"target_shape must have length 2, got {tuple(target_shape)}.")
        rows, cols = int(target_shape[0]), int(target_shape[1])
        return cls(rows=rows, cols=cols, rank=rank, model=model, **kwargs)


def load_matrix_from_mat(
    mat_path: str,
    key: str = "B",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Load target matrix B from a MAT file."""
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


def build_model_for_matrix(
    B_shape: Sequence[int],
    model: ModelName = "chain",
    dims: Optional[Sequence[int]] = None,
    num_factors: Optional[int] = None,
    inner_dim: Optional[int] = None,
    rank: Optional[int] = None,
    constraint: Constraint = "softplus",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> _FactorizationBase:
    """Build a factorization model for a target matrix shape."""
    if model == "chain":
        resolved_dims = resolve_factor_dims(
            target_shape=B_shape,
            dims=dims,
            num_factors=num_factors,
            inner_dim=inner_dim,
        )
        return DeepMatrixFactorization(
            dims=resolved_dims,
            constraint=constraint,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    if rank is None:
        raise ValueError("rank is required when model is 'uv' or 'ugv'.")

    return LowRankMatrixFactorization.from_target_shape(
        target_shape=B_shape,
        rank=rank,
        model=model,
        constraint=constraint,
        device=device,
        dtype=dtype,
        seed=seed,
    )


def factorize_from_mat(
    input_mat_path: str,
    output_mat_path: str,
    dims: Optional[Sequence[int]] = None,
    num_factors: Optional[int] = None,
    inner_dim: Optional[int] = None,
    rank: Optional[int] = None,
    model: ModelName = "chain",
    input_key: str = "B",
    constraint: Constraint = "softplus",
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
    """Load B from MAT, run factorization, and save factors back to MAT."""
    B = load_matrix_from_mat(input_mat_path, key=input_key, device=device, dtype=dtype)
    factor_model = build_model_for_matrix(
        B_shape=B.shape,
        model=model,
        dims=dims,
        num_factors=num_factors,
        inner_dim=inner_dim,
        rank=rank,
        constraint=constraint,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    result = factor_model.fit(
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
    parser = argparse.ArgumentParser(description="Matrix factorization from MAT input.")
    parser.add_argument("--input-mat", type=str, required=False, help="Path to input MAT file")
    parser.add_argument("--output-mat", type=str, required=False, help="Path to output MAT file")
    parser.add_argument(
        "--model",
        choices=["chain", "uv", "ugv"],
        default="chain",
        help="Factorization family: long chain, shallow UV, or deep-flavored UGV.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        required=False,
        help="Chain-model dimensions, e.g. --dims 20 10 10 30.",
    )
    parser.add_argument(
        "--num-factors",
        type=int,
        required=False,
        help="Number of factors L for the chain model.",
    )
    parser.add_argument(
        "--inner-dim",
        type=int,
        required=False,
        help="Shared hidden dimension Q for the chain model.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Low-rank dimension r for model='uv' or model='ugv'.",
    )
    parser.add_argument("--input-key", type=str, default="B")
    parser.add_argument(
        "--constraint",
        choices=["softplus", "cholesky", "projected_psd", "projected_nonnegative"],
        default="projected_nonnegative",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def demo() -> None:
    """Small self-check demo on a sparse, very-low-rank rectangular matrix."""
    torch.manual_seed(1)

    m, n, r = 10, 18, 2
    U_true = torch.rand(m, r, dtype=torch.float64) + 0.1
    G_true = torch.tensor([[1.2, 0.1], [0.1, 0.8]], dtype=torch.float64)
    V_true = torch.zeros(r, n, dtype=torch.float64)
    active_cols = torch.tensor([2, 3, 10, 11, 15])
    V_true[:, active_cols] = torch.rand(r, active_cols.numel(), dtype=torch.float64) + 0.1
    B = U_true @ G_true @ V_true
    B = B + 0.01 * torch.rand_like(B)

    model = LowRankMatrixFactorization(
        rows=m,
        cols=n,
        rank=r,
        model="ugv",
        constraint="projected_nonnegative",
        seed=42,
    )
    result = model.fit(B, lr=0.05, max_steps=2000, optimizer="adam", verbose=False)

    rel_err = torch.linalg.norm(result.reconstruction - B) / torch.linalg.norm(B)
    print("Demo model: ugv")
    print(f"Factor shapes: {[tuple(f.shape) for f in result.factors]}")
    print(f"Final loss: {result.loss_history[-1]:.3e}")
    print(f"Relative reconstruction error: {float(rel_err):.3e}")


if __name__ == "__main__":
    args = _parse_args()
    using_mat_workflow = any(
        value is not None
        for value in (
            args.input_mat,
            args.output_mat,
            args.dims,
            args.num_factors,
            args.inner_dim,
            args.rank,
        )
    )
    if using_mat_workflow:
        if not args.input_mat or not args.output_mat:
            raise SystemExit("Both --input-mat and --output-mat are required for MAT I/O.")
        if args.model == "chain" and args.dims is None and args.num_factors is None:
            raise SystemExit("For model='chain', provide either --dims or --num-factors.")
        if args.model in {"uv", "ugv"} and args.rank is None:
            raise SystemExit("For model='uv' or model='ugv', provide --rank.")

        out = factorize_from_mat(
            input_mat_path=args.input_mat,
            output_mat_path=args.output_mat,
            dims=args.dims,
            num_factors=args.num_factors,
            inner_dim=args.inner_dim,
            rank=args.rank,
            model=args.model,
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
