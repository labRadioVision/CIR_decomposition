import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch

from deep_matrix_factorization import build_model_for_matrix


def sorted_real_eigs(matrix: np.ndarray) -> list[float]:
    eigvals = np.linalg.eigvals(matrix)
    eigvals = np.real_if_close(eigvals, tol=1_000)
    eigvals = np.asarray(eigvals, dtype=np.float64)
    return sorted(eigvals.tolist(), reverse=True)


def sorted_gram_eigs(matrix: np.ndarray, left_gram: bool) -> list[float]:
    gram = matrix @ matrix.T if left_gram else matrix.T @ matrix
    eigvals = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    return sorted(np.asarray(eigvals, dtype=np.float64).tolist(), reverse=True)


def sorted_singular_values(matrix: np.ndarray) -> list[float]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return sorted(np.asarray(singular_values, dtype=np.float64).tolist(), reverse=True)


def vector_l2_diff(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def factorize_window(B_win: np.ndarray, rank: int) -> dict:
    B_t = torch.as_tensor(B_win, dtype=torch.float64)
    model = build_model_for_matrix(
        B_shape=B_t.shape,
        model="ugv",
        rank=rank,
        constraint="projected_nonnegative",
        dtype=torch.float64,
        seed=0,
    )
    result = model.fit(
        B_t,
        lr=0.03,
        max_steps=3000,
        optimizer="adam",
    )

    U, G, V = [np.asarray(f.detach().cpu(), dtype=np.float64) for f in result.factors]
    B_hat = np.asarray(result.reconstruction.detach().cpu(), dtype=np.float64)
    rel_err = float(np.linalg.norm(B_hat - B_win) / np.linalg.norm(B_win)) if np.linalg.norm(B_win) > 0 else 0.0

    return {
        "U": U,
        "G": G,
        "V": V,
        "B_hat": B_hat,
        "rel_err": rel_err,
        "spectra": {
            "G_eigenvalues": sorted_real_eigs(G),
            "UtU_eigenvalues": sorted_gram_eigs(U, left_gram=False),
            "VVt_eigenvalues": sorted_gram_eigs(V, left_gram=True),
            "U_singular_values": sorted_singular_values(U),
            "G_singular_values": sorted_singular_values(G),
            "V_singular_values": sorted_singular_values(V),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare windowed UGV factorizations between two MAT files.")
    parser.add_argument("--input-a", type=Path, default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f\input_B.mat"))
    parser.add_argument("--input-b", type=Path, default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f\input_B_v2.mat"))
    parser.add_argument("--input-key", type=str, default="CIR_linear")
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f"))
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--step", type=int, default=150)
    parser.add_argument("--max-cols", type=int, default=2100)
    parser.add_argument("--rank", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    A = np.asarray(scipy.io.loadmat(args.input_a)[args.input_key], dtype=np.float64)[:, : args.max_cols]
    B = np.asarray(scipy.io.loadmat(args.input_b)[args.input_key], dtype=np.float64)[:, : args.max_cols]

    if A.shape != B.shape:
        raise ValueError(f"Input shapes do not match: {A.shape} vs {B.shape}")
    if args.step <= 0 or args.window_size <= 0:
        raise ValueError("window-size and step must be positive.")
    if A.shape[1] < args.window_size:
        raise ValueError("window-size exceeds the truncated column count.")

    num_windows = 1 + (A.shape[1] - args.window_size) // args.step
    window_results = []

    for idx in range(num_windows):
        start = idx * args.step
        end = start + args.window_size
        A_win = A[:, start:end]
        B_win = B[:, start:end]

        fact_a = factorize_window(A_win, rank=args.rank)
        fact_b = factorize_window(B_win, rank=args.rank)

        comparison = {
            "window_index": idx + 1,
            "col_range_1based": [start + 1, end],
            "relative_reconstruction_error": {
                "input_B": fact_a["rel_err"],
                "input_B_v2": fact_b["rel_err"],
            },
            "raw_input_relative_difference": float(np.linalg.norm(A_win - B_win) / np.linalg.norm(A_win))
            if np.linalg.norm(A_win) > 0
            else 0.0,
            "raw_factor_differences_fro": {
                "U": float(np.linalg.norm(fact_a["U"] - fact_b["U"])),
                "G": float(np.linalg.norm(fact_a["G"] - fact_b["G"])),
                "V": float(np.linalg.norm(fact_a["V"] - fact_b["V"])),
            },
            "spectral_distance_l2": {
                "G_eigenvalues": vector_l2_diff(
                    fact_a["spectra"]["G_eigenvalues"], fact_b["spectra"]["G_eigenvalues"]
                ),
                "UtU_eigenvalues": vector_l2_diff(
                    fact_a["spectra"]["UtU_eigenvalues"], fact_b["spectra"]["UtU_eigenvalues"]
                ),
                "VVt_eigenvalues": vector_l2_diff(
                    fact_a["spectra"]["VVt_eigenvalues"], fact_b["spectra"]["VVt_eigenvalues"]
                ),
                "U_singular_values": vector_l2_diff(
                    fact_a["spectra"]["U_singular_values"], fact_b["spectra"]["U_singular_values"]
                ),
                "G_singular_values": vector_l2_diff(
                    fact_a["spectra"]["G_singular_values"], fact_b["spectra"]["G_singular_values"]
                ),
                "V_singular_values": vector_l2_diff(
                    fact_a["spectra"]["V_singular_values"], fact_b["spectra"]["V_singular_values"]
                ),
            },
            "leading_values": {
                "input_B": {
                    "G_eig1": fact_a["spectra"]["G_eigenvalues"][0],
                    "UtU_eig1": fact_a["spectra"]["UtU_eigenvalues"][0],
                    "VVt_eig1": fact_a["spectra"]["VVt_eigenvalues"][0],
                    "U_sv1": fact_a["spectra"]["U_singular_values"][0],
                    "G_sv1": fact_a["spectra"]["G_singular_values"][0],
                    "V_sv1": fact_a["spectra"]["V_singular_values"][0],
                },
                "input_B_v2": {
                    "G_eig1": fact_b["spectra"]["G_eigenvalues"][0],
                    "UtU_eig1": fact_b["spectra"]["UtU_eigenvalues"][0],
                    "VVt_eig1": fact_b["spectra"]["VVt_eigenvalues"][0],
                    "U_sv1": fact_b["spectra"]["U_singular_values"][0],
                    "G_sv1": fact_b["spectra"]["G_singular_values"][0],
                    "V_sv1": fact_b["spectra"]["V_singular_values"][0],
                },
            },
        }

        comparison["combined_singular_distance"] = (
            comparison["spectral_distance_l2"]["U_singular_values"]
            + comparison["spectral_distance_l2"]["G_singular_values"]
            + comparison["spectral_distance_l2"]["V_singular_values"]
        )
        comparison["combined_eigen_distance"] = (
            comparison["spectral_distance_l2"]["G_eigenvalues"]
            + comparison["spectral_distance_l2"]["UtU_eigenvalues"]
            + comparison["spectral_distance_l2"]["VVt_eigenvalues"]
        )

        window_results.append(comparison)
        print(
            f"window {idx + 1:02d} cols {start + 1}-{end}: "
            f"combined_sv_dist={comparison['combined_singular_distance']:.6f}, "
            f"combined_eig_dist={comparison['combined_eigen_distance']:.6f}"
        )

    strongest_by_sv = max(window_results, key=lambda x: x["combined_singular_distance"])
    weakest_by_sv = min(window_results, key=lambda x: x["combined_singular_distance"])
    strongest_by_eig = max(window_results, key=lambda x: x["combined_eigen_distance"])
    weakest_by_eig = min(window_results, key=lambda x: x["combined_eigen_distance"])

    summary = {
        "rank": args.rank,
        "window_size": args.window_size,
        "step": args.step,
        "max_cols": args.max_cols,
        "num_windows": num_windows,
        "strongest_window_by_combined_singular_distance": strongest_by_sv,
        "weakest_window_by_combined_singular_distance": weakest_by_sv,
        "strongest_window_by_combined_eigen_distance": strongest_by_eig,
        "weakest_window_by_combined_eigen_distance": weakest_by_eig,
        "window_results": window_results,
    }

    json_path = args.output_dir / f"overlap_window_comparison_rank{args.rank}.json"
    mat_path = args.output_dir / f"overlap_window_comparison_rank{args.rank}.mat"
    json_path.write_text(json.dumps(summary, indent=2))
    scipy.io.savemat(
        mat_path,
        {
            "rank": args.rank,
            "window_size": args.window_size,
            "step": args.step,
            "max_cols": args.max_cols,
            "combined_singular_distance": np.asarray(
                [item["combined_singular_distance"] for item in window_results], dtype=np.float64
            ),
            "combined_eigen_distance": np.asarray(
                [item["combined_eigen_distance"] for item in window_results], dtype=np.float64
            ),
            "raw_input_relative_difference": np.asarray(
                [item["raw_input_relative_difference"] for item in window_results], dtype=np.float64
            ),
            "window_ranges_1based": np.asarray(
                [item["col_range_1based"] for item in window_results], dtype=np.float64
            ),
        },
    )
    print(f"summary_json {json_path}")
    print(f"summary_mat {mat_path}")


if __name__ == "__main__":
    main()
