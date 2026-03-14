import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
import torch

from deep_matrix_factorization import factorize_from_mat


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


def factor_metrics(result_path: Path, source_name: str, source_matrix: np.ndarray) -> dict:
    data = scipy.io.loadmat(result_path)
    U = np.asarray(data["H_1"], dtype=np.float64)
    G = np.asarray(data["H_2"], dtype=np.float64)
    V = np.asarray(data["H_3"], dtype=np.float64)
    B_hat = np.asarray(data["B_reconstruction"], dtype=np.float64)

    source_norm = np.linalg.norm(source_matrix)
    recon_error = np.linalg.norm(B_hat - source_matrix)
    rel_error = float(recon_error / source_norm) if source_norm > 0 else 0.0

    return {
        "name": source_name,
        "result_path": str(result_path),
        "relative_reconstruction_error": rel_error,
        "fro_norms": {
            "U": float(np.linalg.norm(U)),
            "G": float(np.linalg.norm(G)),
            "V": float(np.linalg.norm(V)),
            "B_reconstruction": float(np.linalg.norm(B_hat)),
        },
        "spectra": {
            "G_eigenvalues": sorted_real_eigs(G),
            "UtU_eigenvalues": sorted_gram_eigs(U, left_gram=False),
            "VVt_eigenvalues": sorted_gram_eigs(V, left_gram=True),
            "U_singular_values": sorted_singular_values(U),
            "G_singular_values": sorted_singular_values(G),
            "V_singular_values": sorted_singular_values(V),
        },
        "factors": {"U": U, "G": G, "V": V},
    }


def list_diff(a: list[float], b: list[float]) -> list[float]:
    return [float(x - y) for x, y in zip(a, b)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Factorize and compare two CIR matrices.")
    parser.add_argument(
        "--input-a",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f\input_B.mat"),
    )
    parser.add_argument(
        "--input-b",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f\input_B_v2.mat"),
    )
    parser.add_argument("--input-key", type=str, default="CIR_linear")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f"),
    )
    parser.add_argument("--rank", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_specs = [
        ("input_B", args.input_a),
        ("input_B_v2", args.input_b),
    ]

    raw_matrices: dict[str, np.ndarray] = {}
    result_paths: dict[str, Path] = {}

    for name, path in input_specs:
        data = scipy.io.loadmat(path)
        raw_matrices[name] = np.asarray(data[args.input_key], dtype=np.float64)
        output_path = args.output_dir / f"{name}_ugv_rank{args.rank}.mat"
        factorize_from_mat(
            input_mat_path=str(path),
            output_mat_path=str(output_path),
            model="ugv",
            rank=args.rank,
            input_key=args.input_key,
            constraint="projected_nonnegative",
            lr=0.03,
            max_steps=3000,
            optimizer="adam",
        )
        result_paths[name] = output_path
        print(f"factorized {name} -> {output_path}")

    metrics_a = factor_metrics(result_paths["input_B"], "input_B", raw_matrices["input_B"])
    metrics_b = factor_metrics(result_paths["input_B_v2"], "input_B_v2", raw_matrices["input_B_v2"])

    Ua = metrics_a["factors"]["U"]
    Ub = metrics_b["factors"]["U"]
    Ga = metrics_a["factors"]["G"]
    Gb = metrics_b["factors"]["G"]
    Va = metrics_a["factors"]["V"]
    Vb = metrics_b["factors"]["V"]

    comparison = {
        "rank": args.rank,
        "input_key": args.input_key,
        "relative_reconstruction_error": {
            "input_B": metrics_a["relative_reconstruction_error"],
            "input_B_v2": metrics_b["relative_reconstruction_error"],
        },
        "raw_factor_differences_fro": {
            "U": float(np.linalg.norm(Ua - Ub)),
            "G": float(np.linalg.norm(Ga - Gb)),
            "V": float(np.linalg.norm(Va - Vb)),
        },
        "spectral_differences": {
            "G_eigenvalue_diff": list_diff(
                metrics_a["spectra"]["G_eigenvalues"],
                metrics_b["spectra"]["G_eigenvalues"],
            ),
            "UtU_eigenvalue_diff": list_diff(
                metrics_a["spectra"]["UtU_eigenvalues"],
                metrics_b["spectra"]["UtU_eigenvalues"],
            ),
            "VVt_eigenvalue_diff": list_diff(
                metrics_a["spectra"]["VVt_eigenvalues"],
                metrics_b["spectra"]["VVt_eigenvalues"],
            ),
            "U_singular_value_diff": list_diff(
                metrics_a["spectra"]["U_singular_values"],
                metrics_b["spectra"]["U_singular_values"],
            ),
            "G_singular_value_diff": list_diff(
                metrics_a["spectra"]["G_singular_values"],
                metrics_b["spectra"]["G_singular_values"],
            ),
            "V_singular_value_diff": list_diff(
                metrics_a["spectra"]["V_singular_values"],
                metrics_b["spectra"]["V_singular_values"],
            ),
        },
        "metrics": {
            "input_B": {
                "result_path": metrics_a["result_path"],
                "relative_reconstruction_error": metrics_a["relative_reconstruction_error"],
                "fro_norms": metrics_a["fro_norms"],
                "spectra": metrics_a["spectra"],
            },
            "input_B_v2": {
                "result_path": metrics_b["result_path"],
                "relative_reconstruction_error": metrics_b["relative_reconstruction_error"],
                "fro_norms": metrics_b["fro_norms"],
                "spectra": metrics_b["spectra"],
            },
        },
    }

    summary_json_path = args.output_dir / f"factorization_comparison_rank{args.rank}.json"
    summary_mat_path = args.output_dir / f"factorization_comparison_rank{args.rank}.mat"

    summary_json_path.write_text(json.dumps(comparison, indent=2))
    scipy.io.savemat(
        summary_mat_path,
        {
            "rank": args.rank,
            "relative_reconstruction_error_input_B": metrics_a["relative_reconstruction_error"],
            "relative_reconstruction_error_input_B_v2": metrics_b["relative_reconstruction_error"],
            "U_diff_fro": comparison["raw_factor_differences_fro"]["U"],
            "G_diff_fro": comparison["raw_factor_differences_fro"]["G"],
            "V_diff_fro": comparison["raw_factor_differences_fro"]["V"],
            "G_eigenvalues_input_B": np.asarray(metrics_a["spectra"]["G_eigenvalues"], dtype=np.float64),
            "G_eigenvalues_input_B_v2": np.asarray(metrics_b["spectra"]["G_eigenvalues"], dtype=np.float64),
            "UtU_eigenvalues_input_B": np.asarray(metrics_a["spectra"]["UtU_eigenvalues"], dtype=np.float64),
            "UtU_eigenvalues_input_B_v2": np.asarray(metrics_b["spectra"]["UtU_eigenvalues"], dtype=np.float64),
            "VVt_eigenvalues_input_B": np.asarray(metrics_a["spectra"]["VVt_eigenvalues"], dtype=np.float64),
            "VVt_eigenvalues_input_B_v2": np.asarray(metrics_b["spectra"]["VVt_eigenvalues"], dtype=np.float64),
            "U_singular_values_input_B": np.asarray(metrics_a["spectra"]["U_singular_values"], dtype=np.float64),
            "U_singular_values_input_B_v2": np.asarray(metrics_b["spectra"]["U_singular_values"], dtype=np.float64),
            "G_singular_values_input_B": np.asarray(metrics_a["spectra"]["G_singular_values"], dtype=np.float64),
            "G_singular_values_input_B_v2": np.asarray(metrics_b["spectra"]["G_singular_values"], dtype=np.float64),
            "V_singular_values_input_B": np.asarray(metrics_a["spectra"]["V_singular_values"], dtype=np.float64),
            "V_singular_values_input_B_v2": np.asarray(metrics_b["spectra"]["V_singular_values"], dtype=np.float64),
        },
    )

    print(f"summary_json {summary_json_path}")
    print(f"summary_mat {summary_mat_path}")


if __name__ == "__main__":
    main()
