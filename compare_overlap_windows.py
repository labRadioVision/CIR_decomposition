import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io
import torch

from deep_matrix_factorization import build_model_for_matrix

_BASE = Path(r"C:\Users\STEFANOSAVAZZI\Desktop\CIR_decomposition")
_DEFAULT_INPUTS = {
    "metal": _BASE / "CIR_20260209T162335_Pit5x10_NS21x1_RF18001_RF28001_MetalCyl_LinOnly.mat",
    "paper": _BASE / "CIR_20260209T163709_Pit5x10_NS21x1_RF18001_RF28001_PaperCyl_LinOnly.mat",
    "ptfe":  _BASE / "CIR_20260209T165435_Pit5x10_NS21x1_RF18001_RF28001_PTFECyl_LinOnly.mat",
}
_LABELS = list(_DEFAULT_INPUTS.keys())
_PAIRS = list(combinations(_LABELS, 2))  # (metal,paper), (metal,ptfe), (paper,ptfe)


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
            "U_singular_values": sorted_singular_values(U),
            "G_singular_values": sorted_singular_values(G),
            "V_singular_values": sorted_singular_values(V),
        },
    }


def pair_comparison(fa: dict, fb: dict, label_a: str, label_b: str) -> dict:
    """Compute all pairwise metrics between two factorization results."""
    spectral = {
        "U_singular_values": vector_l2_diff(
            fa["spectra"]["U_singular_values"], fb["spectra"]["U_singular_values"]
        ),
        "G_singular_values": vector_l2_diff(
            fa["spectra"]["G_singular_values"], fb["spectra"]["G_singular_values"]
        ),
        "V_singular_values": vector_l2_diff(
            fa["spectra"]["V_singular_values"], fb["spectra"]["V_singular_values"]
        ),
    }
    combined_sv_dist = sum(spectral.values())

    return {
        "pair": f"{label_a}_vs_{label_b}",
        "relative_reconstruction_error": {
            label_a: fa["rel_err"],
            label_b: fb["rel_err"],
        },
        "raw_factor_differences_fro": {
            "U": float(np.linalg.norm(fa["U"] - fb["U"])),
            "G": float(np.linalg.norm(fa["G"] - fb["G"])),
            "V": float(np.linalg.norm(fa["V"] - fb["V"])),
        },
        "spectral_distance_l2": spectral,
        "combined_singular_distance": combined_sv_dist,
        "leading_values": {
            label_a: {
                "U_sv1": fa["spectra"]["U_singular_values"][0],
                "G_sv1": fa["spectra"]["G_singular_values"][0],
                "V_sv1": fa["spectra"]["V_singular_values"][0],
            },
            label_b: {
                "U_sv1": fb["spectra"]["U_singular_values"][0],
                "G_sv1": fb["spectra"]["G_singular_values"][0],
                "V_sv1": fb["spectra"]["V_singular_values"][0],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare windowed UGV factorizations across three MAT files (metal, paper, PTFE)."
    )
    parser.add_argument("--input-metal", type=Path, default=_DEFAULT_INPUTS["metal"])
    parser.add_argument("--input-paper", type=Path, default=_DEFAULT_INPUTS["paper"])
    parser.add_argument("--input-ptfe",  type=Path, default=_DEFAULT_INPUTS["ptfe"])
    parser.add_argument("--input-key", type=str, default="CIR_linear")
    parser.add_argument("--output-dir", type=Path, default=_BASE / "overlap_window_results")
    parser.add_argument("--window-size", type=int, default=300,
                        help="Window width in number of columns (travel-distance samples).")
    parser.add_argument("--step", type=int, default=150,
                        help="Step between consecutive windows in number of columns.")
    parser.add_argument("--max-cols", type=int, default=2100,
                        help="Truncate each matrix to this many columns before windowing.")
    parser.add_argument("--rank", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load matrices and shared axes ---
    paths = {"metal": args.input_metal, "paper": args.input_paper, "ptfe": args.input_ptfe}
    matrices: dict[str, np.ndarray] = {}
    travel_distances: np.ndarray | None = None
    material_offset: np.ndarray | None = None

    for label, path in paths.items():
        raw = scipy.io.loadmat(path)
        matrices[label] = np.asarray(raw[args.input_key], dtype=np.float64)[:, : args.max_cols]
        if travel_distances is None:
            travel_distances = np.asarray(raw["travel_distances"], dtype=np.float64).flatten()[: args.max_cols]
            material_offset = np.asarray(raw["material_offset"], dtype=np.float64).flatten()

    shapes = {lbl: m.shape for lbl, m in matrices.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Input shapes do not match across files: {shapes}")

    n_rows, n_cols = next(iter(matrices.values())).shape

    if args.step <= 0 or args.window_size <= 0:
        raise ValueError("window-size and step must be positive.")
    if n_cols < args.window_size:
        raise ValueError("window-size exceeds the truncated column count.")

    num_windows = 1 + (n_cols - args.window_size) // args.step
    window_results = []

    for idx in range(num_windows):
        start = idx * args.step
        end = start + args.window_size
        td_start = float(travel_distances[start])
        td_end = float(travel_distances[end - 1])

        # Factorize each material's window
        facts: dict[str, dict] = {}
        for label, mat in matrices.items():
            facts[label] = factorize_window(mat[:, start:end], rank=args.rank)

        # Pairwise comparisons
        pairwise = []
        for la, lb in _PAIRS:
            # raw input difference
            win_a = matrices[la][:, start:end]
            win_b = matrices[lb][:, start:end]
            raw_rel_diff = (
                float(np.linalg.norm(win_a - win_b) / np.linalg.norm(win_a))
                if np.linalg.norm(win_a) > 0
                else 0.0
            )
            cmp = pair_comparison(facts[la], facts[lb], la, lb)
            cmp["raw_input_relative_difference"] = raw_rel_diff
            pairwise.append(cmp)

        # Mean combined_singular_distance across all pairs as a single scalar per window
        mean_combined_sv_dist = float(np.mean([c["combined_singular_distance"] for c in pairwise]))

        window_entry = {
            "window_index": idx + 1,
            "col_range": [start, end - 1],
            "travel_distance_range_mm": [td_start, td_end],
            "pairwise": pairwise,
            "mean_combined_singular_distance": mean_combined_sv_dist,
            "reconstruction_errors": {lbl: facts[lbl]["rel_err"] for lbl in _LABELS},
        }
        window_results.append(window_entry)

        pair_strs = "  ".join(
            f"{c['pair']}={c['combined_singular_distance']:.4f}" for c in pairwise
        )
        print(
            f"window {idx + 1:02d}  {td_start:.2f}-{td_end:.2f} mm  "
            f"mean_sv_dist={mean_combined_sv_dist:.6f}  [{pair_strs}]"
        )

    strongest = max(window_results, key=lambda x: x["mean_combined_singular_distance"])
    weakest = min(window_results, key=lambda x: x["mean_combined_singular_distance"])

    summary = {
        "rank": args.rank,
        "window_size_cols": args.window_size,
        "step_cols": args.step,
        "max_cols": args.max_cols,
        "num_windows": num_windows,
        "material_offset_cm": material_offset.tolist(),
        "travel_distances_mm_range": [float(travel_distances[0]), float(travel_distances[-1])],
        "strongest_window_by_mean_combined_singular_distance": strongest,
        "weakest_window_by_mean_combined_singular_distance": weakest,
        "window_results": window_results,
    }

    json_path = args.output_dir / f"overlap_window_comparison_rank{args.rank}.json"
    mat_path = args.output_dir / f"overlap_window_comparison_rank{args.rank}.mat"

    json_path.write_text(json.dumps(summary, indent=2))

    # Build per-pair arrays for .mat output
    mat_data: dict = {
        "rank": args.rank,
        "window_size_cols": args.window_size,
        "step_cols": args.step,
        "max_cols": args.max_cols,
        "material_offset_cm": material_offset,
        "travel_distance_start_mm": np.array(
            [w["travel_distance_range_mm"][0] for w in window_results], dtype=np.float64
        ),
        "travel_distance_end_mm": np.array(
            [w["travel_distance_range_mm"][1] for w in window_results], dtype=np.float64
        ),
        "mean_combined_singular_distance": np.array(
            [w["mean_combined_singular_distance"] for w in window_results], dtype=np.float64
        ),
    }
    for la, lb in _PAIRS:
        key = f"{la}_vs_{lb}"
        pair_entries = [
            next(c for c in w["pairwise"] if c["pair"] == key) for w in window_results
        ]
        mat_data[f"combined_sv_dist_{key}"] = np.array(
            [c["combined_singular_distance"] for c in pair_entries], dtype=np.float64
        )
        mat_data[f"raw_rel_diff_{key}"] = np.array(
            [c["raw_input_relative_difference"] for c in pair_entries], dtype=np.float64
        )

    scipy.io.savemat(mat_path, mat_data)
    print(f"summary_json {json_path}")
    print(f"summary_mat  {mat_path}")


if __name__ == "__main__":
    main()
