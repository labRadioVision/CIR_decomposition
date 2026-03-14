import argparse
import json
from pathlib import Path

import scipy.io
import torch

from deep_matrix_factorization import factorize_from_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brute-force rank sweep for overlapping-window consistency.")
    parser.add_argument(
        "--input-mat",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f\input_B.mat"),
        help="Path to the source MAT file.",
    )
    parser.add_argument("--source-key", type=str, default="CIR_linear")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--step", type=int, default=150)
    parser.add_argument("--max-cols", type=int, default=2100)
    parser.add_argument("--rank-min", type=int, default=2)
    parser.add_argument("--rank-max", type=int, default=21)
    parser.add_argument(
        "--workspace-tmp",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\CIR_decomposition\.codex_tmp_rank_sweep"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\rank_sweep_overlap_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.workspace_tmp.mkdir(parents=True, exist_ok=True)

    data = scipy.io.loadmat(args.input_mat)
    B = data[args.source_key][:, : args.max_cols]

    if args.step <= 0 or args.window_size <= 0:
        raise ValueError("window-size and step must be positive.")
    if args.step >= args.window_size:
        raise ValueError("This sweep is intended for overlapping windows, so require step < window-size.")

    num_windows = 1 + (B.shape[1] - args.window_size) // args.step
    overlap = args.window_size - args.step
    pair_count = num_windows - 1

    summary = {
        "input_mat": str(args.input_mat),
        "source_key": args.source_key,
        "truncated_shape": list(B.shape),
        "window_size": args.window_size,
        "step": args.step,
        "num_windows": num_windows,
        "overlap_width": overlap,
        "rank_results": [],
    }

    per_pair_best = [
        {"pair": [idx + 1, idx + 2], "best_rank": None, "best_disagreement": None}
        for idx in range(pair_count)
    ]

    best_overall_rank = None
    best_overall_mean = None

    for rank in range(args.rank_min, args.rank_max + 1):
        window_recons = []
        window_rel_errs = []

        for idx in range(num_windows):
            start = idx * args.step
            end = start + args.window_size
            B_win = B[:, start:end]
            temp_input = args.workspace_tmp / f"rank_{rank:02d}_window_{idx + 1:02d}_input.mat"
            temp_output = args.workspace_tmp / f"rank_{rank:02d}_window_{idx + 1:02d}_output.mat"
            scipy.io.savemat(temp_input, {"B": B_win})

            result = factorize_from_mat(
                input_mat_path=str(temp_input),
                output_mat_path=str(temp_output),
                model="ugv",
                rank=rank,
                input_key="B",
                constraint="projected_nonnegative",
                lr=0.03,
                max_steps=3000,
                optimizer="adam",
            )

            target = torch.as_tensor(B_win, dtype=result.reconstruction.dtype)
            denom = torch.linalg.norm(target)
            rel_err = torch.linalg.norm(result.reconstruction - target)
            rel_err = float(rel_err / denom) if float(denom) > 0 else 0.0
            window_rel_errs.append(rel_err)
            window_recons.append(result.reconstruction.detach().cpu())

        overlap_disagreements = []
        for idx in range(pair_count):
            left = window_recons[idx][:, args.step : args.window_size]
            right = window_recons[idx + 1][:, :overlap]
            true_overlap = torch.as_tensor(
                B[:, (idx + 1) * args.step : (idx + 1) * args.step + overlap],
                dtype=left.dtype,
            )
            true_norm = torch.linalg.norm(true_overlap)
            disagreement = float(torch.linalg.norm(left - right) / true_norm) if float(true_norm) > 0 else 0.0
            overlap_disagreements.append(disagreement)

            pair_best = per_pair_best[idx]
            if pair_best["best_disagreement"] is None or disagreement < pair_best["best_disagreement"]:
                pair_best["best_disagreement"] = disagreement
                pair_best["best_rank"] = rank

        mean_disagreement = sum(overlap_disagreements) / len(overlap_disagreements)
        rank_result = {
            "rank": rank,
            "mean_window_rel_err": sum(window_rel_errs) / len(window_rel_errs),
            "max_window_rel_err": max(window_rel_errs),
            "min_overlap_disagreement": min(overlap_disagreements),
            "max_overlap_disagreement": max(overlap_disagreements),
            "mean_overlap_disagreement": mean_disagreement,
            "overlap_disagreements": overlap_disagreements,
        }
        summary["rank_results"].append(rank_result)

        if best_overall_mean is None or mean_disagreement < best_overall_mean:
            best_overall_mean = mean_disagreement
            best_overall_rank = rank

        print(
            f"rank={rank:2d}  mean_overlap={mean_disagreement:.6f}  "
            f"min_overlap={min(overlap_disagreements):.6f}  "
            f"max_overlap={max(overlap_disagreements):.6f}"
        )

    summary["best_overall_rank"] = best_overall_rank
    summary["best_overall_mean_overlap_disagreement"] = best_overall_mean
    summary["best_rank_per_overlap_pair"] = per_pair_best

    args.summary_json.write_text(json.dumps(summary, indent=2))
    print(f"summary_json {args.summary_json}")


if __name__ == "__main__":
    main()
