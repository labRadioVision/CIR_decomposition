import argparse
from pathlib import Path

import scipy.io
import torch

from deep_matrix_factorization import factorize_from_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run windowed matrix factorization over a MAT variable.")
    parser.add_argument(
        "--input-mat",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\f\input_B.mat"),
        help="Path to the source MAT file.",
    )
    parser.add_argument(
        "--source-key",
        type=str,
        default="CIR_linear",
        help="Variable name to read from the source MAT file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop"),
        help="Folder where per-window and aggregate MAT files will be saved.",
    )
    parser.add_argument(
        "--workspace-tmp",
        type=Path,
        default=Path(r"C:\Users\STEFANOSAVAZZI\Desktop\CIR_decomposition\.codex_tmp_window_factorization"),
        help="Workspace folder used for temporary per-window input MAT files.",
    )
    parser.add_argument("--window-size", type=int, default=300, help="Window width along columns.")
    parser.add_argument("--step", type=int, default=300, help="Step between consecutive windows.")
    parser.add_argument("--max-cols", type=int, default=2100, help="Use only the first max-cols columns.")
    parser.add_argument("--rank", type=int, default=7, help="UGV rank. Defaults to 7.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="factors_out",
        help="Prefix used for per-window output filenames.",
    )
    parser.add_argument(
        "--aggregate-name",
        type=str,
        default="factors_out_windows_ugv_rank7.mat",
        help="Filename for the aggregate MAT output.",
    )
    parser.add_argument(
        "--compute-overlap-metrics",
        action="store_true",
        help="Compute consistency metrics on overlapping regions between adjacent windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.workspace_tmp.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    factorizer_input_key = "B"

    mat = scipy.io.loadmat(args.input_mat)
    B_full = mat[args.source_key]
    B = B_full[:, : args.max_cols]

    if B.shape[1] < args.window_size:
        raise ValueError(
            f"window_size={args.window_size} exceeds truncated column count {B.shape[1]}."
        )
    if args.step <= 0:
        raise ValueError(f"step must be positive, got {args.step}.")

    num_windows = 1 + (B.shape[1] - args.window_size) // args.step

    final_losses = []
    relative_errors = []
    window_ranges = []
    window_recons = []

    for idx in range(num_windows):
        start = idx * args.step
        end = start + args.window_size
        B_win = B[:, start:end]

        temp_input = args.workspace_tmp / f"window_{idx + 1:02d}_input.mat"
        output_path = args.output_dir / f"{args.output_prefix}_window_{idx + 1:02d}.mat"

        scipy.io.savemat(temp_input, {factorizer_input_key: B_win})

        result = factorize_from_mat(
            input_mat_path=str(temp_input),
            output_mat_path=str(output_path),
            model="ugv",
            rank=args.rank,
            input_key=factorizer_input_key,
            constraint="projected_nonnegative",
            lr=0.03,
            max_steps=3000,
            optimizer="adam",
        )

        target = torch.as_tensor(B_win, dtype=result.reconstruction.dtype)
        denom = torch.linalg.norm(target)
        rel_err = torch.linalg.norm(result.reconstruction - target)
        rel_err = float(rel_err / denom) if float(denom) > 0 else 0.0

        saved = scipy.io.loadmat(output_path)
        saved["window_start_col_1based"] = start + 1
        saved["window_end_col_1based"] = end
        saved["window_index"] = idx + 1
        saved["relative_error"] = rel_err
        saved["source_variable"] = args.source_key
        scipy.io.savemat(output_path, saved)

        final_losses.append(result.loss_history[-1] if result.loss_history else float("nan"))
        relative_errors.append(rel_err)
        window_ranges.append((start + 1, end))
        window_recons.append(result.reconstruction.detach().cpu())

        print(
            f"window {idx + 1}/{num_windows}: "
            f"cols {start + 1}-{end}, rel_err={rel_err:.6f}, output={output_path.name}"
        )

    aggregate_payload = {
        "source_variable": args.source_key,
        "source_shape": B_full.shape,
        "truncated_shape": B.shape,
        "window_size": args.window_size,
        "step": args.step,
        "num_windows": num_windows,
        "rank": args.rank,
        "model": "ugv",
        "constraint": "projected_nonnegative",
        "truncated_matrix": B,
        "window_ranges_1based": window_ranges,
        "relative_errors": relative_errors,
        "final_losses": final_losses,
    }

    for idx in range(num_windows):
        output_path = args.output_dir / f"{args.output_prefix}_window_{idx + 1:02d}.mat"
        window_mat = scipy.io.loadmat(output_path)
        aggregate_payload[f"H1_window_{idx + 1:02d}"] = window_mat["H_1"]
        aggregate_payload[f"H2_window_{idx + 1:02d}"] = window_mat["H_2"]
        aggregate_payload[f"H3_window_{idx + 1:02d}"] = window_mat["H_3"]
        aggregate_payload[f"B_reconstruction_window_{idx + 1:02d}"] = window_mat["B_reconstruction"]

    if args.compute_overlap_metrics and args.step < args.window_size:
        overlap = args.window_size - args.step
        overlap_disagreement = []
        left_overlap_rel_err = []
        right_overlap_rel_err = []
        overlap_pairs = []

        for idx in range(num_windows - 1):
            left = window_recons[idx][:, args.step : args.window_size]
            right = window_recons[idx + 1][:, :overlap]
            true_overlap = torch.as_tensor(
                B[:, (idx + 1) * args.step : (idx + 1) * args.step + overlap],
                dtype=left.dtype,
            )

            true_norm = torch.linalg.norm(true_overlap)
            if float(true_norm) > 0:
                disagreement = float(torch.linalg.norm(left - right) / true_norm)
                left_err = float(torch.linalg.norm(left - true_overlap) / true_norm)
                right_err = float(torch.linalg.norm(right - true_overlap) / true_norm)
            else:
                disagreement = 0.0
                left_err = 0.0
                right_err = 0.0

            overlap_disagreement.append(disagreement)
            left_overlap_rel_err.append(left_err)
            right_overlap_rel_err.append(right_err)
            overlap_pairs.append((idx + 1, idx + 2))

        aggregate_payload["overlap_pairs"] = overlap_pairs
        aggregate_payload["overlap_disagreement_rel"] = overlap_disagreement
        aggregate_payload["left_overlap_rel_err"] = left_overlap_rel_err
        aggregate_payload["right_overlap_rel_err"] = right_overlap_rel_err

        print(f"overlap_disagreement_mean {sum(overlap_disagreement) / len(overlap_disagreement):.6f}")
        print(f"overlap_disagreement_max {max(overlap_disagreement):.6f}")

    aggregate_path = args.output_dir / args.aggregate_name
    scipy.io.savemat(aggregate_path, aggregate_payload)
    print(f"aggregate_output {aggregate_path}")


if __name__ == "__main__":
    main()
