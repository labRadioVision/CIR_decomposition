from pathlib import Path

import scipy.io
import torch

from deep_matrix_factorization import factorize_from_mat


def main() -> None:
    input_path = Path(r"C:\Users\STEFANOSAVAZZI\Desktop\input_B.mat")
    output_dir = Path(r"C:\Users\STEFANOSAVAZZI\Desktop")
    workspace_tmp = Path(r"C:\Users\STEFANOSAVAZZI\Desktop\CIR_decomposition\.codex_tmp_window_factorization")
    workspace_tmp.mkdir(parents=True, exist_ok=True)

    source_key = "CIR_linear"
    factorizer_input_key = "B"
    window_size = 300
    step = 300
    max_cols = 2100
    rank = 2

    mat = scipy.io.loadmat(input_path)
    B_full = mat[source_key]
    B = B_full[:, :max_cols]

    num_windows = 1 + (B.shape[1] - window_size) // step

    final_losses = []
    relative_errors = []
    window_ranges = []

    for idx in range(num_windows):
        start = idx * step
        end = start + window_size
        B_win = B[:, start:end]

        temp_input = workspace_tmp / f"window_{idx + 1:02d}_input.mat"
        output_path = output_dir / f"factors_out_window_{idx + 1:02d}.mat"

        scipy.io.savemat(temp_input, {factorizer_input_key: B_win})

        result = factorize_from_mat(
            input_mat_path=str(temp_input),
            output_mat_path=str(output_path),
            model="ugv",
            rank=rank,
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
        saved["source_variable"] = source_key
        scipy.io.savemat(output_path, saved)

        final_losses.append(result.loss_history[-1] if result.loss_history else float("nan"))
        relative_errors.append(rel_err)
        window_ranges.append((start + 1, end))

        print(
            f"window {idx + 1}/{num_windows}: "
            f"cols {start + 1}-{end}, rel_err={rel_err:.6f}, output={output_path.name}"
        )

    aggregate_payload = {
        "source_variable": source_key,
        "source_shape": B_full.shape,
        "truncated_shape": B.shape,
        "window_size": window_size,
        "step": step,
        "num_windows": num_windows,
        "rank": rank,
        "model": "ugv",
        "constraint": "projected_nonnegative",
        "truncated_matrix": B,
        "window_ranges_1based": window_ranges,
        "relative_errors": relative_errors,
        "final_losses": final_losses,
    }

    for idx in range(num_windows):
        output_path = output_dir / f"factors_out_window_{idx + 1:02d}.mat"
        window_mat = scipy.io.loadmat(output_path)
        aggregate_payload[f"H1_window_{idx + 1:02d}"] = window_mat["H_1"]
        aggregate_payload[f"H2_window_{idx + 1:02d}"] = window_mat["H_2"]
        aggregate_payload[f"H3_window_{idx + 1:02d}"] = window_mat["H_3"]
        aggregate_payload[f"B_reconstruction_window_{idx + 1:02d}"] = window_mat["B_reconstruction"]

    aggregate_path = output_dir / "factors_out_windows_ugv_rank2.mat"
    scipy.io.savemat(aggregate_path, aggregate_payload)
    print(f"aggregate_output {aggregate_path}")


if __name__ == "__main__":
    main()
