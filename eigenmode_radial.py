import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def _finalize_figure(fig, save_path=None, save=True, show=True):
    if save and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"saved radial eigenmode plot at {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _ensure_file_name(param):
    if param.file_name != "":
        return

    if param.basis == "bessel":
        basis = "b"
    elif param.basis == "hermite":
        basis = "h"
    else:
        basis = param.basis

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    param.file_name = (
        f"{date}_n{param.n_start}-{param.n_end}"
        f"_dn{param.n_delta}"
        f"_m{param.m}_p{param.p}_{basis}"
    )


def _get_f_blocks(solve_data):
    if "F_block_final_state" in solve_data:
        return solve_data["F_block_final_state"]
    if "F_blocked" in solve_data:
        return solve_data["F_blocked"]
    raise KeyError("solve_data must contain F_block_final_state or F_blocked.")


def _select_phi_coefficients(solve_data, block_index, radial_mode_count):
    f_blocks = _get_f_blocks(solve_data)
    most_unstable_mode_indexes = solve_data.get("most_unstable_mode_indexes")
    F_block = np.asarray(f_blocks[block_index])

    if most_unstable_mode_indexes is None:
        F = F_block
    else:
        most_unstable_mode_index = int(most_unstable_mode_indexes[block_index])
        F = F_block[:, most_unstable_mode_index] if F_block.ndim == 2 else F_block

    if F.ndim != 1:
        raise ValueError(f"Expected a 1D mode coefficient vector, got shape {F.shape}.")
    if len(F) < radial_mode_count:
        raise ValueError(
            f"Mode coefficient vector is shorter than the radial mode count: "
            f"{len(F)} < {radial_mode_count}."
        )

    return F[:radial_mode_count]


def plot_eigenmode_radial(param, profiles, mode_data, mat_data, solve_data, save=True, show=True):
    W = mat_data["W"]
    rs = profiles["rs"]

    n_values = solve_data["n_values"]
    n_mode_indexes = solve_data["n_mode_indexes"]
    n_count = len(n_values)

    ncols = min(3, n_count)
    nrows = int(np.ceil(n_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, n in enumerate(n_values):
        idx = n_mode_indexes[n]
        phi_k = _select_phi_coefficients(solve_data, i, len(idx))
        Wk = W[idx]

        radial_phi = np.sum(phi_k[:, None] * Wk, axis=0)
        amplitude = np.abs(radial_phi)

        ax = axes_flat[i]
        ax.plot(rs, amplitude)
        ax.set_xlabel("r")
        ax.set_ylabel("amplitude")
        ax.set_title(f"n={n} radial eigenmode")
        ax.grid(True)

    for ax in axes_flat[n_count:]:
        ax.set_visible(False)

    fig.suptitle(f"Radial eigenmode amplitude ({param.basis})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f"{param.file_name}_eigenmode_radial.png"
    save_path = os.path.join(param.save_dir, filename)
    _finalize_figure(fig, save_path=save_path, save=save, show=show)


def main():
    from utils import parse_params
    from parameters import build_profiles
    from modes import build_modes
    from matrices import build_matrices
    from solve import construct_A_matrix

    param = parse_params()
    _ensure_file_name(param)
    print(f"parameter: {param}")

    profiles = build_profiles(param)
    mode_data = build_modes(param, profiles)
    mat_data = build_matrices(param, profiles, mode_data)
    matrix = construct_A_matrix(mode_data, mat_data)

    if param.method == "eigenproblem":
        from solve import solve_eigenvalue_problem

        solve_data = solve_eigenvalue_problem(matrix)
    elif param.method == "time_evolution":
        from solve import solve_time_evolution

        solve_data = solve_time_evolution(param, matrix)
    else:
        raise ValueError(f"Unknown method: {param.method}")

    plot_eigenmode_radial(param, profiles, mode_data, mat_data, solve_data, save=True, show=True)


if __name__ == "__main__":
    main()
