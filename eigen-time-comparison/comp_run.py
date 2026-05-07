from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_method_comparison(param):
    from matrices import build_matrices
    from modes import build_modes
    from parameters import build_profiles
    from solve import construct_A_matrix, solve_eigenvalue_problem, solve_time_evolution

    profiles = build_profiles(param)
    mode_data = build_modes(param, profiles)
    mat_data = build_matrices(param, profiles, mode_data)
    matrix = construct_A_matrix(mode_data, mat_data)

    time_param = replace(param, method="time_evolution")

    eigen_data = solve_eigenvalue_problem(matrix)
    time_data = solve_time_evolution(time_param, matrix)

    return profiles, eigen_data, time_data


def _k_thetas_rho_i(param, n_values):
    q_val = 1.4
    r_val = 0.5
    return n_values * q_val / r_val * param.rhos0


def _normalized_eigenvalue_data(param, profiles, solve_data):
    gamma_factor = float(profiles["R_Lne"]) / param.rmajor
    omega_factor = gamma_factor * 4

    return {
        "n_values": np.asarray(solve_data["n_values"], dtype=int),
        "gammas": np.asarray(solve_data["gammas"], dtype=float) / gamma_factor,
        "omegas": np.asarray(solve_data["omegas"], dtype=float) / omega_factor,
    }


def plot_method_comparison(param, profiles, eigen_data, time_data, save_path=None, show=True):
    eigen = _normalized_eigenvalue_data(param, profiles, eigen_data)
    time = _normalized_eigenvalue_data(param, profiles, time_data)

    if not np.array_equal(eigen["n_values"], time["n_values"]):
        raise ValueError("eigenproblem and time_evolution results have different n_values.")

    k_thetas_rho_i = _k_thetas_rho_i(param, eigen["n_values"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    axes[0].plot(k_thetas_rho_i, eigen["gammas"], "o-", label="eigenproblem")
    axes[0].plot(k_thetas_rho_i, time["gammas"], "s--", label="time_evolution")
    axes[0].set_xlabel(r"$k_{\theta} \rho_i$")
    axes[0].set_ylabel("Growth rate")
    axes[0].set_title("Growth rate comparison")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(k_thetas_rho_i, eigen["omegas"], "o-", label="eigenproblem")
    axes[1].plot(k_thetas_rho_i, time["omegas"], "s--", label="time_evolution")
    axes[1].set_xlabel(r"$k_{\theta} \rho_i$")
    axes[1].set_ylabel("Frequency/4")
    axes[1].set_title("Frequency comparison")
    axes[1].grid(True)
    axes[1].legend()

    text = (
        f"basis: {param.basis}\n"
        f"n={param.n_start}:{param.n_delta}:{param.n_end}, m<= {param.m}, p< {param.p}\n"
        f"dt={param.dt}, T={param.T}, F0={param.F0}"
    )
    fig.text(
        0.5,
        0.01,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
    fig.suptitle("Eigenproblem vs Time Evolution", fontsize=14)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"comparison figure saved at {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return {
        "k_thetas_rho_i": k_thetas_rho_i,
        "eigenproblem_gammas": eigen["gammas"],
        "time_evolution_gammas": time["gammas"],
        "eigenproblem_omegas": eigen["omegas"],
        "time_evolution_omegas": time["omegas"],
    }


def _comparison_file_name(param):
    basis = "b" if param.basis == "bessel" else "h"
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{param.suffix}" if param.suffix else ""
    return (
        f"{date}_n{param.n_start}-{param.n_end}"
        f"_dn{param.n_delta}_m{param.m}_p{param.p}_{basis}"
        f"{suffix}_method_comparison.png"
    )


def main():
    from utils import parse_params

    param = parse_params()
    profiles, eigen_data, time_data = run_method_comparison(param)

    save_path = RESULTS_DIR / _comparison_file_name(param)
    plot_method_comparison(param, profiles, eigen_data, time_data, save_path=save_path, show=True)


if __name__ == "__main__":
    main()
