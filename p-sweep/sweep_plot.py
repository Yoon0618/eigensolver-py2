import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


P_VALUES = [10, 20, 30, 40, 50]
M_VALUE = 50
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_PATH = SCRIPT_DIR / "p_sweep_growth_rates.png"


def _latest_eigenvalues_path(p, m_value=M_VALUE, results_dir=RESULTS_DIR):
    pattern = f"*_m{m_value}_p{p}_*_eigenvalues.npz"
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No eigenvalues npz found for m={m_value}, p={p}: {results_dir / pattern}")
    return paths[-1]


def _load_growth_rate_data(path):
    with np.load(path, allow_pickle=True) as data:
        k_thetas_rho_i = np.asarray(data["k_thetas_rho_i"], dtype=float)
        warning_category = getattr(getattr(np, "exceptions", np), "VisibleDeprecationWarning", Warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", warning_category)
            gammas = np.asarray(data["gammas"], dtype=float)
        return k_thetas_rho_i, gammas


def plot_p_sweep(p_values=P_VALUES, m_value=M_VALUE, results_dir=RESULTS_DIR, save_path=OUTPUT_PATH, show=True):
    fig, ax = plt.subplots(figsize=(10, 6))

    for p in p_values:
        path = _latest_eigenvalues_path(p, m_value=m_value, results_dir=results_dir)
        k_thetas_rho_i, gammas = _load_growth_rate_data(path)
        ax.plot(k_thetas_rho_i, gammas, "o-", label=f"p={p}")

    ax.set_xlabel(r"$k_{\theta} \rho_i$")
    ax.set_ylabel("Growth rate")
    ax.set_title(f"Growth rate vs k_theta_rho_i for p sweep (m={m_value})")
    ax.grid(True)
    ax.legend()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    plot_p_sweep()


if __name__ == "__main__":
    main()
