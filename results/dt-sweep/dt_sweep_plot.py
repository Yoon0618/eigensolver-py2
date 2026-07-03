"""Plot growth-rate diagnostics for dt-sweep results in this directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARISON_OUTPUT = SCRIPT_DIR / "dt_sweep_growth_rate_comparison.png"
DEFAULT_DIAGNOSTICS_OUTPUT = SCRIPT_DIR / "dt_sweep_convergence_diagnostics.png"
FIGURE_FACE_COLOR = "#FAFAFA"
AXES_FACE_COLOR = "white"


def style_figure(fig, axes) -> None:
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    for ax in np.ravel(np.atleast_1d(axes)):
        ax.set_facecolor(AXES_FACE_COLOR)


def load_runs(results_dir: Path, max_dt_exclusive: float | None) -> list[dict]:
    runs = []

    for params_path in sorted(results_dir.glob("*_params.json")):
        npz_path = params_path.with_name(
            params_path.name.replace("_params.json", "_eigenvalues.npz")
        )
        if not npz_path.exists():
            continue

        with params_path.open(encoding="utf-8") as f:
            params = json.load(f)

        if params.get("method") != "time_evolution":
            continue

        dt = float(params["dt"])
        if max_dt_exclusive is not None and dt >= max_dt_exclusive:
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            k = np.asarray(data["k_thetas_rho_i"], dtype=float)
            gamma = np.asarray(data["gammas"], dtype=float)

            if "n_values" in data.files:
                n_values = np.asarray(data["n_values"], dtype=int)
            else:
                n_values = np.arange(
                    params["n_start"],
                    params["n_end"] + 1,
                    params["n_delta"],
                    dtype=int,
                )

        runs.append(
            {
                "dt": dt,
                "params": params,
                "params_path": params_path,
                "npz_path": npz_path,
                "k": k,
                "gamma": gamma,
                "n_values": n_values,
            }
        )

    runs.sort(key=lambda run: run["dt"])
    return runs


def validate_runs(runs: list[dict], reference: dict) -> None:
    if not runs:
        raise ValueError("No dt-sweep runs found.")

    for run in runs:
        if not np.allclose(run["k"], reference["k"]):
            raise ValueError(
                f"k grid mismatch: {run['npz_path'].name} differs from "
                f"{reference['npz_path'].name}"
            )
        if len(run["gamma"]) != len(reference["gamma"]):
            raise ValueError(
                f"gamma length mismatch: {run['npz_path'].name} differs from "
                f"{reference['npz_path'].name}"
            )


def plot_comparison(runs: list[dict], output_path: Path) -> None:
    reference = runs[0]
    validate_runs(runs, reference)

    fig, (ax_growth, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15]},
    )
    style_figure(fig, [ax_growth, ax_diff])

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, run in enumerate(runs):
        label = f"dt={run['dt']:g}"
        if run is reference:
            label += " (reference)"

        ax_growth.plot(
            run["k"],
            run["gamma"],
            marker=markers[i % len(markers)],
            lw=2,
            ms=5,
            label=label,
            color=colors[i],
        )

        if run is not reference:
            ax_diff.plot(
                run["k"],
                run["gamma"] - reference["gamma"],
                marker=markers[i % len(markers)],
                lw=1.6,
                ms=4,
                label=f"dt={run['dt']:g} - dt={reference['dt']:g}",
                color=colors[i],
            )

    params = reference["params"]
    ax_growth.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    ax_growth.set_title(
        "dt sweep growth rate comparison\n"
        f"time evolution, n={params['n_start']}:{params['n_delta']}:{params['n_end']}, "
        f"m={params['m']}, p={params['p']}, {params['basis']} basis, T={params['T']}"
    )
    ax_growth.set_ylabel("Normalized growth rate")
    ax_growth.grid(True, alpha=0.32)
    ax_growth.legend(loc="best", fontsize=9)

    ax_top = ax_growth.secondary_xaxis("top")
    ax_top.set_xticks(reference["k"])
    ax_top.set_xticklabels([str(int(n)) for n in reference["n_values"]], fontsize=8)
    ax_top.set_xlabel("n")

    ax_diff.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    ax_diff.set_xlabel(r"$k_{\theta}\rho_i$")
    ax_diff.set_ylabel("Diff vs ref")
    ax_diff.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_diff.grid(True, alpha=0.32)
    ax_diff.legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_diagnostics(runs: list[dict], output_path: Path) -> None:
    reference = runs[0]
    validate_runs(runs, reference)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 9.2),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 2.2, 1.7]},
    )
    style_figure(fig, axes)
    ax_growth, ax_diff, ax_error = axes

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]
    markers = ["o", "s", "^", "D", "v", "P"]

    ax_growth.plot(
        reference["k"],
        reference["gamma"],
        "o-",
        color="black",
        lw=2.2,
        ms=5,
        label=f"reference dt={reference['dt']:g}",
    )

    for i, run in enumerate(runs):
        if run is reference:
            continue

        ax_growth.plot(
            run["k"],
            run["gamma"],
            marker=markers[i % len(markers)],
            lw=1.2,
            ms=4,
            alpha=0.6,
            label=f"dt={run['dt']:g}",
            color=colors[i],
        )
        ax_diff.plot(
            run["k"],
            run["gamma"] - reference["gamma"],
            marker=markers[i % len(markers)],
            lw=1.8,
            ms=5,
            label=f"dt={run['dt']:g} - {reference['dt']:g}",
            color=colors[i],
        )

    params = reference["params"]
    ax_growth.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    ax_growth.set_title(
        "dt sweep convergence diagnostics\n"
        f"time evolution, n={params['n_start']}:{params['n_delta']}:{params['n_end']}, "
        f"m={params['m']}, p={params['p']}, {params['basis']} basis"
    )
    ax_growth.set_ylabel("Growth rate")
    ax_growth.grid(True, alpha=0.32)
    ax_growth.legend(loc="best", fontsize=8, ncols=2)

    ax_diff.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    ax_diff.set_xlabel(r"$k_{\theta}\rho_i$")
    ax_diff.set_ylabel("Growth-rate difference")
    ax_diff.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_diff.grid(True, alpha=0.32)
    ax_diff.legend(loc="best", fontsize=8)

    ax_diff_top = ax_diff.secondary_xaxis("top")
    ax_diff_top.set_xticks(reference["k"])
    ax_diff_top.set_xticklabels(
        [str(int(n)) for n in reference["n_values"]],
        fontsize=8,
    )
    ax_diff_top.set_xlabel("n")

    dts = np.asarray([run["dt"] for run in runs if run is not reference], dtype=float)
    max_diff = np.asarray(
        [
            np.max(np.abs(run["gamma"] - reference["gamma"]))
            for run in runs
            if run is not reference
        ],
        dtype=float,
    )
    rms_diff = np.asarray(
        [
            np.sqrt(np.mean((run["gamma"] - reference["gamma"]) ** 2))
            for run in runs
            if run is not reference
        ],
        dtype=float,
    )

    ax_error.loglog(dts, max_diff, "o-", lw=2, ms=6, label="max |diff|")
    ax_error.loglog(dts, rms_diff, "s-", lw=2, ms=5, label="RMS diff")

    if len(dts) >= 2:
        coeff = np.polyfit(np.log10(dts), np.log10(max_diff), 1)
        x_fit = np.asarray([dts.min(), dts.max()])
        y_fit = 10 ** np.polyval(coeff, np.log10(x_fit))
        ax_error.loglog(
            x_fit,
            y_fit,
            "--",
            color="0.35",
            lw=1.2,
            label=f"max diff slope ~ {coeff[0]:.2f}",
        )

    ax_error.set_xlabel("dt")
    ax_error.set_ylabel(f"Error vs dt={reference['dt']:g}")
    ax_error.grid(True, which="both", alpha=0.32)
    ax_error.legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def print_summary(runs: list[dict]) -> None:
    reference = runs[0]
    print(f"included dts: {[run['dt'] for run in runs]}")
    print(f"reference dt={reference['dt']:g}")
    for run in runs:
        max_abs_diff = float(np.max(np.abs(run["gamma"] - reference["gamma"])))
        print(
            f"dt={run['dt']:g} "
            f"max_gamma={np.max(run['gamma']):.12g} "
            f"max_abs_diff_vs_dt{reference['dt']:g}={max_abs_diff:.12g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dt-sweep growth-rate comparison from result files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing *_params.json and *_eigenvalues.npz files.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT,
        help="Output PNG path for the comparison plot.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=DEFAULT_DIAGNOSTICS_OUTPUT,
        help="Output PNG path for the diagnostics plot.",
    )
    parser.add_argument(
        "--max-dt-exclusive",
        type=float,
        default=1.0,
        help="Ignore runs with dt greater than or equal to this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.results_dir, args.max_dt_exclusive)
    plot_comparison(runs, args.comparison_output)
    print(f"saved {args.comparison_output}")
    plot_diagnostics(runs, args.diagnostics_output)
    print(f"saved {args.diagnostics_output}")
    print_summary(runs)


if __name__ == "__main__":
    main()
