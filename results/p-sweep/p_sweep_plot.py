"""Plot growth-rate diagnostics for p-sweep results in this directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "p_sweep_growth_rate_comparison.png"
DEFAULT_GROWTH_ONLY_OUTPUT = SCRIPT_DIR / "p_sweep_growth_rate_only.png"
FIGURE_FACE_COLOR = "#FAFAFA"
AXES_FACE_COLOR = "white"


def style_figure(fig, axes) -> None:
    fig.patch.set_facecolor(FIGURE_FACE_COLOR)
    for ax in np.ravel(np.atleast_1d(axes)):
        ax.set_facecolor(AXES_FACE_COLOR)


def load_runs(results_dir: Path) -> list[dict]:
    runs = []

    for params_path in sorted(results_dir.glob("*_params.json")):
        npz_path = params_path.with_name(
            params_path.name.replace("_params.json", "_eigenvalues.npz")
        )
        if not npz_path.exists():
            continue

        with params_path.open(encoding="utf-8") as f:
            params = json.load(f)

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
                "p": int(params["p"]),
                "params": params,
                "params_path": params_path,
                "npz_path": npz_path,
                "k": k,
                "gamma": gamma,
                "n_values": n_values,
            }
        )

    runs.sort(key=lambda run: run["p"])
    return runs


def get_reference_run(runs: list[dict], reference_p: int | None) -> dict:
    if not runs:
        raise ValueError("No runs found.")

    if reference_p is None:
        return runs[-1]

    for run in runs:
        if run["p"] == reference_p:
            return run

    available = ", ".join(str(run["p"]) for run in runs)
    raise ValueError(f"reference p={reference_p} not found. Available p values: {available}")


def validate_runs(runs: list[dict], reference: dict) -> None:
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


def plot_comparison(
    runs: list[dict],
    reference: dict,
    output_path: Path,
) -> None:
    validate_runs(runs, reference)

    p_values = np.asarray([run["p"] for run in runs], dtype=int)
    gamma_matrix = np.vstack([run["gamma"] for run in runs])
    reference_gamma = reference["gamma"]
    diff_matrix = gamma_matrix - reference_gamma[None, :]

    max_indexes = np.nanargmax(gamma_matrix, axis=1)
    max_gammas = gamma_matrix[np.arange(len(runs)), max_indexes]
    max_n_values = reference["n_values"][max_indexes]
    max_abs_diffs = np.max(np.abs(diff_matrix), axis=1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.4, 9.7),
        sharex=False,
        gridspec_kw={"height_ratios": [2.5, 2.1, 1.7]},
    )
    ax_growth, ax_diff, ax_summary = axes
    style_figure(fig, axes)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, run in enumerate(runs):
        label = f"p={run['p']} n_r={run['params'].get('r_num')}"
        if run is reference:
            label += " ref"

        ax_growth.plot(
            run["k"],
            run["gamma"],
            marker=markers[i % len(markers)],
            lw=1.9,
            ms=4.5,
            label=label,
            color=colors[i],
        )

        if run is not reference:
            ax_diff.plot(
                run["k"],
                run["gamma"] - reference_gamma,
                marker=markers[i % len(markers)],
                lw=1.55,
                ms=3.8,
                label=f"p={run['p']} - {reference['p']}",
                color=colors[i],
            )

    ax_growth.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    params = reference["params"]
    ax_growth.set_title(
        "p sweep growth-rate comparison\n"
        f"time evolution, dt={params['dt']}, T={params['T']}, "
        f"n={params['n_start']}:{params['n_delta']}:{params['n_end']}, "
        f"m={params['m']}, {params['basis']} basis"
    )
    ax_growth.set_ylabel("Normalized growth rate")
    ax_growth.grid(True, alpha=0.32)
    ax_growth.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    for i, run in enumerate(runs):
        ax_growth.scatter(
            [run["k"][max_indexes[i]]],
            [max_gammas[i]],
            s=52,
            facecolors="none",
            edgecolors="black",
            lw=0.8,
            zorder=5,
        )

    ax_diff.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    ax_diff.set_ylabel(f"Diff vs p={reference['p']}")
    ax_diff.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax_diff.grid(True, alpha=0.32)
    ax_diff.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax_diff.set_xlabel(r"$k_{\theta}\rho_i$")

    ax_diff_top = ax_diff.secondary_xaxis("top")
    ax_diff_top.set_xticks(reference["k"])
    ax_diff_top.set_xticklabels(
        [str(int(n)) for n in reference["n_values"]],
        fontsize=8,
    )
    ax_diff_top.set_xlabel("n")

    ax_summary.plot(
        p_values,
        max_gammas,
        "o-",
        lw=2,
        ms=5.5,
        label="max gamma",
        color="tab:blue",
    )
    ax_summary.set_xlabel("p")
    ax_summary.set_ylabel("max growth rate")
    ax_summary.set_xticks(p_values)
    ax_summary.grid(True, alpha=0.32)

    ax_summary_diff = ax_summary.twinx()
    ax_summary_diff.set_facecolor(AXES_FACE_COLOR)
    valid_diff = max_abs_diffs > 0
    ax_summary_diff.semilogy(
        p_values[valid_diff],
        max_abs_diffs[valid_diff],
        "s--",
        color="tab:red",
        lw=1.8,
        ms=4.8,
        label=f"max |diff| vs p={reference['p']}",
    )
    ax_summary_diff.set_ylabel(f"max |diff| vs p={reference['p']}")

    for p, gamma, n_value in zip(p_values, max_gammas, max_n_values):
        ax_summary.annotate(
            f"n={int(n_value)}",
            (p, gamma),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    lines_1, labels_1 = ax_summary.get_legend_handles_labels()
    lines_2, labels_2 = ax_summary_diff.get_legend_handles_labels()
    ax_summary.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", fontsize=8)

    summary = f"reference: p={reference['p']}\n" + "\n".join(
        f"p={p}: {diff:.3g}"
        for p, diff in zip(p_values, max_abs_diffs)
        if diff > 0
    )
    ax_diff.text(
        0.012,
        0.04,
        summary,
        transform=ax_diff.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_growth_only(
    runs: list[dict],
    reference: dict,
    output_path: Path,
) -> None:
    validate_runs(runs, reference)

    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    style_figure(fig, [ax])

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(runs) - 1, 1)) for i in range(len(runs))]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for i, run in enumerate(runs):
        label = f"p={run['p']} n_r={run['params'].get('r_num')}"
        if run is reference:
            label += " ref"

        ax.plot(
            run["k"],
            run["gamma"],
            marker=markers[i % len(markers)],
            lw=1.9,
            ms=4.8,
            label=label,
            color=colors[i],
        )

        max_index = int(np.nanargmax(run["gamma"]))
        ax.scatter(
            [run["k"][max_index]],
            [run["gamma"][max_index]],
            s=52,
            facecolors="none",
            edgecolors="black",
            lw=0.8,
            zorder=5,
        )

    ax.axhline(0, color="0.25", lw=0.9, alpha=0.65)
    params = reference["params"]
    ax.set_title(
        "p sweep growth-rate comparison\n"
        f"time evolution, dt={params['dt']}, T={params['T']}, "
        f"n={params['n_start']}:{params['n_delta']}:{params['n_end']}, "
        f"m={params['m']}, {params['basis']} basis"
    )
    ax.set_xlabel(r"$k_{\theta}\rho_i$")
    ax.set_ylabel("Normalized growth rate")
    ax.grid(True, alpha=0.32)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks(reference["k"])
    ax_top.set_xticklabels(
        [str(int(n)) for n in reference["n_values"]],
        fontsize=8,
    )
    ax_top.set_xlabel("n")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def print_summary(runs: list[dict], reference: dict) -> None:
    reference_gamma = reference["gamma"]

    print(f"reference p={reference['p']}")
    for run in runs:
        max_index = int(np.nanargmax(run["gamma"]))
        max_abs_diff = float(np.max(np.abs(run["gamma"] - reference_gamma)))
        print(
            f"p={run['p']} "
            f"r_num={run['params'].get('r_num')} "
            f"dr={run['params'].get('dr')} "
            f"max_gamma={run['gamma'][max_index]:.12g} "
            f"n={int(run['n_values'][max_index])} "
            f"k={run['k'][max_index]:.12g} "
            f"max_abs_diff_vs_p{reference['p']}={max_abs_diff:.12g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot p-sweep growth-rate comparison from result files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing *_params.json and *_eigenvalues.npz files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path for the three-panel comparison.",
    )
    parser.add_argument(
        "--growth-only-output",
        type=Path,
        default=DEFAULT_GROWTH_ONLY_OUTPUT,
        help="Output PNG path for the growth-rate-only plot.",
    )
    parser.add_argument(
        "--reference-p",
        type=int,
        default=None,
        help="Reference p value. Defaults to the largest p found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.results_dir)
    reference = get_reference_run(runs, args.reference_p)
    plot_comparison(runs, reference, args.output)
    print(f"saved {args.output}")
    plot_growth_only(runs, reference, args.growth_only_output)
    print(f"saved {args.growth_only_output}")
    print_summary(runs, reference)


if __name__ == "__main__":
    main()
