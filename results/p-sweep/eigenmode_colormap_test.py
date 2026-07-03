import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eigensolver-py-matplotlib")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

from parameters import Params, build_profiles


DEFAULT_BASE = "20260514_180514_n4-4-48_m150_p80_h_te"
DEFAULT_CMAPS = [
    "viridis",
    "cividis",
    "turbo",
    "magma",
    "RdBu_r",
    "coolwarm",
    "seismic",
    "Spectral_r",
    "PiYG",
    "PRGn",
]


def load_saved_params(path):
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    init_kwargs = {
        field.name: raw[field.name]
        for field in fields(Params)
        if field.init and field.name in raw
    }
    param = Params(**init_kwargs)

    for key, value in raw.items():
        if hasattr(param, key):
            setattr(param, key, value)

    return param


def load_modes_quietly(param, profiles):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        from modes import build_modes

        return build_modes(param, profiles)


def unpack_final_state(eigen_data, target_n):
    n_values = np.asarray(eigen_data["n_values"], dtype=int)
    matches = np.where(n_values == target_n)[0]
    if len(matches) == 0:
        raise ValueError(f"n={target_n} is not present. Available n values: {n_values.tolist()}")

    i = int(matches[0])
    offsets = np.asarray(eigen_data["F_block_final_state_offsets"], dtype=int)
    flat = np.asarray(eigen_data["F_block_final_state"])
    return flat[offsets[i] : offsets[i + 1]]


def hermite_widths(param, rs, mode_radius_indexes, ks):
    base_w_mn = float(param.w_mn)
    widths = np.empty(len(ks), dtype=float)

    for i, (_, _, _) in enumerate(ks):
        r0 = rs[mode_radius_indexes[i]]
        w = base_w_mn

        if r0 < (5.0 / 7.0) * param.p * w:
            w = (7.0 / 5.0) * r0 / param.p
        elif (1.0 - r0) < (4.0 / 7.0) * param.p * w:
            w = (7.0 / 4.0) * (1.0 - r0) / param.p

        widths[i] = w

    return widths


def raw_hermite_w(rs, r0, widths, p_orders):
    p_orders = np.asarray(p_orders, dtype=int)
    x = (rs[None, :] - r0[:, None]) / widths[:, None]
    v_p = (2.0 ** (p_orders / 2.0)) * np.sqrt(sp.gamma(p_orders + 1.0)) * np.pi**0.25
    return (
        sp.eval_hermite(p_orders[:, None], x)
        * np.exp(-0.5 * x * x)
        / (np.sqrt(2.0 * rs[None, :] * widths[:, None]) * v_p[:, None])
    )


def build_radial_coefficients(
    param,
    profiles,
    mode_data,
    n_indexes,
    phi_coefficients,
    plot_rs,
    chunk_size,
):
    rs = profiles["rs"]
    dr = np.full(param.r_num, param.dr)
    dr[-1] = 0.5 * param.dr
    rdr = rs * dr

    ks = mode_data["ks"]
    mode_radius_indexes = mode_data["mode_radius_indexes"]
    widths = hermite_widths(param, rs, mode_radius_indexes[n_indexes], ks[n_indexes])

    m_values = ks[n_indexes, 1]
    unique_m = np.unique(m_values)
    m_to_column = {int(m): i for i, m in enumerate(unique_m)}
    radial_coefficients = np.zeros((len(plot_rs), len(unique_m)), dtype=np.complex128)

    for start in range(0, len(n_indexes), chunk_size):
        stop = min(start + chunk_size, len(n_indexes))
        local = slice(start, stop)
        global_indexes = n_indexes[local]

        r0 = rs[mode_radius_indexes[global_indexes]]
        p_orders = ks[global_indexes, 2]
        chunk_widths = widths[local]

        w_full = raw_hermite_w(rs, r0, chunk_widths, p_orders)
        norms = np.sqrt(np.sum(w_full * w_full * rdr[None, :], axis=1))
        del w_full

        w_plot = raw_hermite_w(plot_rs, r0, chunk_widths, p_orders) / norms[:, None]

        for row, m, coefficient in zip(w_plot, m_values[local], phi_coefficients[local]):
            radial_coefficients[:, m_to_column[int(m)]] += coefficient * row

        print(f"processed modes {stop}/{len(n_indexes)}")

    return unique_m, radial_coefficients


def reconstruct_phi(param, profiles, mode_data, final_state, target_n, radial_samples, chunk_size):
    ks = mode_data["ks"]
    n_indexes = np.where(ks[:, 0] == target_n)[0]

    if final_state.size != 3 * len(n_indexes):
        raise ValueError(
            f"n={target_n} final-state length mismatch: "
            f"got {final_state.size}, expected {3 * len(n_indexes)}"
        )

    phi_coefficients = final_state[: len(n_indexes)]
    rs = profiles["rs"]
    plot_indices = np.unique(np.linspace(0, len(rs) - 1, radial_samples).astype(int))
    plot_rs = rs[plot_indices]
    m_values, radial_coefficients = build_radial_coefficients(
        param,
        profiles,
        mode_data,
        n_indexes,
        phi_coefficients,
        plot_rs,
        chunk_size,
    )

    return plot_rs, m_values, radial_coefficients


def evaluate_on_cartesian_grid(plot_rs, m_values, radial_coefficients, cartesian_samples, point_chunk_size=100000):
    grid = np.linspace(-1.0, 1.0, cartesian_samples)
    xx, yy = np.meshgrid(grid, grid)
    rr = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    mask = rr <= plot_rs[-1]

    flat_r = rr[mask]
    flat_theta = theta[mask]
    flat_phi = np.empty(len(flat_r), dtype=np.float64)

    for start in range(0, len(flat_r), point_chunk_size):
        stop = min(start + point_chunk_size, len(flat_r))
        r_chunk = flat_r[start:stop]
        theta_chunk = flat_theta[start:stop]
        values = np.zeros(stop - start, dtype=np.complex128)

        for j, m in enumerate(m_values):
            radial_real = np.interp(r_chunk, plot_rs, radial_coefficients[:, j].real)
            radial_imag = np.interp(r_chunk, plot_rs, radial_coefficients[:, j].imag)
            values += (radial_real + 1j * radial_imag) * np.exp(1j * int(m) * theta_chunk)

        flat_phi[start:stop] = values.real

    image = np.full_like(rr, np.nan, dtype=np.float64)
    image[mask] = flat_phi
    return image


def plot_colormap_grid(real_phi, cmaps, target_n, output_path):
    ncols = 5
    nrows = int(np.ceil(len(cmaps) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 4.0 * nrows),
        squeeze=False,
        layout="constrained",
    )
    axes_flat = axes.ravel()

    vmax = np.nanmax(np.abs(real_phi))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("real_phi does not contain finite nonzero values")

    for ax, cmap in zip(axes_flat, cmaps):
        cmap_object = plt.get_cmap(cmap).copy()
        cmap_object.set_bad("white")
        image = ax.imshow(
            np.ma.masked_invalid(real_phi),
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap=cmap_object,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_aspect("equal")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(cmap, fontsize=11)
        fig.colorbar(image, ax=ax, shrink=0.78, fraction=0.046, pad=0.02)

    for ax in axes_flat[len(cmaps) :]:
        ax.set_visible(False)

    fig.suptitle(f"n={target_n} eigenmode colormap comparison", fontsize=15)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--radial-samples", type=int, default=2048)
    parser.add_argument("--cartesian-samples", type=int, default=900)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--cmaps", nargs="+", default=DEFAULT_CMAPS)
    args = parser.parse_args()

    result_dir = Path(__file__).resolve().parent
    params_path = result_dir / f"{args.base}_params.json"
    eigen_path = result_dir / f"{args.base}_eigenvalues.npz"
    output_path = result_dir / f"{args.base}_n{args.n}_eigenmode_colormap_test.png"

    param = load_saved_params(params_path)
    profiles = build_profiles(param)
    mode_data = load_modes_quietly(param, profiles)

    with np.load(eigen_path, allow_pickle=True) as eigen_data:
        final_state = unpack_final_state(eigen_data, args.n)

    plot_rs, m_values, radial_coefficients = reconstruct_phi(
        param,
        profiles,
        mode_data,
        final_state,
        args.n,
        args.radial_samples,
        args.chunk_size,
    )
    real_phi = evaluate_on_cartesian_grid(
        plot_rs,
        m_values,
        radial_coefficients,
        args.cartesian_samples,
    )
    plot_colormap_grid(real_phi, args.cmaps, args.n, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
