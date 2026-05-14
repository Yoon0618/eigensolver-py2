import os
import matplotlib.pyplot as plt
import numpy as np

def _finalize_figure(fig, save_path=None, save=True, show=True):
    if save and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    
def plot_eigenvalues(param, profiles, solve_data, save=True, show=True):
    # 결과를 플로팅한다.
    
    n_values = solve_data["n_values"]
    gammas = solve_data["gammas"]
    omegas = solve_data["omegas"]

    # 각 모드 별 성장률 비교를 위해서, x축에 해당하는 값 ktheta_rho_i를 계산한다. 
    # k_theta_rho_i ~ nq/r * rhos0
    # cyclon case parameters
    q_val = 1.4 # q at r=0.5a
    r_val = 0.5 # 0.5a
    k_thetas_rho_i = n_values * q_val / r_val * param.rhos0

    # normalize
    gamma_factor = profiles["R_Lne"] / param.rmajor
    omega_factor = gamma_factor*4
    gammas = gammas / gamma_factor 
    omegas = omegas / omega_factor

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_thetas_rho_i, gammas, 'o-', label='Growth Rate') # 파란색 점
    ax.plot(k_thetas_rho_i, omegas, 's-', label='Frequency/4') # 빨간색 점
    ax.set_xlabel(r'$k_{\theta} \rho_i$')
    ax.set_ylabel('Growth Rate, Frequency/4')
    text = f"basis: {param.basis}\nmethod: {param.method}\nparameters:\n {param.n_start} <= n <= {param.n_end}, $\\Delta$n={param.n_delta}\n 1 <= m <= {param.m}, $\\Delta$m=1\n 0 <= p < {param.p}\n"
    ax.text(0.5, 0.5, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.legend()
    ax.grid()
    
    filename = f"{param.file_name}_eigenvalues.png"
    save_path = os.path.join(param.save_dir, filename)
    _finalize_figure(fig, save_path=save_path, save=save, show=show)     

    return {
        "k_thetas_rho_i": k_thetas_rho_i,
        "gammas": gammas,
        "omegas": omegas,
    }
    
def plot_eigenmodes(param, profiles, mode_data, selected_mat_data, solve_data, save=True, show=True):
    # 모드를 시각화한다.
    # ~ plot_eigenmodes

    # 각 n에 대해 가장 큰 성장률을 가지는 모드의 퍼텐셜을 subplot으로 시각화한다.
    W = selected_mat_data["W"]
    rs = profiles["rs"]
    thetas = np.arange(-np.pi, np.pi, 0.01)

    n_values = solve_data["n_values"]
    most_unstable_mode_indexes = solve_data["most_unstable_mode_indexes"]
    F_blocks = solve_data.get("F_block_final_state", solve_data.get("F_blocked"))
    n_mode_indexes = solve_data["n_mode_indexes"]
    ks = mode_data["ks"]
    n_count = len(n_values)

    x = rs[:, None] * np.cos(thetas)[None, :]
    y = rs[:, None] * np.sin(thetas)[None, :]
    ncols = min(3, n_count)
    nrows = int(np.ceil(n_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, n in enumerate(n_values):
        idx = n_mode_indexes[n] # n 모드에 해당하는 k 인덱스들을 가져온다.
        if most_unstable_mode_indexes is None: # time evolution 방법에서는 각 n별로 계수 F들만 구할 수 있으므로,
            F = F_blocks[i]

        else: # eigenproblem 방법에서는 각 n별로 여러 모드가 나올 수 있는데, 그 중에서 가장 성장률이 큰 모드의 계수 F를 가져온다.
            most_unstable_mode_index = most_unstable_mode_indexes[i] # n 모드에서 가장 성장률이 큰 모드의 인덱스를 가져온다.
            F = F_blocks[i][:, most_unstable_mode_index] # 성장률이 가장 큰 모드의 계수들을 가져온다. shape (k_n,) F_blocked[i] = [phi1, phi2, ... phi_kn, Ti1, Ti2, ... Ti_kn, ne1, ne2, ... ne_kn]
        
        phi_k = F[:len(idx)] # phi에 해당하는 계수들을 가져온다. shape (k_n,)
        Wk = W[idx] # n 모드에 해당하는 Wk 함수들을 가져온다. shape (k_n, r_num)
        m = ks[idx, 1] # n 모드에 해당하는 m 값들을 가져온다. shape (K_n,)
        exp_imtheta = np.exp(1j * m[:, None] * thetas[None, :]) # theta에 대한 exp(i*m*theta) 부분을 계산한다. shape (k_n, 100,)
        
        # phi = sum_k F_k*W_k*exp(1j*m*theta) shape (256, 100)
        # F_k * W_k(r) * exp(i*m*theta) 부분을 계산한다. shape (256, 100)
        exp_imtheta = np.exp(1j * m[:, None] * thetas[None, :])
        phi = Wk.T @ (phi_k[:, None] * exp_imtheta)
        
        ax = axes_flat[i]
        contour = ax.contourf(x, y, phi.real, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax, label='Real part of potential')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f"n={n} mode with max growth rate")

    for ax in axes_flat[n_count:]:
        ax.set_visible(False)

    fig.suptitle(f"Eigenmodes ({param.basis})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    filename = f"{param.file_name}_eigenmodes.png"
    save_path = os.path.join(param.save_dir, filename)
    _finalize_figure(fig, save_path=save_path, save=save, show=show)

def _get_time_fit_info(ts, F_block, info=None):
    amp = np.linalg.norm(F_block, axis=1)
    log_amp = np.log(np.maximum(amp, 1e-300))

    if info is None:
        i0 = int(0.8 * len(ts))
        i1 = len(ts)
        slope, intercept = np.polyfit(ts[i0:i1], log_amp[i0:i1], 1)
    else:
        i0 = int(info["i0"])
        i1 = int(info["i1"])
        slope = float(info["slope"])
        intercept = float(info["intercept"])

    return log_amp, i0, i1, slope, intercept

def plot_time_evolution(param, profiles, solve_data, save=True, show=True):
    """
    Time-evolution 결과를 plot한다.

    새 solve_data 구조:
        ts      : shape (Nt,)
        lnFs    : shape (num_n, Nt)
        alphas  : shape (num_n, Nt), optional
                  alpha(t) = <F, B F> / <F, F>
                  dominant mode에서 alpha -> gamma - i omega
        gammas  : shape (num_n,), optional
        omegas  : shape (num_n,), optional
        fit_info: list of dict, optional

    그리는 것:
        1. ln ||F(t)|| 와 late-time linear fit
        2. alphas가 있으면 omega(t) = -Im(alpha(t)) 와 late-time 평균 omega
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    ts = np.asarray(solve_data["ts"], dtype=float)
    lnFs = np.asarray(solve_data["lnFs"], dtype=float)

    alphas = solve_data.get("alphas", None)
    if alphas is not None:
        alphas = np.asarray(alphas, dtype=complex)

    fit_info = solve_data.get("fit_info", None)
    gammas = solve_data.get("gammas", None)
    omegas = solve_data.get("omegas", None)

    # n_values는 solve_data에 넣는 것을 권장.
    # 없으면 param에서 복원 시도.
    n_values = solve_data.get("n_values", None)
    if n_values is None:
        candidate = np.arange(param.n_start, param.n_end + 1, param.n_delta)
        if len(candidate) == lnFs.shape[0]:
            n_values = candidate
        else:
            # fallback: label만 index로 표시
            n_values = np.arange(lnFs.shape[0])
    else:
        n_values = np.asarray(n_values)

    if lnFs.ndim != 2:
        raise ValueError(f"lnFs must have shape (num_n, Nt), got {lnFs.shape}")

    if lnFs.shape[1] != len(ts):
        raise ValueError(
            f"lnFs.shape[1] must match len(ts). "
            f"lnFs.shape={lnFs.shape}, len(ts)={len(ts)}"
        )

    if alphas is not None and alphas.shape != lnFs.shape:
        raise ValueError(
            f"alphas must have same shape as lnFs. "
            f"alphas.shape={alphas.shape}, lnFs.shape={lnFs.shape}"
        )

    if len(n_values) != lnFs.shape[0]:
        raise ValueError(
            f"len(n_values) must match lnFs.shape[0]. "
            f"len(n_values)={len(n_values)}, lnFs.shape={lnFs.shape}"
        )

    has_alpha = alphas is not None

    if has_alpha:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        ax_amp, ax_omega = axes
    else:
        fig, ax_amp = plt.subplots(figsize=(10, 6))
        ax_omega = None

    fit_t0_global = None
    fit_t1_global = None

    def _get_info(i):
        if fit_info is None:
            return None
        if i >= len(fit_info):
            return None
        return fit_info[i]

    def _fit_from_info_or_recompute(i, y):
        """
        fit_info가 있으면 그 정보를 사용하고,
        없으면 마지막 20% 구간에서 새로 fit한다.
        """
        info = _get_info(i)

        if info is not None:
            t0 = float(info["fit_t_start"])
            t1 = float(info["fit_t_end"])
            gamma = float(info["gamma"])
            intercept = float(info["intercept"])

            mask = (
                np.isfinite(ts)
                & np.isfinite(y)
                & (ts >= t0)
                & (ts <= t1)
            )

            if np.count_nonzero(mask) >= 2:
                return mask, gamma, intercept, info

        # fallback: 마지막 20%에서 fit
        t0 = ts[0] + 0.8 * (ts[-1] - ts[0])
        mask = np.isfinite(ts) & np.isfinite(y) & (ts >= t0)

        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Not enough valid points to fit n index {i}")

        X = np.column_stack([ts[mask], np.ones(np.count_nonzero(mask))])
        gamma, intercept = np.linalg.lstsq(X, y[mask], rcond=None)[0]

        y_pred = gamma * ts[mask] + intercept
        ss_res = np.sum((y[mask] - y_pred) ** 2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        info = {
            "gamma": float(gamma),
            "intercept": float(intercept),
            "fit_t_start": float(ts[mask][0]),
            "fit_t_end": float(ts[mask][-1]),
            "fit_points": int(np.count_nonzero(mask)),
            "r2": float(r2),
        }

        return mask, float(gamma), float(intercept), info

    for i, n in enumerate(n_values):
        y = lnFs[i]

        fit_mask, gamma, intercept, info = _fit_from_info_or_recompute(i, y)

        # label용 gamma/omega
        gamma_label = gamma
        if gammas is not None:
            gamma_label = float(gammas[i])

        omega_label = None
        if omegas is not None:
            omega_label = float(omegas[i])
        elif info is not None and "omega" in info:
            omega_label = float(info["omega"])

        if omega_label is None:
            label = f"n={n}, γ={gamma_label:.3g}"
        else:
            label = f"n={n}, γ={gamma_label:.3g}, ω={omega_label:.3g}"

        line, = ax_amp.plot(ts, y, label=label)

        # fit line
        ax_amp.plot(
            ts[fit_mask],
            gamma * ts[fit_mask] + intercept,
            "--",
            color=line.get_color(),
            alpha=0.85,
        )

        fit_t0 = float(ts[fit_mask][0])
        fit_t1 = float(ts[fit_mask][-1])

        fit_t0_global = fit_t0 if fit_t0_global is None else min(fit_t0_global, fit_t0)
        fit_t1_global = fit_t1 if fit_t1_global is None else max(fit_t1_global, fit_t1)

        # omega(t) = -Im(alpha)
        if has_alpha:
            omega_t = -np.imag(alphas[i])

            ax_omega.plot(
                ts,
                omega_t,
                color=line.get_color(),
                alpha=0.75,
            )

            # late-time omega mean line
            if omega_label is not None:
                ax_omega.plot(
                    ts[fit_mask],
                    np.full(np.count_nonzero(fit_mask), omega_label),
                    "--",
                    color=line.get_color(),
                    alpha=0.85,
                )

    # fit interval shading
    if fit_t0_global is not None and fit_t1_global is not None:
        ax_amp.axvspan(
            fit_t0_global,
            fit_t1_global,
            color="gray",
            alpha=0.12,
            label="fit interval",
        )

        if ax_omega is not None:
            ax_omega.axvspan(
                fit_t0_global,
                fit_t1_global,
                color="gray",
                alpha=0.12,
            )

    ax_amp.set_ylabel(r"$\ln \|F(t)\|$")
    ax_amp.set_title("Time evolution: amplitude growth and frequency estimate")
    ax_amp.grid(True)
    ax_amp.legend(loc="best", fontsize=8)

    if ax_omega is not None:
        ax_omega.set_xlabel("time")
        ax_omega.set_ylabel(r"$-\mathrm{Im}\,\alpha(t)$")
        ax_omega.grid(True)
    else:
        ax_amp.set_xlabel("time")

    # 정보 박스
    text = (
        f"basis: {param.basis}\n"
        f"n={param.n_start}:{param.n_delta}:{param.n_end}, "
        f"m≤{param.m}, p<{param.p}\n"
        f"dt={param.dt}, T={param.T}, F0={param.F0}"
    )

    ax_amp.text(
        0.02,
        0.02,
        text,
        transform=ax_amp.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()

    file_name = f"{param.file_name}_time_evolution.png"
    save_path = os.path.join(param.save_dir, file_name)

    _finalize_figure(fig, save_path=save_path, save=save, show=show)

# def plot_matrices(matrices, titles):
#     """
#     matrices: list of 2D numpy arrays to plot
#     titles: list of titles for each subplot
#     """
#     n = len(matrices)
#     if n == 0:
#         return 0

#     if len(titles) != n:
#         raise ValueError("'titles' length must match 'matrices' length")

#     fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
#     axes = np.atleast_1d(axes)

#     for i in range(n):
#         ax = axes[i]
#         im = ax.imshow(matrices[i], aspect='auto', origin='lower')
#         ax.set_title(titles[i])
#         plt.colorbar(im, ax=ax)

#     plt.tight_layout()
#     plt.show()

# def plot_matrices(matrices, titles):
#     """
#     matrices: list of 2D numpy arrays to plot
#     titles: list of titles for each subplot
#     """
#     n = len(matrices)
#     if n == 0:
#         return 0

#     if len(titles) != n:
#         raise ValueError("'titles' length must match 'matrices' length")

#     fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
#     axes = np.atleast_1d(axes)

#     for i in range(n):
#         ax = axes[i]
#         im = ax.imshow(matrices[i], aspect='auto', origin='lower')
#         ax.set_title(titles[i])
#         plt.colorbar(im, ax=ax)

#     plt.tight_layout()
#     plt.show()
#     plt.close(fig)

def plot_growth_rate_comparison(
    results_dir="results",
    params_paths=None,
    label_keys=("method", "basis", "n", "m", "p", "dt"),
    gamma_key="gammas",
    save_path=None,
    show=True,
):
    """
    여러 run의 *_params.json, *_eigenvalues.npz를 읽어서 growth rate를 한 그래프에 비교한다.

    params_paths가 None이면 results_dir 아래의 *_params.json 전체를 사용한다.
    params_paths에 json 파일 경로들을 직접 넘기면 해당 run들만 그린다.
    gamma_key를 "gammas_raw"로 바꾸면 time-domain solver의 정규화 전 growth rate를 그릴 수 있다.
    """
    import json
    from pathlib import Path

    if params_paths is None:
        params_paths = [
            path for path in sorted(Path(results_dir).glob("*_params.json"))
            if path.with_name(path.name.replace("_params.json", "_eigenvalues.npz")).exists()
        ]
    else:
        params_paths = [Path(path) for path in params_paths]

    fig, ax = plt.subplots(figsize=(10, 6))
    run_count = 0

    for params_path in params_paths:
        npz_path = params_path.with_name(
            params_path.name.replace("_params.json", "_eigenvalues.npz")
        )

        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)

        with np.load(npz_path, allow_pickle=True) as data:
            k_thetas_rho_i = np.asarray(data["k_thetas_rho_i"], dtype=float)
            gammas = np.asarray(data[gamma_key], dtype=float)

        label_parts = []
        for key in label_keys:
            if key == "n":
                label_parts.append(
                    f"n={params['n_start']}:{params['n_delta']}:{params['n_end']}"
                )
            elif key in params:
                label_parts.append(f"{key}={params[key]}")

        label = ", ".join(label_parts) or params_path.stem.replace("_params", "")
        ax.plot(k_thetas_rho_i, gammas, "o-", label=label)
        run_count += 1

    ax.set_xlabel(r"$k_{\theta} \rho_i$")
    ax.set_ylabel("Growth rate")
    ax.set_title("Growth rate comparison")
    ax.grid(True)

    if run_count > 0:
        ax.legend(loc="best", fontsize=8)

    if save_path is not None:
        save_path = os.fspath(save_path)
        if os.path.dirname(save_path) == "":
            save_path = os.path.join(".", save_path)

    _finalize_figure(fig, save_path=save_path, save=save_path is not None, show=show)
    
def main():
    # plot.py를 직접 실행하면 npz, json 파일을 읽어서 결과를 플로팅한다.

    # 0: eigenvalues, 1: eigenmodes, 2: growth rates
    # plot_type = input("플롯 타입을 선택하세요 (1: eigenvalues, 2: eigenmodes, 3: growth rates): ")
    
    plot_growth_rate_comparison(
    results_dir="results-w/results",
    save_path="results-w/growth_rate_comparison.png",
    )

    return 0
    

if __name__ == "__main__":
    main()
    
