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
    ax.xlabel(r'$k_{\theta} \rho_i$')
    ax.ylabel('Growth Rate, Frequency/4')
    text = f"basis: {param.basis}\nparameters:\n {param.n_start} <= n <= {param.n_end}, $\\Delta$n={param.n_delta}\n 1 <= m <= {param.m}, $\\Delta$m=1\n 0 <= p < {param.p}\n"
    filename = f"n{param.n_start}_{param.n_end}_m{param.m}_p{param.p}_{param.basis}"
    ax.text(0.5, 0.5, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
    ax.legend()
    ax.grid()
    
    filename = f"n{param.n_start}_{param.n_end}_m{param.m}_p{param.p}_{param.basis}.png"
    save_path = os.path.join(param.save_dir, filename)
    _finalize_figure(fig, save_path=save_path, save=save, show=show)     

def plot_eigenmodes(param, profiles, mode_data, mat_data, solve_data, save=True, show=True):
    # 모드를 시각화한다.
    # ~ plot_eigenmodes

    # 각 n에 대해 가장 큰 성장률을 가지는 모드의 퍼텐셜을 subplot으로 시각화한다.
    W = mat_data["W"]
    rs = profiles["rs"]
    thetas = np.arange(-np.pi, np.pi, 0.01)

    n_values = solve_data["n_values"]
    most_unstable_mode_indexes = solve_data["most_unstable_mode_indexes"]
    F_blocked = solve_data["F_blocked"]
    n_mode_indexes = solve_data["n_mode_indexes"]
    ks = mode_data["ks"]
    n_count = len(n_values)

    x = rs[:, None] * np.cos(thetas)[None, :]
    y = rs[:, None] * np.sin(thetas)[None, :]
    ncols = min(3, n_count)
    nrows = int(np.ceil(n_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    filename = f"n{param.n_start}_{param.n_end}_m{param.m}_p{param.p}_{param.basis}"

    for i, n in enumerate(n_values):
        idx = n_mode_indexes[n] # n 모드에 해당하는 k 인덱스들을 가져온다.
        if most_unstable_mode_indexes is None: # time evolution 방법에서는 각 n별로 계수 F들만 구할 수 있으므로,
            F = F_blocked[i]
        else: # eigenproblem 방법에서는 각 n별로 여러 모드가 나올 수 있는데, 그 중에서 가장 성장률이 큰 모드의 계수 F를 가져온다.
            most_unstable_mode_index = most_unstable_mode_indexes[i] # n 모드에서 가장 성장률이 큰 모드의 인덱스를 가져온다.
            F = F_blocked[i][:, most_unstable_mode_index] # 성장률이 가장 큰 모드의 계수들을 가져온다. shape (k_n,) F_blocked[i] = [phi1, phi2, ... phi_kn, Ti1, Ti2, ... Ti_kn, ne1, ne2, ... ne_kn]
        
        phi_k = F[:len(idx)] # phi에 해당하는 계수들을 가져온다. shape (k_n,)
        Wk = W[idx] # n 모드에 해당하는 Wk 함수들을 가져온다. shape (k_n, r_num)
        m = ks[idx, 1] # n 모드에 해당하는 m 값들을 가져온다. shape (K_n,)
        exp_imtheta = np.exp(1j * m[:, None] * thetas[None, :]) # theta에 대한 exp(i*m*theta) 부분을 계산한다. shape (k_n, 100,)
        
        # phi = sum_k F_k*W_k*exp(1j*m*theta) shape (256, 100)
        phi = np.sum(phi_k[:, None, None] * Wk[:, :, None] * exp_imtheta[:, None, :], axis=0) # F_k * W_k(r) * exp(i*m*theta) 부분을 계산한다. shape (256, 100)
        
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
    n_values = solve_data["n_values"]
    ts = solve_data["ts"]
    Fs = solve_data["Fs"]
    fit_info = solve_data.get("fit_info")

    fig, ax = plt.subplots(figsize=(10, 6))

    fit_t0 = None
    fit_t1 = None
    for i, n in enumerate(n_values):
        info = None if fit_info is None else fit_info[i]
        log_amp, i0, i1, slope, intercept = _get_time_fit_info(ts, Fs[i], info)

        line, = ax.plot(ts, log_amp, label=f"n={n}")
        ax.plot(ts[i0:i1], slope * ts[i0:i1] + intercept, "--", color=line.get_color(), alpha=0.85)

        fit_t0 = ts[i0] if fit_t0 is None else min(fit_t0, ts[i0])
        fit_t1 = ts[i1 - 1] if fit_t1 is None else max(fit_t1, ts[i1 - 1])

    if fit_t0 is not None and fit_t1 is not None:
        ax.axvspan(fit_t0, fit_t1, color="gray", alpha=0.12, label="fit interval")

    ax.set_xlabel("time")
    ax.set_ylabel("ln(||F||)")
    ax.set_title("Time evolution of mode amplitude and linear fit")
    ax.grid(True)
    ax.legend(loc="best")

    text = (
        f"basis: {param.basis}\n"
        f"n={param.n_start}:{param.n_delta}:{param.n_end}, m<= {param.m}, p< {param.p}\n"
        f"dt={param.dt}, T={param.T}, F0={param.F0}"
    )
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    save_path = os.path.join(param.save_dir, f"time_evolution_gamma{param.suffix}.png")
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
    
def main():
    # plot.py를 직접 실행하면 npz, json 파일을 읽어서 결과를 플로팅한다.

    # 0: eigenvalues, 1: eigenmodes, 2: growth rates
    plot_type = input("플롯 타입을 선택하세요 (1: eigenvalues, 2: eigenmodes, 3: growth rates): ")
    
    return 0
    

if __name__ == "__main__":
    main()
    

