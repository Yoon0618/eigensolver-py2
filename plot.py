import matplotlib.pyplot as plt
import numpy as np

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

    plt.figure(figsize=(10, 6))
    plt.plot(k_thetas_rho_i, gammas, 'o-', label='Growth Rate') # 파란색 점
    plt.plot(k_thetas_rho_i, omegas, 's-', label='Frequency/4') # 빨간색 점
    plt.xlabel(r'$k_{\theta} \rho_i$')
    plt.ylabel('Growth Rate, Frequency/4')
    text = f"basis: {param.basis}\nparameters:\n {param.n_start} <= n <= {param.n_end}, $\\Delta$n={param.n_delta}\n 1 <= m <= {param.m}, $\\Delta$m=1\n 0 <= p < {param.p}\n"
    filename = f"n{param.n_start}_{param.n_end}_m{param.m}_p{param.p}_{param.basis}"
    plt.text(0.5, 0.5, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{param.save_dir}/{filename}.png", dpi=300)
    elif show:
        plt.show()
        
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

    if save:
        fig.savefig(f"{param.save_dir}/{filename}_all_modes.png", dpi=300)
    elif show:
        plt.show()

    plt.close(fig)


def plot_time_evolution(param, profiles, solve_data, save=True, show=True):
    # n_values = solve_data["n_values"]
    # ts = solve_data["ts"]
    # gammas = solve_data["gammas"]

    # # 결과 플로팅
    # plt.figure(figsize=(10, 6))
    # for i, n in enumerate(n_values):
    #     plt.plot(ts, gammas[i], label=f'n={n}')
    # plt.xlabel('Normalized Time')
    # plt.ylabel('Growth Rate (F\'/F)')
    # plt.title('Growth Rate vs Time for Different n Modes')
    # # 파라미터 텍스트로 추가
    # # 시간 스텝과 시뮬레이션 시간, 초기값 크기 등을 텍스트로 추가한다.
    # text1 = f"basis: {param.basis}\nparameters:\n {param.n_start} <= n <= {param.n_end}, $\Delta$n={param.n_delta}\n 1 <= m <= {param.m}, $\Delta$m=1\n 0 <= p < {param.p}\n"
    # text2 = f"dt={param.dt}, total time={param.T}, initial perturbation={param.F0}"
    # plt.text(0.05, 0.95, text1 + text2, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    # plt.legend()
    # plt.grid()

    # if save:
    #     plt.savefig(f"{param.save_dir}/{param.basis}_growth_rates.png", dpi=300)
    # elif show:
    #     plt.show()
    return 0

def plot_matrices(matrices, titles):
    """
    matrices: list of 2D numpy arrays to plot
    titles: list of titles for each subplot
    """
    n = len(matrices)
    if n == 0:
        return 0

    if len(titles) != n:
        raise ValueError("'titles' length must match 'matrices' length")

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes = np.atleast_1d(axes)

    for i in range(n):
        ax = axes[i]
        im = ax.imshow(matrices[i], aspect='auto', origin='lower')
        ax.set_title(titles[i])
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def main():
    # plot.py를 직접 실행하면 npz, json 파일을 읽어서 결과를 플로팅한다.

    # 0: eigenvalues, 1: eigenmodes, 2: growth rates
    plot_type = input("플롯 타입을 선택하세요 (1: eigenvalues, 2: eigenmodes, 3: growth rates): ")
    
    return 0
    
if __name__ == "__main__":
    main()
    

# def plot_modes():
#     # plot rr vs q, (n,m) vs q, (r,n) vs q, (r,m) vs q
#     fig, ax = plt.subplots(2, 2, figsize=(12, 10))

#     # title
#     fig.suptitle(f'(n,m) modes: {np.count_nonzero(~np.isnan(qs))}', fontsize=16)

#     # plot r vs q
#     x = rr.flatten()
#     y = qs.flatten()

#     ax[0,0].scatter(x, y, c='blue', marker='o')
#     ax[0,0].set_xlabel('r/a')
#     ax[0,0].set_ylabel('q')
#     ax[0,0].set_title('r vs q for (n,m) modes')
#     ax[0,0].grid()

#     # plot (n,m) vs q
#     # ax.imshow(qs_, extent=(0, ns[-1], 0, ms[-1]), origin='lower', aspect='auto')
#     ax[0,1].scatter(np.tile(ns, len(ms)), np.repeat(ms, len(ns)), c=qs.flatten(), cmap='viridis', marker='o')
#     ax[0,1].set_xlabel('n')
#     ax[0,1].set_ylabel('m')
#     ax[0,1].set_title('q values for (n,m) modes')
#     ax[0,1].grid()

#     X, Y = np.meshgrid(ns, ms)
#     # plot (r,m) vs q
#     nn = X.flatten()
#     mm = Y.flatten()
#     ax[1,0].scatter(rr.flatten(), mm, c=qs.flatten(), cmap='viridis', marker='o')
#     ax[1,0].set_xlabel('r/a')
#     ax[1,0].set_ylabel('m')
#     ax[1,0].set_title('rho values for (n,m) modes')
#     ax[1,0].grid()

#     # plot (r,n) vs q
#     ax[1,1].scatter(rr.flatten(), nn, c=qs.flatten(), cmap='viridis', marker='o')
#     ax[1,1].set_xlabel('r/a')
#     ax[1,1].set_ylabel('n')
#     ax[1,1].set_title('rho values for (n,m) modes')
#     ax[1,1].grid()
#     plt.show()
