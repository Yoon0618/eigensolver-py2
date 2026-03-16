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
    text = f"basis: {param.basis}\nparameters:\n {param.n_start} <= n <= {param.n_end}, $\Delta$n={param.n_delta}\n m={param.m},\n 0 <= p < {param.p}\n"
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

    # 특정 n에 대해 가장 큰 성장률을 가지는 모드의 퍼텐셜을 시각화한다.
    W = mat_data["W"]
    rs = profiles["rs"]
    thetas = np.arange(-np.pi, np.pi, 0.01)

    n_values = solve_data["n_values"]
    most_unstable_mode_indexes = solve_data["most_unstable_mode_indexes"]
    F_blocked = solve_data["F_blocked"]
    n_mode_indexes = solve_data["n_mode_indexes"]
    ks = mode_data["ks"]

    for i, n in enumerate(n_values):
        idx = n_mode_indexes[n] # n 모드에 해당하는 k 인덱스들을 가져온다.
        most_unstable_mode_index = most_unstable_mode_indexes[i] # n 모드에서 가장 성장률이 큰 모드의 인덱스를 가져온다.
        F = F_blocked[i][:, most_unstable_mode_index] # 성장률이 가장 큰 모드의 계수들을 가져온다. shape (k_n,) F_blocked[i] = [phi1, phi2, ... phi_kn, Ti1, Ti2, ... Ti_kn, ne1, ne2, ... ne_kn]
        phi_k = F[:len(idx)] # phi에 해당하는 계수들을 가져온다. shape (k_n,)
        Wk = W[idx] # n 모드에 해당하는 Wk 함수들을 가져온다. shape (k_n, r_num)
        m = ks[idx, 1] # n 모드에 해당하는 m 값들을 가져온다. shape (K_n,)
        exp_imtheta = np.exp(1j * m[:, None] * thetas[None, :]) # theta에 대한 exp(i*m*theta) 부분을 계산한다. shape (k_n, 100,)
        # phi = sum_k F_k*W_k*exp(1j*m*theta) shape (256, 100)

        phi = np.sum(phi_k[:, None, None] * Wk[:, :, None] * exp_imtheta[:, None, :], axis=0) # F_k * W_k(r) * exp(i*m*theta) 부분을 계산한다. shape (256, 100)
        x = rs[:, None] * np.cos(thetas)[None, :]
        y = rs[:, None] * np.sin(thetas)[None, :]
        
        plt.figure(figsize=(8, 6))
        plt.contourf(x, y, phi.real, levels=50, cmap='viridis')
        plt.colorbar(label='Real part of potential')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"n={n} mode with max growth rate")
        text = f"basis: {param.basis}\nparameters:\n {param.n_start} <= n <= {param.n_end}, $\Delta$n={param.n_delta}\n 1 <= m <= {param.m}, $\Delta$m=1\n 0 <= p < {param.p}\n"
        filename = f"n{param.n_start}_{param.n_end}_m{param.m}_p{param.p}_{param.basis}"
        if save:
            plt.savefig(f"{param.save_dir}/{filename}_n{n}_mode.png", dpi=300)
        elif show:
            plt.show()

def plot_matrices(matrices, titles):
    """
    matrices: list of 2D numpy arrays to plot
    titles: list of titles for each subplot
    """
    n = len(matrices)
    if n == 0:
        return

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