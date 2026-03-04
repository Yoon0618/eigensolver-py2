import matplotlib.pyplot as plt
import numpy as np
import parameters as param

# def plot_eigenmodes(eigenvectors, gammas, gammas_idx):
#     # F가 구해졌으니, 이를 기반으로 함수의 모습을 시각화한다.
#     # 베셀 함수와 에르미트 함수의 경우를 나눠야 한다. 일단은 bessel만 구현

#     theta = np.linspace(-np.pi, np.pi, 0.01)
#     rs = param.rs
#     # F = exp(-iwt) * sum_k F_k * W_k(rho) * exp(i(m*theta - n*zeta))
#     # 가장 큰 성장률만 시각화해보자.
#     # zeta는 어떤 값이든간에 시각화에 큰 영향을 주지 않으므로, zeta=0으로 고정한다.
#     # 시간 t도 시각화에 큰 영향을 주지 않으므로, t=0으로 고정한다.
#     # F = F_k * W_k(rho) * exp(i(m*theta))

#     # 그릴 수 있는 모드가 많은데, n=4 만 그려보자
#     nn = 4

#     W = Wk[i]
#     Theta = np.exp(1j * (nn * theta))
#     F = W * np.real(Theta)

#     x = rs * np.cos(theta)
#     y = rs * np.sin(theta)

#     if param.basis == "bessel":
        
        


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