# solve.py
# A 행렬을 조립하고 블록화시켜 고유값 문제를 풀어서 각 모드의 성장률과 진동수를 구한다. 결과를 플로팅한다.
print("[solve.py]")

import numpy as np
import parameters as param
import modes
import matrices
import scipy.special as sp
import matplotlib.pyplot as plt

ks = modes.ks

L, M, invM, J0, Dc = matrices.L, matrices.M, matrices.invM, matrices.J0, matrices.Dc
k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel = matrices.k_parallel, matrices.n_k_parallel, matrices.Ti_k_parallel, matrices.tau_k_parallel
D_glf, Gp, Gn, GTi, a, b = matrices.D_glf, matrices.Gp, matrices.Gn, matrices.GTi, matrices.a, matrices.b
n_hat, Ti_hat, Te_hat = param.n_hat, param.Ti_hat, param.Te_hat

A11 = invM @ (-a + Gp @ L + Gn @ J0 + 1j*Dc @ M)
A12 = -invM @ b
A13 = -invM @ n_k_parallel
A21 = GTi @ J0
A22 = 1j * Dc + 1j * D_glf
A23 = 2/3 * Ti_k_parallel
A31 = k_parallel + tau_k_parallel
A32 = k_parallel
A33 = 1j * Dc

A = np.block([[A11, A12, A13],
              [A21, A22, A23],
              [A31, A32, A33]])
print(f"complete constructing A matrix. shape: {A.shape}")


# 서로 다른 n 모드들은 독립적이다. 따라서 n 모드별로 A를 블록 대각화할 수 있다. 이를 이용해서 고유값 문제를 더 작은 크기로 나눌 수 있다.
# ks에서 n 모드별로 인덱스를 찾는다.
n_values = np.unique(ks[:, 0])  # ks에서 n 모드들의 고유한 n 값들을 찾는다.
n_mode_indexes = {n: np.where(ks[:, 0] == n)[0] for n in n_values}  # 각 n 값에 대해 ks에서 해당 n 모드들의 인덱스를 찾는다.
# 사용예
# n = n_values[0]  # 예시로 첫 번째 n 모드를 선택한다.
# indexes = n_mode_indexes[n]  # 선택한 n 모드에 해당하는 인덱스들을 가져온다.
# A_n = A[np.ix_(indexes, indexes)]  # A에서 선택한 n 모드에 해당하는 부분 행렬을 추출한다

F_blocked, omega_blocked = [], []
for n in n_values:
    indexes = n_mode_indexes[n]
    A_block = A[np.ix_(indexes, indexes)]

    print(f"find eigenvalues of n={n} block")
    F, omega = np.linalg.eig(A_block)
    F_blocked.append(F)
    omega_blocked.append(omega)

# 고윳값을 실수와 허수로 분리한다.
omega_blocked_re = [np.real(omega) for omega in omega_blocked]
omega_blocked_im = [np.imag(omega) for omega in omega_blocked]

# 성장률에 해당하는 허수 고윳값 중 가장 큰 값만을 모은다.
gammas = [np.max(omega_im) for omega_im in omega_blocked_im]
print(f"gammas for each n mode: {gammas}")
# 진동수에 해당하는 실수 고윳값 중 가장 큰 값만을 모은다.
omegas = [np.max(omega_re) for omega_re in omega_blocked_re]

# 각 모드 별 성장률 비교를 위해서, x축에 해당하는 값 ktheta_rho_i를 계산한다. 
# k_theta_rho_i ~ nq/r * rhos0
# cyclon case parameters
q_val = 1.4 # q at r=0.5a
r_val = 0.5 # 0.5a 
k_thetas_rho_i = n_values * q_val / r_val * param.rhos0

gamma_factor = param.R_Lne / param.rmajor
omega_factor = gamma_factor*4
gammas = np.array(gammas) / gamma_factor
omegas = np.array(omegas) / omega_factor

# 결과를 플로팅한다.
plt.figure(figsize=(10, 6))
plt.plot(k_thetas_rho_i, gammas, 'o-', label='Growth Rate') # 파란색 점
# plt.plot(n_values, omegas*0.25, 's-', label='Frequency/4') # 빨간색 점
plt.xlabel(r'$k_{\theta} \rho_i$')
plt.ylabel('Growth Rate, Frequency/4')
text = f"basis: {param.basis}\nparameters:\n {param.n_start} <= n <= {param.n_end} $Delta$n={param.n_delta}\n {param.m_start} <= m <= {param.m_end} $Delta$m={param.m_delta}\n 0 <= p < {param.p-1}\n"
plt.text(0.5, 0.5, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
plt.legend()
plt.grid()
plt.show()

# k_theta ~ nq/r