# solve_time_evolution.py
# A 행렬을 조립하고 블록화시켜 고유값 문제를 풀어서 각 모드의 성장률과 진동수를 구한다. 결과를 플로팅한다.
# F의 시간 함수를 exp(-iwt)가 아닌 미지 함수 T(t)로 바꿔서, dT/dt = A @ T 형태의 선형 미분 방정식을 풀어서 시간 진화를 시뮬레이션한다.
# dF/dt = -iw F = -i AF이므로 기존 A에 -i를 곱하고 풀어야 한다.

print("[solve_time_evolution.py]")

import numpy as np
import parameters as param
import modes
import matrices
import scipy.special as sp
import matplotlib.pyplot as plt

ks = modes.ks
N = len(ks)

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

A = -1j*np.block([[A11, A12, A13],
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

# time evolution
dt = 0.05 # time step
ts = np.arange(0, 10.0+dt, dt) # 정규화된 시간을 저장한다.
update_term = 20 # update_term 타임스텝마다 행렬 업데이트. 예를 들어, update_term=10이면, 10 타임스텝마다 행렬을 업데이트한다.
FFs = [] # 각 모드 계수들의 시간 진화를 저장할 리스트.
# 모드마다 어레이 크기가 달라 넘파이 어레이 Fs를 파이썬 리스트로 담는다.

# 초기값 F0를 구하기 위해 T(t) = exp(-iwt)를 가정하고 푼다.

# 시간 진화 시뮬레이션
for n in n_values:
    idx = n_mode_indexes[n] # n 모드에 해당하는 인덱스들을 가져온다.
    # A들에서 각각 블록 행렬을 추출해야 하므로, 총 9개의 인덱스 뭉치가 필요하다.
    idx_full = np.concatenate([idx, idx + N, idx + 2*N])
    A_block = A[np.ix_(idx_full, idx_full)]

    print(f"find eigenvalues of n={n} block")
    omega, F = np.linalg.eig(A_block)
    Fs = np.empty((len(ts), len(idx_full), len(idx_full)), dtype=complex) # n 모드에 해당하는 모드 계수들의 시간 진화를 저장할 어레이. shape (K_n, T)
    Fs[0] = F

    print(f"simulate time evolution of n={n} block")
    for i, t in enumerate(ts[:-1]):
        F_now = Fs[i]
        k1 = A_block @ F_now
        k2 = A_block @ (F_now + 0.5*k1*dt)
        k3 = A_block @ (F_now + 0.5*k2*dt)
        k4 = A_block @ (F_now + k3*dt)
        F_next = F_now + 1/6 * (k1 + 2*k2 + 2*k3 + k4)*dt
        Fs[i+1] = F_next

    FFs.append(Fs)

# n=n_values[0]인 모드 계수의 시간 진화를 플로팅해보자.
n = n_values[0]
indexes = n_mode_indexes[n]
Fs = FFs[0] # n=n_values[0]인 모드 계수들의 시간 진화. shape (ts, K_n)
plt.figure(figsize=(10, 6))
for i in range(Fs.shape[1]):
    plt.plot(ts, np.real(Fs[:, i]), label=f'mode {i} real')
    plt.plot(ts, np.imag(Fs[:, i]), label=f'mode {i} imag', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Mode Coefficients')
plt.title(f'Time Evolution of Mode Coefficients for n={n}')
plt.legend()
plt.grid()
plt.show()