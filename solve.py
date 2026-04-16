# solve.py
# A 행렬을 조립하고 블록화시켜 고유값 문제를 풀어서 각 모드의 성장률과 진동수를 구한다. 결과를 플로팅한다.
print("[solve.py]")

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from utils import timed

@timed
def construct_A_matrix(mode_data, mat_data):
    # A 행렬을 조립한다.

    # 미리 계산된 행렬들을 가져온다.
    L, M, invM, J0, Dc = mat_data["L"], mat_data["M"], mat_data["invM"], mat_data["J0"], mat_data["Dc"]
    k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel = mat_data["k_parallel"], mat_data["n_k_parallel"], mat_data["Ti_k_parallel"], mat_data["tau_k_parallel"]
    D_glf, Gp, Gn, GTi, a, b = mat_data["D_glf"], mat_data["Gp"], mat_data["Gn"], mat_data["GTi"], mat_data["a"], mat_data["b"]
 
    A11 = invM @ (-a + Gp @ L + Gn @ J0 + 1j*Dc @ M)
    A12 = -invM @ b
    A13 = invM @ n_k_parallel
    A21 = GTi @ J0
    A22 = 1j * Dc + 1j * D_glf
    A23 = 2/3 * Ti_k_parallel
    A31 = k_parallel + tau_k_parallel
    A32 = k_parallel
    A33 = 1j * Dc

    A = np.block([[A11, A12, A13],
                  [A21, A22, A23],
                  [A31, A32, A33]])

    print(f"matrix A constructed. shape: {A.shape}")

    # 서로 다른 n 모드들은 독립적이다. 따라서 n 모드별로 A를 블록 대각화할 수 있다. 이를 이용해서 고유값 문제를 더 작은 크기로 나눌 수 있다.
    # ks에서 n 모드별로 인덱스를 찾는다.
    ks = mode_data["ks"]
    
    n_values = np.unique(ks[:, 0])  # ks에서 n 모드들의 고유한 n 값들을 찾는다.
    n_mode_indexes = {n: np.where(ks[:, 0] == n)[0] for n in n_values}  # 각 n 값에 대해 ks에서 해당 n 모드들의 인덱스를 찾는다.
    # 사용예
    # n = n_values[0]  # 예시로 첫 번째 n 모드를 선택한다.
    # indexes = n_mode_indexes[n]  # 선택한 n 모드에 해당하는 인덱스들을 가져온다.
    # A_n = A[np.ix_(indexes, indexes)]  # A에서 선택한 n 모드에 해당하는 부분 행렬을 추출한다

    # A의 블록화 결과 출력
    print("Block-diagonalized A matrix:")
    for n in n_values:
        print(f"{n_mode_indexes[n][0]}-{n_mode_indexes[n][-1]}", end=' ')
    print()
    
    return {
        "A": A,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
    }

@timed
def solve_eigenvalue_problem(matrix):
    n_values = matrix["n_values"]
    n_mode_indexes = matrix["n_mode_indexes"]
    A = matrix["A"]
    N = A.shape[0] // 3  # A는 3N x 3N 행렬이므로, N은 A의 크기의 1/3이다.)

    from scipy.sparse.linalg import eigs
    from scipy.sparse import csr_matrix

    # n 별로 A를 블록화하여 각 A 블록에서 고유값 문제를 풀어서 성장률과 진동수를 구한다.
    F_blocked, eigenvalues_blocked = [], [] # 블록 별로 고유값과 고유벡터를 저장할 리스트
    for n in n_values:
        idx = n_mode_indexes[n] # n 모드에 해당하는 인덱스들을 가져온다.
        # A들에서 각각 블록 행렬을 추출해야 하므로, 총 9개의 인덱스 뭉치가 필요하다.
        idx_full = np.concatenate([idx, idx + N, idx + 2*N])
        A_block = A[np.ix_(idx_full, idx_full)]

        print(f"finding eigenvalues of n={n} block, shape: {A_block.shape}")
        # eigenvalues, F = np.linalg.eig(A_block)

        # # sparse check
        # import numpy as np
        print(f"A matrix shape: {A_block.shape}, nnz: {np.count_nonzero(A_block)}, density: {np.count_nonzero(A_block)/A_block.size:.2e}")
        # from plot import plot_matrices
        # plot_matrices([A_block.real, A_block.imag], ["Real part of A", "Imaginary part of A"])

        A_block = csr_matrix(A_block)
        eigenvalues, F = eigs(A_block, k=1, which='LI') # Scipy sparse eigenvalue solver를 사용해서 가장 큰 성장률을 가지는 6개의 모드의 고유값과 고유벡터를 구한다. A는 희소 행렬이므로, eigs 함수를 사용한다. which='LI' 옵션은 가장 큰 실수 부분을 가지는 고유값을 찾도록 지정한다.

        F_blocked.append(F)
        eigenvalues_blocked.append(eigenvalues)

    print("eigenvalue problem solved for all n blocks.")

    """
    eigenvalues_blocked = [ [list of eigenvalues n1], [list of eigenvalues n2], ... ]
    F_blocked = [ [list of F n1], [list of F n2], ... ]

    gammas = [growth rate of most unstable mode n1, growth rate of most unstable mode n2, ... ]
    omegas = [frequency of most unstable mode n1, frequency of most unstable mode n2, ... ]
    most_unstable_mode_indexes = [index of max(gammas n1), index of max(gammas n2), ... ]
    ex) ks[most_unstable_mode_indexes[0]] -> n1 모드에서 가장 성장률이 큰 모드의 [n, m, p] 값
    """

    most_unstable_mode_indexes = np.empty_like(n_values, dtype=int) # 각 n 모드에서 가장 성장률이 큰 모드의 인덱스를 저장할 리스트. shape (len(n_values),)
    gammas = np.empty_like(n_values, dtype=float) # growth rates of most unstable modes
    omegas = np.empty_like(n_values, dtype=float) # frequency of most unstable modes
    for i, eigenvalues in enumerate(eigenvalues_blocked):
        most_unstable_mode_index = np.argmax(eigenvalues.imag) # 가장 성장률이 큰 모드의 인덱스를 찾는다.

        most_unstable_mode_indexes[i] = most_unstable_mode_index
        gammas[i] = eigenvalues[most_unstable_mode_index].imag
        omegas[i] = eigenvalues[most_unstable_mode_index].real

    return {
        "gammas": gammas,
        "omegas": omegas,
        "most_unstable_mode_indexes": most_unstable_mode_indexes,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
        "F_blocked": F_blocked,
        "eigenvalues_blocked": eigenvalues_blocked,
    }

def solve_time_evolution(param, matrix):
    # 시간 진화를 풀어서 모드의 성장률과 진동수를 구한다.
    n_values = matrix["n_values"]
    n_mode_indexes = matrix["n_mode_indexes"]
    A = -1j * matrix["A"] # ddt = -i omega
    # A = 1 * matrix["A"] # ddt = -i omega
    N = A.shape[0] // 3  # A는 3N x 3N 행렬이므로, N은 A의 크기의 1/3이다.)

    dt = param.dt # normalized time step
    ts = np.arange(0, param.T + dt, dt) # 정규화된 시간을 저장한다.

    # A행렬은 희소 행렬이므로 csr_matrix로 변환한다.
    from scipy.sparse import csr_matrix

    # n 별로 A를 블록화하여 각 A 블록에서 고유값 문제를 풀어서 성장률과 진동수를 구한다.
    
    # 내가 얻어야 하는 것은 초기에 매우 작게 주어진 모드 계수들이 시간에 따라 어떻게 진화하는지 이다.
    F0 = np.ones(shape=(3*N,), dtype=np.float64) * param.F0 # 초기값 F0는 아주 작은 섭동으로 주어진다. shape (3N,)
    Fs = [] # 각 모드 계수들의 시간 진화를 저장할 리스트. 모드마다 어레이 크기가 달라 넘파이 어레이 Fs를 파이썬 리스트로 담는다.
    F_blocked = [] # 시간 진화를 마친 n 모드별 모드 계수들을 각각 저장할 리스트.

    for n in n_values:
        idx = n_mode_indexes[n] # n 모드에 해당하는 인덱스들을 가져온다.
        Fs_block = np.empty((len(ts), len(idx)*3), dtype=complex) # n 모드에 해당하는 모드 계수들의 시간 진화를 저장할 어레이. shape (T, K_n)
        Fs_block[0] = F0[np.concatenate([idx, idx + N, idx + 2*N])] # 초기값 F0에서 n 모드에 해당하는 부분을 가져와서 시간 진화 시뮬레이션의 초기값으로 사용한다. shape (K_n,)
        
        # A들에서 각각 블록 행렬을 추출해야 하므로, 총 9개의 인덱스 뭉치가 필요하다.
        idx_full = np.concatenate([idx, idx + N, idx + 2*N])
        A_block = A[np.ix_(idx_full, idx_full)]
        A_block = csr_matrix(A_block) # A 블록을 희소 행렬로 변환한다.

        print(f"simulate time evolution of n={n} block")

        # 시간 진화는 RK4로 계산
        for i, t in enumerate(ts[:-1]):
            F_now = Fs_block[i]
            k1 = A_block @ F_now
            k2 = A_block @ (F_now + 0.5*k1*dt)
            k3 = A_block @ (F_now + 0.5*k2*dt)
            k4 = A_block @ (F_now + k3*dt)
            F_next = F_now + 1/6 * (k1 + 2*k2 + 2*k3 + k4)*dt
            Fs_block[i+1] = F_next

            # 진행 상황 로깅 및 상태 일부 저장
            if (i+1) % 1000 == 0 or i == len(ts)-2:
                print(f"n={n}, time step {i+1}/{len(ts)-1} completed.")

        Fs.append(Fs_block)
        F_blocked.append(Fs_block[-1])  # 마지막 시간 스텝의 모드 계수를 저장
    print("time evolution simulation completed for all n blocks.")

    # 성장률 계산
    gammas = np.empty(len(n_values))
    omegas = np.empty(len(n_values))
    
    fit_info = []

    for i, F in enumerate(Fs):   # F shape: (t, dof)
        a = np.linalg.norm(F, axis=1)
        y = np.log(np.maximum(a, 1e-300))

        i0 = int(0.8 * len(ts))
        i1 = len(ts)

        slope, intercept = np.polyfit(ts[i0:i1], y[i0:i1], 1)
        gammas[i] = slope
        fit_info.append((i0, i1, slope, intercept))

        dF_dt = np.gradient(F, ts, axis=0)
        den = np.maximum(np.sum(np.abs(F)**2, axis=1), 1e-300)
        alpha = np.sum(np.conj(F) * dF_dt, axis=1) / den
        omegas[i] = -np.mean(alpha.imag[i0:i1])

    # 제대로 작동하는지 확인하기 위해을 여기서 바로 plot해보기
    # 시간에 따른 ln(a)와 회귀선을 플로팅한다.
    # 회귀에 포함되는 영역을 나타낸다.
    plt.figure()
    for i, n in enumerate(n_values):
        i0, i1, slope, intercept = fit_info[i]
        a = np.linalg.norm(Fs[i], axis=1)
        y = np.log(np.maximum(a, 1e-300))

        plt.plot(ts, y, label=f"n={n}")
        plt.plot(ts[i0:i1], slope * ts[i0:i1] + intercept, "--")
    plt.xlabel("time")
    plt.ylabel("ln(||F||)")
    plt.title("Time evolution of mode amplitude and linear fit")
    plt.legend()
    plt.grid()
    plt.savefig(f"{param.save_dir}/time_evolution_gamma{param.suffix}.png")
    plt.show()

    return {
        "gammas": gammas,
        "ts": ts,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
        "Fs": Fs,
        "F_blocked": F_blocked,
        "most_unstable_mode_indexes": 0,
    }