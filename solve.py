# solve.py
# A 행렬을 조립하고 블록화시켜 고유값 문제를 풀어서 각 모드의 성장률과 진동수를 구한다. 결과를 플로팅한다.
print("[solve.py]")

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

def construct_A_matrix(matrices):
    # A 행렬을 조립한다.    

    # 미리 계산된 행렬들을 가져온다.
    L, M, invM, J0, Dc = matrices["L"], matrices["M"], matrices["invM"], matrices["J0"], matrices["Dc"]
    k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel = matrices["k_parallel"], matrices["n_k_parallel"], matrices["Ti_k_parallel"], matrices["tau_k_parallel"]
    D_glf, Gp, Gn, GTi, a, b = matrices["D_glf"], matrices["Gp"], matrices["Gn"], matrices["GTi"], matrices["a"], matrices["b"]
 
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

    print(f"complete constructing A matrix. shape: {A.shape}")

    return {
        "A": A
    }

def solve_eigenvalue_problem(mode_data, matrix):
    # 서로 다른 n 모드들은 독립적이다. 따라서 n 모드별로 A를 블록 대각화할 수 있다. 이를 이용해서 고유값 문제를 더 작은 크기로 나눌 수 있다.
    # ks에서 n 모드별로 인덱스를 찾는다.
    ks = mode_data["ks"]
    N = len(ks)
    A = matrix["A"]
    
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

    # n 별로 A 블록에서 고유값 문제를 풀어서 성장률과 진동수를 구한다.
    F_blocked, eigenvalues_blocked = [], [] # 블록 별로 고유값과 고유벡터를 저장할 리스트
    for n in n_values:
        idx = n_mode_indexes[n] # n 모드에 해당하는 인덱스들을 가져온다.
        # A들에서 각각 블록 행렬을 추출해야 하므로, 총 9개의 인덱스 뭉치가 필요하다.
        idx_full = np.concatenate([idx, idx + N, idx + 2*N])
        A_block = A[np.ix_(idx_full, idx_full)]

        print(f"finding eigenvalues of n={n} block, shape: {A_block.shape}")
        eigenvalues, F = np.linalg.eig(A_block)
        F_blocked.append(F)
        eigenvalues_blocked.append(eigenvalues)

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
