# solve.py
# A 행렬을 조립하고 블록화시켜 고유값 문제를 풀어서 각 모드의 성장률과 진동수를 구한다. 결과를 플로팅한다.

print("[solve.py]")

import numpy as np
from utils import timed
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigs, LinearOperator, expm_multiply

@timed
def sparse_check(A):
    """
    density가 0.2보다 작으면 sparse, 그렇지 않으면 dense로 판단한다.

    density = nnz / total_elements    
    """
    nnz = np.count_nonzero(A)
    total_elements = A.size
    density = nnz / total_elements

    print(f"A matrix shape: {A.shape}, nnz: {nnz}, density: {density:.2e}")

    if density < 0.2:
        return "sparse"
    else:
        print(f"A matrix is too dense to be treated as sparse. Consider using a dense solver for this block.")
        return "dense"

@timed
def construct_A_matrix(mode_data, mat_data):
    '''A 행렬을 조립한다. A를 블록대각화하고 희소 행렬로 바꾼다.
    Input:
        - mode_data: 모드 정보가 담긴 딕셔너리.
        - mat_data: 미리 계산된 행렬들이 담긴 딕셔너리.

    Variables:
         - A: 조립된 A 행렬. shape (3N, 3N) N은 모드의 총 개수이다.
         

    Output:
        - blocked_A: n 모드별로 블록화된 A 행렬 딕셔너리
            {n1: A_block_n1, n2: A_block_n2, ...}
        - n_indexes: 해당 n 모드에 해당하는 인덱스들을 저장한 딕셔너리
            {n1: [index1, index2, ...], n2: [index1, index2, ...], ...}
        - n_values: ks에서 n 모드들의 고유한 n 값들을 저장한 배열. shape (num_n_modes,)

        - N: 모드의 총 개수. A 행렬의 크기는 3N x 3N이므로, N은 A의 크기의 1/3이다.
        
    '''

    # 미리 계산된 행렬들을 가져온다.
    L, M, invM, J0, Dc = mat_data["L"], mat_data["M"], mat_data["invM"], mat_data["J0"], mat_data["Dc"]
    k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel = mat_data["k_parallel"], mat_data["n_k_parallel"], mat_data["Ti_k_parallel"], mat_data["tau_k_parallel"]
    D_glf, Gp, Gn, GTi, a, b = mat_data["D_glf"], mat_data["Gp"], mat_data["Gn"], mat_data["GTi"], mat_data["a"], mat_data["b"]
    
    # A 행렬을 조립한다. A는 3N x 3N 행렬로, N은 모드의 총 개수이다.
    A11 = invM @ (-a + Gp @ L + Gn @ J0 + 1j*Dc @ M)
    A12 = -invM @ b
    A13 = invM @ n_k_parallel
    A21 = GTi @ J0
    A22 = 1j * Dc + 1j * D_glf
    A23 = 2/3 * Ti_k_parallel
    A31 = k_parallel + tau_k_parallel
    A32 = k_parallel
    A33 = 1j * Dc

    # matlab 코드는 3N x 3N 크기의 A 행렬을 조립한 다음, n 모드별로 블록 대각화해서 각 n 모드에 해당하는 A 블록을 저장한다.
    # 대신 메모리를 아끼기 위해
    # 블록 대각화 및 희소 행렬 변환 한 다음 각 n 모드별로 A 블록을 저장한다.

    # 서로 다른 n 모드들은 독립적이다. 따라서 n 모드별로 A를 블록 대각화할 수 있다. 이를 이용해서 고유값 문제를 더 작은 크기로 나눌 수 있다.
    # ks에서 n 모드별로 인덱스를 찾는다.
    ks = mode_data["ks"]
    n_values = np.unique(ks[:, 0])  # ks에서 n 모드들의 고유한 n 값들을 찾는다.
    n_mode_indexes = {n: np.where(ks[:, 0] == n)[0] for n in n_values}  # 각 n 값에 대해 ks에서 해당 n 모드들의 인덱스를 찾는다.
    
    # 사용예
    # n1 = n_values[0] # 예시로 첫 번째 n 모드를 선택한다.
    # indexes = n_mode_indexes[n1]  # 선택한 n 모드에 해당하는 인덱스들을 가져온다.
    # A_n = A[np.ix_(indexes, indexes)]  # A에서 선택한 n 모드에 해당하는 부분 행렬을 추출한다

    blocked_A = {}

    for n in n_values:
        idx = n_mode_indexes[n]

        A_block = np.block([
            [
                A11[np.ix_(idx, idx)],
                A12[np.ix_(idx, idx)],
                A13[np.ix_(idx, idx)],
            ],
            [
                A21[np.ix_(idx, idx)],
                A22[np.ix_(idx, idx)],
                A23[np.ix_(idx, idx)],
            ],
            [
                A31[np.ix_(idx, idx)],
                A32[np.ix_(idx, idx)],
                A33[np.ix_(idx, idx)],
            ],
        ])

        if sparse_check(A_block) == "sparse":
            A_block = csr_matrix(A_block)
            A_block.eliminate_zeros() # 희소 행렬로 변환한 후, 0 요소를 제거해서 메모리를 더 절약한다.

        blocked_A[n] = A_block

    # A의 블록화 결과 출력
    print(f"A matrix is block diagonalied by {len(n_values)} blocks:")
    for n in n_values:
        print(f"n={n}: N={len(n_mode_indexes[n])}")
    
    return {
        "blocked_A": blocked_A,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
        "N": len(ks)
    }

def solve_one_block(matrix):
    if issparse(matrix):
        return eigs(matrix, k=1, which='LI')
    
    N = matrix.shape[0]

    # 작은 dense block은 full eigenvalue solver, 큰 dense block은 sparse matvec 기반 eigs solver
    if N <= 2000:
        return np.linalg.eig(matrix)
    else:
        # 큰 dense block이면 dense matvec 기반 LinearOperator
        Aop = LinearOperator(
            shape=matrix.shape,
            matvec=lambda x: matrix @ x,
            dtype=matrix.dtype,
        )

        return eigs(Aop, k=1, which="LI")
    
def get_gamma_omega(eigenvalues_blocked, n_values):
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

    return gammas, omegas, most_unstable_mode_indexes

@timed
def solve_eigenvalue_problem(matrix):
    """A 행렬의 고유값 문제를 풀어서 각 n 모드에서 가장 성장률이 큰 모드의 성장률과 진동수를 구한다.
    
    """
    n_values = matrix["n_values"]
    n_mode_indexes = matrix["n_mode_indexes"]
    blocked_A = matrix["blocked_A"]
    
    # n 별로 A를 블록화하여 각 A 블록에서 고유값 문제를 풀어서 성장률과 진동수를 구한다.
    F_blocked, eigenvalues_blocked = [], [] # 블록 별로 고유값과 고유벡터를 저장할 리스트
    for n in n_values:
        A_block = blocked_A[n]

        print(f"finding eigenvalues of n={n} block, shape: {A_block.shape}")
        
        # A 블록이 희소 행렬인지 여부에 따라 적절한 고유값 솔버를 사용한다.
        eigenvalues, F = solve_one_block(A_block)

        F_blocked.append(F)
        eigenvalues_blocked.append(eigenvalues)

    print("eigenvalue problem solved for all n blocks.")

    # 각 n 모드에서 가장 성장률이 큰 모드의 성장률과 진동수를 구한다.
    print("calculating growth rates and frequencies of most unstable modes for each n block.")
    gammas, omegas, most_unstable_mode_indexes = get_gamma_omega(eigenvalues_blocked, n_values)
    
    return {
        "gammas": gammas,
        "omegas": omegas,
        "most_unstable_mode_indexes": most_unstable_mode_indexes,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
        "F_blocked": F_blocked,
        "eigenvalues_blocked": eigenvalues_blocked,
    }

def lnF(F):
    norm = np.linalg.norm(F)
    return np.log(np.maximum(norm, 1e-300)) # underflow 방지

def lnF_and_alpha(F, BF, eps=1e-300):
    """Return ln(||F||) and alpha=<F,BF>/<F,F> using an already computed BF."""
    denominator = np.vdot(F, F)
    log_amp = 0.5 * np.log(max(denominator.real, eps))

    if abs(denominator) < eps:
        alpha_value = np.nan + 1j * np.nan
    else:
        alpha_value = np.vdot(F, BF) / denominator

    return log_amp, alpha_value

def alpha(F, BF, eps=1e-300):
    # k1 = B_block @ F에서 계산된 k1을 이용해서 alpha를 계산한다. alpha = (F^* @ k1) / (F^* @ F)
    numerator = np.vdot(F, BF)
    denominator = np.vdot(F, F)

    # 분모가 너무 작을 경우 발산 방지
    if abs(denominator) < eps:
        return np.nan + 1j * np.nan
    
    return numerator / denominator

def calc_gamma_omega(lnFs, alphas, ts, n_values, fit_start_fraction=0.8):
    """
    Input:
        lnFs:
            shape (num_n, Nt)
            ln ||F(t)||

        alphas:
            shape (num_n, Nt)
            alpha(t) = <F, B F> / <F, F>
            dF/dt = B F일 때, dominant mode에서는
            alpha -> gamma - i omega

        ts:
            shape (Nt,)

    Output:
        gammas:
            ln ||F||의 late-time slope

        omegas:
            -Im(alpha)의 late-time average

        fit_info:
            fit 품질 확인용 정보
    """
    gammas = np.empty(len(n_values), dtype=float)
    omegas = np.empty(len(n_values), dtype=float)
    fit_info = []

    ts = np.asarray(ts, dtype=float)

    t_fit_start = ts[0] + fit_start_fraction * (ts[-1] - ts[0])

    for i, n in enumerate(n_values):
        y = np.asarray(lnFs[i], dtype=float)
        a = np.asarray(alphas[i], dtype=complex)

        finite = (
            np.isfinite(ts)
            & np.isfinite(y)
            & np.isfinite(a.real)
            & np.isfinite(a.imag)
        )

        fit_mask = finite & (ts >= t_fit_start)

        if np.count_nonzero(fit_mask) < 2:
            raise ValueError(f"n={n}: not enough valid points for gamma/omega fit")

        t_fit = ts[fit_mask]
        y_fit = y[fit_mask]
        a_fit = a[fit_mask]

        # ln ||F|| = gamma * t + intercept
        X = np.column_stack([t_fit, np.ones_like(t_fit)])
        gamma, intercept = np.linalg.lstsq(X, y_fit, rcond=None)[0]

        y_pred = gamma * t_fit + intercept
        ss_res = np.sum((y_fit - y_pred)**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)

        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = np.nan

        # alpha = gamma - i omega
        omega = -np.mean(a_fit.imag)
        gamma_alpha = np.mean(a_fit.real)

        omega_std = np.std(-a_fit.imag)
        gamma_alpha_std = np.std(a_fit.real)

        gammas[i] = gamma
        omegas[i] = omega

        fit_info.append({
            "n": int(n),
            "fit_t_start": float(t_fit[0]),
            "fit_t_end": float(t_fit[-1]),
            "fit_points": int(len(t_fit)),
            "gamma": float(gamma),
            "omega": float(omega),
            "intercept": float(intercept),
            "r2": float(r2),
            "gamma_alpha": float(gamma_alpha),
            "gamma_alpha_std": float(gamma_alpha_std),
            "omega_std": float(omega_std),
        })

        print(
            f"n={n}: "
            f"gamma={gamma:.6e}, "
            f"omega={omega:.6e}, "
            f"R2={r2:.6f}, "
            f"gamma_alpha={gamma_alpha:.6e}, "
            f"omega_std={omega_std:.3e}"
        )

    return gammas, omegas, fit_info

def solve_time_evolution(param, matrix):
    """dF/dt = B @ F 시간 진화를 풀어서 모드의 성장률과 진동수를 구한다.
    B = -j * A

    모든 시간 스텝에 대한 F값들을 저장할 필요는 없으므로, 모든 t에서의 ln|F|, alpha, 최종 F만 저장한다.

    출력의 lnFs, alphas는 처리를 통해 성장률과 진동수를 구하는 데 사용될 것이다. 
    예를 들어, 충분히 수렴한 후의 ln|F|에 선형 회귀를 적용해서 성장률을 계산할 수 있다. 진동수는 F의 시간 변화에서 계산할 수 있다.

    Input:
        - param: 시뮬레이션 파라미터가 담긴 Params 객체
        - matrix: construct_A_matrix 함수에서 반환된 A 행렬과 n 모드 정보가 담긴 딕셔너리
    Variables:
        F0: 아주 작은 섭동의 초기 모드 계수 벡터. shape (3N,)
        Fs: [F_block_n1, F_block_n2, ...]
        F_block_n1: shape (T, K_n1) n1 모드에 해당하는 모드 계수들의 시간 진화.
    Output:
        - ts: 시간 스텝 배열 shape (T,)
        - lnFs: [lnF(F_block_n1), lnF(F_block_n2), ...] shape (len(n_values), T)
        - alphas: [alpha(F_block_n1), alpha(F_block_n2), ...] shape (len(n_values), T)
        - F_final_state: n 모드별로 시간 진화를 마친 최종 모드 계수들을 저장한 리스트. [F_block_n1_final, F_block_n2_final, ...] shape of each F_block_n_final: (K_n,)
    """

    n_values = matrix["n_values"]
    n_mode_indexes = matrix["n_mode_indexes"]
    N = matrix["N"]

    dt = param.dt # normalized time step
    steps = int(round(param.T / dt))
    ts = dt * np.arange(steps + 1) # 정규화된 시간을 저장한다.
    
    F0 = np.ones(shape=(3*N,), dtype=np.complex128) * param.F0 # 초기값 F0는 아주 작은 섭동으로 주어진다. shape (3N,)
    
    lnFs = np.empty((len(n_values), len(ts)))
    alphas = np.empty((len(n_values), len(ts)), dtype=complex)
    F_block_final_state = []
    
    for i, n in enumerate(n_values):
        # 초기화
        idx = n_mode_indexes[n] # n 모드에 해당하는 인덱스들을 가져온다.
        idx_full = np.concatenate([idx, idx + N, idx + 2*N]) # n 모드에 해당하는 인덱스들의 전체 인덱스. shape (3*K_n,)
        F_now = F0[idx_full].copy() # 초기값 F0에서 n 모드에 해당하는 부분을 가져와서 시간 진화 시뮬레이션의 초기값으로 사용한다. shape (K_n,)
        B_block = -1j * matrix["blocked_A"][n]
        lnFs_block = np.zeros_like(ts) # 시간 스텝마다 ln(||F||) 값을 저장할 어레이. shape (T,)
        alphas_block = np.zeros_like(ts, dtype=complex) # 시간 스텝마다 alpha 값을 저장할 어레이. shape (T,)

        print(f"calculating time evolution of n={n} block, shape: {B_block.shape}")

        # 시간 적분
        for j in range(steps):
            # RK4 계수 게산
            k1 = B_block @ F_now

            # 계량값 저장
            lnFs_block[j], alphas_block[j] = lnF_and_alpha(F_now, k1)

            k2 = B_block @ (F_now + 0.5*k1*dt)
            k3 = B_block @ (F_now + 0.5*k2*dt)
            k4 = B_block @ (F_now + k3*dt)

            # 상태 업데이트
            F_now = F_now + 1/6 * (k1 + 2*k2 + 2*k3 + k4)*dt

            # logging
            if (j+1) % 1000 == 0 or (j+1) == steps:
                print(f"n={n}, time step {j+1}/{steps}")
        
        # 마지막 인덱스 계량값 저장
        BF_final = B_block @ F_now
        lnFs_block[-1], alphas_block[-1] = lnF_and_alpha(F_now, BF_final)

        print(f"completed.")
        
        # 결과 저장
        lnFs[i] = lnFs_block
        alphas[i] = alphas_block
        F_block_final_state.append(F_now) # 마지막 시간 스텝의 모드 계수를 저장
    
    print("time evolution completed for all blocks.")

    gammas, omegas, fit_info = calc_gamma_omega(lnFs, alphas, ts, n_values)

    return {
        "ts": ts,
        "lnFs": lnFs,
        "alphas": alphas,
        "F_block_final_state": F_block_final_state,
        "gammas": gammas,
        "omegas": omegas,
        "fit_info": fit_info,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
        "most_unstable_mode_indexes": None,
    }

@timed
def solve_matrix_exponential(param, matrix):
    """dF/dt = B @ F, B = -1j*A 를 matrix exponential으로 시간 진화시킨다.

    RK4와 같은 출력 형식을 사용하되, 전체 Fs(t)는 저장하지 않는다.

    expm_multiply는 지정된 시간 구간의 여러 time sample을 한 번에 반환하므로,
    전체 time sample을 한 번에 요청하면 다시 큰 메모리를 쓰게 된다.
    따라서 param.expm_chunk_steps 단위로 나누어 호출한다.
    """

    n_values = matrix["n_values"]
    n_mode_indexes = matrix["n_mode_indexes"]
    N = matrix["N"]

    dt = param.dt
    steps = int(round(param.T / dt))
    ts = dt * np.arange(steps + 1)

    chunk_steps = int(getattr(param, "expm_chunk_steps", 1000))
    if chunk_steps <= 0:
        chunk_steps = steps
    chunk_steps = max(1, min(chunk_steps, max(1, steps)))

    F0 = np.ones(shape=(3*N,), dtype=np.complex128) * param.F0

    lnFs = np.empty((len(n_values), len(ts)), dtype=float)
    alphas = np.empty((len(n_values), len(ts)), dtype=complex)
    F_block_final_state = []

    print(
        "matrix exponential time evolution: "
        f"steps={steps}, dt={dt}, chunk_steps={chunk_steps}"
    )

    for i, n in enumerate(n_values):
        idx = n_mode_indexes[n]
        idx_full = np.concatenate([idx, idx + N, idx + 2*N])

        F_now = F0[idx_full].copy()
        B_block = -1j * matrix["blocked_A"][n]

        lnFs_block = np.empty_like(ts, dtype=float)
        alphas_block = np.empty_like(ts, dtype=complex)

        print(f"calculating matrix exponential evolution of n={n} block, shape: {B_block.shape}")

        step0 = 0
        while step0 < steps:
            local_steps = min(chunk_steps, steps - step0)

            # states[j] = exp(B_block * (j*dt)) @ F_now, j=0..local_steps
            # Only this chunk is materialized, then discarded.
            states = expm_multiply(
                B_block,
                F_now,
                start=0.0,
                stop=local_steps * dt,
                num=local_steps + 1,
                endpoint=True,
            )

            # Record all states except the last one. The last state becomes the
            # initial condition of the next chunk and will be recorded there.
            for local_j in range(local_steps):
                global_j = step0 + local_j
                F = states[local_j]
                BF = B_block @ F
                lnFs_block[global_j], alphas_block[global_j] = lnF_and_alpha(F, BF)

            F_now = np.asarray(states[-1], dtype=np.complex128).copy()
            step0 += local_steps

            print(f"n={n}, expm time step {step0}/{steps}")

        # 마지막 시각 T 기록
        BF_final = B_block @ F_now
        lnFs_block[-1], alphas_block[-1] = lnF_and_alpha(F_now, BF_final)

        print("completed.")

        lnFs[i] = lnFs_block
        alphas[i] = alphas_block
        F_block_final_state.append(F_now)

    print("matrix exponential evolution completed for all blocks.")

    gammas, omegas, fit_info = calc_gamma_omega(lnFs, alphas, ts, n_values)

    return {
        "ts": ts,
        "lnFs": lnFs,
        "alphas": alphas,
        "F_final_state": F_block_final_state,
        "F_block_final_state": F_block_final_state,
        "gammas": gammas,
        "omegas": omegas,
        "fit_info": fit_info,
        "n_values": n_values,
        "n_mode_indexes": n_mode_indexes,
        "most_unstable_mode_indexes": None,
    }
