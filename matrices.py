# matrices.py
# 행렬 A를 구성하는 각각의 연산자 행렬을 계산한다.
print("[matrices.py]")

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from utils import timed


def _build_nm_groups(ks):
    groups = {}

    for idx, (n, m, _) in enumerate(ks):
        key = (int(n), int(m))
        groups.setdefault(key, []).append(idx)

    for key, idxs in groups.items():
        idxs.sort(key=lambda i: int(ks[i, 2]))
        groups[key] = np.asarray(idxs, dtype=int)

    return groups


def _weighted_cross(W_left, W_right, weight):
    return (W_left * weight[None, :]) @ W_right.T


@timed
def build_matrices(param, profiles, mode_data):
    # mode data 불러오기
    ks = mode_data["ks"]
    mode_radius_indexes = mode_data["mode_radius_indexes"]
    mode_q_values = mode_data["mode_q_values"]
    p_minus = mode_data["p_minus"]
    p_plus = mode_data["p_plus"]
    p_2minus = mode_data["p_2minus"]
    p_2plus = mode_data["p_2plus"]
    same_nm = mode_data["same_nm"]
    
    # %% parameters 불러오기

    # 반경 변수 정의
    dr_ = param.dr # dr value
    rs = profiles["rs"] # r values, shape (r_num,)
    dr = np.full(param.r_num, dr_) # dr list
    dr[-1] = 0.5*dr_ # 마지막 가중치는 절반
    rdr = rs*dr # r*dr 적분에 사용

    rho_s = param.rho_s
    rmajor = param.rmajor
    basis = param.basis

    # 평형 프로파일 불러오기
    q = profiles["q_profile"](rs)
    n_hat = profiles["n_hat"]
    Te_hat = profiles["Te_hat"]
    Ti_hat = profiles["Ti_hat"]
    pi_hat = profiles["pi_hat"]
    tau = profiles["tau"]

    dpdr = profiles["dpi_dr"]
    dndr = profiles["dn_dr"]
    dTidr = profiles["dTi_dr"]

    d_lnn_dr = profiles["d_lnn_dr"]
    d_lnTi_dr = profiles["d_lnTi_dr"]
    d_lnTe_dr = profiles["d_lnTe_dr"]
    d_lnpi_dr = profiles["d_lnpi_dr"]
    d_lntau_dr = profiles["d_lntau_dr"]

    # %% basis 종류에 따라 W, dWdr, L, M, invM, J0, Dc 계산하기

    # W, dWdr, L, M, invM, J0, Dc는 basis에 따라 달라지므로, basis마다 다르게 계산해준다.

    # initialize matrices
    W = np.empty((len(ks), len(rs)), dtype=float) # W[k, r] = Wk(r)
    dWdr = np.empty_like(W) # dWdr[k, r] = dWk/dr(r)

    N = len(ks) # mode의 개수
    L = np.zeros((N, N), dtype=float) # L[k, k] = 
    J0 = np.zeros_like(L) # J0[k, k] = 1/(1-0.5*L[k, k])
    Dc = np.zeros_like(L) # Dc[k, k] = mu1 * L[k, k] - mu2 * L[k, k]^2
    M = np.zeros_like(L) #
    invM = np.zeros_like(L)

    # 베셀 함수의 기저 경우.
    if basis=="bessel": 
        # 베셀 함수 영점 alpha[m, p] 계산
        # alpha[m, p] = p+1번째 영점 of J_m
        alpha = np.empty((param.m+1, param.p)) # j_m (alpha[m, p]) = 0
        for m in range(param.m+1):
            alpha[m] = sp.jn_zeros(m, param.p)

        # Wk(r)와 그 도함수들을 미리 계산한다.

        # Wk는 k 값에 따라 달라짐. 이를 미리 다 계산해줌.
        # W = [Wk(k1), Wk(k2), ..., Wk(kK)], shape (K, r_num)
        # Wk(r) = sqrt(2) / J_{m+1}(alpha[m, p]) * J_m(alpha[m, p] * r)

        for i, (n, m, p) in enumerate(ks):
            Wk = np.sqrt(2) / sp.jv(m+1, alpha[m, p]) * sp.jv(m, alpha[m, p] * rs)
            dWdrk = np.sqrt(2) / sp.jv(m+1, alpha[m, p]) * sp.jvp(m, alpha[m, p] * rs) * alpha[m, p]

            W[i] = Wk
            dWdr[i] = dWdrk

            # 대각 행렬들 만들기
            rho_mn_index = mode_radius_indexes[i] # mode의 반지름에 가장 가까운 r의 인덱스

            L[i, i] = - rho_s**2 * alpha[m, p]**2
            J0[i, i] = 1/(1-0.5*L[i, i])
            Dc[i, i] = param.damping_c * (param.mu1 * L[i, i] - param.mu2 * L[i, i]**2)
            M[i, i] = n_hat[rho_mn_index] / Te_hat[rho_mn_index] - n_hat[rho_mn_index] * L[i, i]
            invM[i, i] = 1.0 / M[i, i] if M[i, i] != 0 else 0.0

        print(f"W, dWdr, d2Wdr2 computed. Shapes: {W.shape}")
        print(f"L, M, invM, J0, Dc matrices computed. Shapes: {L.shape}")


    # 에르미트 함수 기저의 경우
    elif basis=="hermite":
        #######################################################
        # Wnmp(rho) = 1/sqrt(2 rho w_mn) 1/v_p * H_p(x) * exp(-x^2/2)
        # x = (rho - rho_mn) / w_mn
        # w_mn = 5 rho_s
        # v_p = 2^(p/2) Gamma(p+1)^1/2 pi^1/4

        # 모드별 w_mn 계산
        base_w_mn = param.w_mn 
        p = param.p

        w_modes = np.empty(len(ks), dtype=float)

        for i, (n, m, _) in enumerate(ks):
            r0 = rs[mode_radius_indexes[i]] # mode의 반지름에 가장 가까운 r 값

            w = base_w_mn

            # Same logic as original MATLAB eigensolver_init.m
            if r0 < (5.0 / 7.0) * p * w:
                w = (7.0 / 5.0) * r0 / p

            elif (1.0 - r0) < (4.0 / 7.0) * p * w:
                w = (7.0 / 4.0) * (1.0 - r0) / p

            w_modes[i] = w

        # W 계산
        for i, (n, m, p) in enumerate(ks):
            w_mn = w_modes[i]
            x = (rs - rs[mode_radius_indexes[i]])/w_mn
            v_p = 2**(p/2) * np.sqrt(sp.gamma(p+1)) * np.pi**0.25
            Wk = 1.0/(np.sqrt(2 * rs * w_mn) * v_p) * sp.eval_hermite(p, x) * np.exp(-0.5 * x*x)
            W[i] = Wk

        # dWdr 계산
        for i, (n, m, p) in enumerate(ks):
            w_mn = w_modes[i]
            p_minus_i = p_minus[i]
            p_plus_i = p_plus[i]
            p_2minus_i = p_2minus[i]
            p_2plus_i = p_2plus[i]

            Wk_p_minus = W[p_minus_i] if p_minus_i != -1 else 0
            Wk_p_plus = W[p_plus_i] if p_plus_i != -1 else 0
            Wk_p_2minus = W[p_2minus_i] if p_2minus_i != -1 else 0
            Wk_p_2plus = W[p_2plus_i] if p_2plus_i != -1 else 0

            dWdrk = 1/w_mn * ( np.sqrt(p/2) * Wk_p_minus - w_mn/(2*rs) * W[i] - np.sqrt((p+1)*0.5) * Wk_p_plus )
            dWdr[i] = dWdrk

        # normalize W, dWdr
        for i in range(len(ks)):
            norm = np.sum(W[i] * W[i] * rdr)
            W[i] /= np.sqrt(norm)
            dWdr[i] /= np.sqrt(norm)

        # Laplacian
        for i, (n, m, p) in enumerate(ks):
            rho_mn_index = mode_radius_indexes[i] # mode의 반지름에 가장 가까운 r의 인덱스
            p_2minus_i = p_2minus[i] # n, m이 같고 p2 = p-2인 모드의 인덱스
            p_2plus_i = p_2plus[i] # n, m이 같고 p2 = p+2인 모드의 인덱스
            same_nm_ks = same_nm[i] # n, m이 같은 모드들의 인덱스

            kappa = rho_s/w_modes[i]
            ky = rho_s * m / rs[rho_mn_index]

            # delta p',p, diagonal
            L[i, i] += -(ky**2 + kappa**2 * (p + 0.5))
            M[i, i] += n_hat[rho_mn_index] / Te_hat[rho_mn_index] - n_hat[rho_mn_index] * L[i, i]

            # delta p',p-2
            if p_2minus_i != -1:
                j = p_2minus_i
                L[i, j] += kappa**2/2 * np.sqrt(p*(p-1))
                M[i, j] += - n_hat[rho_mn_index] *L[i, j]

            # delta p',p+2
            if p_2plus_i != -1:
                j = p_2plus_i
                L[i, j] += kappa**2/2 * np.sqrt((p+1)*(p+2))
                M[i, j] += - n_hat[rho_mn_index] *L[i, j]

        # Laplacian-like 행렬들 계산
        I = np.eye(L.shape[0]) # identity matrix
        J0 = np.linalg.solve(I - param.gyroaverage*0.5*L, I) # AX = I -> X = inv(A)
        Dc = param.damping_c * (param.mu1 * L - param.mu2 * L @ L) # or mu1 * L - mu2 * diag(ky^4) 매트랩 코드는 이렇게 구현함.
        invM = np.linalg.solve(M, I) # M = I * n_hat/Te_hat - n_hat * L

    print(f"W, dWdr, Laplacian-like matrices computed.")

    W = np.ascontiguousarray(W)
    dWdr = np.ascontiguousarray(dWdr)

    # %% 다음은 수치 적분이 필요한 grad_parallel(k_parallel), D_glf, 등등 행렬

    # F k_parallel kk' = i < k | F grad_parallel | k' > = delta(m, m') * delta(n, n') * a/R int_0^1 F(r) ( m/q(r) - n ) Wk Wk' rdr 

    # D_glf kk' = - < k | sqrt(8T_i_hat) / pi) |grad_parallel| k' > = - delta(m, m') * delta(n, n') * sqrt(8/pi) * a/R * int_0^1 sqrt(T_i_hat(r)) |m/q(r) - n| Wk Wk' rdr

    k_parallel = np.zeros_like(L) # k_parallel[k1, k2] = <k1|grad_parallel|k2>
    n_k_parallel = np.zeros_like(L) #
    Ti_k_parallel = np.zeros_like(L) # 
    tau_k_parallel = np.zeros_like(L) # 
    D_glf = np.zeros_like(L) # D_glf[k1, k2] = <k1|D_glf|k2>
    Gp = np.zeros_like(L) # Gp[k1, k2] = <k1|Gp|k2>
    Gn = np.zeros_like(L) # Gn[k1, k2] = <k1|Gn|k2>
    GTi = np.zeros_like(L) # GT[k1, k2] = <k1|GT|k2>
    a = np.zeros_like(L)
    b = np.zeros_like(L)

    groups = _build_nm_groups(ks)

    for (n, m), S in groups.items():
        Wg = W[S]
        mq_n = m / q - n
        ix = np.ix_(S, S)

        k_parallel[ix] = _weighted_cross(Wg, Wg, mq_n * rdr) / rmajor
        n_k_parallel[ix] = _weighted_cross(Wg, Wg, n_hat * mq_n * rdr) / rmajor
        Ti_k_parallel[ix] = _weighted_cross(Wg, Wg, Ti_hat * mq_n * rdr) / rmajor
        tau_k_parallel[ix] = _weighted_cross(Wg, Wg, tau * mq_n * rdr) / rmajor
        D_glf[ix] = (
            -np.sqrt(8.0 / np.pi)
            * _weighted_cross(Wg, Wg, np.sqrt(Ti_hat) * np.abs(mq_n) * rdr)
            / rmajor
        )
        Gp[ix] = -rho_s * m * _weighted_cross(Wg, Wg, dpdr * dr)
        Gn[ix] = -rho_s * m * _weighted_cross(Wg, Wg, dndr * dr)
        GTi[ix] = -rho_s * m * _weighted_cross(Wg, Wg, dTidr * dr)

    pref = rho_s / rmajor

    for (n, m), S in groups.items():
        Wi = W[S]

        T = groups.get((n, m + 1))
        if T is not None:
            Wj = W[T]
            dWj = dWdr[T]
            ix = np.ix_(S, T)

            weight_a1 = (
                n_hat
                * ((m + 1) * (1.0 + tau) + (d_lnn_dr + d_lntau_dr) * tau)
                * dr
            )
            weight_a2 = n_hat * (1.0 + tau) * rdr
            a[ix] += pref * (
                _weighted_cross(Wi, Wj, weight_a1)
                + _weighted_cross(Wi, dWj, weight_a2)
            )

            weight_b1 = n_hat * (m + 1 + d_lnn_dr) * dr
            weight_b2 = n_hat * rdr
            b[ix] += pref * (
                _weighted_cross(Wi, Wj, weight_b1)
                + _weighted_cross(Wi, dWj, weight_b2)
            )

        T = groups.get((n, m - 1))
        if T is not None:
            Wj = W[T]
            dWj = dWdr[T]
            ix = np.ix_(S, T)

            weight_a1 = (
                n_hat
                * ((m - 1) * (1.0 + tau) - (d_lnn_dr + d_lntau_dr) * tau)
                * dr
            )
            weight_a2 = n_hat * (1.0 + tau) * rdr
            a[ix] += pref * (
                _weighted_cross(Wi, Wj, weight_a1)
                - _weighted_cross(Wi, dWj, weight_a2)
            )

            weight_b1 = n_hat * (m - 1 - d_lnn_dr) * dr
            weight_b2 = n_hat * rdr
            b[ix] += pref * (
                _weighted_cross(Wi, Wj, weight_b1)
                - _weighted_cross(Wi, dWj, weight_b2)
            )

    D_glf = param.damping_glf*D_glf

    print(f"k_parallel, D_glf, Gp, Gn, GTi, a, b matrices computed. Shapes: {k_parallel.shape}")

    return {
        "W": W,
        "L": L,
        "M": M,
        "invM": invM,
        "J0": J0,
        "Dc": Dc,
        "k_parallel": k_parallel,
        "n_k_parallel": n_k_parallel,
        "Ti_k_parallel": Ti_k_parallel,
        "tau_k_parallel": tau_k_parallel,
        "D_glf": D_glf,
        "Gp": Gp,
        "Gn": Gn,
        "GTi": GTi,
        "a": a,
        "b": b,
    }


# def plot_matrices():
#     # plot all matrices
#     # L, M, invM, J0, Dc, k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel, D_glf, Gp, Gn, GTi, a, b
#     import matplotlib.pyplot as plt
#     matrices = [L, M, invM, J0, Dc, k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel, D_glf, Gp, Gn, GTi, a, b]
#     titles = ['L', 'M', 'invM', 'J0', 'Dc', 'k_parallel', 'n_k_parallel', 'Ti_k_parallel', 'tau_k_parallel', 'D_glf', 'Gp', 'Gn', 'GTi', 'a', 'b']
#     fig, axes = plt.subplots(4, 4, figsize=(20, 20))
#     for i, (matrix, title) in enumerate(zip(matrices, titles)):
#         ax = axes[i//4, i%4]
#         im = ax.imshow(matrix, cmap='viridis', aspect='auto')
#         ax.set_title(title)
#         fig.colorbar(im, ax=ax)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     plot_matrices()
# %%
