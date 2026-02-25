# matrices.py
# 행렬 A를 구성하는 각각의 연산자 행렬을 계산한다.
print("[matrices.py]")

import numpy as np
import parameters as param
import modes
import scipy.special as sp
import matplotlib.pyplot as plt

# A의 각 행렬들을 수치적분으로 만든다.

# 반경 변수 정의
dr_ = param.dr # dr value
rs = param.rs # r values, shape (r_num,)
dr = np.full(param.r_num, dr_) # dr list
dr[-1] = 0.5*dr_ # 마지막 가중치는 절반
rdr = rs*dr # r*dr 적분에 사용

rho_s = param.rho_s
rmajor = param.rmajor

# 평형 프로파일 불러오기
q = param.q_profile(rs)
n_hat = param.n_hat
Te_hat = param.Te_hat
Ti_hat = param.Ti_hat
pi_hat = param.pi_hat
tau = param.tau

dpdr = param.dpdr
dndr = param.dndr
dTidr = param.dTidr

d_lnn_dr = param.d_lnn_dr
d_lnTi_dr = param.d_lnTi_dr
d_lnTe_dr = param.d_lnTe_dr
d_lnpi_dr = param.d_lnpi_dr
d_lntau_dr = param.d_lntau_dr

# mode 변수들 불러오기
ks = modes.ks
mode_radius_indexes = modes.mode_radius_indexes
mode_q_values = modes.mode_q_values

# 쉬운 대각 행렬들 만들기
N = len(ks) # mode의 개수
L = - rho_s**2 * np.eye(N, dtype=float) # L[k1, k2] = - rho_s^2 * delta(k, k')
M = np.zeros_like(L) # 
invM = np.zeros_like(L)

# for i, k1 in enumerate(ks):
#     m1, n1, p1 = k1
#     for j, k2 in enumerate(ks):
#         m2, n2, p2 = k2
#         if m1 == m2 and n1 == n2 and p1 == p2:
#             rho_mn_index = mode_radius_indexes[i] # mode의 반지름에 가장 가까운 r의 인덱스
#             M[i, j] = n_hat[rho_mn_index] / Te_hat[rho_mn_index] - n_hat[rho_mn_index] * L[i, j]
#             invM[i, j] = 1.0 / M[i, j] if M[i, j] != 0 else 0.0
# J0 = (1.0 / (1.0 + 0.5*rho_s**2)) * np.eye(N, dtype=float) # L이 대각행렬이라 단순 역수가 역행렬이다.
# Dc = param.mu1 * L  - param.mu2 * L**2


for i, k in enumerate(ks):
    n, m, p = k
    rho_mn_index = mode_radius_indexes[i] # mode의 반지름에 가장 가까운 r의 인덱스
    M[i, i] = n_hat[rho_mn_index] / Te_hat[rho_mn_index] - n_hat[rho_mn_index] * L[i, i]
    invM[i, i] = 1.0 / M[i, i] if M[i, i] != 0 else 0.0
J0 = (1.0 / (1.0 + 0.5*rho_s**2)) * np.eye(N, dtype=float) # L이 대각행렬이라 단순 역수가 역행렬이다.
Dc = param.mu1 * L  - param.mu2 * L**2

print(f"L, M, invM, J0, Dc matrices computed. Shapes: {L.shape}, {M.shape}, {invM.shape}, {J0.shape}, {Dc.shape}")
# 다음은 수치 적분이 필요한 grad_parallel(k_parallel), D_glf 행렬

# F k_parallel kk' = i < k | F grad_parallel | k' > = delta(m, m') * delta(n, n') * a/R int_0^1 F(r) ( m/q(r) - n ) Wk Wk' rdr 

# D_glf kk' = - < k | sqrt(8T_i_hat) / pi) |grad_parallel| k' > = - delta(m, m') * delta(n, n') * sqrt(8/pi) * a/R * int_0^1 sqrt(T_i_hat(r)) |m/q(r) - n| Wk Wk' rdr

# 베셀 함수 영점 어레이
# alpha[m, p] = p+1번째 영점 of J_m
alpha = np.empty((param.m_end+1, param.p)) # j_m (alpha[m, p]) = 0
for m in range(param.m_end+1):
    alpha[m] = sp.jn_zeros(m, param.p)

# Wk는 k 값에 따라 달라짐. 이를 미리 다 계산해줌.
# W = [Wk(k1), Wk(k2), ..., Wk(kK)], shape (K, r_num)
# Wk(r) = sqrt(2) / J_{m+1}(alpha[m, p]) * J_m(alpha[m, p] * r)

W = np.empty((len(ks), len(rs)), dtype=float) # W[k, r] = Wk(r)
dWdr = np.empty_like(W) # dWdr[k, r] = dWk/dr(r)
d2Wdr2 = np.empty_like(W) # d2Wdr2[k, r] = d^2Wk/dr^2(r)

for i, k in enumerate(ks):
    n, m, p = k
    Wk = np.sqrt(2) / sp.jv(m+1, alpha[m, p]) * sp.jv(m, alpha[m, p] * rs)
    dWdrk = np.sqrt(2) / sp.jv(m+1, alpha[m, p]) * sp.jvp(m, alpha[m, p] * rs) * alpha[m, p]
    d2Wdr2k = np.sqrt(2) / sp.jv(m+1, alpha[m, p]) * sp.jvp(m, alpha[m, p] * rs, n=2) * alpha[m, p]**2

    W[i] = Wk
    dWdr[i] = dWdrk
    d2Wdr2[i] = d2Wdr2k

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

m_plus, m_minus, index_of_mode, same_mn = modes.m_plus, modes.m_minus, modes.index_of_mode, modes.same_mn

for i, k1 in enumerate(ks):
    n1, m1, p1 = k1
    for j, k2 in enumerate(ks):
        n2, m2, p2 = k2
        if n1 == n2 and m1 == m2:
            # k_parallel kk' 계산
            integrand_k_parallel = (m1/q - n1) * W[i] * W[j] * rdr
            k_parallel_kk = 1/rmajor * np.sum(integrand_k_parallel)

            # n_k_parallel kk' 계산
            integrand_n_k_parallel = n_hat * (m1/q - n1) * W[i] * W[j] * rdr
            n_k_parallel_kk = 1/rmajor * np.sum(integrand_n_k_parallel)

            # Ti_k_parallel kk' 계산
            integrand_Ti_k_parallel = Ti_hat * (m1/q - n1) * W[i] * W[j] * rdr
            Ti_k_parallel_kk = 1/rmajor * np.sum(integrand_Ti_k_parallel)

            # tau_k_parallel kk' 계산
            integrand_tau_k_parallel = tau * (m1/q - n1) * W[i] * W[j] * rdr
            tau_k_parallel_kk = 1/rmajor * np.sum(integrand_tau_k_parallel)

            # D_glf kk' 계산
            integrand_D_glf = np.sqrt(Ti_hat) * np.abs(m1/q - n1) * W[i] * W[j] * rdr
            D_glf_kk = - np.sqrt(8/np.pi) * 1/rmajor * np.sum(integrand_D_glf)
            
            # Gp_kk' 계산
            integrand_Gp = dpdr * W[i] * W[j] * dr
            Gp_kk = - rho_s * m1 * np.sum(integrand_Gp)

            # Gn_kk' 계산
            integrand_Gn = dndr * W[i] * W[j] * dr
            Gn_kk = - rho_s * m1 * np.sum(integrand_Gn)

            # GTi_kk' 계산
            integrand_GTi = dTidr * W[i] * W[j] * dr
            GTi_kk = - rho_s * m1 * np.sum(integrand_GTi)

            
            k_parallel[i, j] = k_parallel_kk
            n_k_parallel[i, j] = n_k_parallel_kk
            Ti_k_parallel[i, j] = Ti_k_parallel_kk
            tau_k_parallel[i, j] = tau_k_parallel_kk
            D_glf[i, j] = D_glf_kk
            Gp[i, j] = Gp_kk
            Gn[i, j] = Gn_kk
            GTi[i, j] = GTi_kk

        elif n1 == n2 and m1+1 == m2:
            k_ = index_of_mode[n1, m1+1, p2]

            # a_plus_kk' 계산
            integrand_a_plus_1 = n_hat * ( (m1+1)*(1+tau) + (d_lnn_dr + d_lntau_dr) * tau * W[i] * W[k_] * dr          )
            integrand_a_plus_2 = n_hat * (1+tau) * W[i] * dWdr[k_] * rdr
            a_plus_kk = rho_s * 1/rmajor * np.sum(integrand_a_plus_1 + integrand_a_plus_2)

            # b_plus_kk' 계산
            integrand_b_plus_1 = n_hat * (m1+1+d_lnn_dr) * W[i] * W[k_] * dr
            integrand_b_plus_2 = n_hat * W[i] * dWdr[k_] * rdr
            b_plus_kk = rho_s * 1/rmajor * np.sum(integrand_b_plus_1 + integrand_b_plus_2)

            a[i, j] += a_plus_kk
            b[i, j] += b_plus_kk

        elif m1-1 == m2 and n1 == n2:
            k_ = index_of_mode[n1, m1-1, p2]

            # a_minus_kk' 계산
            integrand_a_minus_1 = n_hat * ( (m1-1)*(1+tau) - (d_lnn_dr + d_lntau_dr) * tau * W[i] * W[k_] * dr          )
            integrand_a_minus_2 = n_hat * (1+tau) * W[i] * dWdr[k_] * rdr
            a_minus_kk = rho_s * 1/rmajor * np.sum(integrand_a_minus_1 - integrand_a_minus_2)

            # b_minus_kk' 계산
            integrand_b_minus_1 = n_hat * (m1-1+d_lnn_dr) * W[i] * W[k_] * dr
            integrand_b_minus_2 = n_hat * W[i] * dWdr[k_] * rdr
            b_minus_kk = rho_s * 1/rmajor * np.sum(integrand_b_minus_1 - integrand_b_minus_2)

            a[i, j] += a_minus_kk
            b[i, j] += b_minus_kk

print(f"k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel matrices computed. Shapes: {k_parallel.shape}, {n_k_parallel.shape}, {Ti_k_parallel.shape}, {tau_k_parallel.shape}")
print(f"D_glf, Gp, Gn, GTi, a, b matrices computed. Shapes: {k_parallel.shape}, {D_glf.shape}, {Gp.shape}, {Gn.shape}, {GTi.shape}, {a.shape}, {b.shape}")

def plot_matrices():
    # plot all matrices
    # L, M, invM, J0, Dc, k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel, D_glf, Gp, Gn, GTi, a, b
    import matplotlib.pyplot as plt
    matrices = [L, M, invM, J0, Dc, k_parallel, n_k_parallel, Ti_k_parallel, tau_k_parallel, D_glf, Gp, Gn, GTi, a, b]
    titles = ['L', 'M', 'invM', 'J0', 'Dc', 'k_parallel', 'n_k_parallel', 'Ti_k_parallel', 'tau_k_parallel', 'D_glf', 'Gp', 'Gn', 'GTi', 'a', 'b']
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes[i//4, i%4]
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_matrices()