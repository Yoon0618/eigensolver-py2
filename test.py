import numpy as np
import parameters as param
import modes
import scipy.special as sp

dr_ = 1/param.r_num # 적분 구간이 정규화되어 있으므로 균일한 dr은 단순히 1/r_num이다. rs = [0, dr, 2*dr, ..., r_end]
# rs = [dr, 2*dr, ..., r_end] -> 적분 가중치가 r*dr이라 r=0은 영향이 없으므로 dr부터 시작한다.
rs = np.linspace(dr_, param.r_end, param.r_num, endpoint=True, )
fac = np.ones_like(rs) 
fac[-1] = 0.5 # 사다리꼴 리만합
rdr = fac*rs*dr_ # r*dr 적분에 사용
dr = fac*dr_ # dr 적분에 사용

alpha = np.empty((param.m_end+1, param.p)) # j_m (alpha[m, p]) = 0
for m in range(param.m_end+1):
    alpha[m] = sp.jn_zeros(m, param.p)

W = lambda k: np.sqrt(2) / sp.jv(k[0]+1, alpha[k[0], k[2]]) * sp.jv(k[0], alpha[k[0], k[2]] * rs)
dWdr = lambda k: np.sqrt(2) / sp.jv(k[0]+1, alpha[k[0], k[2]]) * sp.jvp(k[0], alpha[k[0], k[2]] * rs) * alpha[k[0], k[2]]
d2Wdr2 = lambda k: np.sqrt(2) / sp.jv(k[0]+1, alpha[k[0], k[2]]) * sp.jvp(k[0], alpha[k[0], k[2]] * rs, n=2) * alpha[k[0], k[2]]**2

LW = lambda k: param.rho_s**2 * (d2Wdr2(k) + 1/rs * dWdr(k) - k[0]**2 / rs**2 * W(k))
LWrdr = lambda k: LW(k) * rdr

# 반지름에 대해 미리 계산해 놓기
W_kk = np.empty((len(modes.ks), len(rs)), dtype=float)
dWdr_kk = np.empty_like(W_kk)
d2Wdr2_kk = np.empty_like(W_kk)
LW_kk = np.empty_like(W_kk)
LWrdr_kk = np.empty_like(W_kk)
for i, k in enumerate(modes.ks):
    W_kk[i] = W(k)
    dWdr_kk[i] = dWdr(k)
    d2Wdr2_kk[i] = d2Wdr2(k)
    LW_kk[i] = LW(k)
    LWrdr_kk[i] = LW_kk[i] * rdr

LL = LW_kk @ LWrdr_kk.T # shape (K, K)
# plot LL
import matplotlib.pyplot as plt
plt.imshow(LL, cmap="viridis")
plt.colorbar()
plt.title("LL")
plt.xlabel("k2")
plt.ylabel("k1")
plt.show()
print("precomputation done")

def inner_product(k1, k2, OW_kk, weight):
    m1, n1, p1 = k1
    m2, n2, p2 = k2
    
    Wk2 = W_kk[k2]
    OW = OW_kk[k1]
    return np.sum(OW * Wk2 * weight)

L_kk = np.empty((len(modes.ks), len(modes.ks)), dtype=float)
for i, k1 in enumerate(modes.ks):
    print(f"computing row {i+1}/{len(modes.ks)}")
    for j, k2 in enumerate(modes.ks):
        L_kk[i, j] = inner_product(k1, k2, LW_kk, rdr)

# plot L_kk
import matplotlib.pyplot as plt
plt.imshow(L_kk, cmap="viridis")
plt.colorbar()
plt.title("L_kk")
plt.xlabel("k2")
plt.ylabel("k1")
plt.show()
