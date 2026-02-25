# modes.py
# 이 코드는 가능한 (n,m) 모드들을 찾아 2D 어레이로 저장한다.
print("[modes.py]")
# %%
from matplotlib import pyplot as plt
import numpy as np
import parameters as param

# ns = [n_start, n_start + n_delta, ..., n_end]
# ms = [m_start, m_start + m_delta, ..., m_end]
ns = np.arange(param.n_start, param.n_end + 1, param.n_delta)
ms = np.arange(param.m_start, param.m_end + 1, param.m_delta)

# if n=0 or m=0, q is not defined. 따라서 n과 m이 모두 양수인 경우에 대해서만 q를 계산한다.
if param.n_start <= 0:
    ns = ns[ns > 0]
if param.m_start <= 0:
    ms = ms[ms > 0]

rs = param.rs
q_min, q_max = np.min(param.q_profile(rs)), np.max(param.q_profile(rs))
ks = [] # [ [n1, m1, p1], [n1, m1, p2], [n1, m2, p1], [n1, m2, p2], [n1, m1, p1], ...,[m, m, p] ], 형태로 모드의 인덱스를 저장할 리스트
#  n1 < n2, m1 < m2, p1 < p2

mode_radius_indexes = [] # mode의 반지름에 가장 가까운 r의 인덱스를 저장할 리스트
mode_q_values = [] # mode의 q 값을 저장할 리스트

def find_index_of_nearst_r(r):
    for i, r_test in enumerate(rs): # r_test이 r보다 커지는 순간이 r에 가장 가까운 rs의 원소가 된다.
        if r_test >= r:
            return i

for i, n in enumerate(ns):
    for j, m in enumerate(ms):
        q = m/n
        
        # q profile이 monotonic일 때와 reversed일 때를 구분해서 다룬다.
        if param.q_profile_type == "monotonic":
            if q - param.q0 < 0: # 근이 없는 경우
                continue
            else:
                r = np.sqrt((q - param.q0) / param.q1) # 근이 있는 경우, q가 monotonic하므로 역함수로 r을 구할 수 있다.
                k_theta_rho_i = m/r*param.rhos0

            # q 값이 q_min, q_max 범위 안에 있는지, k_theta_rho_i 조건을 만족하는 확인한다. 범위를 벗어나면 NaN으로 마스킹한다.
            if q > q_min and q < q_max and k_theta_rho_i <= param.k_theta_rho_i_cut:
                mode_radius_indexes.append(find_index_of_nearst_r(r)) # mode의 반지름에 가장 가까운 r의 인덱스를 저장한다.
                mode_q_values.append(q)

                for p in range(param.p):
                    ks.append((n, m, p))

        elif param.q_profile_type == "reversed":
            raise NotImplementedError("reversed q profile is not implemented yet.")

ks = np.array(ks, dtype=int)   # shape (K, 3)

print(f"ks ({ks.shape}): \n{ks}")

mode_radius_indexes = np.array(mode_radius_indexes, dtype=int) # shape (K/p,)
mode_radius_indexes = np.repeat(mode_radius_indexes, param.p) # shape (K,)

mode_q_values = np.array(mode_q_values, dtype=float) # shape (K/p,)
mode_q_values = np.repeat(mode_q_values, param.p) # shape (K,)


def build_mode_maps(k: np.ndarray, p_max: int | None = None):
    """
    Parameters
    ----------
    k : (K,3) int array
        Each row is [m,n,p]. Assume no duplicates.
    p_max : int | None
        If None, use max(p)+1 inferred from k.

    Returns
    -------
    m_plus  : (K,) int array
    m_minus : (K,) int array
    index_of_mode : (n_max+1, m_max+1, p_max) int array filled with -1 if missing
    same_mn : (K, p_max) int array
        same_mn[i, pp] = index j such that k[j] = [n_i, m_i, pp], or -1 if missing
    """
    k = np.asarray(k)
    if k.ndim != 2 or k.shape[1] != 3:
        raise ValueError("k must have shape (K,3).")
    if not np.issubdtype(k.dtype, np.integer):
        raise ValueError("k must be an integer array.")

    K = k.shape[0]
    if p_max is None:
        p_max = int(k[:, 2].max()) + 1

    n_max = int(k[:, 0].max())
    m_max = int(k[:, 1].max())

    # (n,m,p) -> index
    index_map = {tuple(row): i for i, row in enumerate(k)}

    def lookup_targets(targets: np.ndarray) -> np.ndarray:
        out = np.full(K, -1, dtype=int)
        for i, key in enumerate(map(tuple, targets)):
            out[i] = index_map.get(key, -1)
        return out

    # m_plus, m_minus
    t_plus = k.copy()
    t_plus[:, 0] += 1
    m_plus = lookup_targets(t_plus)

    t_minus = k.copy()
    t_minus[:, 0] -= 1
    m_minus = lookup_targets(t_minus)

    # index_of_mode[n,m,p] = i  (없으면 -1)
    index_of_mode = np.full((n_max + 1, m_max + 1, p_max), -1, dtype=int)
    n = k[:, 0]
    m = k[:, 1]
    p = k[:, 2]
    valid_p = (0 <= p) & (p < p_max)
    index_of_mode[n[valid_p], m[valid_p], p[valid_p]] = np.arange(K, dtype=int)[valid_p]

    # same_mn: 각 (n,m)에 대해 p=0..p_max-1 인덱스 벡터를 만들고, 이를 각 i에 복제
    mn = k[:, :2]
    mn_unique, inv = np.unique(mn, axis=0, return_inverse=True)  # inv: (K,)
    U = mn_unique.shape[0]

    # (U, p_max) 테이블: pair u의 [n,m,pp] 인덱스
    per_pair = np.empty((U, p_max), dtype=int)
    for u, (mu, nu) in enumerate(mn_unique):
        if 0 <= mu <= m_max and 0 <= nu <= n_max:
            per_pair[u, :] = index_of_mode[nu, mu, :]
        else:
            per_pair[u, :] = -1

    same_mn = per_pair[inv]  # (K, p_max)

    return m_plus, m_minus, index_of_mode, same_mn

m_plus, m_minus, index_of_mode, same_mn = build_mode_maps(ks)


# def plot_modes():
#     # plot rr vs q, (n,m) vs q, (r,n) vs q, (r,m) vs q
#     fig, ax = plt.subplots(2, 2, figsize=(12, 10))

#     # title
#     fig.suptitle(f'(n,m) modes: {np.count_nonzero(~np.isnan(qs))}', fontsize=16)

#     # plot r vs q
#     x = rr.flatten()
#     y = qs.flatten()

#     ax[0,0].scatter(x, y, c='blue', marker='o')
#     ax[0,0].set_xlabel('r/a')
#     ax[0,0].set_ylabel('q')
#     ax[0,0].set_title('r vs q for (n,m) modes')
#     ax[0,0].grid()

#     # plot (n,m) vs q
#     # ax.imshow(qs_, extent=(0, ns[-1], 0, ms[-1]), origin='lower', aspect='auto')
#     ax[0,1].scatter(np.tile(ns, len(ms)), np.repeat(ms, len(ns)), c=qs.flatten(), cmap='viridis', marker='o')
#     ax[0,1].set_xlabel('n')
#     ax[0,1].set_ylabel('m')
#     ax[0,1].set_title('q values for (n,m) modes')
#     ax[0,1].grid()

#     X, Y = np.meshgrid(ns, ms)
#     # plot (r,m) vs q
#     nn = X.flatten()
#     mm = Y.flatten()
#     ax[1,0].scatter(rr.flatten(), mm, c=qs.flatten(), cmap='viridis', marker='o')
#     ax[1,0].set_xlabel('r/a')
#     ax[1,0].set_ylabel('m')
#     ax[1,0].set_title('rho values for (n,m) modes')
#     ax[1,0].grid()

#     # plot (r,n) vs q
#     ax[1,1].scatter(rr.flatten(), nn, c=qs.flatten(), cmap='viridis', marker='o')
#     ax[1,1].set_xlabel('r/a')
#     ax[1,1].set_ylabel('n')
#     ax[1,1].set_title('rho values for (n,m) modes')
#     ax[1,1].grid()
#     plt.show()

# if __name__ == "__main__":
#     plot_modes()
