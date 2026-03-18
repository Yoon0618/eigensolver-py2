# modes.py
# 이 코드는 가능한 (n,m) 모드들을 찾아 2D 어레이로 저장한다.
print("[modes.py]")

#####################################################
"""
이 코드가 저장하는 것

ks: shape (K, 3) int ndarray, 각 행은 (n, m, p) 모드의 인덱스. n은 poloidal mode number, m은 toroidal mode number, p는 radial mode number.
ex) ks[0] = [n1, m1, p1], ks[1] = [n1, m1, p2], ks[2] = [n1, m2, p1], ks[3] = [n1, m2, p2], ks[4] = [n2, m1, p1], ...

mode_radius_indexes: shape (K,) int ndarray, 각 모드의 반지름에 가장 가까운 r의 인덱스. p 모드마다 같은 값이 반복된다.
ex) mode_radius_indexes[k1] = k1 모드의 반지름, 정수
ex) rs[mode_radius_indexes[k1]] = k1 모드의 반지름, 실수

mode_q_values: shape (K,) float ndarray, 각 모드의 q 값. p 모드마다 같은 값이 반복된다.
ex) mode_q_values[k1] = k1 모드의 q 값, 실수

m_plus: shape (K, p_max) int ndarray
m_plus[k, :] = n은 같고 m2 = m1 + 1인 모든 p에 대한 인덱스들.
해당 모드가 없으면 None 배열

m_2plus: shape (K, p_max) int ndarray
m_2plus[k, :] = n은 같고 m2 = m1 + 2인 모든 p에 대한 인덱스들.
없으면 None 배열

m_minus: shape (K, p_max) int ndarray
m_minus[k, :] = n은 같고 m2 = m1 - 1인 모든 p에 대한 인덱스들.
없으면 None 배열

m_2minus: shape (K, p_max) int ndarray
m_2minus[k, :] = n은 같고 m2 = m1 - 2인 모든 p에 대한 인덱스들.
없으면 None 배열

index_of_mode: shape (n_max+1, m_max+1, p_max) int ndarray,
n,m은 0이 없어서 +1 shape를 가짐
k의 모드가 (n, m, p)일 때, index_of_mode[n, m, p]는 k 모드의 인덱스. 만약 (n, m, p) 모드가 존재하지 않으면 -1.
ex) index_of_mode[n, m, p] = k1, where ks[k1] = (n, m, p)

same_nm: shape (K, p_max) int ndarray,
k의 모드가 (n, m, p)일 때, same_nm[k, pp]는 (n, m, pp) 모드의 인덱스. 만약 (n, m, pp) 모드가 존재하지 않으면 -1.
ex) same_nm[k1, pp] = k2, where ks[k1] = (n, m, p) and ks[k2] = (n, m, pp)
"""

# %%
from matplotlib import pyplot as plt
import numpy as np
from utils import timed

@timed
def build_modes(param, profiles):
    # ns = [n_start, n_start + n_delta, ..., n_end]
    # ms = [1, 2, 3, ..., m]
    ns = np.arange(param.n_start, param.n_end + 1, param.n_delta, dtype=int)
    ms = np.arange(1, param.m + 1, dtype=int)
    p_max = param.p

    rs = profiles["rs"]
    q_min = profiles["q_profile"](param.r_start)
    q_max = profiles["q_profile"](param.r_end)

    nm_modes_list = []              # [[n1, m1], [n1, m2], ...]
    mode_radius_indexes_list = []   # length M
    mode_q_values_list = []         # length M

    def find_index_of_nearest_r(r):
        # searchsorted 기반으로 가장 가까운 index 선택
        idx = np.searchsorted(rs, r)
        if idx <= 0:
            return 0
        if idx >= len(rs):
            return len(rs) - 1
        return idx - 1 if abs(rs[idx - 1] - r) <= abs(rs[idx] - r) else idx

    for n in ns:
        for m in ms:
            q = m / n

            if param.q_profile_type == "monotonic":
                if q <= param.q0:
                    continue

                r = np.sqrt((q - param.q0) / param.q1)
                k_theta_rho_i = m / r * param.rhos0

                if q_min < q < q_max and k_theta_rho_i <= param.k_theta_rho_i_cut:
                    nm_modes_list.append([n, m])
                    mode_radius_indexes_list.append(find_index_of_nearest_r(r))
                    mode_q_values_list.append(q)

            elif param.q_profile_type == "reversed":
                raise NotImplementedError("reversed q profile is not implemented yet.")

    nm_modes = np.asarray(nm_modes_list, dtype=int)              # shape (M, 2)
    mode_radius_nm = np.asarray(mode_radius_indexes_list, dtype=int)  # shape (M,)
    mode_q_nm = np.asarray(mode_q_values_list, dtype=float)           # shape (M,)

    if nm_modes.ndim != 2 or nm_modes.shape[1] != 2:
        raise ValueError("nm_modes must have shape (M, 2).")

    M = len(nm_modes)

    # ks 구성: 각 (n,m)에 대해 p=0..p_max-1 반복
    # ks shape = (M*p_max, 3)
    ks = np.empty((M * p_max, 3), dtype=int)
    ks[:, 0:2] = np.repeat(nm_modes, p_max, axis=0)
    ks[:, 2] = np.tile(np.arange(p_max, dtype=int), M)

    print(f"ks ({ks.shape}): \n{ks}")

    # mode_radius_indexes, mode_q_values도 p 방향으로 반복
    mode_radius_indexes = np.repeat(mode_radius_nm, p_max)   # shape (K,)
    mode_q_values = np.repeat(mode_q_nm, p_max)              # shape (K,)

    # ---------- lookup tables ----------
    n_arr = ks[:, 0]
    m_arr = ks[:, 1]
    p_arr = ks[:, 2]

    if np.unique(ks, axis=0).shape[0] != len(ks):
        raise ValueError("Duplicate (n, m, p) entries found in ks.")

    n_max = int(n_arr.max())
    m_max = int(m_arr.max())
    K = len(ks)

    # index_of_mode[n, m, p] = k, 없으면 -1
    index_of_mode = np.full((n_max + 1, m_max + 1, p_max), -1, dtype=int)
    index_of_mode[n_arr, m_arr, p_arr] = np.arange(K, dtype=int)

    # same_nm[k, pp] = (same n, same m, p=pp)의 index, 없으면 -1
    same_nm = index_of_mode[n_arr, m_arr, :]   # shape (K, p_max)

    # m-shift lookup:
    # out[i]는 "ks를 인덱싱하는 1D int ndarray"
    # 없으면 빈 배열
    def collect_m_shift(delta_m: int):
        out = np.empty(len(ks), dtype=object)

        for i in range(len(ks)):
            n1, m1, p1 = ks[i]
            m_target = m1 + delta_m

            mask = (ks[:, 0] == n1) & (ks[:, 1] == m_target)

            # ks의 행이 아니라, ks의 인덱스를 저장
            out[i] = np.where(mask)[0].astype(int)

        return out

    m_plus   = collect_m_shift(+1)
    m_2plus  = collect_m_shift(+2)
    m_minus  = collect_m_shift(-1)
    m_2minus = collect_m_shift(-2)

    return {
        "ks": ks,
        "mode_radius_indexes": mode_radius_indexes,
        "mode_q_values": mode_q_values,
        "m_plus": m_plus,
        "m_2plus": m_2plus,
        "m_minus": m_minus,
        "m_2minus": m_2minus,
        "index_of_mode": index_of_mode,
        "same_nm": same_nm,
    }
