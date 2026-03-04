import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import eval_hermite, gammaln

import parameters as param
import modes


def nu_p(p: int) -> float:
    # ν_p = 2^(p/2) * sqrt(Γ(p+1)) * π^(1/4)
    return float(np.exp(0.5 * p * np.log(2.0) + 0.5 * gammaln(p + 1.0) + 0.25 * np.log(np.pi)))


def resolve_w_mn(n: int, m: int) -> float:
    """
    w_mn 해석:
    1) param.w_mn 이 callable이면 w_mn(n,m)
    2) scalar이면 상수 폭
    3) dict[(n,m)]이면 해당 값
    4) 2D ndarray면 modes.ns/modes.ms 인덱스로 접근
    5) 없으면 rs 간격 기반 기본값
    """
    w_obj = getattr(param, "w_mn", None)

    if callable(w_obj):
        w = float(w_obj(n, m))
        return max(w, 1e-12)

    if w_obj is not None:
        if np.isscalar(w_obj):
            return max(float(w_obj), 1e-12)

        if isinstance(w_obj, dict):
            if (n, m) in w_obj:
                return max(float(w_obj[(n, m)]), 1e-12)

        if isinstance(w_obj, np.ndarray) and w_obj.ndim == 2:
            i = np.where(modes.ns == n)[0]
            j = np.where(modes.ms == m)[0]
            if len(i) > 0 and len(j) > 0:
                return max(float(w_obj[i[0], j[0]]), 1e-12)

    # fallback
    if hasattr(param, "rs") and len(param.rs) > 1:
        dr = float(np.mean(np.diff(param.rs)))
        return max(3.0 * dr, 1e-12)
    return 0.05


def fallback_rho_from_q(n: int, m: int):
    """modes에 없는 경우 q=m/n에서 rho_mn 근사 추정."""
    if n <= 0:
        return None, None
    q_target = m / n

    # monotonic + q0/q1이면 역함수 사용
    if getattr(param, "q_profile_type", "") == "monotonic" and hasattr(param, "q0") and hasattr(param, "q1"):
        if (q_target - param.q0) >= 0 and param.q1 != 0:
            rho = float(np.sqrt((q_target - param.q0) / param.q1))
            return rho, q_target

    # 일반적으로 rs grid에서 |q(r)-q_target| 최소값 선택
    rs = np.asarray(param.rs)
    q_vals = np.asarray(param.q_profile(rs))
    idx = int(np.argmin(np.abs(q_vals - q_target)))
    return float(rs[idx]), float(q_vals[idx])


def get_mode_rho_q(n: int, m: int, p: int):
    """선택한 (n,m,p)의 rho_mn, q를 반환. 없으면 fallback 사용."""
    iom = modes.index_of_mode
    idx = -1

    if (
        n >= 0
        and m >= 0
        and p >= 0
        and n < iom.shape[0]
        and m < iom.shape[1]
        and p < iom.shape[2]
    ):
        idx = int(iom[n, m, p])

    if idx >= 0:
        ridx = int(modes.mode_radius_indexes[idx])
        ridx = np.clip(ridx, 0, len(param.rs) - 1)
        rho_mn = float(param.rs[ridx])
        q_val = float(modes.mode_q_values[idx])
        return rho_mn, q_val, True

    rho_mn, q_val = fallback_rho_from_q(n, m)
    if rho_mn is None:
        return None, None, False
    return rho_mn, q_val, False


def hermite_basis(rho: np.ndarray, n: int, m: int, p: int):
    rho_mn, q_val, from_mode_table = get_mode_rho_q(n, m, p)
    if rho_mn is None:
        return np.full_like(rho, np.nan), np.nan, np.nan, np.nan, False

    w = resolve_w_mn(n, m)
    x = (rho - rho_mn) / w
    H = eval_hermite(p, x)
    denom = np.sqrt(2.0 * np.clip(rho, 1e-12, None) * w) * nu_p(p)
    W = H * np.exp(-0.5 * x * x) / denom
    return W, rho_mn, w, q_val, from_mode_table


def main():
    rho = np.asarray(param.rs, dtype=float)

    # 슬라이더 범위
    n_min = max(1, int(getattr(param, "n_start", 1)))
    n_max = int(getattr(param, "n_end", max(n_min, 10)))
    m_min = max(1, int(getattr(param, "m_start", 1)))
    m_max = int(getattr(param, "m_end", max(m_min, 10)))
    p_min = 0
    p_max = int(getattr(param, "p", 1)) - 1
    p_max = max(0, p_max)

    n0, m0, p0 = n_min, m_min, 0
    W0, rho0, w0, q0, ok0 = hermite_basis(rho, n0, m0, p0)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.27)

    (line,) = ax.plot(rho, W0, lw=2, label=r"$W_{mnp}(\rho)$")
    vline = ax.axvline(rho0 if np.isfinite(rho0) else rho[0], color="tab:red", ls="--", alpha=0.8, label=r"$\rho_{mn}$")
    status = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$W_{mnp}(\rho)$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    def refresh_title(n, m, p, rho_mn, w, q, in_table):
        src = "modes.py table" if in_table else "fallback"
        ax.set_title(
            rf"Hermite basis: n={n}, m={m}, p={p} | q≈{q:.6g}, $\rho_{{mn}}$≈{rho_mn:.6g}, w={w:.6g} ({src})"
        )

    refresh_title(n0, m0, p0, rho0, w0, q0, ok0)

    # sliders
    ax_n = plt.axes([0.1, 0.17, 0.8, 0.03])
    ax_m = plt.axes([0.1, 0.12, 0.8, 0.03])
    ax_p = plt.axes([0.1, 0.07, 0.8, 0.03])

    s_n = Slider(ax_n, "n", n_min, n_max, valinit=n0, valstep=1)
    s_m = Slider(ax_m, "m", m_min, m_max, valinit=m0, valstep=1)
    s_p = Slider(ax_p, "p", p_min, p_max, valinit=p0, valstep=1)

    def update(_):
        n = int(s_n.val)
        m = int(s_m.val)
        p = int(s_p.val)

        W, rho_mn, w, q, in_table = hermite_basis(rho, n, m, p)

        if np.all(np.isnan(W)):
            status.set_text("invalid (n,m,p)")
            return

        line.set_ydata(W)
        vline.set_xdata([rho_mn, rho_mn])

        ymin, ymax = np.nanpercentile(W, [1, 99])
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            pad = 0.15 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        refresh_title(n, m, p, rho_mn, w, q, in_table)
        status.set_text("" if in_table else "mode table에 없음: q-profile 기반 fallback 사용")
        fig.canvas.draw_idle()

    s_n.on_changed(update)
    s_m.on_changed(update)
    s_p.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()