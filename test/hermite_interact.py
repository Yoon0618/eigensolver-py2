import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import eval_hermite, gammaln
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parameters import build_profiles
from modes import build_modes
from utils import parse_params


def nu_p(p: int) -> float:
    # ν_p = 2^(p/2) * sqrt(Γ(p+1)) * π^(1/4)
    return float(np.exp(0.5 * p * np.log(2.0) + 0.5 * gammaln(p + 1.0) + 0.25 * np.log(np.pi)))


def effective_w_mn(param, rho_mn: float) -> float:
    """matrices.py의 Hermite w_mn 보정과 같은 로직."""
    w = float(param.w_mn)
    p_count = int(param.p)

    if p_count > 0:
        if rho_mn < (5.0 / 7.0) * p_count * w:
            w = (7.0 / 5.0) * rho_mn / p_count
        elif (1.0 - rho_mn) < (4.0 / 7.0) * p_count * w:
            w = (7.0 / 4.0) * (1.0 - rho_mn) / p_count

    return max(float(w), 1e-12)


def fallback_rho_from_q(param, profiles, n: int, m: int):
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
    rs = np.asarray(profiles["rs"])
    q_vals = np.asarray(profiles["q_profile"](rs))
    idx = int(np.argmin(np.abs(q_vals - q_target)))
    return float(rs[idx]), float(q_vals[idx])


def get_mode_rho_q(param, profiles, mode_data, n: int, m: int, p: int):
    """선택한 (n,m,p)의 rho_mn, q를 반환. 없으면 fallback 사용."""
    iom = mode_data["index_of_mode"]
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
        rs = profiles["rs"]
        ridx = int(mode_data["mode_radius_indexes"][idx])
        ridx = np.clip(ridx, 0, len(rs) - 1)
        rho_mn = float(rs[ridx])
        q_val = float(mode_data["mode_q_values"][idx])
        return rho_mn, q_val, True

    rho_mn, q_val = fallback_rho_from_q(param, profiles, n, m)
    if rho_mn is None:
        return None, None, False
    return rho_mn, q_val, False


def hermite_basis(param, profiles, mode_data, rho: np.ndarray, n: int, m: int, p: int):
    rho_mn, q_val, from_mode_table = get_mode_rho_q(param, profiles, mode_data, n, m, p)
    if rho_mn is None:
        return np.full_like(rho, np.nan), np.nan, np.nan, np.nan, False

    w = effective_w_mn(param, rho_mn)
    x = (rho - rho_mn) / w
    H = eval_hermite(p, x)
    denom = np.sqrt(2.0 * np.clip(rho, 1e-12, None) * w) * nu_p(p)
    W = H * np.exp(-0.5 * x * x) / denom
    dr = np.full_like(rho, float(param.dr), dtype=float)
    dr[-1] = 0.5 * float(param.dr)
    norm = np.sum(W * W * rho * dr)
    if np.isfinite(norm) and norm > 0:
        W /= np.sqrt(norm)
    return W, rho_mn, w, q_val, from_mode_table


def main():
    param = parse_params()
    profiles = build_profiles(param)
    mode_data = build_modes(param, profiles)
    rho = np.asarray(profiles["rs"], dtype=float)

    # 슬라이더 범위
    n_min = max(1, int(getattr(param, "n_start", 1)))
    n_max = int(getattr(param, "n_end", max(n_min, 10)))
    m_min = max(1, int(getattr(param, "m_start", 1)))
    m_max = int(getattr(param, "m", max(m_min, 10)))
    p_min = 0
    p_max = int(getattr(param, "p", 1)) - 1
    p_max = max(0, p_max)

    ks = mode_data["ks"]
    n_values = np.unique(ks[:, 0]) if len(ks) > 0 else np.arange(n_min, n_max + 1)
    if len(n_values) > 0:
        n_min = int(n_values[0])
        n_max = int(n_values[-1])

    if len(ks) > 0:
        n0, m0, p0 = map(int, ks[0])
    else:
        n0, m0, p0 = n_min, m_min, 0

    W0, rho0, w0, q0, ok0 = hermite_basis(param, profiles, mode_data, rho, n0, m0, p0)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.27)

    (line,) = ax.plot(rho, W0, lw=2, label=r"$W_{mnp}(\rho)$")
    vline = ax.axvline(rho0 if np.isfinite(rho0) else rho[0], color="tab:red", ls="--", alpha=0.8, label=r"$\rho_{mn}$")
    status = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    mode_warning = ax.text(
        0.5,
        0.86,
        "",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#5f3b00",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#fff3cd",
            "edgecolor": "#c98a00",
            "linewidth": 1.4,
            "alpha": 0.95,
        },
    )

    def set_mode_warning(n, m, p, in_table):
        if in_table:
            mode_warning.set_visible(False)
            mode_warning.set_text("")
            return

        mode_warning.set_text(
            f"선택한 모드가 mode table에 없습니다.\n"
            f"n={n}, m={m}, p={p}"
        )
        mode_warning.set_visible(True)

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
    set_mode_warning(n0, m0, p0, ok0)

    # sliders
    ax_n = plt.axes([0.1, 0.17, 0.8, 0.03])
    ax_m = plt.axes([0.1, 0.12, 0.8, 0.03])
    ax_p = plt.axes([0.1, 0.07, 0.8, 0.03])

    s_n = Slider(ax_n, "n", n_min, n_max, valinit=n0, valstep=n_values)
    s_m = Slider(ax_m, "m", m_min, m_max, valinit=m0, valstep=1)
    s_p = Slider(ax_p, "p", p_min, p_max, valinit=p0, valstep=1)

    def update(_):
        n = int(s_n.val)
        m = int(s_m.val)
        p = int(s_p.val)

        W, rho_mn, w, q, in_table = hermite_basis(param, profiles, mode_data, rho, n, m, p)

        if np.all(np.isnan(W)):
            status.set_text("invalid (n,m,p)")
            set_mode_warning(n, m, p, False)
            fig.canvas.draw_idle()
            return

        line.set_ydata(W)
        vline.set_xdata([rho_mn, rho_mn])

        ymin, ymax = np.nanpercentile(W, [1, 99])
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin < ymax:
            pad = 0.15 * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        refresh_title(n, m, p, rho_mn, w, q, in_table)
        set_mode_warning(n, m, p, in_table)
        status.set_text("" if in_table else "mode table에 없음: q-profile 기반 fallback 사용")
        fig.canvas.draw_idle()

    s_n.on_changed(update)
    s_m.on_changed(update)
    s_p.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
