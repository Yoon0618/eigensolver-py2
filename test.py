import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy import special

# -----------------------------
# Normalization: v_p
# v_p = 2^{p/2} * Gamma(p+1)^{1/2} * pi^{1/4}
# so v_p^2 = 2^p p! sqrt(pi)
# -----------------------------
def v_p(p: int) -> float:
    return (2.0 ** (p / 2.0)) * np.sqrt(special.gamma(p + 1.0)) * (np.pi ** 0.25)

# Physicists' Hermite polynomial H_p(x)
def H(p: int, x):
    return special.eval_hermite(p, x)

# -----------------------------
# 1) Exact delta test in x-space:
# I_pp' = (1/(2 v_p v_p')) * ∫ H_p(x) H_p'(x) e^{-x^2} dx
# Use Gauss-Hermite quadrature, which is exact for polynomials under e^{-x^2}.
# -----------------------------
def delta_test_x(pmax: int = 10, N_gh: int = 80):
    # nodes, weights for ∫ f(x) e^{-x^2} dx
    xg, wg = hermgauss(N_gh)

    M = np.zeros((pmax + 1, pmax + 1), dtype=float)
    for p in range(pmax + 1):
        Hp = H(p, xg)
        vp = v_p(p)
        for q in range(pmax + 1):
            Hq = H(q, xg)
            vq = v_p(q)
            integral = np.sum(wg * Hp * Hq)  # ≈ ∫ H_p H_q e^{-x^2} dx
            M[p, q] = integral / (2.0 * vp * vq)

    return M

# -----------------------------
# 2) Approx delta test in rho-space for W_p(rho):
# W_p(rho) = [1/sqrt(2*rho*w*v_p)] * H_p(x) * exp(-x^2/2),
# x = (rho - rho0)/w
# Check: ∫_{rho_min}^{rho_max} W_p W_q rho drho ≈ δ_pq
# Use Gauss-Legendre quadrature on rho.
# -----------------------------
def W(p: int, rho, rho0: float, w: float):
    x = (rho - rho0) / w
    return (H(p, x) * np.exp(-0.5 * x * x)) / np.sqrt(2.0 * rho * w * v_p(p))

def delta_test_rho(
    pmax: int = 10,
    rho0: float = 0.5,
    w: float = 0.03,
    rho_min: float = 1e-8,
    rho_max: float = 1.0,
    N_gl: int = 4000,
):
    # Gauss-Legendre nodes/weights for ∫_{-1}^{1} ...
    t, wt = leggauss(N_gl)
    # map to [rho_min, rho_max]
    rho = 0.5 * (rho_max - rho_min) * t + 0.5 * (rho_max + rho_min)
    drho = 0.5 * (rho_max - rho_min)

    M = np.zeros((pmax + 1, pmax + 1), dtype=float)
    for p in range(pmax + 1):
        Wp = W(p, rho, rho0=rho0, w=w)
        for q in range(pmax + 1):
            Wq = W(q, rho, rho0=rho0, w=w)
            integral = drho * np.sum(wt * (rho * Wp * Wq))
            M[p, q] = integral
    return M

# -----------------------------
# Run + report
# -----------------------------
if __name__ == "__main__":
    pmax = 12

    # 1) x-space: should be identity (Kronecker delta)
    Mx = delta_test_x(pmax=pmax, N_gh=80)
    Ix = np.eye(pmax + 1)
    err_x = Mx - Ix
    print("[x-space] max |offdiag| =", np.max(np.abs(err_x - np.diag(np.diag(err_x)))))
    print("[x-space] max |diag-1|  =", np.max(np.abs(np.diag(Mx) - 1.0)))

    # 2) rho-space: approximate identity if localized well
    Mr = delta_test_rho(
        pmax=pmax,
        rho0=0.5,   # center away from boundaries
        w=0.03,     # narrower -> better approx
        rho_min=1e-8,
        rho_max=1.0,
        N_gl=4000
    )
    Ir = np.eye(pmax + 1)
    err_r = Mr - Ir
    print("[rho-space] max |offdiag| =", np.max(np.abs(err_r - np.diag(np.diag(err_r)))))
    print("[rho-space] max |diag-1|  =", np.max(np.abs(np.diag(Mr) - 1.0)))

    # Optional: print matrices if you want
    # np.set_printoptions(precision=3, suppress=True)
    # print("Mx=\n", Mx)
    # print("Mr=\n", Mr)