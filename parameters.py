import numpy as np
from numpy import exp, cosh, tanh
import sympy as sp
from dataclasses import dataclass

@dataclass
class Params:
    n_start: int = 4
    n_delta: int = 10
    n_end: int = 48
    m: int = 50
    p: int = 10

    basis: str = "hermite" # "bessel" or "hermite"
    method: str = "time_evolution" # "eigenproblem" or "time_evolution"
    dt: float = 1e-3 # time step for time evolution method
    T: float = 10.0 # total simulation time for time evolution method
    F0: float = 1e-9 # initial perturbation amplitude for time evolution method

    suffix: str = "" # file name suffix for saving images

    r_start : float = 0.2
    r_end : float = 0.9
    r_num : int = 256
    dr : float = 1/r_num
    
    rhos0 : float = 7.086026e-03
    rho_s : float = rhos0 # rho_s = rhos0/a 인데 왜인지 코드에는 둘이 같게 되어있다...

    a : float = 0.5 # minor radius. 정확히는 0.48을 쓰는 것 같은데 편의상인지 0.5를 쓴다.
    R : float = 1.3 # major radius
    rmajor : float = 1.3/0.48 # R/a, aspect ratio

    k_theta_rho_i_cut : float = 1.3

    mu1 : float = 0.1 # viscosity coefficient
    mu2 : float = 0.0 # hyper-viscosity coefficient

    w_mn : float = 5*rho_s # Hermite 기저 함수 폭

    save_dir : str = "results" # directory to save results, relative to the current working directory

    q_profile_type : str = "monotonic" # "monotonic" or "reversed"
    q0 : float = 0.854 # q(r=0) = q0
    q1 : float = 2.184 # q(r=1) = q1

def build_profiles(param):
    rs = np.linspace(param.dr, 1.0, param.r_num)

    # q profile type
    if param.q_profile_type == "monotonic":
        q_profile = lambda r: param.q0+param.q1*r**2; 

    elif param.q_profile_type == "reversed":
        raise NotImplementedError("reversed q profile is not implemented yet.")

    # ----------- equilibrium profiles based on cyclone case -----------

    R_Lne = sp.Float(1/0.45)
    R_Lte = sp.Float(6.92)
    R_Lti = sp.Float(6.92)
    Te_T0 = sp.Float(1.0)

    # x = r/a (0~1)
    x = sp.Symbol("x", real=True)

    # use exact rationals for 0.5 and 0.3 if you want
    xx = (x - sp.Rational(1, 2)) / sp.Float(0.3)

    rmajor = sp.Float(param.rmajor)
    n_hat_sp   = sp.exp(-0.3*R_Lne/rmajor*sp.tanh(xx))
    Te_hat_sp  = Te_T0*sp.exp(-0.3*R_Lte/rmajor*sp.tanh(xx))
    Ti_hat_sp  = sp.exp(-0.3*R_Lti/rmajor*sp.tanh(xx))
    pi_hat_sp  = n_hat_sp*Ti_hat_sp
    tau_sp     = Ti_hat_sp/Te_hat_sp

    d_lnn_dr_sp    = sp.simplify(sp.diff(sp.log(n_hat_sp),  x))   # matches rln
    d_lnTi_dr_sp   = sp.simplify(sp.diff(sp.log(Ti_hat_sp), x))   # matches rlt
    d_lnTe_dr_sp   = sp.simplify(sp.diff(sp.log(Te_hat_sp), x))   # matches rlte
    d_lnpi_dr_sp   = sp.simplify(sp.diff(sp.log(pi_hat_sp), x))   # ~ rln+rlt
    d_lntau_dr_sp  = sp.simplify(sp.diff(sp.log(tau_sp),   x))    # rlt - rlte

    dn_dr_sp   = sp.simplify(sp.diff(n_hat_sp,  x))
    dTi_dr_sp  = sp.simplify(sp.diff(Ti_hat_sp, x))
    dpi_dr_sp  = sp.simplify(sp.diff(pi_hat_sp, x))

    # ---- quick check at x=0.5 ----
    check = sp.simplify(d_lnn_dr_sp.subs(x, sp.Rational(1,2)))  # -> -R_Lne/rmajor

    # ---- lambdify (keyword is modules) ----
    n_hat_np      = sp.lambdify(x, n_hat_sp,     modules="numpy")
    Te_hat_np     = sp.lambdify(x, Te_hat_sp,    modules="numpy")
    Ti_hat_np     = sp.lambdify(x, Ti_hat_sp,    modules="numpy")
    pi_hat_np     = sp.lambdify(x, pi_hat_sp,    modules="numpy")
    tau_np        = sp.lambdify(x, tau_sp,       modules="numpy")

    d_lnn_dr_np   = sp.lambdify(x, d_lnn_dr_sp,  modules="numpy")
    d_lnTi_dr_np  = sp.lambdify(x, d_lnTi_dr_sp, modules="numpy")
    d_lnTe_dr_np  = sp.lambdify(x, d_lnTe_dr_sp, modules="numpy")
    d_lnpi_dr_np  = sp.lambdify(x, d_lnpi_dr_sp, modules="numpy")
    d_lntau_dr_np = sp.lambdify(x, d_lntau_dr_sp,modules="numpy")

    dn_dr_np      = sp.lambdify(x, dn_dr_sp,     modules="numpy")
    dTi_dr_np     = sp.lambdify(x, dTi_dr_sp,    modules="numpy")
    dpi_dr_np     = sp.lambdify(x, dpi_dr_sp,    modules="numpy")

    n_hat = n_hat_np(rs)
    Te_hat = Te_hat_np(rs)
    Ti_hat = Ti_hat_np(rs)
    pi_hat = pi_hat_np(rs)
    tau = tau_np(rs)

    dpi_dr = dpi_dr_np(rs)
    dn_dr = dn_dr_np(rs)
    dTi_dr = dTi_dr_np(rs)

    d_lnn_dr = d_lnn_dr_np(rs)
    d_lnTi_dr = d_lnTi_dr_np(rs)
    d_lnTe_dr = d_lnTe_dr_np(rs)
    d_lnpi_dr = d_lnpi_dr_np(rs)
    d_lntau_dr = d_lntau_dr_np(rs)

    return {
        "rs": rs,
        "q_profile": q_profile,
        "R_Lne": R_Lne,
        "n_hat": n_hat,
        "Te_hat": Te_hat,
        "Ti_hat": Ti_hat,
        "pi_hat": pi_hat,
        "tau": tau,
        "d_lnn_dr": d_lnn_dr,
        "d_lnTi_dr": d_lnTi_dr,
        "d_lnTe_dr": d_lnTe_dr,
        "d_lnpi_dr": d_lnpi_dr,
        "d_lntau_dr": d_lntau_dr,
        "dn_dr": dn_dr,
        "dTi_dr": dTi_dr,
        "dpi_dr": dpi_dr,
    }

"""
original MATLAB code for profiles:
%-----------------------------------------------------------------------
% Set equilibrium profiles : cyclone case used
%-----------------------------------------------------------------------
status='   Setting equilibrium profiles'

      R_Lne=1./0.45;      % R/L_ne, R=major radius, L_ne=1/(dlog(n)/dr) at r=0.5a 
      R_Lte=6.92;         % R/L_Te, L_Te=1/(dlog(Te)/dr) at r=0.5a
      R_Lti=6.92;         % R/L_Ti, L_Ti=1/(dlog(Ti)/dr) at r=0.5a
      Te_T0=1.0;          % Te/Ti ratio
      rmajor=1.3/0.48;    % R/a, a=minor radius
      rhos0=7.086026e-03; % rho_star=rho_i/a
      q0=0.854;           % q(r=0)
      q1=2.184;           % q(r=1)
      for j=1:jmax
        xx=(r(j)-0.5)/0.3;
        enhat(j)=exp(-0.3*R_Lne/rmajor*tanh(xx)); % density profile
        tehat(j)=Te_T0*exp(-0.3*R_Lte/rmajor*tanh(xx)); % Te profile
        that(j)=exp(-0.3*R_Lti/rmajor*tanh(xx)); % Ti profile
        pihat(j)=enhat(j)*that(j); % ion pressure profile
        tau(j)=that(j)/tehat(j);   % tau=Ti/Te
        rln(j)=-R_Lne/rmajor/cosh(xx)^2;  % a*dlog(n)/dr
        rlt(j)=-R_Lti/rmajor/cosh(xx)^2;  % a*dlog(Ti)/dr
        rlte(j)=-R_Lte/rmajor/cosh(xx)^2; % a*dlog(Te)/dr
        rltau(j)=rlt(j)-rlte(j);          % a*dlog(tau)/dr
        dpihat(j)=fac(j)*drm(j)*pihat(j)*(rln(j)+rlt(j)); % dPi/dr * dr
        dthat(j)=fac(j)*drm(j)*that(j)*rlt(j);            % dTi/dr * dr
        denhat(j)=fac(j)*drm(j)*enhat(j)*rln(j);          % dn/dr * dr
        q(j)=q0+q1*r(j)^2; % q-profile
      end
"""
