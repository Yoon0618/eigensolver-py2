# Introduction to Eigenvalue Solver for ITG and Gyrofluid Simulation using BOUT++ GLF code in Core Region 

S. S. Kim, H. Jhang, P. H. Diamond, T. Rhee, G. Y. Park  
WCI Center for Fusion Theory, NFRI, Korea

X. Q. Xu, P. W. Xi, A. Dimits, M. Umansky  
LLNL

BOUT++ 2013 Workshop

## Part I. Eigenvalue Solver for ITG

## Governing equations for ITG (Ottaviani et al., PoP ’99)

### Vorticity equation

$$
\frac{\partial \Omega_i}{\partial t} + \mathbf{V}_E \cdot \nabla \Omega_i
= -n\nabla_{\parallel} V_{\parallel i}
+ n(\mathbf{V}_E + \mathbf{V}_{p_i})\cdot(\boldsymbol{\kappa} + \nabla \ln B)
+ n\mathbf{V}_{p_i}\cdot\nabla\left(\frac{n_1-\Omega_i}{n}\right)
- \mathbf{V}_E\cdot\nabla n_{eq}
+ D_c \Omega_i .
$$

**Definitions**

- Generalized vorticity: $\Omega_i = n_1 - n_{pol}$
- Polarization density: $n_{pol}=\frac{ne}{m_i\omega_{ci}^2}\nabla_{\perp}^2\phi$
- Adiabatic response: $n_1 = n_{eq}\frac{e\phi_1}{T_e}$
 
### Ion temperature equation

$$
\frac{\partial T_i}{\partial t} + \mathbf{V}_E \cdot \nabla T_i
= -\frac{2}{3}T_i\nabla_{\parallel}V_{\parallel i}
+ D_c T_i + D_{glf}T_i .
$$

### Ion parallel velocity equation

$$
\frac{\partial V_{\parallel i}}{\partial t} + \mathbf{V}_E \cdot \nabla V_{\parallel i}
= -\frac{e}{m_i}\nabla_{\parallel}\phi
- \frac{1}{m_i n}\nabla_{\parallel}p_i
+ D_c V_{\parallel i} .
$$

### Drift velocities

$$
\mathbf{V}_E = \frac{c}{B}\,\mathbf{b}\times\nabla\phi,
\qquad
\mathbf{V}_{p_i} = \frac{c}{neB}\,\mathbf{b}\times\nabla p_i .
$$

### Damping operators

$$
D_c F = \mu_1\nabla_{\perp}^2F - \mu_2\nabla_{\perp}^4F
\qquad \text{(viscous damping)},
$$

$$
D_{glf}T_i = -\sqrt{\frac{8T_{i,eq}}{\pi m_i}}\;\left|\nabla_{\parallel}\right|\;T_{i1}
\qquad \text{(Landau damping)}.
$$

## Linearized equations

### Linearized ITG system (normalized variables)

$$
\frac{\partial \tilde{\Omega}_i}{\partial t}
=
-\hat{n}_i \hat{\nabla}_{\parallel}\tilde{V}_{\parallel i}
+i\left(\hat{n}_i\omega_{di}\tilde{\phi}+\omega_{di}\tilde{p}_i\right)
+\left[\frac{\hat{p}_i}{\rho_*},\,\hat{\nabla}_{\perp}^{2}\tilde{\phi}\right]
-\left[\tilde{\Psi},\,\frac{\hat{n}_i}{\rho_*}\right]
+\bar{D}_c\,\tilde{\Omega}_i
$$

$$
\frac{\partial \tilde{T}_i}{\partial t}
+\left[\tilde{\Psi},\,\frac{\hat{T}_i}{\rho_*}\right]
=
-\frac{2}{3}\hat{T}_i\hat{\nabla}_{\parallel}\tilde{V}_{\parallel i}
+\bar{D}_c\,\tilde{T}_i
+\bar{D}_{glf}\,\tilde{T}_i
$$

$$
\frac{\partial \tilde{V}_{\parallel i}}{\partial t}
=
-\hat{\nabla}_{\parallel}\tilde{\phi}
-\frac{1}{\hat{n}_i}\hat{\nabla}_{\parallel}\tilde{p}_i
+\bar{D}_c\,\tilde{V}_{\parallel i}
$$

### Perturbation definitions and gyroaveraging

$$
\tilde{\Omega}_i=\tilde{n}_i-\hat{n}_i\hat{\nabla}_{\perp}^{2}\tilde{\phi},
\qquad
\tilde{n}_i=\frac{\hat{n}_i}{\hat{T}_e}\tilde{\phi}
$$

$$
\tilde{\Psi}
=\Gamma_0^{1/2}\tilde{\phi}
=\left(1-\frac{1}{2}\hat{\nabla}_{\perp}^{2}\right)^{-1}\tilde{\phi}
$$

`Ψ̃` is the gyroaveraged potential.

### Normalizations used

$$
F=F_{eq}+F_1,
\qquad
F=(\Omega,\,V_{\parallel},\,n,\,T,\,\phi),
\qquad
F_0=\left(n_0,\,c_{s0},\,n_0,\,T_0,\,\frac{T_0}{e}\right)
$$

$$
\hat{F}=\frac{F_{eq}}{F_0},
\qquad
\tilde{F}=\frac{F_1}{\rho_*F_0},
\qquad
\hat{\nabla}_{\perp}=\rho_*a\nabla_{\perp},
\qquad
\hat{\nabla}_{\parallel}=a\nabla_{\parallel},
\qquad
\rho=\frac{r}{a},
\qquad
\hat{t}=\frac{t}{t_0}
$$

$$
\rho_*=\frac{\rho_{s0}}{a},
\qquad
\rho_{s0}=\frac{c_{s0}}{\omega_{c0}},
\qquad
c_{s0}=\sqrt{\frac{T_0}{m_i}},
\qquad
\omega_{c0}=\frac{e_iB_0}{m_ic},
\qquad
t_0=\frac{a}{c_{s0}}
$$

$$
[f,g]\equiv \rho_{s0}^2\,\mathbf{b}\times\nabla f\cdot\nabla g
=\frac{\rho_*^2}{\rho}
\left(
\frac{\partial f}{\partial\rho}\frac{\partial g}{\partial\theta}
-\frac{\partial g}{\partial\rho}\frac{\partial f}{\partial\theta}
\right)
$$

$$
i\omega_{di}\tilde{F}
\equiv
2\rho_{s0}a\,\mathbf{b}\times\nabla\tilde{F}\cdot\nabla\ln B
=
2\rho_*\frac{a}{R}
\left(
\frac{1}{\rho}\cos\theta\,\frac{\partial\tilde{F}}{\partial\theta}
+\sin\theta\,\frac{\partial\tilde{F}}{\partial\rho}
\right)
$$

Note: `F0` is constant (not the value on the magnetic axis).

---

## From linearized PDEs to an eigenvalue problem

### Normal-mode substitution and operator notation

$$
\frac{\partial}{\partial t}\equiv -i\omega,
\qquad
\hat{\nabla}_{\perp}^{2}\tilde{\phi}\equiv L\tilde{\phi},
\qquad
\hat{\nabla}_{\parallel}\equiv -ik_{\parallel}
$$

$$
\tilde{\Omega}_i
=
\left(
\frac{\hat{n}_i}{\hat{T}_e}-\hat{n}_i\hat{\nabla}_{\perp}^{2}
\right)\tilde{\phi}
\equiv
M\tilde{\phi}
$$

$$
\tilde{\Psi}
=
\left(1-\frac{1}{2}\hat{\nabla}_{\perp}^{2}\right)^{-1}\tilde{\phi}
\equiv
J_0\tilde{\phi}
$$

$$
\left[\frac{\hat{F}}{\rho_*},\,\cdot\,\right]
=
\rho_{s0}^2\,\mathbf{b}\times\nabla\left(\frac{\hat{F}}{\rho_*}\right)\cdot\nabla
\equiv
-iG_{\hat{F}}
$$

$$
i\left(\hat{n}_i\omega_{di}\tilde{\phi}+\omega_{di}\tilde{p}_i\right)
\equiv
i\left(a_w\tilde{\phi}+b_w\tilde{T}_i\right)
$$

$$
\tilde{p}_i=\frac{\hat{p}_i}{\hat{T}_e}\tilde{\phi}+\hat{n}_i\tilde{T}_i
$$

### Frequency-domain linear system (operator form)

$$
\omega M\tilde{\phi}
=
\hat{n}_i k_{\parallel}\tilde{V}_{\parallel i}
-a_w\tilde{\phi}
-b_w\tilde{T}_i
+G_{\hat{p}_i}L\tilde{\phi}
+G_{\hat{n}_i}J_0\tilde{\phi}
+i\bar{D}_c\,M\tilde{\phi}
$$

$$
\omega\tilde{T}_i
=
G_{\hat{T}_i}J_0\tilde{\phi}
+\frac{2}{3}\hat{T}_i k_{\parallel}\tilde{V}_{\parallel i}
+i\bar{D}_c\,\tilde{T}_i
+i\bar{D}_{glf}\,\tilde{T}_i
$$

$$
\omega\tilde{V}_{\parallel i}
=
\left(1+\frac{\hat{T}_i}{\hat{T}_e}\right)k_{\parallel}\tilde{\phi}
+k_{\parallel}\tilde{T}_i
+i\bar{D}_c\,\tilde{V}_{\parallel i}
$$

(Equivalently, solving the first equation for `ω φ̃`)

$$
\omega\tilde{\phi}
=
M^{-1}\left(-a_w+G_{\hat{p}_i}L+G_{\hat{n}_i}J_0+i\bar{D}_c M\right)\tilde{\phi}
-M^{-1}b_w\tilde{T}_i
+M^{-1}\hat{n}_ik_{\parallel}\tilde{V}_{\parallel i}
$$

These linear operators (`L, M, J0, G, D`) can be represented as matrices once a functional basis is chosen.

---

## Final form of eigenvalue equation

### Vector–matrix form

$$
\omega\tilde{\phi}=A_{11}\tilde{\phi}+A_{12}\tilde{T}_i+A_{13}\tilde{V}_{\parallel i}
$$

$$
\omega\tilde{T}_i=A_{21}\tilde{\phi}+A_{22}\tilde{T}_i+A_{23}\tilde{V}_{\parallel i}
$$

$$
\omega\tilde{V}_{\parallel i}=A_{31}\tilde{\phi}+A_{32}\tilde{T}_i+A_{33}\tilde{V}_{\parallel i}
$$

$$
\mathbf{F}=
\begin{pmatrix}
\tilde{\phi}\\
\tilde{T}_i\\
\tilde{V}_{\parallel i}
\end{pmatrix},
\qquad
\mathbf{A}=
\begin{pmatrix}
A_{11}&A_{12}&A_{13}\\
A_{21}&A_{22}&A_{23}\\
A_{31}&A_{32}&A_{33}
\end{pmatrix}
$$

$$
\omega\mathbf{F}=\mathbf{A}\cdot\mathbf{F}
$$

A typical discretization is to expand in a basis `|k⟩`:

$$
\tilde{\phi}=\sum_k\tilde{\phi}_k|k\rangle
=
\begin{pmatrix}
\tilde{\phi}_1\\
\tilde{\phi}_2\\
\tilde{\phi}_3\\
\vdots
\end{pmatrix},
\qquad
|k\rangle\ \text{is a basis function}
$$

### Matrix elements

$$
A_{11}=M^{-1}\left(-a_w+G_{\hat{p}_i}L+G_{\hat{n}_i}J_0+i\bar{D}_c M\right),
\qquad
A_{12}=-M^{-1}b_w,
\qquad
A_{13}=M^{-1}\hat{n}_ik_{\parallel}
$$

$$
A_{21}=G_{\hat{T}_i}J_0,
\qquad
A_{22}=i\bar{D}_c+i\bar{D}_{glf},
\qquad
A_{23}=\frac{2}{3}\hat{T}_ik_{\parallel}
$$

$$
A_{31}=\left(1+\frac{\hat{T}_i}{\hat{T}_e}\right)k_{\parallel},
\qquad
A_{32}=k_{\parallel},
\qquad
A_{33}=i\bar{D}_c
$$

### Computing eigenpairs

$$
[\mathbf{F},\omega]=\mathrm{eig}(\mathbf{A}) \qquad (\text{in matlab})
$$

## Basis functions

### Mode expansion in concentric–circular geometry

$$
F(\rho,\theta,\zeta,t)=e^{-i\omega t}\sum_{k=mnp}F_k\,W_k(\rho)\,e^{i(m\theta-n\zeta)}.
$$

$$
k\equiv(m,n,p),\qquad k'\equiv(m',n',p').
$$

Bra–ket notation:

$$
\mathbf{F}=|F\rangle=e^{-i\omega t}\sum_k F_k\,|k\rangle,
\qquad
|k\rangle=|mnp\rangle=W_{mnp}(\rho)\,e^{i(m\theta-n\zeta)}.
$$

### Inner product and orthonormality

$$
\langle k|F\rangle \equiv \frac{1}{4\pi^2}\int_{0}^{1}\rho\,d\rho
\int_{0}^{2\pi}d\theta\int_{0}^{2\pi}d\zeta\;
W_{mnp}(\rho)e^{-i(m\theta-n\zeta)}\,F,
$$

$$
\langle k|k'\rangle=\delta_{kk'}.
$$

### Radial basis functions

Hermite basis:

$$
W_{mnp}(\rho)=\frac{1}{\sqrt{2\rho w_{mn}}\,\nu_p}\;H_p(x)\,e^{-x^2/2},
$$

$$
x=\frac{\rho-\rho_{mn}}{w_{mn}},
$$

$$
\rho_{mn}\ \text{is the radial position of the rational surface satisfying}\quad q=\frac{m}{n},
$$

$$
\nu_p=2^{p/2}\,\Gamma(p+1)^{1/2}\,\pi^{1/4}.
$$

Bessel basis:

$$
W_{mnp}(\rho)=\frac{\sqrt{2}}{J_{m+1}(\alpha_{mp})}\,J_m(\alpha_{mp}\rho),
\qquad
J_m(\alpha_{mp})=0.
$$

---

## Matrix (M, J0, Dc) involving Laplacian L

### Perpendicular Laplacian matrix element (Bessel basis)

$$
L_{kk'}=\left\langle k\left|\tilde{\nabla}_\perp^2\right|k'\right\rangle
=\rho_*^2\left\langle k\left|
\left(\frac{\partial^2}{\partial\rho^2}+\frac{1}{\rho}\frac{\partial}{\partial\rho}-\frac{m^2}{\rho^2}\right)
\right|k'\right\rangle
=-k_\perp^2\,\delta_{kk'}.
$$

$$
k_\perp=\rho_*\,\alpha_{mp}.
$$

### Derived matrices: M, J0, and Dc

$$
M_{kk'}\approx
\left.\frac{\hat n_i}{\hat T_e}\right|_{\rho=\rho_{mn}}\delta_{kk'}
-
\left.\hat n_i\right|_{\rho=\rho_{mn}}\,L_{kk'}.
$$

$$
J_{0,kk'}=\left(1-\frac{1}{2}L_{kk'}\right)^{-1}.
$$

$$
\bar D_{c,kk'}=\bar\mu_1\,L_{kk'}-\bar\mu_2\,L_{kk'}^2.
$$

(Operator form)

$$
L\tilde{\phi}=\tilde{\nabla}_\perp^2\tilde{\phi},
$$

$$
M\tilde{\phi}=\left(\frac{\hat n_i}{\hat T_e}-\hat n_i\tilde{\nabla}_\perp^2\right)\tilde{\phi}=\tilde{\Omega}_i,
$$

$$
J_0\tilde{\phi}=\left(1-\frac{1}{2}\tilde{\nabla}_\perp^2\right)^{-1}\tilde{\phi}=\tilde{\Psi},
$$

$$
\bar D_c F=\bar\mu_1\tilde{\nabla}_\perp^2F-\bar\mu_2\tilde{\nabla}_\perp^4F.
$$

---

## Matrix (k∥, Dglf) involving parallel wavenumber

### Parallel wavenumber matrix

$$
\hat F\,k_{\parallel,kk'}
\equiv i\left\langle k\left|\hat F\,\hat{\nabla}_\parallel\right|k'\right\rangle
=
\delta_{mm'}\delta_{nn'}\frac{a}{R}
\int_0^1 \hat F(\rho)\left(\frac{m}{q(\rho)}-n\right)
W_{mnp}(\rho)\,W_{mn p'}(\rho)\;\rho\,d\rho.
$$

### Landau damping matrix

$$
\bar D_{glf,kk'}
\equiv
-\left\langle k\left|\sqrt{\frac{8\hat T_i}{\pi}}\;\left|\hat{\nabla}_\parallel\right|\right|k'\right\rangle
=
-\delta_{mm'}\delta_{nn'}\sqrt{\frac{8}{\pi}\frac{a}{R}}
\int_0^1 \sqrt{\hat T_i(\rho)}\left|\frac{m}{q(\rho)}-n\right|
W_{mnp}(\rho)\,W_{mn p'}(\rho)\;\rho\,d\rho.
$$

---

## Matrix (G) for ExB and diamagnetic drifts

$$
G_{\hat F}\tilde{\phi}\equiv i\left[\frac{\hat F}{\rho_*},\,\tilde{\phi}\right].
$$

$$
\begin{aligned}
G_{\hat F,kk'}
&\equiv i\left\langle k\left|\left[\frac{\hat F}{\rho_*},\,\cdot\,\right]\right|k'\right\rangle
= i\rho_*\left\langle k\left|\frac{1}{\rho}\frac{\partial \hat F}{\partial\rho}\frac{\partial}{\partial\theta}\right|k'\right\rangle \\
&= -\delta_{mm'}\delta_{nn'}\,\rho_*\,m\int_0^1 \frac{\partial\hat F}{\partial\rho}\,W_{mnp}(\rho)\,W_{mn p'}(\rho)\,d\rho.
\end{aligned}
$$

---

## Matrix (aw, bw) involving curvature

### Generic curvature coupling (selection rule m → m±1)

$$
\hat\beta\,\omega_{di}\hat\alpha\,|k'\rangle
=
-2i\rho_*\frac{a}{R}\hat\beta\left(\frac{1}{\rho}\cos\theta\,\frac{\partial}{\partial\theta}+\sin\theta\,\frac{\partial}{\partial\rho}\right)\left(\hat\alpha\,|k'\rangle\right).
$$

$$
\hat\beta\,\omega_{di}\hat\alpha\,|k'\rangle
=
\rho_*\frac{a}{R}\left[
\left(\frac{m'}{\rho}\hat\beta\hat\alpha W_{k'}+\hat\beta\frac{\partial(\hat\alpha W_{k'})}{\partial\rho}\right)e^{i(m'-1)\theta}
+
\left(\frac{m'}{\rho}\hat\beta\hat\alpha W_{k'}-\hat\beta\frac{\partial(\hat\alpha W_{k'})}{\partial\rho}\right)e^{i(m'+1)\theta}
\right]e^{-in'\zeta}.
$$

$$
\begin{aligned}
\left\langle k\left|\hat\beta\,\omega_{di}\hat\alpha\right|k'\right\rangle
&=
\rho_*\frac{a}{R}\Bigg[
\Bigg\{(m+1)\int_0^1 \hat\beta\hat\alpha\,W_{mnp}\,W_{m+1,n,p'}\,d\rho
+\int_0^1 \hat\beta\,W_{mnp}\frac{\partial(\hat\alpha W_{m+1,n,p'})}{\partial\rho}\,\rho\,d\rho\Bigg\}\delta_{m+1,m'}\delta_{nn'} \\
&\qquad+
\Bigg\{(m-1)\int_0^1 \hat\beta\hat\alpha\,W_{mnp}\,W_{m-1,n,p'}\,d\rho
-\int_0^1 \hat\beta\,W_{mnp}\frac{\partial(\hat\alpha W_{m-1,n,p'})}{\partial\rho}\,\rho\,d\rho\Bigg\}\delta_{m-1,m'}\delta_{nn'}
\Bigg].
\end{aligned}
$$

### Definitions for the ITG curvature terms

$$
a_{w,kk'}
\equiv
\left\langle k\left|\left(\hat n_i\omega_{di}+\omega_{di}\frac{\hat p_i}{\hat T_e}\right)\right|k'\right\rangle
=
a_w^+(k,p')\,\delta_{m+1,m'}\delta_{nn'}
+
a_w^-(k,p')\,\delta_{m-1,m'}\delta_{nn'}.
$$

$$
b_{w,kk'}
\equiv
\left\langle k\left|\omega_{di}\hat n_i\right|k'\right\rangle
=
b_w^+(k,p')\,\delta_{m+1,m'}\delta_{nn'}
+
b_w^-(k,p')\,\delta_{m-1,m'}\delta_{nn'}.
$$

$$
a_w\tilde{\phi}=\hat n_i\omega_{di}\tilde{\phi}+\omega_{di}\frac{\hat p_i}{\hat T_e}\tilde{\phi},
\qquad
b_w\tilde{T}_i=\omega_{di}\hat n_i\tilde{T}_i.
$$

where

$$
\begin{aligned}
a_w^{\pm}(k,p')
&=\rho_*\frac{a}{R}\Bigg[
\int_0^1 \hat n_i\left\{(m\pm1)(1+\tau)\pm\left(\frac{\partial\ln\hat n_i}{\partial\rho}+\frac{\partial\ln\tau}{\partial\rho}\right)\tau\right\}
W_{mnp}\,W_{m\pm1,n,p'}\,d\rho \\
&\qquad\pm
\int_0^1 \hat n_i(1+\tau)\,W_{mnp}\,\frac{\partial W_{m\pm1,n,p'}}{\partial\rho}\;\rho\,d\rho
\Bigg],
\end{aligned}
$$

$$
\begin{aligned}
b_w^{\pm}(k,p')
&=\rho_*\frac{a}{R}\Bigg[
\int_0^1 \left(m\pm1+\frac{\partial\ln\hat n_i}{\partial\rho}\right)\hat n_i\,
W_{mnp}\,W_{m\pm1,n,p'}\,d\rho \\
&\qquad\pm
\int_0^1 \hat n_i\,W_{mnp}\,\frac{\partial W_{m\pm1,n,p'}}{\partial\rho}\;\rho\,d\rho
\Bigg],
\qquad
\tau=\frac{\hat T_i}{\hat T_e}.
\end{aligned}
$$

---

## Comparison of BOUT++ with eigenvalue solver

- In the eigenvalue solver, the following basis function is used:

$$
W_{mnp}=\frac{\sqrt{2}}{J_{m+1}(\alpha_{mp})}\,J_m\!\left(\frac{\alpha_{mp}}{a}r\right)\,e^{i(m\theta-n\zeta)},
\qquad
(p=1\sim N).
$$

- The fluid equations are projected onto the set of basis functions.
- Eigenvalues are obtained using MATLAB.
- BOUT++ results agree well with the eigenvalues, implying BOUT++ correctly solves the given equations.

### Figures

![Convergence test for eigenvalue solver (N = number of radial basis functions)](sandbox:/mnt/data/rendered_9_13/fig_convergence.png)

![Cyclone base case: dashed = eigenvalue solver, solid = BOUT++](sandbox:/mnt/data/rendered_9_13/fig_cyclone_full.png)

