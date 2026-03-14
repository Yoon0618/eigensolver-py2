**O Spectral 방법을 활용한 자기 유체 난류 시뮬레이션 코드 개발**

- TRB spectral 코드에 기반한 비선형 난류 코드 개발
  * TRB 코드는 기본적으로 Hermite (resonant modes) 및 Bessel (non-resonant modes) basis 함수들을 사용하여 3-field Landau 유체 모델 (Landau-fluid model)을 전개한 후 시간에 대해 explicit 방법 (midpoint leap-frog method)으로 풀어 냄으로써 매우 빠른 난류 시뮬레이션 가능.
  * 본 연구에서는 TRB코드와 동일한 basis 함수 및 시간 적분 방법을 사용하여 Beer-Hammett의 자이로 유체 모델 (gyro-fluid model)을 spectral 코드로 구현하고자 함. 이때 Beer-Hammett의 “3+1” 자이로 유체 모델을 TRB 코드의 3-field Landau 유체 모델에 맞추어 “3+0” 자이로 유체 모델로 축약함 (isotropic pressure). 따라서, 3-field Landau 유체 방정식은 “3+0” 자이로 유체 방정식의 부분집합 (subset)에 해당. 장파장 ($k_\perp \rho_i \ll 1$) 영역에서는 Beer-Hammett의 “3+0” 자이로 유체 방정식이 TRB 코드의 3-field Landau 유체 방정식으로 귀결됨을 확인함. “3+0” 자이로 유체 모델은 자이로 중심 밀도 ($n_i$) 방정식, 자이로 중심 평행 속도 ($\tilde{V}_{i \parallel}$) 방정식, 그리고 자이로 중심 압력 ($P_i$) 방정식 등으로 구성되는데, 구체적인 식은 다음과 같음 ($\nu$: 토로이달 닫음 계수).

$$ \frac{dn_i}{dt} = -n_0 B \nabla_{\parallel} \frac{\tilde{V}_{i \parallel}}{B} - \frac{2en_0}{T_{i0}} i\omega_d \Phi - \frac{2}{T_{i0}} i\omega_d \tilde{P}_i + \frac{en_0}{T_{i0}} \eta_i i\omega_* \delta\Phi - \frac{en_0}{T_{i0}} i\omega_d \delta\Phi $$

$$ \frac{d\tilde{V}_{i \parallel}}{dt} = - \frac{e}{m_i} \nabla_{\parallel} \Phi - \frac{1}{m_i n_0} \nabla_{\parallel} \tilde{P}_i - \frac{e}{m_i} \delta\Phi \nabla_{\parallel} \ln B - 4 i\omega_d \tilde{V}_{i \parallel} - 2 |\omega_d| \nu_5 \tilde{V}_{i \parallel} $$

$$ \frac{dP_i}{dt} = - \frac{2}{3} B \nabla_{\parallel} \frac{\tilde{Q}_{i \parallel}^{LD}}{B} - \frac{10}{3} n_0 i\omega_d \tilde{T}_i + \frac{5}{3} T_{i0} \frac{dn_i}{dt} + \frac{2}{3} e n_0 i\omega_* (\delta\Phi + \eta_i \delta\chi) - \frac{2}{3} e n_0 i\omega_d (2\delta\Phi + \delta\chi) - 2 n_0 |\omega_d| \nu \tilde{T}_i $$

여기서 $f = f_0 + \tilde{f}$, $d/dt = \partial/\partial t + \vec{V}_{\Phi} \cdot \nabla$, $\vec{V}_{f} = (1/B)\hat{b} \times \nabla f$, $\Phi = \Gamma_1 \phi = \Gamma_0^{1/2} \phi$, $\Gamma_0 = 1/(1+b)$,<br>
$b = -\rho_i^2 \nabla_{\perp}^2$, $\delta\Phi = \Gamma_2 \phi = b \partial\Gamma_1 / \partial b \phi$, $\delta\chi = \Gamma_3 \phi = (b \partial/\partial b)^2 \Gamma_1 \phi$, $i\omega_* = \frac{T_{i0}}{eB} \hat{b} \times \nabla \ln n_0 \cdot \nabla$,<br>
$i\omega_d = \frac{T_{i0}}{2eB} (\hat{b} \times \vec{\chi} + \hat{b} \times \nabla \ln B) \cdot \nabla$, $\eta_i = (n_0/T_{i0})(T_{i0}'/n_0')$, $\tilde{Q}_{i \parallel}^{LD} = -2n_0\sqrt{2/\pi} V_{th} (i k_{\parallel} \tilde{T}_i) / |k_{\parallel}|$.

- Beer & Hammett “3+0” gyrofluid 모델의 수치적 구현
Beer & Hammett “3+0” 자이로 유체 모델을 수치적으로 구현하기 위해 다음과 같이 자이로 평균 연산자 (Gyro-average operator) 및 자이로 유체 모델의 FLR (Finite Larmor Radius) 및 토로이달 항들을 기존의 TRB 코드에 추가하는 작업을 수행하고 있음. 새로이 추가되는 네가지 주요 항들을 코드에서 구현하는 기본적인 알고리즘은 다음과 같음.
  * **자이로 평균 연산자 ($\Gamma_1$)** : $\Phi = \Gamma_1 \phi$에서 자이로 평균 연산자의 분모를 양변에 곱하면 $(1+b/2)\Phi = \phi$의 라플라시안 방정식을 얻는데, 이를 TRB basis 공간에서 전개하면 삼중 대각 행렬식 (tridiagonal matrix equation)이 됨. 이 행렬의 역변환 (TRB 코드내의 invtridiag 루틴)을 취해서 최종적으로 자이로 평균된 퍼텐셜 $\Phi$를 구함.
  * **고차 (Higher-order) FLR 항 ($\propto \delta\Phi, \delta\chi$)** : $\Gamma_2, \Gamma_3$는 자이로 평균 연산자들 ($\Gamma_1, \Gamma_1^2, \Gamma_1^3$)의 차로 나타낼 수 있으므로 $\delta\Phi (= \Gamma_2 \phi), \delta\chi (= \Gamma_3 \phi)$는 자이로 평균 연산자의 역변환을 순차적으로 적용해서 구하게 됨.
  * **자이로키네틱 (gyrokinetic) Poisson 방정식** : 작은 Debye 길이 가정하에서 자이로키네틱 Poisson 방정식은 준중성 상태 방정식 $\tilde{n}_e + n_0(1-\Gamma_0)(e\phi/T_{i0}) = \bar{n}_i = \Gamma_1 \tilde{n}_i + (n_0/T_{i0})\Gamma_2 \tilde{T}_i$이 됨. 전자에 대해 단열반응을 가정하면 이 식은 퍼텐셜 $\phi$에 대해 자이로 평균 연산자의 라플라시안 방정식과 비슷한 형태의 방정식으로 변환. 따라서, 그 해도 삼중 대각 행렬의 역변환을 취해서 비슷한 방식으로 도출할 수 있음.
  * **토로이달 닫음 (toroidal closure) 항 ($\propto |\omega_d|$)** : $|\omega_d|$는 local wave vector $\vec{k}$를 가진 각각의 basis mode에 대해 $(\rho V_{ti}/R)|k_\theta \cos\theta + k_r \sin\theta|$로 표현될 수 있으므로, 각각의 basis mode의 $k_\theta = m/r$ & $k_r = -i\partial/\partial r \ln W_k(r)$을 치환해서 모든 basis mode에 대해 더해주면 임의의 함수에 대한 토로이달 닫음 항을 얻게 됨.

<br>

**향후 연구 계획**

O Spectral gyrofluid 코드를 이용한 ITG 시뮬레이션 연구
- 다른 코드와의 비교를 통한 ITG linear benchmark 수행
- 고차 FLR 항 및 토로이달 닫음 항에 의한 선형 및 비선형 난류 시뮬레이션 영향 연구

O AI ML 모델 개발을 위한 KSTAR 노심 MHD 시뮬레이션 DB 구축
- 현재까지 축적된 KSTAR H-mode 평형 DB 이용 [Y.S. Park et al., NF(2020)]
- DCON/RDCON 및 M3D-C1 코드를 통한 n=1 no-wall stability criteria $-\delta W$ 및 linear stability index $\Delta'$을 계산하고 이에 대한 MHD 시뮬레이션 DB 구축

<br>

**학회 참가 계획**

APS-DPP (미국, 일리노이, 시카고, '26.11.02 ~ '26.11.06)