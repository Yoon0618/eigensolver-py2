# Vorticity Equation and Hasegawa-Mima Equation

## Vorticity Equation Derivation

We start with the vorticity equation $b\cdot\nabla\times V$. The momentum equation is given by:


$$\frac{\partial mnV}{\partial t}+\nabla\cdot P=en\left(E+\frac{1}{c}V\times B\right)+F+S_{m}$$

where the pressure tensor is $P=pI+\Pi+mnVV$.

Using the continuity equation, we obtain the convective form:


$$\left(\frac{\partial}{\partial t}+V\cdot\nabla\right)V=\frac{e}{m}\left(E+\frac{1}{c}V\times B\right)+\frac{1}{mn}(-\nabla p-\nabla\cdot\Pi+F+S_{m}-mVS_{n})$$



Taking the $\nabla\times$ operation , and using the continuity equation properties $\frac{\partial n}{\partial t}=S_{n}-\nabla\cdot(nV)=S_{n}-n\nabla\cdot V-V\cdot\nabla n$ which yields $\nabla\cdot V=\frac{S_{n}}{n}-\left(\frac{\partial}{\partial t}+V\cdot\nabla\right)\ln n$, we get:


$$\frac{\partial}{\partial t}\nabla\times V+\nabla\times(V\cdot\nabla V)=\frac{e}{m}\left[\nabla\times E+\frac{1}{c}\nabla\times(V\times B)\right]+\frac{1}{mn}\nabla \ln n\times\nabla p+\nabla\times\frac{1}{mn}(-\nabla\cdot\Pi+F+S_{m}-mVS_{n})$$



From the vector identity $V\cdot\nabla V=\frac{1}{2}\nabla V^{2}-V\times(\nabla\times V)$ and Faraday's law $\nabla\times E=-\frac{1}{c}\frac{\partial B}{\partial t}$, the equation can be rewritten in terms of vorticity $\omega=\nabla\times V$:


$$\frac{\partial\omega}{\partial t}-\nabla\times(V\times\omega)=-\left[\frac{\partial\omega_{c}}{\partial t}-\nabla\times(V\times\omega_{c})\right]+\frac{1}{mn}\nabla \ln n\times\nabla p+\nabla\times\frac{1}{mn}(f-\nabla\cdot\Pi_{g})$$


or equivalently,


$$\frac{\partial\Omega}{\partial t}-\nabla\times(V\times\Omega)=\frac{1}{mn}\nabla \ln n\times\nabla p+\nabla\times\frac{1}{mn}(f-\nabla\cdot\Pi_{g})$$



Here, $\omega=\nabla\times V$, the cyclotron frequency is $\omega_c = \frac{eB}{mc} = \nabla\times\frac{eA}{mc}$, and the absolute vorticity is $\Omega=\omega+\omega_c=\nabla\times\left(V+\frac{eA}{mc}\right)$.
The effective force $f$ is defined as $f = -\nabla\cdot\Pi_{cano} + F + S_m - mVS_n$.

Using the identity $\nabla\times(V\times\Omega)=-\Omega\nabla\cdot V+\Omega\cdot\nabla V-V\cdot\nabla\Omega$, we find:


$$\left(\frac{\partial}{\partial t}+V\cdot\nabla\right)\Omega+\Omega(\nabla\cdot V)=\Omega\cdot\nabla V+\frac{1}{mn}\nabla \ln n\times\nabla p+\nabla\times\frac{1}{mn}(f-\nabla\cdot\Pi_{g})$$



## Hasegawa-Mima Equation

For the Hasegawa-Mima limit:


$$\left(\frac{d}{dt}-\frac{d \ln n}{dt}\right)\Omega=\Omega\cdot\nabla V+\frac{1}{mn}\nabla \ln n\times\nabla p+\nabla\times\frac{1}{mn}(f-\nabla\cdot\Pi_{g})-\Omega\frac{S_{n}}{n}=0$$


Which implies:


$$\frac{1}{\Omega}\left(\frac{d}{dt}-\frac{d \ln n}{dt}\right)\Omega=0 \rightarrow \frac{d}{dt}\ln\frac{\Omega}{n}=0 \rightarrow \frac{d}{dt}\left[\ln\left(1+\frac{\omega}{\omega_{c}}\right)-\ln\frac{n}{\omega_{c}}\right]=0 \approx \frac{d}{dt}\left(\frac{\omega}{\omega_{c}}-\ln\frac{n}{B}\right)=0$$



For generalized vorticity:


$$\left[\frac{d}{dt}+(\nabla\cdot V)\right]\Omega=\Omega\cdot\nabla V+\frac{1}{mn}\nabla \ln n\times\nabla p+\nabla\times\frac{1}{mn}(f-\nabla\cdot\Pi_{g})$$



$$\frac{n}{\Omega}\left[\frac{d}{dt}+(\nabla\cdot V)\right]\Omega \approx \frac{d}{dt}\Omega_{general}+n\frac{d}{dt}\ln B$$



---

# Evolution of Generalized Vorticity

Using the relation for the diamagnetic vorticity $\Omega_p$:


$$\frac{d\Omega_p}{dt}+\Omega_p(\nabla\cdot V)-(\Omega_p\cdot\nabla)V=\frac{\partial\Omega_p}{\partial t}-\nabla\times(V\times\Omega_p)=\nabla\times\left[\frac{\partial V_p}{\partial t}-V\times(\nabla\times V_p)\right]=\nabla\times\left(\frac{\partial V_p}{\partial t}+V\cdot\nabla V_p-(\nabla V_p)\cdot V\right)=\nabla\times\frac{dV_p}{dt}-\nabla\times(\nabla V_p)V$$



We approximate the gyroviscous term (Ref. Chang, PF1992):


$$-\nabla\times\frac{\nabla\cdot\Pi_{g}}{mn}\approx\nabla\times\frac{dV_{p}}{dt}+\frac{1}{mn}(\nabla \ln n\times\nabla\chi)+\nabla\times(V_{p}\cdot\nabla V_{||})b \leftarrow mn\frac{dV_{p}}{dt}+\nabla\cdot\Pi_{g}\approx\nabla\chi-mn(V_{p}\cdot\nabla V_{||})b$$

where $\chi\sim\frac{\omega_{||}}{\omega_{c}}p$.

Substituting these, we obtain:


$$\frac{d\Omega^{\prime}}{dt}+\Omega^{\prime}(\nabla\cdot V)=\Omega^{\prime}\cdot\nabla V+\frac{1}{mn}\nabla \ln n\times\nabla(p+\chi)+\nabla\times\frac{f}{mn}+(V_{p}\cdot\nabla V_{||})\nabla\times b - b\times\nabla(V_{p}\cdot\nabla V_{||}) + \nabla\times(\nabla V_{p})\cdot V_{\perp}^{\prime}+\nabla\times(\nabla V_{p})\cdot V_{p}=0$$

Note that $\nabla\times(\nabla V_{p})V_{||}=0$.
Here, $\Omega^{\prime}=\Omega-\Omega_{p}$, $\Omega_{p}=\nabla\times V_{p}$, $V_{\perp}^{\prime} = V_{\perp}-V_{p}$, $V_{p} = \frac{b\times\nabla p}{mn\omega_{c}}$ is the diamagnetic velocity, and $\frac{d}{dt}=\frac{\partial}{\partial t}+V\cdot\nabla$.

Taking the $b\cdot$ operation:


$$b\cdot\frac{d\Omega^{\prime}}{dt}+\Omega_{||}^{\prime}(\nabla\cdot V)=b\cdot(\Omega^{\prime}\cdot\nabla)V+\frac{1}{mn}b\cdot\nabla \ln n\times\nabla(p+\chi)+b\cdot\nabla\times\frac{f}{mn}+(V_{p}\cdot\nabla V_{||})(b\cdot\nabla\times b)+b\cdot\nabla\times(\nabla V_{p})\cdot V_{\perp}^{\prime}$$



Expanding the time derivative:


$$\frac{d\Omega_{||}^{\prime}}{dt}-\omega_{\perp}^{\prime}\cdot\frac{db}{dt}+\Omega_{||}(\nabla\cdot V)=\Omega_{||}bb:\nabla V+b\cdot(\omega_{\perp}^{\prime}\cdot\nabla)V-\omega_{c}(V_{p}+V_{\chi}+V_{f})\cdot\nabla \ln n+\frac{b\cdot\nabla\times f}{mn}+(V_{p}\cdot\nabla V_{||})(b\cdot\nabla\times b)+b\cdot\nabla\times(\nabla V_{p})\cdot V_{\perp}^{\prime}$$

where $\Omega_{||}^{\prime}=b\cdot\Omega^{\prime}=\omega_{||}^{\prime}+\omega_{c}$, $\omega_{||}^{\prime}=b\cdot(\omega-\Omega_{p})$, $V_{\chi}=\frac{b\times\nabla\chi}{mn\omega_{c}}$, $\frac{1}{\omega_{c}}b\cdot\nabla\times\frac{f}{mn}=\frac{b\cdot\nabla\times f}{mn\omega_{c}}-V_{f}\cdot\nabla \ln n$, and $V_{f}=-\frac{b\times f}{mn\omega_{c}}$.

Applying the approximation $\frac{\omega_{\perp}^{\prime}}{\omega_{c}}\ll\frac{\omega_{||}^{\prime}}{\omega_{c}}\sim\frac{\omega_{s}}{\omega_{c}}\ll 1$:


$$\frac{d}{dt}(\omega_{||}^{\prime}+\omega_{c})-\omega_{\perp}^{\prime}\cdot\frac{db}{dt}+(\omega_{||}^{\prime}+\omega_{c})(\nabla\cdot V)=(\omega_{||}^{\prime}+\omega_{c})bb:\nabla V+b\cdot(\omega_{\perp}^{\prime}\cdot\nabla)V-\omega_{c}(V_{p}+V_{\chi}+V_{f})\cdot\nabla \ln n+\frac{b\cdot\nabla\times f}{mn}+(V_{p}\cdot\nabla V_{||})(b\cdot\nabla\times b)+b\cdot\nabla\times(\nabla V_{p})\cdot V_{\perp}^{\prime}$$



Dividing by $\omega_c$:


$$\frac{d}{dt}\frac{\omega_{||}^{\prime}}{\omega_{c}}+\left(1-\frac{\omega_{||}^{\prime}}{\omega_{c}}\right)\frac{d \ln \omega_{c}}{dt}+\nabla\cdot V=bb:\nabla V-(V_{p}+V_{\chi}+V_{f})\cdot\nabla \ln n+\frac{b\cdot\nabla\times f}{mn\omega_{c}}+\left(\frac{1}{\omega_{c}}V_{p}\cdot\nabla V_{||}\right)(b\cdot\nabla\times b)+\frac{1}{\omega_{c}}b\cdot\nabla\times(\nabla V_{p})V_{\perp}^{\prime}$$

where we used the identity $\frac{1}{\omega_{c}}\frac{d\omega_{||}}{dt}=\frac{d}{dt}\frac{\omega_{||}}{\omega_{c}}-\frac{\omega_{||}}{\omega_{c}}\frac{d \ln \omega_{c}}{dt}$.

The geometric terms scale as:


$$-\frac{1}{\omega_{c}}\frac{\omega_{c}\rho_{s}^{2}}{a}\frac{V_{||}}{\rho_{s}}\frac{1}{R}\sim\rho^{*}\nabla_{||}V_{||}\leftarrow V_{p}\sim\frac{\omega_{c}\rho_{s}^{2}}{a} \sim\frac{\delta V_{||}}{\omega_{c}}\frac{\delta V_{||}}{\rho_{s}}\frac{1}{R}\sim\frac{\rho_{s}}{R}\frac{\omega_{||}}{\omega_{c}}\omega_{||}$$

