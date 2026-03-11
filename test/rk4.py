import numpy as np

# 간단한 미방을 RK4로 풀어보자
# equation
# dx/dt = - k * sin(t) * x

x0 = 1
k = 1
dt = 0.1

ts = np.arange(0, 10+dt, dt)
xs = np.empty_like(ts)
xs[0] = x0

def f(t, x):
    return - k * np.sin(t) * x


for i, t in enumerate(ts[:-1]):
    x_now = xs[i]
    k1 = f(t, x_now)
    k2 = f(t+0.5*dt, x_now+0.5*k1*dt)
    k3 = f(t+0.5*dt, x_now+0.5*k2*dt)
    k4 = f(t+dt, x_now+k3*dt)
    x_next = x_now + 1/6 * (k1 + 2*k2 + 2*k3 + k4)*dt
    xs[i+1] = x_next

import matplotlib.pyplot as plt
plt.plot(ts, xs)
plt.sdtow()