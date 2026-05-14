import numpy as np

T = 30.0
dt = 7e-3
n_steps = int(round(T / dt))
ts = dt * np.arange(n_steps)

print(ts)