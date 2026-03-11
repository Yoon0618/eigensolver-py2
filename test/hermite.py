import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial

# x 범위
x_min, x_max = -6, 6
x = np.linspace(x_min, x_max, 800)
orders = range(6)  # n = 0,1,2,3,4,5

# 에르미트 함수(정규직교) 정의
def hermite_function(n, x):
    Hn = eval_hermite(n, x)
    norm = 1.0 / (np.pi**0.25 * np.sqrt(2.0**n * factorial(n)))
    return norm * Hn * np.exp(-x**2 / 2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1) 에르미트 다항식
for n in orders:
    axes[0].plot(x, eval_hermite(n, x), label=f"n={n}")
axes[0].set_title("Hermite Polynomials $H_n(x)$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("$H_n(x)$")
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(-100, 100)  # y축 범위를 제한하여 그래프가 더 잘 보이도록 한다.
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 2) 에르미트 함수
for n in orders:
    axes[1].plot(x, hermite_function(n, x), label=f"n={n}")
axes[1].set_title("Hermite Functions $\\psi_n(x)$")
axes[1].set_xlabel("x")
axes[1].set_ylabel("$\\psi_n(x)$")
axes[1].set_xlim(x_min, x_max)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()