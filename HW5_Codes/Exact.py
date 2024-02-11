import numpy as np
import matplotlib.pyplot as plt

# Define constants
a = 0.2
L = 1.0
N = 201
dx = L / (N - 1)
x = np.linspace(0, L, N)
T = 2.5

# Define initial condition
def u0(x):
    return np.sin(2 * np.pi * x)

# Define exact solution for the linear wave equation
def exact_solution(x, a, t):
    return u0((x - a * t) % L)

# Calculate the exact solution at t = T
u_exact = exact_solution(x, a, T)

# Calculate wave energy
def wave_energy(u, dx):
    return np.sum(0.5 * (u[1:] - u[:-1])**2) * dx

E_initial = wave_energy(u0(x), dx) / N
E_final = wave_energy(u_exact, dx) / N
print("Wave energy (initial):", round(E_initial, 10))
print("Wave energy (final):", round(E_final, 10))

# Plot the solutions
plt.plot(x, u0(x), label="Initial wave shape")
plt.plot(x, u_exact, label=f"Exact wave shape at t={T}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.show()
