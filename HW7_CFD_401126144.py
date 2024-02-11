import numpy as np
import matplotlib.pyplot as plt

# Function defining f(u) for the given wave equation
def f_func(u):
    return 0.5 * u**2

# Function defining the initial condition
def initial_condition(x):
    return np.piecewise(x, [x < 0.35, (0.35 <= x) & (x <= 0.65), x > 0.65], [0, 1, 0])

# Godunov Flux Function
def f_Godunov(u1, u0):
    u = np.array([u0, u1])
    return np.min(f_func(u)) if u0 < u1 else np.max(f_func(u))

# Lax-Friedrichs Flux Function
def f_LF(u1, u0, lamda):
    return 0.5 * (f_func(u1) + f_func(u0)) - 0.5 * lamda**(-1) * (u1 - u0)

# Lax-Wendroff Flux Function
def f_LW(u1, u0, lamda):
    return 0.5 * (f_func(u1) + f_func(u0)) - 0.5 * lamda * ((u1 + u0) / 2)**2 * (u1 - u0)

# Roe Flux Function
def f_Roe(u1, u0):
    return min(f_func(u0), f_func(u1)) if u0 < u1 else max(f_func(u0), f_func(u1))

# REM - Phi Function
def phi(u1, u0, u_1):
    epsilon = 1e-20
    r = (u0 - u_1) / (u1 - u0 + epsilon)
    return max(0, min(1, r))

# Flux Blending Function using Lax-Friedrichs and Lax-Wendroff
def f_LFLW(u1, u0, u_1, lamda):
    return f_LF(u1, u0, lamda) + phi(u1, u0, u_1) * (f_LW(u1, u0, lamda) - f_LF(u1, u0, lamda))

# Upwind Flux Calculation
def upwind_flux(u, dt, dx):
    result = np.zeros_like(u)
    for i in range(1, len(u)):
        result[i] = u[i] - dt/dx * (f_func(u[i]) - f_func(u[i - 1]))
    return result

# Define the domain
x_L = 0
x_R = 1
L = x_R - x_L
N = 201      # number of grid points
dx = L / (N - 1)
x = np.linspace(x_L, x_R, N)
lamda = 0.5  # lamda = dt/dx
dt = lamda * dx
T = 0.2
t = np.arange(0, T + dt, dt)
M = len(t)

# Pre-allocate arrays for different methods
u0 = initial_condition(x)
u_g = np.zeros((N, M))
u_lf = np.zeros((N, M))
u_lw = np.zeros((N, M))
u_roe = np.zeros((N, M))
u_b = np.zeros((N, M))
u_u = np.zeros((N, M))

# Initial condition
u_g[:, 0] = u0
u_lf[:, 0] = u0
u_lw[:, 0] = u0
u_roe[:, 0] = u0
u_b[:, 0] = u0
u_u[:, 0] = u0

# Time-stepping loop for each method
for j in range(M - 1):
    for i in range(1, N - 1):
        # Godunov method
        u_g[i, j + 1] = u_g[i, j] - lamda * (f_Godunov(u_g[i + 1, j], u_g[i, j]) - f_Godunov(u_g[i, j], u_g[i - 1, j]))

        # Lax-Friedrichs method
        u_lf[i, j + 1] = u_lf[i, j] - lamda * (f_LF(u_lf[i + 1, j], u_lf[i, j], lamda) - f_LF(u_lf[i, j], u_lf[i - 1, j], lamda))

        # Lax-Wendroff method
        u_lw[i, j + 1] = max((u_lw[i, j] - lamda * (f_LW(u_lw[i + 1, j], u_lw[i, j], lamda) - f_LW(u_lw[i, j], u_lw[i - 1, j], lamda))),0)

        # Roe method
        u_roe[i, j + 1] = u_roe[i, j] - lamda * (f_Roe(u_roe[i + 1, j], u_roe[i, j]) - f_Roe(u_roe[i, j], u_roe[i - 1, j]))

        # Flux Blending method
        u_b[i, j + 1] = u_b[i, j] - lamda * (f_LFLW(u_b[i + 1, j], u_b[i, j], u_b[i - 1, j], lamda) - f_LFLW(u_b[i, j], u_b[i - 1, j], u_b[i - 1, j], lamda))

        # Upwind method
        u_u[:, j + 1] = upwind_flux(u_u[:, j], dt, dx)

# Plot the final solutions for all methods along with the initial condition
plt.plot(x, u0,           label='Initial Condition',          linewidth=2, color='black')
plt.plot(x, u_g[:, -1],   label='Godunov Flux Method',        linewidth=2, markersize=8, linestyle='-',  marker='o')
plt.plot(x, u_lf[:, -1],  label='Lax-Friedrichs Flux Method', linewidth=2, markersize=8, linestyle='--', marker='s')
plt.plot(x, u_lw[:, -1],  label='Lax-Wendroff Flux Method',   linewidth=2, markersize=6, linestyle='-.', marker='^')
plt.plot(x, u_roe[:, -1], label='Roe Flux Method',            linewidth=2, markersize=6, linestyle='--', marker='v')
plt.plot(x, u_b[:, -1],   label='Flux Blending Method',       linewidth=2, markersize=6, linestyle='-',  marker='D')
plt.plot(x, u_u[:, -1],   label='Upwind Flux Method',         linewidth=2, markersize=9, linestyle='--', marker='x')
plt.title('final solutions for all methods along with the initial condition in t=0.2')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.show()
